"""DDXPlus synthetic binary-symptom yes/no/unknown Two-Tower experiment."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from disease_retrieval import ROOT, TwoTower, metrics, save_json, score

OUT = ROOT / 'artifacts' / 'disease_retrieval_tristate'
SOURCE = ROOT / 'DDXPlus'
SOURCE_URL = 'https://github.com/mila-iqia/ddxplus'


def load_vocab():
    return json.loads((OUT / 'vocabulary.json').read_text(encoding='utf-8'))


def prepare(cap):
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((SOURCE / 'release_evidences.json').read_text(encoding='utf-8'))
    conditions = json.loads((SOURCE / 'release_conditions.json').read_text(encoding='utf-8'))
    # E_150 asks about the ability to pass stools, so absence of its code cannot
    # safely be described to a caller as absence of a symptom without review.
    features = sorted(k for k, v in evidence.items() if v['data_type'] == 'B'
                      and not v['is_antecedent'] and v['code_question'] == k and k != 'E_150')
    assert all(evidence[k]['default_value'] == 0 for k in features)
    diseases = sorted(v['condition_name'] for v in conditions.values())
    labels = {d: i for i, d in enumerate(diseases)}
    nodes = pd.read_csv(ROOT / 'data/project_kg/nodes.csv')
    kg = nodes[nodes.node_type.eq('disease')]
    norm = lambda s: str(s).strip().casefold().replace('_', ' ')
    mapping = {d: kg.loc[kg.name.map(norm).eq(norm(d)), 'node_index'].astype(int).tolist() for d in diseases}
    save_json(OUT / 'vocabulary.json', {'evidence_ids': features,
              'questions': {k: evidence[k]['question_en'] for k in features}, 'diseases': diseases,
              'icd10': {v['condition_name']: v['icd10-id'] for v in conditions.values()},
              'kg_exact_candidates': mapping, 'source': SOURCE_URL,
              'excluded_binary_symptoms': {'E_150': 'question polarity needs review'}})
    arrays = {}
    report = {'source': SOURCE_URL, 'diseases': len(diseases), 'binary_symptoms': len(features),
              'input_dimensions': 2*len(features), 'seed': 42,
              'negative_provenance': 'synthetic binary default states revealed as answers; not documented real patient denials',
              'original_413_model': 'preserved; this model supports a separate 49-disease vocabulary',
              'sampling': 'lowest seeded random priorities per disease within each official split', 'splits': {}}
    earlier_states = set()
    earlier_binary = set()
    for split, file, limit, seed in [('train', 'release_train_patients', cap, 42),
                                    ('val', 'release_validate_patients', max(1, cap//6), 43),
                                    ('test', 'release_test_patients', max(1, cap//6), 44)]:
        rng = np.random.default_rng(seed)
        kept = None
        total = 0
        for chunk in pd.read_csv(SOURCE / file, usecols=['PATHOLOGY', 'EVIDENCES'], chunksize=100000):
            total += len(chunk)
            chunk['priority'] = rng.random(len(chunk))
            pool = pd.concat([kept, chunk], ignore_index=True) if kept is not None else chunk
            kept = pool.sort_values('priority').groupby('PATHOLOGY', sort=False).head(limit)
        rows, ys, keys = [], [], []
        for row in kept.itertuples():
            items = ast.literal_eval(row.EVIDENCES)
            if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
                raise ValueError('Invalid EVIDENCES')
            key = hashlib.sha256(json.dumps(sorted(items)).encode()).hexdigest()
            present = set(items)
            rows.append([int(k in present) for k in features])
            ys.append(labels[row.PATHOLOGY])
            keys.append(key)
        frame = pd.DataFrame({'key': keys, 'y': ys})
        dedup = ~frame.duplicated(['key', 'y'])
        cross = frame.key.isin(earlier_states)
        selected = np.flatnonzero((dedup & ~cross).to_numpy())
        x = np.asarray(rows, dtype=np.uint8)[selected]
        y = np.asarray(ys, dtype=np.int64)[selected]
        states = set(frame.iloc[selected].key)
        assert not states & earlier_states
        earlier_states.update(states)
        binary = [hashlib.sha256(v.tobytes()).hexdigest() for v in x]
        overlap = sum(v in earlier_binary for v in binary)
        earlier_binary.update(binary)
        missing = sorted(set(range(len(diseases))) - set(y))
        if missing and split == 'train':
            raise ValueError(f'{split} missing classes: {missing}; increase cap')
        arrays[f'{split}_x'], arrays[f'{split}_y'] = x, y
        report['splits'][split] = {'official_rows': total, 'sampled_rows': len(frame),
                                  'within_sample_duplicates': int((~dedup).sum()),
                                  'prior_split_full_evidence_overlap_removed': int((dedup & cross).sum()),
                                  'retained_rows': len(x), 'missing_classes': [diseases[i] for i in missing],
                                  'projected_binary_pattern_overlap_rows': overlap,
                                  'zero_selected_positive_rows': int((x.sum(axis=1)==0).sum())}
        print(split, report['splits'][split], flush=True)
    np.savez_compressed(OUT / 'dataset.npz', **arrays)
    report['projection_overlap_note'] = 'Distinct complete evidence states may project to identical 95-symptom vectors; retained and reported as limited-input ambiguity.'
    save_json(OUT / 'audit.json', report)


def observed(truth, positive_count, negative_count, seed):
    """Reveal simulated answers; all unrevealed features stay unknown."""
    rng = np.random.default_rng(seed)
    pos = np.zeros_like(truth, dtype=np.float32)
    neg = np.zeros_like(truth, dtype=np.float32)
    for i, row in enumerate(truth):
        yes = np.flatnonzero(row)
        no = np.flatnonzero(row == 0)
        pos[i, rng.choice(yes, min(positive_count, len(yes)), replace=False)] = 1
        neg[i, rng.choice(no, min(negative_count, len(no)), replace=False)] = 1
    return np.concatenate([pos, neg], axis=1)


def training_observation(truth, rng):
    n, s = truth.shape
    pos_count = rng.integers(1, 6, size=(n, 1))
    neg_count = rng.integers(0, 11, size=(n, 1))
    pos_keys = rng.random((n, s))
    pos_keys[truth == 0] = 2
    neg_keys = rng.random((n, s))
    neg_keys[truth == 1] = 2
    pos_ranks = np.argsort(np.argsort(pos_keys, axis=1), axis=1)
    neg_ranks = np.argsort(np.argsort(neg_keys, axis=1), axis=1)
    pos = ((pos_ranks < pos_count) & (truth == 1)).astype(np.float32)
    neg = ((neg_ranks < neg_count) & (truth == 0)).astype(np.float32)
    return np.concatenate([pos, neg], axis=1)


def train(epochs):
    torch.set_num_threads(4)
    data, vocab = np.load(OUT / 'dataset.npz'), load_vocab()
    s = len(vocab['evidence_ids'])
    models, histories = {}, {}
    for mode in ['positive_only', 'tristate']:
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        model = TwoTower(s if mode == 'positive_only' else s*2, len(vocab['diseases']))
        opt = torch.optim.Adam(model.parameters(), lr=0.002)
        val = observed(data['val_x'], 3, 5, 123)
        if mode == 'positive_only':
            val = val[:, :s]
        best, history = -1, []
        for epoch in range(epochs):
            model.train()
            ids = rng.permutation(len(data['train_y']))
            total = 0
            for start in range(0, len(ids), 512):
                idx = ids[start:start+512]
                inp = training_observation(data['train_x'][idx], rng)
                if mode == 'positive_only':
                    inp = inp[:, :s]
                loss = nn.functional.cross_entropy(model(torch.from_numpy(inp)), torch.from_numpy(data['train_y'][idx]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())*len(idx)
            result = metrics(score(model, val), data['val_y'])
            history.append({'epoch': epoch+1, 'loss': total/len(ids), 'validation': result})
            print(mode, json.dumps(history[-1]), flush=True)
            if result['MRR'] > best:
                best = result['MRR']
                torch.save(model.state_dict(), OUT / f'{mode}.pt')
        model.load_state_dict(torch.load(OUT / f'{mode}.pt', weights_only=True))
        models[mode], histories[mode] = model, history
    results = {}
    truth, y = data['test_x'], data['test_y']
    for count in [0, 1, 3, 5, 10]:
        runs = []
        for seed in [101, 102, 103]:
            # Separate RNG draws keep positive input fixed across negative counts.
            pos = observed(truth, 3, 0, seed)[:, :s]
            neg = observed(truth, 0, count, seed+1000)[:, s:]
            combined = np.concatenate([pos, neg], axis=1)
            runs.append({'positive_only': metrics(score(models['positive_only'], pos), y),
                         'tristate': metrics(score(models['tristate'], combined), y)})
        results[str(count)] = {m: {k: float(np.mean([r[m][k] for r in runs])) for k in runs[0][m]} for m in runs[0]}
    # Paired ablation: same trained weights, same positives; only erase negatives.
    pos = observed(truth, 3, 0, 101)[:, :s]
    neg = observed(truth, 0, 5, 1101)[:, s:]
    full_scores = score(models['tristate'], np.concatenate([pos, neg], axis=1))
    erased_scores = score(models['tristate'], np.concatenate([pos, np.zeros_like(neg)], axis=1))
    top_changed = float((full_scores.argmax(axis=1) != erased_scores.argmax(axis=1)).mean())
    save_json(OUT / 'metrics.json', {'test_up_to_3_positives_by_negative_count': results,
              'history': histories, 'paired_ablation_seed101': {
                  'with_5_negatives': metrics(full_scores, y), 'negatives_erased': metrics(erased_scores, y),
                  'top1_changed_fraction': top_changed},
              'protocol': 'uniform random reveals of synthetic binary answers; same positives across negative budgets; three input seeds',
              'limitations': 'Different 49-disease dataset; no direct comparison to original 413-disease model; no real-world validation'})
    write_report(results, metrics(erased_scores, y), metrics(full_scores, y), top_changed)
    print(json.dumps(results, indent=2), flush=True)


def write_report(results, ablated, with_negative, top_changed):
    lines = ['# 三态召回实验结果', '',
             '输入固定为最多三个阳性症状，逐步加入随机模拟阴性答案。三个固定输入种子取平均。', '',
             '| 阴性答案数 | 三态 Hits@1 | 三态 Hits@5 | 三态 MRR | 阳性对照 Hits@1 |',
             '| --- | --- | --- | --- | --- |']
    for count, result in results.items():
        tri, baseline = result['tristate'], result['positive_only']
        lines.append(f"| {count} | {tri['Hits@1']:.2%} | {tri['Hits@5']:.2%} | {tri['MRR']:.4f} | {baseline['Hits@1']:.2%} |")
    lines += ['', '同一个三态模型的配对消融（固定输入种子 101）：', '',
              f"- 移除阴性答案：Hits@1 {ablated['Hits@1']:.2%}，MRR {ablated['MRR']:.4f}。",
              f"- 加入五个阴性答案：Hits@1 {with_negative['Hits@1']:.2%}，MRR {with_negative['MRR']:.4f}。",
              f'- Top-1 疾病改变比例：{top_changed:.2%}。', '',
              '模型支持 49 种疾病；严格清洗后的测试仅覆盖 48 种，喉痉挛无独立测试病例。',
              '阴性证据来自 DDXPlus 的合成患者默认状态，不是真实问诊否认记录。',
              '完整证据已去重和跨集合排重，但投影到 95 个二元症状后仍有大量相同输入，详见 audit.json。',
              '每疾病抽样上限改变了疾病分布；不能与原 413 疾病模型直接比较，也不代表现实临床准确率。',
              '不同患者加入阴性证据后不保证正确疾病排名都提高；这里只检验总体效果。', '',
              '本轮随机阴性答案实验未显示总体召回提升：加入五个阴性答案后的 Hits@1 和 MRR 均略低于未输入阴性的同模型结果。',
              '因此三态输入和推理支持已完成，但阴性证据的有效利用仍需优化，不能宣称正确性或准确率已因阴性输入提高。', '',
              '[运行与数据说明](DISEASE_RETRIEVAL_TRISTATE_README.md)']
    (ROOT / 'DISEASE_RETRIEVAL_TRISTATE_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


ALIASES = {'fever': 'E_91', '发热': 'E_91', '发烧': 'E_91',
           'cough': 'E_201', '咳嗽': 'E_201', 'shortness of breath': 'E_66', '呼吸困难': 'E_66',
           'sore throat': 'E_97', '咽痛': 'E_97', 'diarrhea': 'E_51', '腹泻': 'E_51'}


def encode(present, absent, vocab):
    ids = vocab['evidence_ids']
    questions = {v.casefold(): k for k, v in vocab['questions'].items()}
    def resolve(items):
        result = set()
        for item in items:
            key = item.strip()
            key = ALIASES.get(key.casefold(), questions.get(key.casefold(), key))
            if key not in ids:
                raise ValueError(f'Unsupported symptom: {item}; use evidence IDs or vocabulary questions')
            result.add(key)
        return result
    yes, no = resolve(present), resolve(absent)
    if yes & no:
        raise ValueError(f'Contradictory present/absent evidence: {sorted(yes & no)}')
    if not yes and not no:
        raise ValueError('At least one known symptom state is required')
    vector = np.zeros((1, 2*len(ids)), dtype=np.float32)
    for key in yes:
        vector[0, ids.index(key)] = 1
    for key in no:
        vector[0, len(ids)+ids.index(key)] = 1
    return vector, sorted(yes), sorted(no)


class DiseaseRetriever:
    """Load once, call retrieve repeatedly. Explicit lists or evidence IDs only."""
    def __init__(self, model_variant='cross_entropy'):
        self.vocab = load_vocab()
        pair_dir = ROOT / 'artifacts' / 'disease_retrieval_pairwise'
        if model_variant == 'selected':
            model_variant = json.loads((pair_dir / 'selection.json').read_text(encoding='utf-8'))['selected_model']
        paths = {'cross_entropy': OUT / 'tristate.pt',
                 'pairwise': pair_dir / 'pairwise.pt', 'ce_pairwise': pair_dir / 'ce_pairwise.pt'}
        if model_variant not in paths:
            raise ValueError(f'Unsupported model variant: {model_variant}')
        self.model_variant = model_variant
        self.model = TwoTower(2*len(self.vocab['evidence_ids']), len(self.vocab['diseases']))
        self.model.load_state_dict(torch.load(paths[model_variant], weights_only=True))
        self.model.eval()

    def retrieve(self, present_symptoms=(), absent_symptoms=(), topk=5):
        if not isinstance(topk, int) or topk < 1:
            raise ValueError('topk must be a positive integer')
        if isinstance(present_symptoms, str) or isinstance(absent_symptoms, str):
            raise ValueError('Symptoms must be lists, not free text')
        vector, yes, no = encode(present_symptoms, absent_symptoms, self.vocab)
        scores = score(self.model, vector)[0]
        candidates = []
        for i in np.argsort(-scores)[:topk]:
            disease = self.vocab['diseases'][i]
            candidates.append({'disease_id': f'DDX_{i:03d}', 'disease_name': disease,
                'icd10': self.vocab['icd10'][disease], 'score': float(scores[i]),
                'kg_candidate_node_indices': self.vocab['kg_exact_candidates'][disease]})
        return {'model': f'ddxplus_tristate_{self.model_variant}', 'supported_diseases': len(self.vocab['diseases']),
                'known_present': yes, 'known_absent': no, 'candidates': candidates,
                'score_type': 'uncalibrated matching logit'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'predict', 'report'])
    parser.add_argument('--cap-per-disease', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--present', nargs='*', default=[])
    parser.add_argument('--absent', nargs='*', default=[])
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--model', choices=['cross_entropy', 'pairwise', 'ce_pairwise', 'selected'], default='cross_entropy')
    args = parser.parse_args()
    if min(args.cap_per_disease, args.epochs, args.topk) < 1:
        parser.error('cap, epochs and topk must be positive')
    if args.command == 'prepare':
        prepare(args.cap_per_disease)
    elif args.command == 'train':
        train(args.epochs)
    elif args.command == 'report':
        result = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
        ablation = result['paired_ablation_seed101']
        write_report(result['test_up_to_3_positives_by_negative_count'], ablation['negatives_erased'],
                     ablation['with_5_negatives'], ablation['top1_changed_fraction'])
    else:
        print(json.dumps(DiseaseRetriever(args.model).retrieve(args.present, args.absent, args.topk), ensure_ascii=False, indent=2))
