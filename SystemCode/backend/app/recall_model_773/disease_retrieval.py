"""Reproducible positive-evidence Two-Tower baseline. Run --help for commands."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts' / 'disease_retrieval'
DATA = ROOT / 'Disease-Symptom Dataset' / 'Final_Augmented_dataset_Diseases_and_Symptoms.csv'


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA)
    features = raw.columns[1:].tolist()
    if raw.isna().any().any() or not np.isin(raw[features].values, [0, 1]).all():
        raise ValueError('Expected complete binary symptom data')
    unique = raw.drop_duplicates().copy()
    counts = unique.diseases.value_counts()
    counts.rename_axis('disease').to_csv(OUT / 'unique_disease_counts.csv')
    selected = unique[unique.diseases.isin(counts[counts >= 50].index)].copy()
    selected = selected[selected[features].sum(axis=1) > 0].reset_index(drop=True)
    diseases = sorted(selected.diseases.unique())
    x = selected[features].to_numpy(dtype=np.float32)
    y = selected.diseases.map({d: i for i, d in enumerate(diseases)}).to_numpy()
    groups = pd.util.hash_pandas_object(selected[features], index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train, held = next(splitter.split(x, y, groups))
    splitter2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=43)
    v, t = next(splitter2.split(x[held], y[held], groups[held]))
    val, test = held[v], held[t]
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(groups[a]) & set(groups[b])
    missing = {name: sorted(set(range(len(diseases))) - set(y[idx])) for name, idx in [('train', train), ('val', val), ('test', test)]}
    if missing['train']:
        raise ValueError('Some diseases lack training samples; inspect group distribution')
    np.savez_compressed(OUT / 'dataset.npz', x=x, y=y, train=train, val=val, test=test)
    nodes = pd.read_csv(ROOT / 'data/project_kg/nodes.csv')
    kg = nodes[nodes.node_type.eq('disease')]
    norm = lambda s: str(s).strip().lower().replace('_', ' ')
    lookup = {}
    for row in kg.itertuples():
        lookup.setdefault(norm(row.name), []).append(int(row.node_index))
    mapping = {d: lookup.get(norm(d), []) for d in diseases}
    save_json(OUT / 'vocabulary.json', {'symptoms': features, 'diseases': diseases, 'kg_exact_candidates': mapping})
    report = {'raw_rows': len(raw), 'duplicate_rows': len(raw)-len(unique), 'unique_rows': len(unique),
              'conflicting_symptom_patterns': int(unique.groupby(pd.util.hash_pandas_object(unique[features], index=False)).diseases.nunique().gt(1).sum()),
              'min_unique_rows_per_disease': 50, 'supported_diseases': len(diseases), 'selected_rows': len(selected),
              'symptoms': len(features), 'split_sizes': {n: len(i) for n, i in [('train', train), ('val', val), ('test', test)]},
              'missing_classes': {n: [diseases[i] for i in ids] for n, ids in missing.items()},
              'kg_exact_matched_diseases': sum(bool(v) for v in mapping.values()),
              'seed': 42, 'zero_semantics': 'unobserved; not explicit denial',
              'split_policy': 'deduplicated rows; identical symptom vectors grouped across labels; approximately 80/10/10',
              'provenance_limit': 'augmentation provenance unavailable; grouping cannot exclude all near-duplicate leakage'}
    save_json(OUT / 'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


class TwoTower(nn.Module):
    def __init__(self, symptoms, diseases):
        super().__init__()
        self.patient = nn.Sequential(nn.Linear(symptoms, 128), nn.ReLU(), nn.Linear(128, 64))
        self.disease = nn.Embedding(diseases, 64)

    def forward(self, x):
        return self.patient(x) @ self.disease.weight.T / 8


def subset(x, count, rng):
    if count is None:
        return x.copy()
    result = np.zeros_like(x)
    for i, row in enumerate(x):
        present = np.flatnonzero(row)
        chosen = rng.choice(present, min(count, len(present)), replace=False)
        result[i, chosen] = 1
    return result


def metrics(scores, y):
    ranks = 1 + (scores > scores[np.arange(len(y)), y, None]).sum(axis=1)
    # Conservatively count tied diseases ahead of the target.
    ranks += (scores == scores[np.arange(len(y)), y, None]).sum(axis=1) - 1
    hit5 = ranks <= 5
    return {**{f'Hits@{k}': float((ranks <= k).mean()) for k in [1, 3, 5]},
            'MRR': float((1 / ranks).mean()),
            'macro_Hits@5': float(pd.DataFrame({'y': y, 'hit': hit5}).groupby('y').hit.mean().mean())}


def score(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+1024])).numpy() for i in range(0, len(x), 1024)])


def train(epochs):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    rng = np.random.default_rng(42)
    data = np.load(OUT / 'dataset.npz')
    vocab = json.loads((OUT / 'vocabulary.json').read_text(encoding='utf-8'))
    x, y = data['x'], data['y']
    model = TwoTower(x.shape[1], len(vocab['diseases']))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    validation = subset(x[data['val']], 3, np.random.default_rng(123))
    best = -1
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0
        ids = rng.permutation(data['train'])
        for start in range(0, len(ids), 512):
            batch = ids[start:start+512]
            bx = torch.from_numpy(x[batch].copy())
            keep = torch.rand(len(batch), 1) * 0.8 + 0.2
            masked = bx * (torch.rand_like(bx) < keep)
            empty = masked.sum(dim=1) == 0
            masked[empty] = bx[empty]
            loss = nn.functional.cross_entropy(model(masked), torch.from_numpy(y[batch]).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        result = metrics(score(model, validation), y[data['val']])
        history.append({'epoch': epoch+1, 'loss': total/len(ids), 'validation_3_symptoms': result})
        print(json.dumps(history[-1]), flush=True)
        if result['MRR'] > best:
            best = result['MRR']
            torch.save(model.state_dict(), OUT / 'model.pt')
    model.load_state_dict(torch.load(OUT / 'model.pt', weights_only=True))
    # Train-only disease symptom profiles form the non-neural comparison.
    profiles = np.stack([x[data['train']][y[data['train']] == i].mean(axis=0) for i in range(len(vocab['diseases']))])
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-8)
    results = {}
    tx, ty = x[data['test']], y[data['test']]
    for count in [1, 2, 3, 5, None]:
        rows = []
        for seed in [101, 102, 103] if count is not None else [101]:
            observed = subset(tx, count, np.random.default_rng(seed))
            rows.append({'two_tower': metrics(score(model, observed), ty),
                         'profile_baseline': metrics(observed @ profiles.T, ty)})
        results[str(count or 'all')] = {name: {key: float(np.mean([r[name][key] for r in rows])) for key in rows[0][name]} for name in rows[0]}
    start = time.perf_counter()
    for _ in range(100):
        score(model, tx[:1])
    save_json(OUT / 'metrics.json', {'test': results, 'history': history, 'single_query_ms': (time.perf_counter()-start)*10,
                                   'partial_input_policy': 'up to N positives, three fixed seeds; no negative evidence',
                                   'selection': 'best validation MRR with up to three symptoms'})
    print(json.dumps(results, indent=2), flush=True)


def predict(symptoms, topk):
    vocab = json.loads((OUT / 'vocabulary.json').read_text(encoding='utf-8'))
    names = {s.lower(): i for i, s in enumerate(vocab['symptoms'])}
    unknown = [s for s in symptoms if s.lower() not in names]
    if unknown:
        raise ValueError(f'Unknown symptoms: {unknown}; use vocabulary.json standard names')
    if not symptoms:
        raise ValueError('At least one positive symptom is required')
    x = np.zeros((1, len(names)), dtype=np.float32)
    for s in symptoms:
        x[0, names[s.lower()]] = 1
    model = TwoTower(len(names), len(vocab['diseases']))
    model.load_state_dict(torch.load(OUT / 'model.pt', weights_only=True))
    scores = score(model, x)[0]
    result = []
    for i in np.argsort(-scores)[:topk]:
        name = vocab['diseases'][i]
        result.append({'disease_id': f'DS_{i:04d}', 'disease_name': name, 'score': float(scores[i]),
                       'kg_candidate_node_indices': vocab['kg_exact_candidates'][name]})
    print(json.dumps({'candidates': result, 'score_type': 'uncalibrated matching logit'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'predict'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--symptoms', nargs='+', default=[])
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1 or args.topk < 1:
        parser.error('epochs and topk must be positive')
    if args.command == 'prepare':
        prepare()
    elif args.command == 'train':
        train(args.epochs)
    else:
        predict(args.symptoms, args.topk)
