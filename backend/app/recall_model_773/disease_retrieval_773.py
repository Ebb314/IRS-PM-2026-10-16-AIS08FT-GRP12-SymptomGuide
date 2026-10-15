"""All 773 original disease labels, 377 positive/unknown symptoms, fixed group split."""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from disease_retrieval import ROOT, DATA, TwoTower, save_json, score, subset

OUT = ROOT / 'artifacts' / 'disease_retrieval_773'
CATALOG = ROOT / 'data/unified_catalog'


def evaluate(scores, y):
    own = scores[np.arange(len(y)), y, None]
    ranks = 1 + (scores > own).sum(axis=1) + (scores == own).sum(axis=1)-1
    rows = pd.DataFrame({'class': y, 'rr': 1/ranks, 'hit5': ranks <= 5})
    return {**{f'Hits@{k}': float((ranks <= k).mean()) for k in [1,3,5,10,20]},
            'MRR': float((1/ranks).mean()), 'macro_MRR': float(rows.groupby('class').rr.mean().mean()),
            'macro_Hits@5': float(rows.groupby('class').hit5.mean().mean()),
            'evaluated_classes': int(rows['class'].nunique()), 'rows': len(y)}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA)
    features = raw.columns[1:].tolist()
    if raw.isna().any().any() or not np.isin(raw[features].to_numpy(), [0,1]).all():
        raise ValueError('Expected complete binary features')
    unique = raw.drop_duplicates().reset_index(drop=True)
    diseases = sorted(unique.diseases.unique())
    if len(diseases) != 773 or len(features) != 377:
        raise ValueError('Source dimensions changed; explicit scope review required')
    x = unique[features].to_numpy(dtype=np.float32)
    if (x.sum(axis=1)==0).any():
        raise ValueError('Zero-positive cases need an explicit prior-only policy')
    y = unique.diseases.map({d:i for i,d in enumerate(diseases)}).to_numpy(dtype=np.int64)
    hashes = pd.util.hash_pandas_object(unique[features], index=False).to_numpy()
    groups, group_keys = pd.factorize(hashes, sort=True)
    rng = np.random.default_rng(42)
    draws = rng.random(len(group_keys))
    assignments = np.where(draws < 0.8, 0, np.where(draws < 0.9, 1, 2))
    sizes = np.bincount(groups)
    moved = []
    # Preserve group isolation, including conflicting labels sharing one vector.
    # Move one complete group to training when a disease has no training rows.
    for label in rng.permutation(len(diseases)):
        candidates = np.unique(groups[y==label])
        if not (assignments[candidates]==0).any():
            chosen = int(candidates[np.argmin(sizes[candidates])])
            moved.append({'disease': diseases[label], 'group_rows': int(sizes[chosen]),
                          'old_split': 'val' if assignments[chosen]==1 else 'test'})
            assignments[chosen] = 0
    train, val, test = [np.flatnonzero(assignments[groups]==i) for i in range(3)]
    assert set(y[train]) == set(range(773))
    for a,b in [(train,val),(train,test),(val,test)]:
        assert not set(groups[a]) & set(groups[b])
    np.savez_compressed(OUT/'dataset.npz', x=x, y=y, groups=groups, train=train, val=val, test=test)
    mappings = json.loads((CATALOG/'original_model_mapping.json').read_text(encoding='utf-8'))
    assert set(mappings['diseases']) == set(diseases)
    assert set(mappings['symptoms']) == set(features)
    save_json(OUT/'vocabulary.json', {'version':'original_773_v1','diseases': diseases,'symptoms': features,
        'disease_mapping': mappings['diseases'], 'symptom_mapping': mappings['symptoms'],
        'zero_semantics':'unknown/unobserved; not explicit denial'})
    count_rows = []
    for i,disease in enumerate(diseases):
        record = {'disease_id':f'DSA_{i:04d}','disease_name':disease,'unique_records':int((y==i).sum())}
        for name, idx in [('train',train),('val',val),('test',test)]:
            record[name+'_rows'] = int((y[idx]==i).sum())
        record['independently_tested'] = record['test_rows'] > 0
        count_rows.append(record)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT/'disease_support.csv',index=False)
    report = {'raw_rows':len(raw),'unique_rows':len(unique),'duplicate_rows':len(raw)-len(unique),
        'supported_diseases':773,'symptoms':377,'min_sample_threshold':None,
        'split_sizes':{name:len(idx) for name,idx in [('train',train),('val',val),('test',test)]},
        'split_class_counts':{name:int(np.unique(y[idx]).size) for name,idx in [('train',train),('val',val),('test',test)]},
        'train_only_classes':counts.loc[(counts.val_rows==0)&(counts.test_rows==0),'disease_name'].tolist(),
        'no_test_classes':counts.loc[counts.test_rows==0,'disease_name'].tolist(),
        'diseases_with_one_unique_record':int((counts.unique_records==1).sum()),
        'diseases_with_under_10_unique_records':int((counts.unique_records<10).sum()),
        'coverage_moves':moved,'zero_semantics':'unknown; original zeros never used as denials',
        'split_policy':'seeded 80/10/10 assignment by complete symptom vector; train coverage repaired by moving entire groups',
        'limitations':'Distinct records are not verified independent patients; augmentation provenance unavailable; no negatives or mixed-DDXPlus patient training.',
        'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest()}
    save_json(OUT/'audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['train_only_classes','no_test_classes','coverage_moves']},indent=2),flush=True)


def train(epochs):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    with np.load(OUT/'dataset.npz') as archive:
        data = {key:archive[key] for key in archive.files}
    x,y = data['x'],data['y']
    model = TwoTower(377,773)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
    counts = np.bincount(y[data['train']],minlength=773)
    weights = np.clip(np.sqrt(len(data['train'])/(773*counts)),0.25,5.0)
    weights = weights/weights.mean()
    weight_tensor = torch.tensor(weights,dtype=torch.float32)
    validation = subset(x[data['val']],3,np.random.default_rng(123))
    best,history = -1,[]
    config = {'model':'TwoTower 377->128->64, 773x64 disease embedding','epochs':epochs,
        'learning_rate':0.002,'batch_size':512,'seed':42,
        'loss':'weighted full softmax Cross-Entropy',
        'class_weight':'clip(sqrt(train_rows/(773*class_train_rows)),0.25,5), normalized to mean 1',
        'selection':'validation macro_MRR with up to 3 positive symptoms',
        'masking':'per-case keep probability uniform 0.2..1; empty masks restore original positives',
        'input':'positive and unknown only; explicit denials rejected'}
    save_json(OUT/'config.json',config)
    np.save(OUT/'class_weights.npy',weights)
    for epoch in range(epochs):
        model.train()
        ids = rng.permutation(data['train'])
        total = 0
        for start in range(0,len(ids),512):
            idx = ids[start:start+512]
            bx = torch.from_numpy(x[idx].copy())
            keep = torch.rand(len(idx),1)*0.8+0.2
            masked = bx*(torch.rand_like(bx)<keep)
            empty = masked.sum(dim=1)==0
            masked[empty] = bx[empty]
            loss = nn.functional.cross_entropy(model(masked),torch.from_numpy(y[idx]),weight=weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()*len(idx)
        result = evaluate(score(model,validation),y[data['val']])
        history.append({'epoch':epoch+1,'loss':total/len(ids),'validation':result})
        print(json.dumps(history[-1]),flush=True)
        if result['macro_MRR']>best:
            best = result['macro_MRR']
            torch.save(model.state_dict(),OUT/'model.pt')
    model.load_state_dict(torch.load(OUT/'model.pt',weights_only=True))
    results = {}
    tx,ty = x[data['test']],y[data['test']]
    # Train-only symptom profile comparison; all 773 output candidates present.
    profiles = np.stack([x[data['train']][y[data['train']]==i].mean(axis=0) for i in range(773)])
    profiles /= np.maximum(np.linalg.norm(profiles,axis=1,keepdims=True),1e-8)
    detail_rows = []
    for count in [1,2,3,5,None]:
        runs = []
        for seed in ([101,102,103] if count is not None else [101]):
            observed = subset(tx,count,np.random.default_rng(seed))
            scores = score(model,observed)
            runs.append({'two_tower':evaluate(scores,ty),'profile_baseline':evaluate(observed@profiles.T,ty)})
            if count==3 and seed==101:
                own = scores[np.arange(len(ty)),ty,None]
                ranks = 1+(scores>own).sum(axis=1)+(scores==own).sum(axis=1)-1
                for label in np.unique(ty):
                    ids = ty==label
                    detail_rows.append({'disease_index':int(label),'test_rows':int(ids.sum()),
                        'Hits@5':float((ranks[ids]<=5).mean()),'MRR':float((1/ranks[ids]).mean())})
        results[str(count or 'all')] = {m:{k:float(np.mean([r[m][k] for r in runs])) for k in runs[0][m]} for m in runs[0]}
    pd.DataFrame(detail_rows).to_csv(OUT/'per_disease_test_3_symptoms.csv',index=False)
    save_json(OUT/'metrics.json',{'test':results,'history':history,'selection':config['selection'],
        'evaluation_scope':'Only held-out labels with test rows; all 773 diseases remain retrieval candidates',
        'input_seeds':[101,102,103],'tie_policy':'conservative rank; insufficient positives use up to N'})
    audit = json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    lines = ['# 原数据集全量 773 疾病训练结果','',
        f"模型输出 773 种疾病，输入 377 种症状。去重后 {audit['unique_rows']:,} 条记录全部保留，不设置疾病最小样本数门槛。",'',
        f"训练/验证/测试记录数：{len(data['train']):,}/{len(data['val']):,}/{len(data['test']):,}。",
        f"对应疾病覆盖：773/{audit['split_class_counts']['val']}/{audit['split_class_counts']['test']}。",
        f"仅训练集覆盖的类别：{len(audit['train_only_classes'])}；无独立测试记录的类别：{len(audit['no_test_classes'])}。",'',
        '| 已知阳性症状数 | Hits@1 | Hits@3 | Hits@5 | Hits@10 | Hits@20 | MRR | macro MRR | 画像基线 Hits@5 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,values in results.items():
        m,b = values['two_tower'],values['profile_baseline']
        lines.append(f"| {name} | {m['Hits@1']:.2%} | {m['Hits@3']:.2%} | {m['Hits@5']:.2%} | {m['Hits@10']:.2%} | {m['Hits@20']:.2%} | {m['MRR']:.4f} | {m['macro_MRR']:.4f} | {b['Hits@5']:.2%} |")
    lines += ['', '部分输入为最多 N 个阳性症状，三个固定输入种子平均；all 使用全部原有阳性症状。',
        '这些指标只评价实际有测试病例的类别，不能声称已验证全部 773 类。',
        '类别权重用于缓解长尾；不会为只有一条记录的疾病创造独立证据。',
        '原始 0 按未知处理，模型不能使用明确否认症状。',
        '统一目录提供实体链接，DDXPlus 病例没有混入此训练，KG 关系也未直接加入损失。',
        '新切分、新类别空间和加权目标与原 413 类版本不同，不能直接把指标差异归因于单一因素。',
        '无法排除增强数据的近似重复或模板来源泄漏。','',
        '[方法与运行说明](DISEASE_RETRIEVAL_773_README.md)']
    (ROOT/'DISEASE_RETRIEVAL_773_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


class DiseaseRetriever773:
    def __init__(self):
        self.vocab = json.loads((OUT/'vocabulary.json').read_text(encoding='utf-8'))
        self.model = TwoTower(377,773)
        self.model.load_state_dict(torch.load(OUT/'model.pt',weights_only=True))
        self.lookup = {name.casefold():i for i,name in enumerate(self.vocab['symptoms'])}

    def retrieve(self,present_symptoms=(),absent_symptoms=(),topk=5):
        if absent_symptoms:
            raise ValueError('773 model has no verified negative training evidence; explicit denials are unsupported, never silently discarded')
        if isinstance(present_symptoms,str) or not present_symptoms:
            raise ValueError('Provide a nonempty list of standardized positive symptoms')
        if not isinstance(topk,int) or topk<1:
            raise ValueError('topk must be a positive integer')
        x = np.zeros((1,377),dtype=np.float32)
        for symptom in present_symptoms:
            key = symptom.strip().casefold()
            if key not in self.lookup:
                raise ValueError(f'Unsupported symptom: {symptom}; use vocabulary.json')
            x[0,self.lookup[key]] = 1
        scores = score(self.model,x)[0]
        candidates = []
        support = pd.read_csv(OUT/'disease_support.csv').set_index('disease_name')
        for i in np.argsort(-scores)[:topk]:
            name = self.vocab['diseases'][i]
            mapping = self.vocab['disease_mapping'][name]
            parse = lambda text:[int(v) for v in text.split(';') if v]
            link_sources = {'direct_mondo':parse(mapping['kg_direct_nodes']),
                            'exact_name_candidate':parse(mapping['kg_exact_name_nodes']),
                            'group_membership':parse(mapping['kg_grouped_nodes'])}
            linked_nodes = sorted({node for values in link_sources.values() for node in values})
            candidates.append({'disease_id':f'DSA_{i:04d}','disease_name':name,'score':float(scores[i]),
                'canonical_mondo_id':mapping['canonical_id'] or None,
                'kg_direct_or_exact_candidates':sorted(set(parse(mapping['kg_direct_nodes'])+parse(mapping['kg_exact_name_nodes']))),
                'kg_group_membership_candidates':parse(mapping['kg_grouped_nodes']),
                'kg_links':[{'node_index':node,'link_types':[kind for kind,values in link_sources.items() if node in values]} for node in linked_nodes],
                'mapping_status':mapping['resolution'],'unique_training_records':int(support.loc[name,'train_rows']),
                'independently_tested':bool(support.loc[name,'independently_tested'])})
        return {'model':'original_773_v1','supported_diseases':773,'supported_symptoms':377,
                'evidence_semantics':'positive/unknown; explicit denial unsupported','score_type':'uncalibrated matching logit','candidates':candidates}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','run','predict'])
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--present',nargs='*',default=[])
    parser.add_argument('--absent',nargs='*',default=[])
    parser.add_argument('--topk',type=int,default=5)
    args = parser.parse_args()
    if args.epochs<1:
        parser.error('epochs must be positive')
    if args.command in ['prepare','run']:
        if not (CATALOG/'original_model_mapping.json').exists():
            from build_unified_catalog import build
            build()
        prepare()
    if args.command in ['train','run']:
        train(args.epochs)
    if args.command=='predict':
        print(json.dumps(DiseaseRetriever773().retrieve(args.present,args.absent,args.topk),ensure_ascii=False,indent=2))
