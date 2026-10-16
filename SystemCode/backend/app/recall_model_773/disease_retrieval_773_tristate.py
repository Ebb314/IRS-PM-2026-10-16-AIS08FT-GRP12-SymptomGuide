"""773-label tri-state research model: explicitly authorized synthetic negatives."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from disease_retrieval import ROOT, TwoTower, save_json, score, subset
from disease_retrieval_773 import OUT as BASE, evaluate
from symptom_evidence import encode_evidence

OUT = ROOT/'artifacts/disease_retrieval_773_tristate_v2'
POLICY = 'synthetic_zero_assumption'


def warm_start():
    old = torch.load(BASE/'model.pt',weights_only=True)
    model = TwoTower(754,773)
    state = {key:value.clone() for key,value in old.items()}
    state['patient.0.weight'] = torch.cat([old['patient.0.weight'],torch.zeros_like(old['patient.0.weight'])],dim=1)
    model.load_state_dict(state)
    return model


def reveal_batch(truth, generator):
    """75% positive-only views; 25% reveal 1-3 synthetic negative answers."""
    n = len(truth)
    keep = torch.rand((n,1),generator=generator)*0.8+0.2
    pos = truth*(torch.rand(truth.shape,generator=generator)<keep)
    empty = pos.sum(dim=1)==0
    pos[empty] = truth[empty]
    active = torch.rand((n,1),generator=generator)<0.25
    budget = torch.randint(1,4,(n,1),generator=generator)
    keys = torch.rand(truth.shape,generator=generator).masked_fill(truth.bool(),2)
    rank = keys.argsort(dim=1).argsort(dim=1)
    neg = ((rank<budget)&active&(~truth.bool())).float()
    return torch.cat([pos,neg],dim=1)


def fixed_view(truth, positives, negatives, seed):
    pos = subset(truth,positives,np.random.default_rng(seed))
    rng = np.random.default_rng(seed+1000)
    neg = np.zeros_like(truth,dtype=np.float32)
    for i,row in enumerate(truth):
        candidates = np.flatnonzero(row==0)
        chosen = rng.choice(candidates,min(negatives,len(candidates)),replace=False)
        neg[i,chosen] = 1
    return np.concatenate([pos,neg],axis=1)


def negative_candidates(vocab):
    allowed = np.ones((773,773),dtype=bool)
    np.fill_diagonal(allowed,False)
    groups = {}
    for i,name in enumerate(vocab['diseases']):
        record = vocab['disease_mapping'][name]
        for node in filter(None,record['kg_grouped_nodes'].split(';')):
            groups.setdefault(node,[]).append(i)
    for labels in groups.values():
        allowed[np.ix_(labels,labels)] = False
    return torch.from_numpy(allowed)


def pair_auxiliary(scores, target, allowed_by_class, generator):
    """One contrast disease for only 25% of patient rows, never the true label."""
    allowed = allowed_by_class[target]
    hard = scores.detach().masked_fill(~allowed,-torch.inf).argmax(dim=1)
    weights = allowed.float()
    empty = weights.sum(dim=1)==0
    weights[empty] = 1
    random = torch.multinomial(weights,1,replacement=True,generator=generator).squeeze(1)
    use_hard = torch.rand(len(target),generator=generator)<0.5
    indices = torch.where(use_hard,hard,random)
    active = (torch.rand(len(target),generator=generator)<0.25)&(~empty)
    active &= allowed.gather(1,indices[:,None]).squeeze(1)
    pos = scores.gather(1,target[:,None]).squeeze(1)
    neg = scores.gather(1,indices[:,None]).squeeze(1)
    losses = nn.functional.softplus(neg-pos)
    loss = (losses*active).sum()/active.sum().clamp_min(1)
    return loss, int(active.sum())


def train(epochs, negative_policy):
    if negative_policy!=POLICY:
        raise ValueError('Explicit synthetic_zero_assumption policy is required; no verified clinical denials are available')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    vocab = json.loads((BASE/'vocabulary.json').read_text(encoding='utf-8'))
    vocab['base_vocabulary_version'] = vocab['version']
    vocab['version'] = 'original_773_tristate_v2'
    vocab['input_encoding'] = {'dimensions':754,'positive_channel':[0,376],'negative_channel':[377,753],
                               'unknown':'both channel bits zero'}
    vocab['training_negative_policy'] = POLICY
    vocab['zero_semantics'] = 'Input zero means unknown; raw dataset zeros are simulated negatives only when revealed under the authorized policy'
    with np.load(BASE/'dataset.npz') as archive:
        data = {key:archive[key] for key in archive.files}
    x,y = data['x'],data['y']
    train_x,train_y = x[data['train']],y[data['train']]
    allowed = negative_candidates(vocab)
    class_weights = torch.tensor(np.load(BASE/'class_weights.npy'),dtype=torch.float32)
    config = {'version':'original_773_tristate_v2','diseases':773,'symptoms':377,'input_dimensions':754,
        'negative_policy':POLICY,'authorization':'User explicitly accepted simulation on 2026-10-15',
        'negative_provenance':'Original 0 treated as simulated absence ONLY when revealed; not clinical denial labels',
        'view_policy':'75% positive-only; 25% add 1-3 simulated negatives; positive masking keep probability 0.2..1',
        'pair_policy':'correct disease supervision on every row; auxiliary for 25% of rows, one 50/50 hard/random contrast; weight 0.05',
        'pair_protection':'correct label and same KG group member candidates excluded from pair negatives; CE remains single-label',
        'epochs':epochs,'batch_size':512,'learning_rate':0.0005,'seed':42,
        'initialization':'Warm start positive 773 model; negative input columns zero-initialized',
        'base_weights_sha256':hashlib.sha256((BASE/'model.pt').read_bytes()).hexdigest(),
        'selection':'mean validation macro_MRR for same 3 positives with 0 and 2 negatives',
        'limitations':'Synthetic-label experiment, no real-world validation; same held-out split reused as follow-up benchmark; 156 classes have no test rows.'}
    save_json(OUT/'config.json',config)
    save_json(OUT/'vocabulary.json',vocab)
    (OUT/'disease_support.csv').write_bytes((BASE/'disease_support.csv').read_bytes())
    base_audit = json.loads((BASE/'audit.json').read_text(encoding='utf-8'))
    save_json(OUT/'base_data_audit.json',base_audit)
    audit = dict(base_audit)
    audit['base_zero_semantics'] = base_audit['zero_semantics']
    audit['zero_semantics'] = 'Unrevealed zero stays unknown; revealed original zero is an authorized simulated negative'
    audit['negative_policy'] = POLICY
    audit['clinical_negative_labels_verified'] = False
    audit['split_inherited_without_changes'] = True
    audit['limitations'] = 'Distinct records are not verified independent patients; augmentation provenance unavailable; negative answers are authorized simulations, not verified denials; no mixed DDXPlus patient training.'
    save_json(OUT/'audit.json',audit)
    val_inputs = {n:fixed_view(x[data['val']],3,n,123) for n in [0,2]}
    models,histories,selection_values = {},{},{}
    for mode in ['ce','ce_pairwise']:
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        observation_generator = torch.Generator().manual_seed(42)
        pair_generator = torch.Generator().manual_seed(31415)
        model = warm_start()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
        best,history = -1,[]
        for epoch in range(epochs):
            model.train()
            permutation = rng.permutation(len(train_y))
            total,negative_answers,negative_rows,pair_count = 0,0,0,0
            for start in range(0,len(permutation),512):
                idx = permutation[start:start+512]
                inputs = reveal_batch(torch.from_numpy(train_x[idx]),observation_generator)
                target = torch.from_numpy(train_y[idx])
                scores = model(inputs)
                loss = nn.functional.cross_entropy(scores,target,weight=class_weights)
                if mode=='ce_pairwise':
                    pair,count = pair_auxiliary(scores,target,allowed,pair_generator)
                    loss = loss+0.05*pair
                    pair_count += count
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()*len(idx)
                negative_answers += int(inputs[:,377:].sum())
                negative_rows += int((inputs[:,377:].sum(dim=1)>0).sum())
            evaluations = {str(n):evaluate(score(model,inp),y[data['val']]) for n,inp in val_inputs.items()}
            selected_metric = float(np.mean([r['macro_MRR'] for r in evaluations.values()]))
            row = {'epoch':epoch+1,'loss':total/len(permutation),'validation':evaluations,
                   'selection_macro_MRR':selected_metric,'views_with_negative':negative_rows,
                   'negative_answers':negative_answers,'auxiliary_negative_pairs':pair_count}
            history.append(row)
            print(mode,json.dumps(row),flush=True)
            if selected_metric>best:
                best = selected_metric
                torch.save(model.state_dict(),OUT/f'{mode}.pt')
        model.load_state_dict(torch.load(OUT/f'{mode}.pt',weights_only=True))
        models[mode],histories[mode],selection_values[mode] = model,history,best
    selected = max(selection_values,key=selection_values.get)
    # Freeze model choice on validation, before evaluating the follow-up test.
    save_json(OUT/'selection.json',{'selected_model':selected,'validation':selection_values,
              'selection_metric':config['selection'],'negative_policy':POLICY})
    baseline = TwoTower(377,773)
    baseline.load_state_dict(torch.load(BASE/'model.pt',weights_only=True))
    results = {}
    test_x,test_y = x[data['test']],y[data['test']]
    per_feature = train_x.shape[0]-train_x.sum(axis=0)
    save_json(OUT/'negative_support.json',{'policy':POLICY,'symptom_training_zero_counts':{
        name:int(per_feature[i]) for i,name in enumerate(vocab['symptoms'])},
        'meaning':'Potential simulated negative states, not independent confirmed denials'})
    for count in [0,1,3,5]:
        runs = []
        for seed in [101,102,103]:
            inputs = fixed_view(test_x,3,count,seed)
            run = {mode:evaluate(score(model,inputs),test_y) for mode,model in models.items()}
            run['positive_baseline'] = evaluate(score(baseline,inputs[:,:377]),test_y)
            runs.append(run)
        results[str(count)] = {mode:{key:float(np.mean([r[mode][key] for r in runs])) for key in runs[0][mode]} for mode in runs[0]}
    paired_yes = fixed_view(test_x,3,3,101)
    paired_unknown = paired_yes.copy()
    paired_unknown[:,377:] = 0
    a,b = score(models[selected],paired_yes),score(models[selected],paired_unknown)
    ablation = {'negatives_known':evaluate(a,test_y),'negatives_unknown':evaluate(b,test_y),
                'top1_changed_fraction':float((a.argmax(axis=1)!=b.argmax(axis=1)).mean())}
    save_json(OUT/'metrics.json',{'test_3_positives_by_negative_count':results,'history':histories,
              'selected_model':selected,'paired_ablation_seed101':ablation,'config':config})
    lines = ['# 773 疾病三态模拟阴性实验结果','',
        '全部 773 类参与训练与候选排序，377 个症状编码为 754 维。测试实际覆盖 617 类。',
        '用户已接受把原始 0 作为模拟阴性的实验假设；这不是获得了真实患者明确否认标签。',
        '训练 75% 输入只含阳性，25% 输入加入最多三个模拟阴性。配对辅助权重 0.05，25% 行抽取一个对照疾病。','',
        f'按验证集选出的版本：**{selected}**。','',
        '| 阴性数 | 模型 | Hits@1 | Hits@5 | Hits@10 | MRR | macro MRR |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for count,values in results.items():
        for mode,r in values.items():
            lines.append(f"| {count} | {mode} | {r['Hits@1']:.2%} | {r['Hits@5']:.2%} | {r['Hits@10']:.2%} | {r['MRR']:.4f} | {r['macro_MRR']:.4f} |")
    lines += ['',f"同模型固定输入的消融：加入三个模拟阴性后 Top-1 改变 {ablation['top1_changed_fraction']:.2%}。",
        '同模型消融的阴性未知/已知指标：','```json',json.dumps(ablation,ensure_ascii=False,indent=2),'```','',
        '本轮重用了已查看的基准切分，且从阳性模型继续训练。输入、额外训练和辅助损失的影响不能混为一个因素。',
        '156 类无独立测试记录，模拟 0/1 状态及增强来源限制仍在，结果不代表真实医疗准确率。',
        '全未知输入不返回疾病排名；阴性输入调整相对分数，不硬排除疾病。','',
        '[方法与接口说明](DISEASE_RETRIEVAL_773_TRISTATE_README.md)']
    (ROOT/'DISEASE_RETRIEVAL_773_TRISTATE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Selected:',selected,flush=True)
    print(json.dumps(results,indent=2),flush=True)


class DiseaseRetriever773Tristate:
    def __init__(self,variant='selected'):
        self.vocab = json.loads((OUT/'vocabulary.json').read_text(encoding='utf-8'))
        self.config = json.loads((OUT/'config.json').read_text(encoding='utf-8'))
        if variant=='selected':
            variant = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected_model']
        if variant not in ['ce','ce_pairwise']:
            raise ValueError('Unknown model variant')
        self.variant = variant
        self.model = TwoTower(754,773)
        self.model.load_state_dict(torch.load(OUT/f'{variant}.pt',weights_only=True))
        self.support = pd.read_csv(OUT/'disease_support.csv').set_index('disease_name')

    def retrieve(self,present_symptoms=(),absent_symptoms=(),unknown_symptoms=(),symptom_states=None,topk=5):
        if not isinstance(topk,int) or isinstance(topk,bool) or topk<1:
            raise ValueError('topk must be a positive integer')
        inputs,evidence = encode_evidence(self.vocab['symptoms'],present_symptoms,absent_symptoms,unknown_symptoms,symptom_states)
        response = {'model':f'original_773_tristate_v2_{self.variant}',
                    'supported_diseases':773,'symptom_types':377,'input_dimensions':754,
                    'negative_policy':self.config['negative_policy'],'evidence':evidence,
                    'score_type':'uncalibrated matching logit'}
        if inputs.sum()==0:
            return {**response,'status':'needs_known_evidence','candidates':[]}
        scores = score(self.model,inputs)[0]
        candidates = []
        for i in np.argsort(-scores)[:topk]:
            name = self.vocab['diseases'][i]
            mapping = self.vocab['disease_mapping'][name]
            links = {'direct_mondo':mapping['kg_direct_nodes'],'exact_name_candidate':mapping['kg_exact_name_nodes'],
                     'group_membership':mapping['kg_grouped_nodes']}
            parsed = {kind:{int(v) for v in values.split(';') if v} for kind,values in links.items()}
            nodes = sorted(set().union(*parsed.values()))
            candidates.append({'disease_id':f'DSA_{i:04d}','disease_name':name,'score':float(scores[i]),
                'canonical_mondo_id':mapping['canonical_id'] or None,
                'kg_links':[{'node_index':n,'link_types':[kind for kind,values in parsed.items() if n in values]} for n in nodes],
                'unique_training_records':int(self.support.loc[name,'train_rows']),
                'independently_tested':bool(self.support.loc[name,'independently_tested'])})
        return {**response,'status':'ok','candidates':candidates}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','predict'])
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--negative-policy',choices=[POLICY])
    parser.add_argument('--model',choices=['selected','ce','ce_pairwise'],default='selected')
    parser.add_argument('--present',nargs='*',default=[])
    parser.add_argument('--absent',nargs='*',default=[])
    parser.add_argument('--unknown',nargs='*',default=[])
    parser.add_argument('--topk',type=int,default=5)
    args = parser.parse_args()
    if args.epochs<1:
        parser.error('epochs must be positive')
    if args.command=='train':
        train(args.epochs,args.negative_policy)
    else:
        print(json.dumps(DiseaseRetriever773Tristate(args.model).retrieve(args.present,args.absent,args.unknown,topk=args.topk),ensure_ascii=False,indent=2))
