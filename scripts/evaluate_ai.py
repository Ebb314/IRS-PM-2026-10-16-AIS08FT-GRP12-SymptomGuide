"""Real local inference evaluation. Failures are reported, never hidden by thresholds."""
import sys,json,statistics,platform,hashlib
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'backend'))
from app.services.model_runtime import LocalGenerator
from app.services.retrieval_service import load_encoder
from app.services.nlp_service import NLPService
from app.services.context_nlp import ContextNLP
from app.services.dialogue_ai import DialogueAI
from app.models.chat_models import SessionState

def metrics(rows,key):
    tp=fp=fn=exact=0
    for r in rows:
        gold=set(r['gold'].items()); pred=set(r[key].items())
        tp+=len(gold&pred); fp+=len(pred-gold); fn+=len(gold-pred); exact+=gold==pred
    return dict(tp=tp,fp=fp,fn=fn,micro_f1=round(2*tp/max(1,2*tp+fp+fn),4),exact_match=round(exact/len(rows),4))

def main():
    source=root/'evaluation/ai_cases.json'; cases=json.loads(source.read_text(encoding='utf-8-sig'))
    generator=LocalGenerator(root/'models/Qwen2.5-1.5B-Instruct')
    encoder=load_encoder(); rules=NLPService(root/"backend/app/data/demo_symptoms.csv",use_models=False)
    nlp=ContextNLP(rules,generator,encoder); dialogue=DialogueAI(generator,encoder)
    rows=[]; retrieval=[]; replies=[]
    for case in cases['nlu']:
        context={}
        if case.get('pending'): context={'pending':rules.nodes['DEMO:'+case['pending']].model_dump()}
        before=perf_counter(); result=nlp.analyze_context(case['text'],1,context)
        row={**case,'rules':{e.kg_id.removeprefix('DEMO:'):e.status for e in rules.analyze(case['text']).evidence},
            'hybrid_llm':{e.kg_id.removeprefix('DEMO:'):e.status for e in result.evidence},
            'evidence':[e.model_dump() for e in result.evidence],'trace':result.ai_trace,'latency_ms':round((perf_counter()-before)*1000,2)}
        rows.append(row); print(case['text'],row['hybrid_llm'],result.ai_trace.get('validation'),flush=True)
    for case in cases['retrieval']:
        retrieval.append({**case,**{mode:[h['id'] for h in dialogue.index.search(case['query'],k=3,mode=mode)] for mode in ['bm25','dense','hybrid']}})
    for query in cases['dialogue']:
        answer,trace=dialogue.conversational(query,SessionState(session_id='evaluation'),dialogue.route(query))
        replies.append(dict(question=query,answer=answer,trace=trace)); print(query,trace['response_mode'],flush=True)
    slots=[(r,slot) for r in rows for slot in ['duration','severity'] if slot in r]
    report={'created_at':datetime.now(timezone.utc).isoformat(),'provenance':cases['provenance'],
        'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'environment':{'python':platform.python_version(),
        'model':generator.name,'device':generator.device,'gpu':generator.torch.cuda.get_device_name() if generator.device=='cuda' else None},
        'summary':{'cases':len(rows),'rules':metrics(rows,'rules'),'hybrid_llm':metrics(rows,'hybrid_llm'),
        'accepted':sum(r['trace'].get('validation')=='accepted' for r in rows),'fallback':sum(r['trace'].get('validation')=='fallback' for r in rows),
        'slots_correct':sum(any(e[slot+'_text']==r[slot] for e in r['evidence']) for r,slot in slots),'slots_total':len(slots),
        'nlu_median_ms':round(statistics.median(r['latency_ms'] for r in rows),2),
        'retrieval':{mode:{'recall_at_3':sum(r['gold'] in r[mode] for r in retrieval)/len(retrieval),
            'mrr_at_3':sum(1/(r[mode].index(r['gold'])+1) if r['gold'] in r[mode] else 0 for r in retrieval)/len(retrieval)} for mode in ['bm25','dense','hybrid']},
        'dialogue_modes':{mode:sum(r['trace']['response_mode']==mode for r in replies) for mode in set(r['trace']['response_mode'] for r in replies)}},
        'nlu':rows,'retrieval':retrieval,'dialogue':replies}
    (root/'evaluation/ai_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__': main()

