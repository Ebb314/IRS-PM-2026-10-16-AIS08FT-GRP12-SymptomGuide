"""Exhaustive canonical-label and assertion coverage, not natural-language generalization."""
import sys,json,hashlib,time,argparse
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.services.nlp_service import NLPService, MODIFIER_TERMS
from app.services.random_reasoning_service import RandomRecallService,RandomSearchService
from app.services.dialogue_ai import DialogueAI
from app.models.chat_models import SessionState,Symptom

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    path=ROOT/'backend/app/data/kg_nodes.csv'; nlp=NLPService(path,use_models=False)
    recall=RandomRecallService(); search=RandomSearchService(nlp.nodes)
    dialogue=DialogueAI(vocabulary=nlp.nodes,mention_finder=nlp._mentions)
    templates=[('present','I have {}.'),('absent','I do not have {}.'),('uncertain','I am not sure whether I have {}.'),('family','My father has {}.'),('history','I used to have {} last year.')]
    failures=[];passed={k:0 for k,_ in templates};coverage=[];random_rows=[];started=time.perf_counter()
    output=ROOT/'evaluation/full_kg_cases.jsonl'
    with output.open('w',encoding='utf-8') as handle:
        for i,node in enumerate(nlp.nodes.values(),1):
            ok=True
            for status,template in templates:
                text=template.format(node.kg_name);result=nlp.analyze(text)
                predicted={e.kg_id:e.status for e in result.evidence}
                # Clinical-modifier terms (Severe, Acute, Left...) are deliberately never matched as symptoms.
                modifier=node.kg_name.casefold() in MODIFIER_TERMS
                expected={} if status in {'family','history'} or modifier else {node.kg_id:status}
                valid=predicted==expected and (modifier or status not in {'family','history'} or any(e.kg_id==node.kg_id for e in result.excluded))
                row={'id':node.kg_id,'name':node.kg_name,'case':status,'text':text,'expected':expected,'predicted':predicted,'pass':valid}
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                if valid:passed[status]+=1
                else:failures.append(row);ok=False
            coverage.append({'id':node.kg_id,'name':node.kg_name,'nlp_all_pass':ok})
            if i%1000==0: print('Covered',i,'nodes',flush=True)
    # Every node is tested as an eligible random question by excluding the rest.
    # This checks coverage/exclusion; independent random draws below check variability.
    all_ids=set(nlp.nodes)
    for node in nlp.nodes.values():
        candidates=recall.recall([node],[])
        selected=search.select_next(candidates,[],[],all_ids-{node.kg_id})
        assert selected and selected.kg_id==node.kg_id
        assert all(c.source=='seeded-random-fixture-v1' for c in candidates)
        state=SessionState(session_id='coverage',knowledge_focus=node)
        answer,trace=dialogue.conversational('Tell me about my latest symptom.',state,'knowledge')
        assert trace['topic']['kg_id']==node.kg_id
    for seed in range(30):
        node=list(nlp.nodes.values())[seed]
        candidates=RandomRecallService(seed).recall([node],[])
        question=RandomSearchService(nlp.nodes,seed).select_next(candidates,[node],[],[])
        assert question.kg_id!=node.kg_id
        random_rows.append({'seed':seed,'candidates':[c.id for c in candidates],'question':question.kg_id})
    report={'created_at':datetime.now(timezone.utc).isoformat(),'source':str(path.relative_to(ROOT)),
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nodes':len(nlp.nodes),'templates_per_node':len(templates),
        'cases':len(nlp.nodes)*len(templates),'passed_by_type':passed,'failures':failures,'node_coverage':coverage,
        'random_question_eligibility_nodes':len(nlp.nodes),'knowledge_topic_coverage_nodes':len(nlp.nodes),
        'random_draws':random_rows,'duration_seconds':round(time.perf_counter()-started,2),
        'limitations':'Canonical-template regression; does not validate all paraphrases, medical correctness or LLM inference on every node. Catalog sources provide labels only. Random diseases are synthetic fixtures.'}
    (ROOT/'evaluation/full_kg_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['node_coverage','failures','random_draws']},indent=2));print('Failures:',len(failures))
    if args.check and failures:raise SystemExit(1)
if __name__=='__main__':main()
