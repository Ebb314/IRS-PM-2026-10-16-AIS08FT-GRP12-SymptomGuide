"""Real HTTP/LLM spot checks over the full catalog. Start the local AI server first."""
import argparse,csv,json,random,time
from pathlib import Path
from uuid import uuid4
from datetime import datetime,timezone
import httpx
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8000')
parser.add_argument('--count',type=int,default=20,help='0 tests every CSV node through the real API; may take hours')
parser.add_argument('--seed',type=int,default=20260915)
args=parser.parse_args()
rows=list(csv.DictReader((ROOT/'backend/app/data/kg_nodes.csv').open(encoding='utf-8-sig')))
selected=rows if args.count==0 else random.Random(args.seed).sample(rows,min(args.count,len(rows)))
results=[]
with httpx.Client(base_url=args.url,timeout=240) as client:
    health=client.get('/api/health').json()
    assert health['ai_profile']=='local' and health['vocabulary_size']==len(rows)
    for i,row in enumerate(selected):
        status,prefix=[('present','I have '),('absent','I do not have '),('uncertain','I am not sure whether I have ')][i%3]
        text=prefix+row['node_name']+'.';started=time.perf_counter();session=None
        try:
            response=client.post('/api/chat/start',json={'message':text,'request_id':uuid4().hex});response.raise_for_status();data=response.json();session=data['session_id']
            predicted={e['kg_id']:e['status'] for e in data['evidence']}
            results.append({'id':row['node_id'],'text':text,'expected':{row['node_id']:status},'predicted':predicted,
                'pass':predicted=={row['node_id']:status},'trace':data['ai_trace'],'latency_ms':round((time.perf_counter()-started)*1000,2)})
            print(i+1,row['node_id'],results[-1]['pass'],flush=True)
        except Exception as exc:
            results.append({'id':row['node_id'],'text':text,'pass':False,'error_type':type(exc).__name__})
        finally:
            if session:client.delete('/api/chat/session/'+session)
        report={'created_at':datetime.now(timezone.utc).isoformat(),'health':health,'seed':args.seed,'cases':len(results),
            'passed':sum(r['pass'] for r in results),'requested_count':args.count,'rows':results,
            'scope':'Real local LLM integration on canonical templates. count=0 selects the entire CSV; default is a seeded sample, separate from exhaustive offline coverage.'}
        (ROOT/'evaluation/full_kg_llm_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
