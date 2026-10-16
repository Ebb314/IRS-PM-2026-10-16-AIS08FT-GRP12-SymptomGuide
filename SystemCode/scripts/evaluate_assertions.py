"""Real local API assertion regression. Uses and deletes only its own sessions."""
import argparse,json,urllib.request,uuid,time
from pathlib import Path

SCENARIOS=[
 [('I dont think i have a fever',{'fever':'absent'})],
 [('I dont have Drowsiness.',{'drowsiness':'absent'})],
 [('Drowsiness is uncertain.',{'drowsiness':'uncertain'})],
 [("Drowsiness? I'm not sure.",{'drowsiness':'uncertain'})],
 [('I have Drowsiness.',{'drowsiness':'present'}),("I don't have Drowsiness.",{'drowsiness':'absent'}),("I'm not sure whether I have Drowsiness",{'drowsiness':'uncertain'})],
 [("I have a cough, no fever, and I'm not sure about headache.",{'cough':'present','fever':'absent','headache':'uncertain'})],
 [('What is fever?',{})],
 [('My sister has fever.',{})],
 [("I'm not sure whether I have a fever",{'fever':'uncertain'})],
]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'evaluation/assertion-status-live.json')
    args=parser.parse_args()
    def call(route,body=None,method=None):
        req=urllib.request.Request(args.url+'/api/chat/'+route,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'},method=method)
        with urllib.request.urlopen(req,timeout=240) as response:
            data=response.read()
            return json.loads(data) if data else None
    rows=[]
    for sequence in SCENARIOS:
        state=None
        try:
            for message,expected in sequence:
                body={'message':message,'request_id':uuid.uuid4().hex}
                if state:body.update(session_id=state['session_id'],expected_version=state['version'])
                started=time.perf_counter()
                state=call('message' if state else 'start',body)
                actual={e['kg_name'].lower():e['status'] for e in state['evidence']}
                persisted=call('session/'+state['session_id'])
                partitions={name:[e['kg_name'].lower() for e in state[name]] for name in ('positive_symptoms','negative_symptoms','uncertain_symptoms')}
                partition_ok=all(sorted(partitions[field])==sorted(k for k,v in actual.items() if v==status) for field,status in [('positive_symptoms','present'),('negative_symptoms','absent'),('uncertain_symptoms','uncertain')])
                row=dict(message=message,expected=expected,actual=actual,passed=actual==expected and persisted['evidence']==state['evidence'] and partition_ok,
                         answer=state['message'],wall_ms=round((time.perf_counter()-started)*1000,1),partitions=partitions,trace=state['ai_trace'])
                rows.append(row)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps({'passed':all(r['passed'] for r in rows),'complete':False,'rows':rows},indent=2),encoding='utf-8')
                print(json.dumps({k:v for k,v in row.items() if k not in ('trace','partitions')}),flush=True)
        finally:
            if state:call('session/'+state['session_id'],method='DELETE')
    report={'passed':all(r['passed'] for r in rows),'complete':True,'count':len(rows),'rows':rows}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']:raise SystemExit('Assertion regression failed; inspect the report')

if __name__=='__main__':main()
