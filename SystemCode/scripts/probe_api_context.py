import httpx,json
from pathlib import Path
from uuid import uuid4
base='http://127.0.0.1:8003/api/chat'
with httpx.Client(timeout=180) as c:
    a=c.post(base+'/start',json={'message':'I have a mild cough for three days.','request_id':uuid4().hex}).json()
    print('PENDING',a['next_symptom'])
    b=c.post(base+'/message',json={'session_id':a['session_id'],'expected_version':a['version'],'request_id':uuid4().hex,'message':'That has stopped completely.'}).json()
    print(json.dumps({'trace':b['ai_trace'],'evidence':b['evidence']},indent=2))
    Path('evaluation/context_failure.json').write_text(json.dumps({'start':a,'reply':b},indent=2),encoding='utf-8')
    c.delete(base+'/session/'+a['session_id'])
