import os,sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'backend'))
os.environ.update(AI_PROFILE='local',DIALOGUE_ENGINE='agent',LOCAL_MODEL_4BIT='true',
    LOCAL_MODEL_PATH=str(root/'models/Qwen3-4B-Instruct-2507'),
    EMBEDDING_MODEL_PATH=str(root/'models/all-MiniLM-L6-v2'),
    SESSION_DB=':memory:',HF_HUB_OFFLINE='1',REASONING_MODE='random')
from app.dependencies import get_chatbot_service
from app.models.chat_models import StartChatRequest,MessageRequest
from uuid import uuid4
service=get_chatbot_service()
scenarios=[['How to describe?','I feel bad but i dont know how to say','What?','Whats fever','Whats cold'],
['I dont get waht you said','yes','yes'],['I have Drowsiness.','Actually, I no longer have Drowsiness.','What have you recorded about me?'],
['Can you plan a trip to Tokyo?','How do I change an answer in this app?']]
rows=[]
for pipeline in ('reviewed','direct'):
    os.environ['AGENT_REPLY_PIPELINE']=pipeline
    for sequence in scenarios:
        state=None
        for text in sequence:
            start=time.perf_counter()
            state=service.start(StartChatRequest(message=text)) if state is None else service.message(MessageRequest(message=text,session_id=state.session_id,expected_version=state.version,request_id=uuid4().hex))
            row=dict(pipeline=pipeline,user=text,answer=state.message,wall_ms=round(1000*(time.perf_counter()-start),1),evidence=[(e.kg_name,e.status) for e in state.evidence],trace=state.ai_trace)
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
            (root/'evaluation/performance-pipelines.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        service.sessions.delete(state.session_id)

