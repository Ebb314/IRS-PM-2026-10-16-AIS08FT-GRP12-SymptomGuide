import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"backend"))
from app.services.model_runtime import LocalGenerator
from app.services.retrieval_service import load_encoder
from app.services.nlp_service import NLPService
from app.services.context_nlp import ContextNLP
from app.services.dialogue_ai import DialogueAI
from app.models.chat_models import SessionState
generator=LocalGenerator(root/"models/Qwen2.5-1.5B-Instruct")
print(generator.torch.cuda.get_device_name(),flush=True)
original_generate=generator.generate
def debug_generate(*args,**kwargs):
    response=original_generate(*args,**kwargs)
    print('SYNTHETIC PROBE RAW:',repr(response[0]),flush=True)
    return response
generator.generate=debug_generate
encoder=load_encoder()
nlp=ContextNLP(NLPService(use_models=False),generator,encoder)
for text in ["I feel like I am going to throw up.","I have a severe headache for two days.","It has gone away."]:
    result=nlp.analyze_context(text,1,{"pending":{"kg_id":"DEMO:cough","kg_name":"cough"}})
    print(json.dumps({"text":text,"result":result.model_dump()},ensure_ascii=False),flush=True)
dialogue=DialogueAI(generator,encoder)
for text in ["What is a fever?","Why are you asking that question?"]:
    print(json.dumps({"text":text,"response":dialogue.conversational(text,SessionState(session_id="probe"),dialogue.route(text))}),flush=True)


