from pathlib import Path
from uuid import uuid4
import pytest
from app.services.nlp_service import NLPService
from app.services.chatbot_service import ChatbotService
from app.services.session_service import InMemorySessionService
from app.services.random_reasoning_service import RandomRecallService,RandomSearchService
from app.services.mock_reasoning_service import MockDrugService
from app.services.dialogue_ai import DialogueAI
from app.models.chat_models import StartChatRequest,MessageRequest,AnswerChatRequest,CorrectionRequest,Symptom
PATH=Path(__file__).parents[1]/'app/data/kg_nodes.csv'

@pytest.fixture(scope='module')
def full_nlp():return NLPService(PATH,use_models=False)

@pytest.fixture
def full_chat(full_nlp):
    store=InMemorySessionService()
    yield ChatbotService(full_nlp,store,RandomRecallService(),RandomSearchService(full_nlp.nodes),MockDrugService(),mode='random',
        dialogue=DialogueAI(vocabulary=full_nlp.nodes,mention_finder=full_nlp._mentions))
    store.close()

def msg(service,state,text):
    return service.message(MessageRequest(session_id=state.session_id,expected_version=state.version,request_id=uuid4().hex,message=text))

def test_full_catalog_loaded(full_nlp):
    assert len(full_nlp.nodes)==15135
    assert full_nlp.nodes['P835'].kg_name=='Fever'
    assert all(not k.startswith('DEMO:') for k in full_nlp.nodes)

def test_latest_topic_follows_reports_not_pending_question(full_chat):
    a=full_chat.start(StartChatRequest(message='I have Fever.'))
    a=msg(full_chat,a,'I have Cough.')
    pending=a.next_symptom
    b=msg(full_chat,a,'Tell me about my latest symptom.')
    assert b.ai_trace['topic']['kg_id']=='P4018'
    assert b.ai_trace['citations'][0]['id']=='nhs-cough'
    assert b.evidence==a.evidence and b.round==a.round and b.next_symptom==pending
    c=msg(full_chat,b,'I have Headache.')
    d=msg(full_chat,c,'Tell me about my latest symptom.')
    assert d.ai_trace['topic']['kg_id']=='P885'
    assert d.ai_trace['response_mode']=='llm-unavailable'
    assert 'Fever' not in d.message
    e=msg(full_chat,d,'I have Cough.')
    assert msg(full_chat,e,'Tell me about my latest symptom.').ai_trace['topic']['kg_id']=='P4018'

def test_explicit_question_overrides_latest(full_chat):
    a=full_chat.start(StartChatRequest(message='I have Headache.'))
    b=msg(full_chat,a,'What is Fever?')
    assert b.ai_trace['topic']['kg_id']=='P835'
    assert b.knowledge_focus.kg_id=='P885'

def test_unknown_new_report_does_not_reuse_old_topic(full_chat):
    a=full_chat.start(StartChatRequest(message='I have Fever.'))
    b=msg(full_chat,a,'Something unfamiliar happened.')
    assert b.knowledge_focus is None
    c=msg(full_chat,b,'Tell me about my latest symptom.')
    assert c.ai_trace['response_mode']=='topic-clarification'
    assert not c.ai_trace['citations']

def test_answer_and_correction_update_topic(full_chat):
    a=full_chat.start(StartChatRequest(message='I have Cough.'))
    q=a.next_symptom
    b=full_chat.answer(AnswerChatRequest(session_id=a.session_id,expected_version=a.version,request_id=uuid4().hex,answer='unknown',symptom=Symptom(kg_id=q.kg_id,kg_name=q.kg_name)))
    assert b.knowledge_focus.kg_id==q.kg_id
    assert full_chat.inspect(b.session_id).knowledge_focus==b.knowledge_focus
    c=full_chat.correct(CorrectionRequest(session_id=b.session_id,expected_version=b.version,request_id=uuid4().hex,kg_id='P4018',status='absent'))
    assert c.knowledge_focus.kg_id=='P4018'

def test_random_adapters_reproducible_and_exclude_known(full_nlp):
    node=full_nlp.nodes['P4018']; r=RandomRecallService(seed=7)
    a=r.recall([node],[]);assert a==r.recall([node],[])
    assert a!=RandomRecallService(seed=8).recall([node],[])
    search=RandomSearchService(full_nlp.nodes,seed=7)
    q=search.select_next(a,[node],[],['P835'])
    assert q.kg_id not in {'P4018','P835'} and q.kg_id in full_nlp.nodes
    assert search.select_next(a,[],[],list(full_nlp.nodes)) is None
    assert all(c.id.startswith('SIM:') for c in a)

def test_temporal_words_inside_canonical_label_are_not_scope_cues(full_nlp):
    assert {e.kg_id:e.status for e in full_nlp.analyze('I have History of recent insect bite.').evidence}=={'P3606':'present'}
    assert {e.kg_id:e.status for e in full_nlp.analyze('I do not have Family history of cancer.').evidence}=={'P12366':'absent'}
    assert full_nlp.analyze('I used to have Family history of cancer last year.').evidence==[]

def test_model_cannot_expand_an_exact_catalog_span_into_another_concept(tmp_path):
    import json
    from app.services.context_nlp import ContextNLP
    path=tmp_path/'nodes.csv'
    path.write_text('node_id,node_name,node_type\nA,Limited knee flexion,symptom\nB,knee flexion,symptom\n',encoding='utf-8')
    class Model:
        def generate(self,*args):return json.dumps({'mentions':[{'concept':'B','quote':'knee flexion','status':'present'}]}),{}
    result=ContextNLP(NLPService(path,use_models=False),Model()).analyze('I have Limited knee flexion.')
    assert [e.kg_id for e in result.evidence]==['A']
    assert result.ai_trace['rejected_overlapping_concepts']==['B']

def test_assertion_words_alone_do_not_ground_an_extra_entity(tmp_path):
    import json
    from app.services.context_nlp import ContextNLP
    path=tmp_path/'nodes.csv'
    path.write_text('node_id,node_name,node_type\nA,Limited knee flexion,symptom\nB,knee flexion,symptom\n',encoding='utf-8')
    class Model:
        def generate(self,*args):return json.dumps({'mentions':[{'concept':'B','quote':'not sure','status':'uncertain'}]}),{}
    result=ContextNLP(NLPService(path,use_models=False),Model()).analyze('I am not sure whether I have Limited knee flexion.')
    assert [e.kg_id for e in result.evidence]==['A']
    assert result.ai_trace['rejected_assertion_only_quotes']==['B']

@pytest.mark.parametrize('output,mode',[
    ('Drowsiness means feeling sleepy or less alert.','local-llm-general'),
    ('{"answer":"Drowsiness means feeling sleepy or less alert.","citations":[]}','local-llm-general'),
    ('{"answer":"Drowsiness means sleepiness.","citations":["invented"]}','llm-generation-failed'),
    ('{"answer":"Headache is pain in the head.","citations":[]}','llm-generation-failed'),
    ('not json','llm-generation-failed'),
])
def test_general_knowledge_generation_and_validation(full_nlp,output,mode):
    from app.models.chat_models import SessionState
    class Generator:
        def generate(self,*args):return output,{'model':'test-model'}
    node=full_nlp.nodes['P986']
    service=DialogueAI(Generator(),vocabulary=full_nlp.nodes,mention_finder=full_nlp._mentions)
    answer,trace=service.conversational('Tell me about my latest symptom.',SessionState(session_id='t',knowledge_focus=node),'knowledge')
    assert trace['response_mode']==mode
    assert trace['citations']==[] and trace['model_call']['model']=='test-model'
    assert 'listed as a symptom node' not in answer

def test_service_initialization_is_serialized(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from functools import lru_cache
    from time import sleep
    from threading import Barrier
    from app import dependencies
    calls=[]
    barrier=Barrier(4)
    @lru_cache(maxsize=1)
    def build():
        calls.append(1)
        sleep(0.05)
        return object()
    monkeypatch.setattr(dependencies,'_build_chatbot_service',build)
    def request(_):
        barrier.wait()
        return dependencies.get_chatbot_service()
    with ThreadPoolExecutor(max_workers=4) as pool:
        services=list(pool.map(request,range(4)))
    assert len(calls)==1
    assert all(service is services[0] for service in services)
