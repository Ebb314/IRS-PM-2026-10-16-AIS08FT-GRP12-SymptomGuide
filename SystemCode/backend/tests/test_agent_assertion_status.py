"""Evidence status must survive model validation, persistence and recall inputs."""
import json
from uuid import uuid4
import pytest
from app.models.chat_models import StartChatRequest, MessageRequest, SessionState, ChatMessage, NextSymptom
from app.services.conversation_agent import ConversationAgent
from app.services.nlp_service import NLPService

class Proposals:
    def __init__(self, observations): self.observations=observations
    def generate(self,*args): return json.dumps({'observations':self.observations}),{'model':'fixture'}
    def chat(self,*args): return 'Your symptom log reflects your latest report.',{'model':'fixture'}

def proposal(status,quote='fever',concept='DEMO:fever'):
    return dict(concept_id=concept,quote=quote,status=status)

@pytest.mark.parametrize('text,status',[
    ('I dont think i have a fever','absent'),
    ("I don't think I have a fever",'absent'),
    ('I dont have fever','absent'),
    ('fever is absent','absent'),
    ('fever is uncertain','uncertain'),
    ("I'm not sure whether I have fever",'uncertain'),
    ("I'm not sure whether I have fever?",'uncertain'),
    ("fever? I'm not sure.",'uncertain'),
    ('I have fever','present'),
])
def test_status_is_not_replaced_by_keyword_baseline(chatbot,text,status):
    chatbot.dialogue=ConversationAgent(Proposals([proposal(status)]))
    state=chatbot.start(StartChatRequest(message=text))
    assert [(e.kg_id,e.status) for e in state.evidence]==[('DEMO:fever',status)]
    assert state.ai_trace['understanding']['proposed'][0]['status']==status
    assert state.ai_trace['understanding']['accepted'][0]['status']==status
    assert chatbot.inspect(state.session_id).evidence==state.evidence

@pytest.mark.parametrize('text',[
    'What is fever?', 'What is fever? I am not sure.',
    "I'm not sure what fever means?", 'Could I have fever?',
    'My sister has fever', 'I had fever last year',
])
def test_uncertainty_cannot_bypass_knowledge_or_person_guards(chatbot,text):
    chatbot.dialogue=ConversationAgent(Proposals([proposal('uncertain')]))
    state=chatbot.start(StartChatRequest(message=text))
    assert not state.evidence
    assert state.ai_trace['understanding']['rejected']

def test_mixed_statuses_remain_independent(chatbot):
    observations=[proposal('present','cough','DEMO:cough'),proposal('absent'),proposal('uncertain','headache','DEMO:headache')]
    chatbot.dialogue=ConversationAgent(Proposals(observations))
    result=chatbot.start(StartChatRequest(message="I have cough, no fever, and I'm not sure about headache"))
    assert {e.kg_name:e.status for e in result.evidence}=={'cough':'present','fever':'absent','headache':'uncertain'}
    assert [e.kg_name for e in result.positive_symptoms]==['cough']
    assert [e.kg_name for e in result.negative_symptoms]==['fever']
    assert [e.kg_name for e in result.uncertain_symptoms]==['headache']

def test_latest_report_changes_status_without_magic_correction_word(chatbot):
    model=Proposals([proposal('present')]);chatbot.dialogue=ConversationAgent(model)
    seen=[];original=chatbot.recall.recall
    def recall(positive,negative):
        seen.append(([s.kg_id for s in positive],[s.kg_id for s in negative]))
        return original(positive,negative)
    chatbot.recall.recall=recall
    result=chatbot.start(StartChatRequest(message='I have fever'))
    for status,text in [('absent','I dont think i have a fever'),('uncertain',"I'm not sure whether I have fever")]:
        model.observations=[proposal(status)]
        request=MessageRequest(session_id=result.session_id,expected_version=result.version,request_id=uuid4().hex,message=text)
        result=chatbot.message(request)
        assert result.evidence[0].status==status
        assert result.ai_trace['understanding']['accepted'][0]['status']==status
        assert chatbot.message(request)==result
    assert [e.status for e in result.evidence_history]==['present','absent','uncertain']
    assert seen==[(['DEMO:fever'],[]),([],['DEMO:fever']),([],[])]

def test_not_sure_answers_the_actual_symptom_question():
    nlp=NLPService(use_models=False)
    state=SessionState(session_id='fixture',last_question=NextSymptom(kg_id='DEMO:fever',kg_name='fever',heuristic_score=.5),
        messages=[ChatMessage(sender='assistant',text='Do you have fever?'),ChatMessage(sender='user',text='Not sure')])
    agent=ConversationAgent(Proposals([proposal('uncertain','Not sure')]))
    evidence,trace=agent.understand('Not sure',state,nlp)
    assert evidence[0].status=='uncertain' and not state.pending_evidence
