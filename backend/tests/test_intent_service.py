import json
import pytest
from app.models.chat_models import SessionState, StartChatRequest, MessageRequest
from app.services.intent_service import IntentService
from app.services.dialogue_ai import DialogueAI

class Generator:
    def __init__(self,outputs): self.outputs=iter(outputs)
    def generate(self,*args): return next(self.outputs),{'model':'test','output_tokens':10}

@pytest.mark.parametrize('intent',['report','knowledge','summary','explain','help','social','out_of_scope','clarify'])
def test_structured_intents(intent):
    result,trace=IntentService(Generator([json.dumps({'intent':intent,'confidence':0.9})])).classify('input',SessionState(session_id='t'))
    assert result==intent and trace['model_call']['output_tokens']==10

@pytest.mark.parametrize('output',['invalid','{"intent":"delete","confidence":1}','{"intent":"report","confidence":0.3}','{"intent":"report","confidence":2}'])
def test_uncertain_or_invalid_route_cannot_modify_evidence(output):
    intent,_=IntentService(Generator([output])).classify('ambiguous',SessionState(session_id='t'))
    assert intent=='clarify'

@pytest.mark.parametrize('verdict,expected',[('YES','help'),('NO','clarify')])
def test_semantic_conflict_requires_second_model_confirmation(verdict,expected):
    service=IntentService(Generator(['explain',verdict]))
    service.index.search=lambda *args,**kwargs:[{'title':'Where do I edit an answer?','intent':'help','cosine':0.9}]
    intent,trace=service.classify('Where can I change my answer?',SessionState(session_id='t'))
    assert intent==expected
    assert trace['consistency_check']['verdict']==verdict

def test_plain_enum_is_accepted_without_inventing_confidence():
    intent,trace=IntentService(Generator(['social'])).classify('Hello',SessionState(session_id='t'))
    assert intent=='social' and trace['confidence'] is None

def test_scope_notice_keeps_valid_model_reply():
    answer,trace=IntentService(Generator(['I can help you understand health terms.'])).respond('Write code',SessionState(session_id='t'),'out_of_scope')
    assert answer.startswith('That request is outside')
    assert 'understand health terms' in answer
    assert trace['scope_notice_added'] and trace['response_mode']=='local-llm-dialogue'

@pytest.mark.parametrize('intent',['help','social','out_of_scope','clarify','summary','explain'])
def test_non_report_messages_preserve_state(chatbot,intent):
    started=chatbot.start(StartChatRequest(message='I have fever.'))
    reply=('That request is outside this app.' if intent=='out_of_scope' else
           'Are you reporting symptoms, asking health knowledge, or asking about the app?' if intent=='clarify' else 'I can help explain the app.')
    chatbot.dialogue=DialogueAI(Generator([json.dumps({'intent':intent,'confidence':0.95}),reply]))
    result=chatbot.message(MessageRequest(session_id=started.session_id,expected_version=started.version,request_id='intent-test',message='Can you help me?'))
    assert result.evidence==started.evidence
    assert result.round==started.round
    assert result.next_symptom==started.next_symptom
    assert result.knowledge_focus==started.knowledge_focus
    assert result.ai_trace['response_mode']=='local-llm-dialogue'
    assert result.ai_trace['routing']['selected_intent']==intent
