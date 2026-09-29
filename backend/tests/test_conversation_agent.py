import json
from uuid import uuid4
import pytest
from app.models.chat_models import StartChatRequest,MessageRequest,ChatMessage,SessionState,NextSymptom
from app.services.conversation_agent import ConversationAgent,Understanding
from app.services.nlp_service import NLPService

@pytest.fixture(autouse=True)
def reviewed_pipeline(monkeypatch):
    monkeypatch.setenv('AGENT_REPLY_PIPELINE','reviewed')

class Model:
    def __init__(self,plan=None):
        self.plan=plan or {'user_goal':'Help the person describe what feels wrong','reference_resolution':'','unresolved':'Which sensation they mean','observations':[]}
        self.calls=[]
    def generate(self,*args):return json.dumps(self.plan),{'model':'test'}
    def chat(self,messages,*args):
        self.calls.append(messages)
        return 'You can use everyday words. What feels different right now?',{'model':'test'}

def send(chatbot,state,text):
    return chatbot.message(MessageRequest(session_id=state.session_id,expected_version=state.version,request_id=uuid4().hex,message=text))

def test_conversation_is_not_dispatched_through_labels(chatbot):
    model=Model();chatbot.dialogue=ConversationAgent(model)
    state=chatbot.start(StartChatRequest(message='I feel bad but cannot describe it'))
    state=send(chatbot,state,'What do you mean by different?')
    assert 'intent' not in state.ai_trace and 'routing' not in state.ai_trace
    roles=[m['role'] for m in model.calls[-2]]
    assert roles==['system','user','assistant','user']
    assert 'cannot describe it' in model.calls[-2][1]['content']
    assert 'conversation' in json.loads(model.calls[-1][1]['content'])
    assert not state.evidence

@pytest.mark.parametrize('message,quote',[('What is fever?','fever'),('My sister has fever','fever'),('I have fever','invented quote')])
def test_unsafe_proposal_does_not_write(chatbot,message,quote):
    plan={'observations':[{'concept_id':'DEMO:fever','quote':quote,'status':'present'}]}
    chatbot.dialogue=ConversationAgent(Model(plan))
    state=chatbot.start(StartChatRequest(message=message))
    assert not state.evidence
    assert state.ai_trace['understanding']['rejected']

def test_exact_self_report_is_committed(chatbot):
    chatbot.dialogue=ConversationAgent(Model({'observations':[{'concept_id':'DEMO:fever','quote':'fever','status':'present'}]}))
    state=chatbot.start(StartChatRequest(message='I have fever'))
    assert state.evidence[0].kg_id=='DEMO:fever'
    assert state.evidence[0].source=='agent-validated-proposal'

def test_bad_analysis_does_not_break_reply_or_use_rule_fallback(chatbot):
    model=Model();model.generate=lambda *args:('broken JSON',{})
    chatbot.dialogue=ConversationAgent(model)
    state=chatbot.start(StartChatRequest(message='What is fever?'))
    assert state.ai_trace['understanding']['validation']=='no-write-on-analysis-failure'
    assert state.ai_trace['response_mode']=='multi-turn-llm'
    assert not state.evidence

def test_yes_to_general_help_cannot_confirm_adapter_question():
    nlp=NLPService(use_models=False)
    class Index:
        def search(self,*args,**kwargs):return []
    nlp.index=Index()
    state=SessionState(session_id='t',last_question=NextSymptom(kg_id='DEMO:fever',kg_name='fever',heuristic_score=0.5),
        messages=[ChatMessage(sender='assistant',text='Would you like help finding the right words?'),ChatMessage(sender='user',text='yes')])
    agent=ConversationAgent(Model({'observations':[{'concept_id':'DEMO:fever','quote':'yes','status':'present'}]}))
    evidence,trace=agent.understand('yes',state,nlp)
    assert not evidence and not state.pending_evidence

def test_legacy_context_flags_do_not_route_agent(chatbot):
    chatbot.dialogue=ConversationAgent(Model())
    state=chatbot.start(StartChatRequest(message='Tell me what this app does'))
    stored=chatbot.sessions.get(state.session_id);stored.dialogue_prompt='menu';stored.clarification_attempts=10;chatbot.sessions.save(stored)
    reply=send(chatbot,state,'yes')
    assert reply.ai_trace['engine']=='conversation-agent-v1'

def test_recent_context_keeps_complete_roles():
    model=Model();agent=ConversationAgent(model)
    history=[]
    for i in range(12):
        history.extend([ChatMessage(sender='user',text=f'message {i}'),ChatMessage(sender='assistant',text=f'reply {i}')])
    history.append(ChatMessage(sender='user',text='latest request'))
    state=SessionState(session_id='t',messages=history)
    agent.reply('latest request',state,{})
    sent=model.calls[-2]
    assert sent[1]['role']=='user' and sent[-1]['content']=='latest request'
    assert len(sent)<=17

def test_generated_adapter_suggestion_does_not_activate_buttons(chatbot):
    chatbot.dialogue=ConversationAgent(Model({'observations':[{'concept_id':'DEMO:fever','quote':'fever','status':'present'}]}))
    state=chatbot.start(StartChatRequest(message='I have fever'))
    assert state.next_symptom and state.symptom_question_active is False

def test_optional_analysis_descriptions_accept_empty_values():
    result=Understanding.model_validate({'unresolved':[],'reference_resolution':None})
    assert result.unresolved==result.reference_resolution==''

def test_invalid_observation_status_remains_rejected():
    with pytest.raises(ValueError):
        Understanding.model_validate({'observations':[{'concept_id':'DEMO:fever','quote':'fever','status':'yes'}]})

def test_repeated_reply_retry_receives_revision_feedback():
    model=Model();reviews=[]
    def chat(messages,*args):
        if messages[0]['content'].startswith('You edit'):
            payload=json.loads(messages[1]['content']);reviews.append(payload)
            return ('Please describe one symptom.' if len(reviews)==1 else 'For example, you can write: my head feels heavy.'),{'model':'test'}
        return 'Please describe one symptom.',{'model':'test'}
    model.chat=chat
    state=SessionState(session_id='t',messages=[ChatMessage(sender='assistant',text='Please describe one symptom.'),ChatMessage(sender='user',text='yes')])
    answer,trace=ConversationAgent(model).reply('yes',state,{})
    assert reviews[0]['revision_feedback']=='' and reviews[1]['revision_feedback']
    assert reviews[1]['latest_user_message']=='yes'
    assert answer.startswith('For example') and trace['response_mode']=='multi-turn-llm'

def test_model_generated_meanings_are_context_not_intent_labels():
    model=Model();agent=ConversationAgent(model)
    state=SessionState(session_id='t',messages=[ChatMessage(sender='user',text='What does that mean?')])
    agent.reply('What does that mean?',state,{'interpretation':{'possible_meanings':['a bodily sensation','a named illness']}})
    assert 'a bodily sensation' in model.calls[0][0]['content']
    assert 'a named illness' in model.calls[0][0]['content']
    assert not state.evidence

def test_direct_pipeline_keeps_context_and_skips_extra_generation(monkeypatch):
    monkeypatch.setenv('AGENT_REPLY_PIPELINE','direct')
    model=Model()
    state=SessionState(session_id='t',messages=[ChatMessage(sender='user',text='How can I describe it?')])
    answer,trace=ConversationAgent(model).reply('How can I describe it?',state,{'interpretation':{'user_goal':'needs ordinary words'}})
    assert len(model.calls)==1 and not trace.get('review_calls')
    assert 'needs ordinary words' in model.calls[0][1]['content']
    assert trace['reply_pipeline']=='direct' and answer.startswith('You can')

def test_direct_pipeline_reviews_detected_narration(monkeypatch):
    monkeypatch.setenv('AGENT_REPLY_PIPELINE','direct')
    model=Model()
    def chat(messages,*args):
        model.calls.append(messages)
        return ('The user feels bad.' if len(model.calls)==1 else 'You can start with where you feel uncomfortable.'),{}
    model.chat=chat
    state=SessionState(session_id='t',messages=[ChatMessage(sender='user',text='I feel bad')])
    answer,trace=ConversationAgent(model).reply('I feel bad',state,{})
    assert len(model.calls)==2 and trace['review_reason']=='output-validation'
    assert answer.startswith('You can')
