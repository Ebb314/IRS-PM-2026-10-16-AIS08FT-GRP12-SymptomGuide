import pytest
from app.models.chat_models import SessionState,ChatMessage,StartChatRequest,MessageRequest
from app.services.conversation_context import normalize,conversation_act,is_meta_narration
from app.services.dialogue_ai import DialogueAI

@pytest.mark.parametrize('text,act',[('How to describe?','guide'),('I feel bad but i dont know how to say','guide'),('What?','repair'),('Whats cold','clarify_term'),('What is fever?',None)])
def test_context_cues(text,act):
    state=SessionState(session_id='t',messages=[ChatMessage(sender='assistant',text='Describe what feels different.')])
    assert conversation_act(text,state)==act

def test_informal_input_normalization():
    assert normalize("Whats fever &#x20;")=='what is fever'
    assert normalize("I dont know") == "I don't know"

@pytest.mark.parametrize('text',['The user feels unwell but cannot describe it.','The question "What?" has been asked.','The question was answered.','[Insert symptom here]'])
def test_meta_narration_rejected(text):
    assert is_meta_narration(text)

def test_support_preserves_state_and_passes_history(chatbot):
    started=chatbot.start(StartChatRequest(message='I have fever.'))
    calls=[]
    class Generator:
        def generate(self,system,user,*args):
            calls.append(user)
            return ('explain' if len(calls)==1 else 'No medical words needed. What feels different in your body?'),{'model':'test'}
    chatbot.dialogue=DialogueAI(Generator())
    result=chatbot.message(MessageRequest(session_id=started.session_id,expected_version=started.version,request_id='guide-test',message='I dont know how to describe it'))
    assert result.evidence==started.evidence and result.next_symptom==started.next_symptom
    assert result.round==started.round and result.knowledge_focus==started.knowledge_focus
    assert result.ai_trace['response_mode']=='local-llm-support'
    assert 'Recent conversation' in calls[-1] and 'fever' in calls[-1]

def test_support_rejects_narration_and_asks_useful_question():
    class Generator:
        def generate(self,*args):return 'The user feels bad.',{}
    from app.services.intent_service import IntentService
    answer,trace=IntentService(Generator()).support('I feel bad',SessionState(session_id='t'),'guide')
    assert trace['response_mode']=='support-fallback'
    assert answer.endswith('?') and not is_meta_narration(answer)
