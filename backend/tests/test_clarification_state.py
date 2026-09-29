from uuid import uuid4
from app.models.chat_models import SessionState,ChatMessage,StartChatRequest,MessageRequest
from app.services.conversation_context import normalize,conversation_act
from app.services.dialogue_ai import DialogueAI

def test_typo_and_repair_phrase():
    state=SessionState(session_id='t',messages=[ChatMessage(sender='assistant',text='Please describe your symptoms.')])
    assert normalize('I dont get waht you said &#x20;')=="I don't get what you said"
    assert conversation_act('I dont get waht you said',state)=='repair'

def test_old_menu_and_numbered_choice():
    state=SessionState(session_id='t',messages=[ChatMessage(sender='assistant',text='Are you here to report symptoms, seek information on a health term, or need assistance with the app?')])
    assert conversation_act('yes',state)=='clarification_retry'
    assert conversation_act('2',state)=='ask_term'

def test_fields_survive_serialization_and_default_for_old_sessions():
    state=SessionState(session_id='t',dialogue_prompt='description',clarification_attempts=2)
    restored=SessionState.model_validate_json(state.model_dump_json())
    assert restored.dialogue_prompt=='description' and restored.clarification_attempts==2
    assert SessionState.model_validate({'session_id':'old'}).dialogue_prompt=='none'

def test_yes_during_repair_never_confirms_pending_symptom(chatbot):
    initial=chatbot.start(StartChatRequest(message='I have fever.'))
    assert initial.next_symptom
    class Generator:
        def generate(self,system,*args):
            return ('report' if 'category' in system else 'Please describe what feels different in your body.'),{'model':'test'}
    chatbot.dialogue=DialogueAI(Generator())
    result=initial
    replies=[]
    for message,expected in [('I dont get waht you said','repair'),('yes','clarification_retry'),('yes','clarification_stop')]:
        result=chatbot.message(MessageRequest(session_id=result.session_id,expected_version=result.version,request_id=uuid4().hex,message=message))
        assert result.ai_trace['intent']==expected
        assert result.evidence==initial.evidence and result.next_symptom==initial.next_symptom and result.round==initial.round
        replies.append(result.message)
    assert result.clarification_attempts==2
    assert replies[-1]!=replies[-2]
    result=chatbot.message(MessageRequest(session_id=result.session_id,expected_version=result.version,request_id=uuid4().hex,message='I have cough.'))
    assert result.dialogue_prompt=='none' and result.clarification_attempts==0
