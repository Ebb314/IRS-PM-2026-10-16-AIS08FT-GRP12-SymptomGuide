from uuid import uuid4
import pytest
from app.models.chat_models import StartChatRequest, CorrectionRequest, ExtractedSymptom
from app.services.chatbot_service import InvalidAnswerError

def request(state, status, kg_id='DEMO:fever'):
    return CorrectionRequest(session_id=state.session_id, expected_version=state.version, request_id=uuid4().hex, kg_id=kg_id, status=status)

def pending(chatbot):
    state=chatbot.start(StartChatRequest(message='hello'))
    stored=chatbot.sessions.get(state.session_id)
    stored.pending_evidence=[ExtractedSymptom(kg_id='DEMO:fever',kg_name='fever',original_text='burning up',source='unconfirmed-agent-proposal')]
    chatbot.sessions.save(stored)
    # Explicit record edits must not call the conversation model.
    class Agent:
        def understand(self,*args): raise AssertionError('No inference for UI edit')
    chatbot.dialogue=Agent()
    return state

@pytest.mark.parametrize('status,field',[('present','positive_symptoms'),('absent','negative_symptoms'),('uncertain','uncertain_symptoms')])
def test_confirm_pending_persists_and_replays(chatbot,status,field):
    state=pending(chatbot); req=request(state,status)
    result=chatbot.correct(req)
    assert result.evidence[0].status==status and not result.pending_evidence
    assert getattr(result,field)[0].kg_id=='DEMO:fever'
    assert result.evidence_history[-1].source=='user-correction'
    assert not result.symptom_question_active
    assert chatbot.correct(req)==result
    assert chatbot.inspect(state.session_id)==result
    with pytest.raises(InvalidAnswerError): chatbot.correct(request(state,'absent'))

def test_ignore_match_does_not_mean_absent(chatbot):
    state=pending(chatbot)
    result=chatbot.correct(request(state,'dismiss'))
    assert not result.pending_evidence and not result.evidence and not result.negative_symptoms
    assert not result.evidence_history

def test_cannot_fabricate_symptom_or_dismiss_saved_record(chatbot):
    state=pending(chatbot)
    with pytest.raises(InvalidAnswerError): chatbot.correct(request(state,'present','UNKNOWN'))
    result=chatbot.correct(request(state,'present'))
    with pytest.raises(InvalidAnswerError): chatbot.correct(request(result,'dismiss'))
    updated=chatbot.correct(request(result,'uncertain'))
    assert not updated.positive_symptoms and not updated.negative_symptoms
    assert len(updated.uncertain_symptoms)==1 and len(updated.evidence_history)==2
