from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from app.models.chat_models import StartChatRequest, MessageRequest, AnswerChatRequest, CorrectionRequest, Symptom
from app.services.chatbot_service import InvalidAnswerError
from app.services.remote_adapters import AdapterError

def base(response):
    return dict(session_id=response.session_id, expected_version=response.version, request_id=uuid4().hex)

def answer(service, response, value):
    return service.answer(AnswerChatRequest(**base(response), symptom=Symptom(kg_id=response.next_symptom.kg_id, kg_name="untrusted client label"), answer=value))

def test_complete_workflow_and_unknown(chatbot):
    first = chatbot.start(StartChatRequest(message="I have fever and cough."))
    second = answer(chatbot, first, "unknown")
    assert second.uncertain_symptoms[0].kg_name == "shortness of breath"
    assert "DEMO:breathlessness" not in [s.kg_id for s in second.negative_symptoms]
    assert second.next_symptom.kg_id == "DEMO:chest_pain"
    third = answer(chatbot, second, "yes")
    final = answer(chatbot, third, "no")
    assert final.status == "COMPLETE"
    assert final.stop_reason == "candidate_threshold"
    assert not final.drug_recommendations

def test_no_match_continues_same_session(chatbot):
    initial = chatbot.start(StartChatRequest(message="hello"))
    updated = chatbot.message(MessageRequest(**base(initial), message="I am coughing"))
    assert updated.session_id == initial.session_id
    assert len(updated.positive_symptoms) == 1
    assert len(updated.messages) == 4

def test_free_text_answer_and_correction(chatbot):
    first = chatbot.start(StartChatRequest(message="fever"))
    second = chatbot.message(MessageRequest(**base(first), message="not sure"))
    assert second.uncertain_symptoms
    changed = chatbot.correct(CorrectionRequest(**base(second), kg_id="DEMO:fever", status="absent"))
    assert not changed.positive_symptoms
    assert changed.negative_symptoms[0].kg_id == "DEMO:fever"
    assert len(changed.evidence_history) == 3

def test_implicit_conflict_requires_confirmation(chatbot):
    a = chatbot.start(StartChatRequest(message="fever"))
    b = chatbot.message(MessageRequest(**base(a), message="no fever"))
    assert b.uncertain_symptoms[0].kg_id == "DEMO:fever"
    assert b.warnings

def test_explicit_correction_and_complete_reopen(chatbot):
    a = chatbot.start(StartChatRequest(message="fever cough headache fatigue"))
    assert a.status == "COMPLETE"
    b = chatbot.message(MessageRequest(**base(a), message="Actually no fever"))
    assert "DEMO:fever" in [s.kg_id for s in b.negative_symptoms]

def test_idempotency_and_payload_collision(chatbot):
    request = StartChatRequest(message="fever")
    a = chatbot.start(request)
    assert chatbot.start(request) == a
    with pytest.raises(InvalidAnswerError):
        chatbot.start(request.model_copy(update={"message":"cough"}))
    request2 = MessageRequest(**base(a), message="cough")
    b = chatbot.message(request2)
    assert chatbot.message(request2) == b
    assert chatbot.inspect(a.session_id).version == 2

def test_stale_version_and_wrong_question(chatbot):
    a = chatbot.start(StartChatRequest(message="fever"))
    with pytest.raises(InvalidAnswerError):
        chatbot.answer(AnswerChatRequest(**base(a), symptom=Symptom(kg_id="wrong",kg_name="x"), answer="yes"))
    answer(chatbot, a, "no")
    with pytest.raises(InvalidAnswerError):
        chatbot.message(MessageRequest(**base(a),message="cough"))

def test_simultaneous_updates_only_one_commit(chatbot):
    a = chatbot.start(StartChatRequest(message="fever"))
    def send(text):
        try:
            return chatbot.message(MessageRequest(**base(a),message=text)).version
        except InvalidAnswerError:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(send, ["cough","headache"]))
    assert sorted(map(str, results)) == ["2","conflict"]

def test_failure_rolls_back_state_and_can_retry(chatbot):
    a = chatbot.start(StartChatRequest(message="fever"))
    original = chatbot.recall.recall
    chatbot.recall.recall = lambda *args: (_ for _ in ()).throw(AdapterError("offline"))
    req = MessageRequest(**base(a), message="cough")
    with pytest.raises(AdapterError):
        chatbot.message(req)
    assert chatbot.inspect(a.session_id) == a
    chatbot.recall.recall = original
    assert chatbot.message(req).version == 2

def test_no_repeated_questions_max_rounds_and_no_candidates(chatbot):
    a = chatbot.start(StartChatRequest(message="fever"))
    chatbot.max_rounds = 1
    b = answer(chatbot, a, "unknown")
    assert b.stop_reason == "max_rounds"
    chatbot.recall.recall = lambda *args: []
    c = chatbot.start(StartChatRequest(message="cough"))
    assert c.stop_reason == "no_candidates"

def test_unknown_exhaustion_and_bad_search_contract(chatbot):
    a = chatbot.start(StartChatRequest(message="fever"))
    for _ in range(3):
        a = answer(chatbot, a, "unknown")
    assert a.stop_reason == "no_questions"
    chatbot.search.select_next = lambda *args: type("Q", (), {"kg_id":"not-in-vocabulary"})()
    with pytest.raises(AdapterError):
        chatbot.start(StartChatRequest(message="cough"))

