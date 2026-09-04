import pytest

from app.models.chat_models import AnswerChatRequest, SessionStatus, StartChatRequest, Symptom
from app.services.chatbot_service import ChatbotService, InvalidAnswerError
from app.services.mock_reasoning_service import MockDrugService, MockInformedSearchService, MockRecallService
from app.services.nlp_service import NLPService
from app.services.session_service import InMemorySessionService


@pytest.fixture()
def chatbot():
    return ChatbotService(
        nlp=NLPService(use_models=False),
        sessions=InMemorySessionService(),
        recall=MockRecallService(),
        search=MockInformedSearchService(),
        drugs=MockDrugService(),
    )


def test_complete_multi_turn_workflow(chatbot):
    first = chatbot.start(StartChatRequest(message="I have fever and cough."))
    assert first.status == SessionStatus.RUNNING
    assert first.next_symptom.kg_name == "shortness of breath"

    second = chatbot.answer(
        AnswerChatRequest(
            session_id=first.session_id,
            answer="yes",
            symptom=Symptom(kg_id="P006", kg_name="shortness of breath"),
        )
    )
    assert second.status == SessionStatus.RUNNING
    assert {item.kg_id for item in second.positive_symptoms} == {"P001", "P002", "P006"}

    final = chatbot.answer(
        AnswerChatRequest(
            session_id=first.session_id,
            answer="no",
            symptom=Symptom(kg_id="P003", kg_name="chest pain"),
        )
    )
    assert final.status == SessionStatus.COMPLETE
    assert len(final.candidate_diseases) == 2
    assert len(final.drug_recommendations) == 2


def test_rejects_answer_for_a_different_question(chatbot):
    first = chatbot.start(StartChatRequest(message="I have fever."))
    with pytest.raises(InvalidAnswerError):
        chatbot.answer(
            AnswerChatRequest(
                session_id=first.session_id,
                answer="yes",
                symptom=Symptom(kg_id="P003", kg_name="chest pain"),
            )
        )
