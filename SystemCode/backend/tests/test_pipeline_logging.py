"""Every pipeline step logs, and logs carry neither raw user text nor session IDs by default."""
import logging
from pathlib import Path
from uuid import uuid4

import pytest

from app.models.chat_models import AnswerChatRequest, StartChatRequest, Symptom
from app.services.chatbot_service import ChatbotService
from app.services.dialogue_ai import DialogueAI
from app.services.nlp_service import NLPService
from app.services.recall773_reasoning_service import EXTRA_NODES_FILE, build_recall773_services
from app.services.session_service import InMemorySessionService

DATA = Path(__file__).parents[1] / "app/data"


@pytest.fixture
def records(caplog):
    root = logging.getLogger("chatbot")
    root.addHandler(caplog.handler)   # works whether or not setup_logging() disabled propagation
    caplog.set_level(logging.DEBUG, logger="chatbot")
    yield caplog
    root.removeHandler(caplog.handler)


def test_each_step_logs_without_user_text_or_session_id(records, monkeypatch):
    monkeypatch.delenv("LOG_USER_TEXT", raising=False)
    nlp = NLPService(DATA / "kg_nodes.csv", use_models=False, extra_vocab_file=str(EXTRA_NODES_FILE))
    recall, search, drugs = build_recall773_services(nlp.nodes, str(DATA / "kg_reasoning.json.gz"), backend="numpy")
    store = InMemorySessionService()
    chat = ChatbotService(nlp, store, recall, search, drugs, mode="recall773", max_rounds=12,
                          dialogue=DialogueAI(vocabulary=nlp.nodes, mention_finder=nlp._mentions))
    secret = "I have diarrhea and vomiting since my trip to Lisbon."
    try:
        r = chat.start(StartChatRequest(message=secret))
        while r.status == "RUNNING" and r.next_symptom:
            q = r.next_symptom
            r = chat.answer(AnswerChatRequest(session_id=r.session_id, expected_version=r.version,
                request_id=uuid4().hex, answer="yes", symptom=Symptom(kg_id=q.kg_id, kg_name=q.kg_name)))
    finally:
        store.close()
    assert r.stop_reason == "candidate_threshold"
    steps = {rec.name for rec in records.records}
    assert {"chatbot.flow", "chatbot.nlp", "chatbot.recall", "chatbot.search", "chatbot.drugs"} <= steps
    text = "\n".join(rec.getMessage() for rec in records.records)
    assert "Lisbon" not in text and r.session_id not in text
    for expected in ("model input", "model top-5", "top-3 mass", "chose ", "STOP reason=candidate_threshold", "drugs ->"):
        assert expected in text
