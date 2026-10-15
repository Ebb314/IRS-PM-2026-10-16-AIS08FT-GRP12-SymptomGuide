"""Agent dialogue keeps the informed-search interview going (kg / ddxplus / recall773 modes).

Regression for two reported bugs:
  1. confirming a suggested symptom (Present / Absent / Not sure) ended with "Saved: ..." and no next question;
  2. after small talk (no symptom in the message) the pending question was never asked again.
"""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.models.chat_models import AnswerChatRequest, CorrectionRequest, ExtractedSymptom, MessageRequest, StartChatRequest, Symptom
from app.services.chatbot_service import ChatbotService
from app.services.conversation_agent import ConversationAgent
from app.services.mock_reasoning_service import MockDrugService
from app.services.nlp_service import NLPService
from app.services.random_reasoning_service import RandomRecallService, RandomSearchService
from app.services.recall773_reasoning_service import EXTRA_NODES_FILE, build_recall773_services
from app.services.session_service import InMemorySessionService

DATA = Path(__file__).parents[1] / "app/data"


class ScriptedGenerator:
    """LLM stand-in: understand() returns the next scripted observation list; replies are fixed."""
    name = "fake"

    def __init__(self, nlp):
        self.ids = {n.kg_name: k for k, n in nlp.nodes.items()}
        self.script = []

    def generate(self, system, user, max_tokens=220):
        names = self.script.pop(0) if self.script else []
        obs = [{"concept_id": self.ids[n], "quote": q, "status": "present"} for n, q in names]
        return json.dumps({"user_goal": "chat", "observations": obs}), {"model": "fake"}

    def chat(self, messages, max_tokens=260, context_tokens=4096, sampling=True):
        return "Happy to help.", {"model": "fake"}


@pytest.fixture(scope="module")
def nlp():
    return NLPService(DATA / "kg_nodes.csv", use_models=False, extra_vocab_file=str(EXTRA_NODES_FILE))


@pytest.fixture(scope="module")
def recall773(nlp):
    return build_recall773_services(nlp.nodes, str(DATA / "kg_reasoning.json.gz"), backend="numpy")


def make(nlp, services, mode):
    gen = ScriptedGenerator(nlp)
    store = InMemorySessionService()
    return ChatbotService(nlp, store, *services, mode=mode, max_rounds=12, dialogue=ConversationAgent(gen)), gen, store


def suggest(store, r, nlp, name):
    """Put an unconfirmed suggested match on the session, as the agent does for a vague phrase."""
    state = store.get(r.session_id)
    node = next(n for n in nlp.nodes.values() if n.kg_name == name)
    state.pending_evidence = [ExtractedSymptom(**node.model_dump(), status="present", original_text="my body feels hot",
                                               turn=state.version, source="unconfirmed-agent-proposal")]
    store.save(state)
    return state.version


@pytest.mark.parametrize("status", ["present", "absent", "uncertain"])
def test_confirming_a_suggested_first_symptom_continues_with_a_question(nlp, recall773, status):
    chat, gen, store = make(nlp, recall773, "recall773")
    r = chat.start(StartChatRequest(message="I have a cough."))          # scripted: nothing recognised by the LLM
    version = suggest(store, r, nlp, "Fever")
    gen.script = []
    r = chat.correct(CorrectionRequest(session_id=r.session_id, expected_version=version, request_id=uuid4().hex,
                                       kg_id=next(k for k, n in nlp.nodes.items() if n.kg_name == "Fever"), status=status))
    assert r.message.startswith("Saved: Fever")
    if status == "present":
        assert r.status == "COMPLETE" or (r.next_symptom and r.symptom_question_active
                                          and f"you experiencing {r.next_symptom.kg_name}?" in r.message)
    else:
        # Nothing present on record yet: there is no question to ask, and no result either.
        assert not r.candidate_diseases


def test_small_talk_re_asks_the_pending_question(nlp, recall773):
    chat, gen, store = make(nlp, recall773, "recall773")
    gen.script = [[("Fever", "fever"), ("Cough", "cough")]]
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    q = r.next_symptom
    assert q and r.symptom_question_active
    gen.script = [[]]                                                       # small talk: no symptom
    r2 = chat.message(MessageRequest(session_id=r.session_id, expected_version=r.version, request_id=uuid4().hex,
                                     message="Thanks! By the way, is it normal to feel this tired in winter?"))
    assert r2.message.startswith("Happy to help.")
    assert f"To continue: are you experiencing {q.kg_name}?" in r2.message
    assert r2.next_symptom.kg_id == q.kg_id and r2.symptom_question_active
    # The buttons still work afterwards.
    r3 = chat.answer(AnswerChatRequest(session_id=r2.session_id, expected_version=r2.version, request_id=uuid4().hex,
                                       answer="yes", symptom=Symptom(kg_id=q.kg_id, kg_name=q.kg_name)))
    assert r3.status == "COMPLETE" or (r3.next_symptom and r3.next_symptom.kg_id != q.kg_id)


def test_small_talk_before_any_symptom_asks_nothing(nlp, recall773):
    chat, gen, store = make(nlp, recall773, "recall773")
    gen.script = [[]]
    r = chat.start(StartChatRequest(message="Hello, how does this app work?"))
    assert r.message == "Happy to help." and not r.symptom_question_active and not r.next_symptom


def test_small_talk_after_the_result_does_not_restart_questions(nlp, recall773, monkeypatch):
    chat, gen, store = make(nlp, recall773, "recall773")
    model = recall773[0].data.model
    import numpy as np
    p = np.full(len(model.diseases), 0.05 / (len(model.diseases) - 3))
    for d, v in [("pneumonia", 0.6), ("acute bronchitis", 0.2), ("flu", 0.15)]:
        p[model.diseases.index(d)] = v
    monkeypatch.setattr(model, "probabilities", lambda *a, **k: p)
    gen.script = [[("Fever", "fever"), ("Cough", "cough")], []]
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    assert r.status == "COMPLETE"
    r2 = chat.message(MessageRequest(session_id=r.session_id, expected_version=r.version, request_id=uuid4().hex,
                                     message="Thank you!"))
    assert r2.message == "Happy to help." and not r2.symptom_question_active


def test_random_mode_confirmation_unchanged(nlp):
    services = (RandomRecallService(), RandomSearchService(nlp.nodes), MockDrugService())
    chat, gen, store = make(nlp, services, "random")
    r = chat.start(StartChatRequest(message="I have a cough."))
    version = suggest(store, r, nlp, "Fever")
    r = chat.correct(CorrectionRequest(session_id=r.session_id, expected_version=version, request_id=uuid4().hex,
                                       kg_id=next(k for k, n in nlp.nodes.items() if n.kg_name == "Fever"), status="present"))
    assert r.message == "Saved: Fever — Present. You can change this again in Symptom log."
    assert r.symptom_question_active is False
