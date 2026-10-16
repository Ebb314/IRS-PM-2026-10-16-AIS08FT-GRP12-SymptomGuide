"""recall773 stop rule: end the interview when the top-3 diseases hold >= 0.9 of the probability.

Probabilities are pinned with monkeypatch so every boundary is exact; the last tests use the real model.
"""
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.config import settings
from app.models.chat_models import AnswerChatRequest, StartChatRequest, Symptom
from app.services.chatbot_service import ChatbotService
from app.services.ddxplus_reasoning_service import build_ddxplus_services
from app.services.dialogue_ai import DialogueAI
from app.services.nlp_service import NLPService
from app.services.recall773_reasoning_service import EXTRA_NODES_FILE, build_recall773_services
from app.services.session_service import InMemorySessionService

DATA = Path(__file__).parents[1] / "app/data"
KG = str(DATA / "kg_reasoning.json.gz")


@pytest.fixture(scope="module")
def nlp():
    return NLPService(DATA / "kg_nodes.csv", use_models=False, extra_vocab_file=str(EXTRA_NODES_FILE))


@pytest.fixture(scope="module")
def services(nlp):
    return build_recall773_services(nlp.nodes, KG, backend="numpy")


@pytest.fixture
def chat(nlp, services):
    store = InMemorySessionService()
    yield ChatbotService(nlp, store, *services, mode="recall773", max_rounds=12,
                         dialogue=DialogueAI(vocabulary=nlp.nodes, mention_finder=nlp._mentions))
    store.close()


def sym(nlp, name):
    node = next(n for n in nlp.nodes.values() if n.kg_name == name)
    return Symptom(kg_id=node.kg_id, kg_name=node.kg_name)


def pinned(model, top):
    """Probability vector: the given {disease: p}; the rest spread evenly over the other diseases."""
    p = np.zeros(len(model.diseases))
    rest = (1 - sum(top.values())) / (len(model.diseases) - len(top))
    p[:] = rest
    for disease, value in top.items():
        p[model.diseases.index(disease)] = value
    return p


def pin(monkeypatch, services, top):
    model = services[0].data.model
    vector = pinned(model, top)
    monkeypatch.setattr(model, "probabilities", lambda *a, **k: vector)


def answer(chat, r, value):
    q = r.next_symptom
    return chat.answer(AnswerChatRequest(session_id=r.session_id, expected_version=r.version, request_id=uuid4().hex,
                                         answer=value, symptom=Symptom(kg_id=q.kg_id, kg_name=q.kg_name)))


# ---- ST01-ST03: configuration ----
def test_ST01_default_threshold_is_0_9_and_result_size_is_three(services, chat):
    assert services[0].stop_top3_mass == 0.9 and settings.recall773_stop_top3_mass == 0.9
    assert chat.finish_threshold == 3


def test_ST02_threshold_is_configurable(nlp):
    recall = build_recall773_services(nlp.nodes, KG, backend="numpy", stop_top3_mass=0.75)[0]
    assert recall.stop_top3_mass == 0.75


def test_ST03_ddxplus_mode_keeps_its_own_rule(nlp):
    recall = build_ddxplus_services(nlp.nodes, KG, backend="numpy")[0]
    assert recall.stop_top3_mass is None and not hasattr(recall, "finish_threshold")


# ---- ST04-ST07: the rule itself (recall output) ----
@pytest.mark.parametrize("top, stops", [
    ({"flu": 0.5, "common cold": 0.3, "acute bronchitis": 0.1}, True),      # exactly 0.90 -> stop
    ({"flu": 0.5, "common cold": 0.3, "acute bronchitis": 0.099}, False),   # 0.899 -> continue
    ({"flu": 0.85, "common cold": 0.04}, False),                            # strong top-1 alone is not enough
    ({"flu": 0.34, "common cold": 0.33, "acute bronchitis": 0.33}, True),   # three-way tie still converges
])
def test_ST04_boundary(monkeypatch, services, nlp, top, stops):
    pin(monkeypatch, services, top)
    out = services[0].recall([sym(nlp, "Fever"), sym(nlp, "Cough")], [])
    if stops:
        assert {c.name for c in out} == set(sorted(top, key=top.get, reverse=True)[:3]) and len(out) == 3
        assert [c.score for c in out] == sorted((c.score for c in out), reverse=True)   # ties: stable order
    else:
        assert len(out) >= 4


def test_ST05_not_converged_keeps_at_least_four(monkeypatch, services, nlp):
    pin(monkeypatch, services, {"flu": 0.3, "common cold": 0.2})
    out = services[0].recall([sym(nlp, "Fever")], [])
    assert 4 <= len(out) <= services[0].top_k
    assert [c.score for c in out] == sorted((c.score for c in out), reverse=True)


def test_ST06_denials_alone_never_produce_candidates(services, nlp):
    assert services[0].recall([], [sym(nlp, "Fever"), sym(nlp, "Cough")]) == []


# ---- ST07-ST11: whole sessions ----
def test_ST07_converged_session_shows_top3_disclaimer_and_drugs(monkeypatch, services, chat):
    pin(monkeypatch, services, {"pneumonia": 0.6, "acute bronchitis": 0.2, "flu": 0.15})
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    assert r.status == "COMPLETE" and r.stop_reason == "candidate_threshold"
    assert [c.name for c in r.candidate_diseases] == ["pneumonia", "acute bronchitis", "flu"]
    assert "not a diagnosis" in r.message.lower()
    assert r.drug_recommendations and all(d.source and d.relation for d in r.drug_recommendations)


def test_ST08_unconverged_session_asks_and_shows_no_drugs(monkeypatch, services, chat):
    pin(monkeypatch, services, {"pneumonia": 0.3, "acute bronchitis": 0.2})
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    assert r.status == "RUNNING" and r.next_symptom and r.symptom_question_active
    assert len(r.candidate_diseases) >= 4 and not r.drug_recommendations


def test_ST09_max_rounds_ends_without_drugs(monkeypatch, services, chat):
    pin(monkeypatch, services, {"pneumonia": 0.3, "acute bronchitis": 0.2})
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    while r.status == "RUNNING" and r.next_symptom:
        assert not r.drug_recommendations
        r = answer(chat, r, "no")
    assert r.stop_reason in {"max_rounds", "no_questions"} and r.round <= 12
    assert not r.drug_recommendations and "insufficient" in r.message.lower()


def test_ST10_rule_reached_mid_interview(monkeypatch, services, chat):
    model = services[0].data.model
    diffuse = pinned(model, {"pneumonia": 0.3, "acute bronchitis": 0.2})
    peaked = pinned(model, {"pneumonia": 0.7, "acute bronchitis": 0.15, "flu": 0.1})
    # The second recall (after one Yes) becomes confident.
    monkeypatch.setattr(model, "probabilities", lambda present, absent=(), t=1.0: peaked if len(present) > 2 else diffuse)
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    assert r.status == "RUNNING"
    r = answer(chat, r, "yes")
    assert r.status == "COMPLETE" and r.stop_reason == "candidate_threshold" and len(r.candidate_diseases) == 3


def test_ST11_not_sure_is_not_evidence(monkeypatch, services, chat):
    calls = []
    model = services[0].data.model
    diffuse = pinned(model, {"pneumonia": 0.3, "acute bronchitis": 0.2})
    monkeypatch.setattr(model, "probabilities", lambda present, absent=(), t=1.0: calls.append((list(present), list(absent))) or diffuse)
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    r = answer(chat, r, "unknown")
    assert r.status == "RUNNING"
    assert calls[-1] == calls[0]   # same model input: "Not sure" is neither present nor absent


# ---- ST12-ST13: real model ----
def test_ST12_real_model_converges_to_three(chat):
    r = chat.start(StartChatRequest(message="I have diarrhea and vomiting."))
    while r.status == "RUNNING" and r.next_symptom:
        r = answer(chat, r, "yes")
    assert r.stop_reason == "candidate_threshold" and len(r.candidate_diseases) == 3
    assert sum(c.score for c in r.candidate_diseases) >= 0.9 - 1e-3


def test_ST13_real_model_all_no_terminates_cleanly(chat):
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    while r.status == "RUNNING" and r.next_symptom:
        r = answer(chat, r, "no")
    assert r.stop_reason in {"candidate_threshold", "max_rounds", "no_questions"} and r.round <= 12
    if r.stop_reason == "candidate_threshold":
        assert len(r.candidate_diseases) == 3 and sum(c.score for c in r.candidate_diseases) >= 0.9 - 1e-3
    else:
        assert not r.drug_recommendations
