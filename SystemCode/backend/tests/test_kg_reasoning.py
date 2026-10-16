"""REASONING_MODE=kg: PrimeKG recall, informed search and drug relations behind the adapter interfaces."""
from pathlib import Path
from uuid import uuid4

import pytest

from app.models.chat_models import AnswerChatRequest, DiseaseCandidate, StartChatRequest, Symptom
from app.services.chatbot_service import ChatbotService
from app.services.dialogue_ai import DialogueAI
from app.services.kg_reasoning_service import (
    KGDrugService, KGSearchService, build_kg_services, load_kg_data,
)
from app.services.nlp_service import NLPService
from app.services.session_service import InMemorySessionService

DATA = Path(__file__).parents[1] / "app/data"
KG_FILE = str(DATA / "kg_reasoning.json.gz")


@pytest.fixture(scope="module")
def full_nlp():
    return NLPService(DATA / "kg_nodes.csv", use_models=False)


@pytest.fixture(scope="module")
def services(full_nlp):
    return build_kg_services(KG_FILE, full_nlp.nodes)


@pytest.fixture(scope="module")
def data():
    return load_kg_data(KG_FILE)


def sym(nlp, name):
    node = next(n for n in nlp.nodes.values() if n.kg_name == name)
    return Symptom(kg_id=node.kg_id, kg_name=node.kg_name)


def test_data_has_no_non_symptom_terms(data):
    names = {s for v in data.disease_symptoms.values() for s in v}
    for bad in ("Autosomal recessive inheritance", "Autosomal dominant inheritance",
                "Variable expressivity", "Infantile onset", "All"):
        assert bad not in names
    assert len(data.disease_symptoms) > 6000


def test_recall_ranks_and_reports_evidence(services, full_nlp):
    recall = services[0]
    fever, cough = sym(full_nlp, "Fever"), sym(full_nlp, "Cough")
    out = recall.recall([fever, cough], [])
    assert 0 < len(out) <= 10
    assert [c.score for c in out] == sorted((c.score for c in out), reverse=True)
    assert abs(sum(c.score for c in out) - 1) < 0.01
    for c in out:
        assert set(c.evidence_ids) <= {fever.kg_id, cough.kg_id} and c.evidence_ids
        assert c.id.startswith("PRIMEKG:")
    assert recall.recall([], [fever]) == []  # denials alone explain nothing


def test_no_answer_removes_every_disease_with_that_symptom(services, full_nlp, data):
    recall = services[0]
    fever, cough, fatigue = (sym(full_nlp, n) for n in ("Fever", "Cough", "Fatigue"))
    before = recall.recall([fever, cough], [])
    assert any("Fatigue" in data.disease_symptoms[c.name] for c in before)
    after = recall.recall([fever, cough], [fatigue])
    assert after and all("Fatigue" not in data.disease_symptoms[c.name] for c in after)


def test_confirmed_symptom_removes_disease_that_lacks_it(full_nlp, data):
    from app.services.kg_reasoning_service import KGRecallService
    recall = KGRecallService(data, top_k=100_000)  # inspect the whole pool, not only the top 10
    vocab = {n.kg_name for n in full_nlp.nodes.values()}
    disease, lacked = next((d, s[0]) for d, s in data.disease_lacks.items()
                           if d in data.disease_symptoms and s[0] in vocab)
    shared = data.disease_symptoms[disease][0]
    assert disease in {c.name for c in recall.recall([sym(full_nlp, shared)], [])}
    with_lacked = recall.recall([sym(full_nlp, shared), sym(full_nlp, lacked)], [])
    assert disease not in {c.name for c in with_lacked}


def test_search_returns_unasked_vocabulary_symptom(services, full_nlp):
    recall, search, _ = services
    fever, cough = sym(full_nlp, "Fever"), sym(full_nlp, "Cough")
    cands = recall.recall([fever, cough], [])
    q = search.select_next(cands, [fever, cough], [], [])
    assert q.kg_id in full_nlp.nodes and q.kg_name == full_nlp.nodes[q.kg_id].kg_name
    assert q.kg_id not in {fever.kg_id, cough.kg_id}
    assert 0 < q.heuristic_score <= 0.5 and "best splits" in q.rationale
    q2 = search.select_next(cands, [fever, cough], [], [q.kg_id])
    assert q2 is None or q2.kg_id != q.kg_id
    assert search.select_next(cands[:1], [fever], [], []) is None  # nothing left to separate


def test_search_whitelist_limits_questions(services, full_nlp):
    recall, search, _ = services
    fever, cough = sym(full_nlp, "Fever"), sym(full_nlp, "Cough")
    cands = recall.recall([fever, cough], [])
    first = search.select_next(cands, [fever, cough], [], [])
    restricted = KGSearchService(search.data, full_nlp.nodes, whitelist={"Fatigue"})
    q = restricted.select_next(cands, [fever, cough], [], [])
    assert q is None or q.kg_name == "Fatigue"
    if first.kg_name != "Fatigue":
        assert q is None or q.kg_name != first.kg_name


def test_drugs_exclude_contraindications_and_label_relations(data):
    a, b = next((a, b) for a in data.disease_drugs for b in data.disease_drugs
                if a != b and set(data.disease_drugs[a]) & set(data.contraindications.get(b, [])))
    unsafe = set(data.contraindications[b])
    drugs = KGDrugService(data, per_disease=100).recommend([
        DiseaseCandidate(id=data.disease_ids[a], name=a, score=.6),
        DiseaseCandidate(id=data.disease_ids[b], name=b, score=.4)])
    assert drugs and not {d.name for d in drugs} & unsafe
    assert {d.relation for d in drugs} <= {"indication", "off-label use"}
    assert all(d.source == "PrimeKG project KG" for d in drugs)
    capped = KGDrugService(data, per_disease=1).recommend([DiseaseCandidate(id="x", name=a, score=1)])
    assert len(capped) <= 1


def test_kg_mode_rejects_demo_vocabulary():
    demo = NLPService(DATA / "demo_symptoms.csv", use_models=False)
    with pytest.raises(ValueError, match="full KG vocabulary"):
        build_kg_services(KG_FILE, demo.nodes)


def test_full_session_converges_through_orchestrator(full_nlp, services, data):
    recall, search, drugs = services
    store = InMemorySessionService()
    chat = ChatbotService(full_nlp, store, recall, search, drugs, mode="kg", max_rounds=12,
                          dialogue=DialogueAI(vocabulary=full_nlp.nodes, mention_finder=full_nlp._mentions))
    try:
        r = chat.start(StartChatRequest(message="I have a fever and a cough."))
        assert r.mode == "kg" and r.candidate_diseases and r.next_symptom
        true = r.candidate_diseases[0].name
        seen = set()
        while r.status == "RUNNING" and r.next_symptom:
            assert r.next_symptom.kg_id not in seen
            seen.add(r.next_symptom.kg_id)
            answer = "yes" if r.next_symptom.kg_name in data.disease_symptoms[true] else "no"
            r = chat.answer(AnswerChatRequest(session_id=r.session_id, expected_version=r.version,
                request_id=uuid4().hex, answer=answer,
                symptom=Symptom(kg_id=r.next_symptom.kg_id, kg_name=r.next_symptom.kg_name)))
        assert r.stop_reason in {"candidate_threshold", "no_questions", "max_rounds"}
        assert true in {c.name for c in r.candidate_diseases}
        negatives = {s.kg_name for s in r.negative_symptoms}
        assert all(not negatives & set(data.disease_symptoms[c.name]) for c in r.candidate_diseases)
    finally:
        store.close()
