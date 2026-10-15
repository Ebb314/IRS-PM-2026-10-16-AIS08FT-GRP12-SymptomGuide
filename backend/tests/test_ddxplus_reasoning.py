"""REASONING_MODE=ddxplus: teammate's tri-state recall model behind the chatbot interfaces."""
import csv
import gzip
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.models.chat_models import AnswerChatRequest, DiseaseCandidate, StartChatRequest, Symptom
from app.services.chatbot_service import ChatbotService
from app.services.ddxplus_reasoning_service import (
    MAPPING_DIR, DDXPlusData, DDXPlusRecallService, DDXPlusSearchService, build_ddxplus_services,
)
from app.services.ddxplus_runtime import ARTIFACTS, TristateRecallModel
from app.services.dialogue_ai import DialogueAI
from app.services.nlp_service import NLPService
from app.services.session_service import InMemorySessionService

DATA = Path(__file__).parents[1] / "app/data"
KG_FILE = str(DATA / "kg_reasoning.json.gz")


@pytest.fixture(scope="module")
def nlp():
    return NLPService(DATA / "kg_nodes.csv", use_models=False)


@pytest.fixture(scope="module")
def model():
    return TristateRecallModel(backend="numpy")


@pytest.fixture(scope="module")
def services(nlp):
    return build_ddxplus_services(nlp.nodes, KG_FILE, backend="numpy")


def sym(nlp, name):
    node = next(n for n in nlp.nodes.values() if n.kg_name == name)
    return Symptom(kg_id=node.kg_id, kg_name=node.kg_name)


def test_numpy_runtime_reproduces_teammate_predictions(model):
    examples = json.loads((ARTIFACTS / "example_predictions.json").read_text(encoding="utf-8"))
    for case in examples.values():
        logits = model.logits(case["known_present"], case["known_absent"])
        order = np.argsort(-logits)[: len(case["candidates"])]
        assert [model.diseases[i] for i in order] == [c["disease_name"] for c in case["candidates"]]
        assert np.allclose([logits[i] for i in order], [c["score"] for c in case["candidates"]], atol=1e-5)


def test_torch_and_numpy_backends_agree_when_torch_is_installed(model):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    torch_model = TristateRecallModel(backend="torch")
    for present, absent in [(["E_201", "E_66"], ["E_91"]), (["E_103"], []), (["E_14", "E_66"], ["E_91", "E_201"])]:
        assert np.allclose(torch_model.logits(present, absent), model.logits(present, absent), atol=1e-5)


def test_mapping_tables_match_vocabularies(nlp, model):
    kg = json.load(gzip.open(KG_FILE, "rt", encoding="utf-8"))
    names = {n.kg_name: k for k, n in nlp.nodes.items()}
    with open(MAPPING_DIR / "symptom_map.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["evidence_id"] for r in rows} == set(model.evidence_ids)
    for r in rows:
        assert r["mapping_status"] in {"draft-exact", "draft-close", "draft-synonym", "unmapped"}
        if r["mapping_status"] != "unmapped":
            assert names[r["kg_name"]] == r["kg_id"]
    with open(MAPPING_DIR / "disease_map.csv", encoding="utf-8") as f:
        diseases = list(csv.DictReader(f))
    assert [d["ddx_name"] for d in diseases] == model.diseases
    for d in diseases:
        if d["primekg_name"]:
            assert kg["disease_ids"][d["primekg_name"]] == d["primekg_id"]


def test_unknown_is_never_sent_and_unmapped_input_is_ignored(services, nlp):
    recall = services[0]
    assert recall.recall([sym(nlp, "Headache")], []) == []          # not a DDXPlus symptom
    assert recall.recall([], [sym(nlp, "Fever")]) == []              # denials alone explain nothing
    with_fever_denied = recall.recall([sym(nlp, "Cough")], [sym(nlp, "Fever")])
    cough_only = recall.recall([sym(nlp, "Cough")], [])
    assert [c.score for c in with_fever_denied] != [c.score for c in cough_only]  # absent channel is used


def test_candidates_are_probability_ranked_and_supported(services, nlp):
    recall = services[0]
    fever, cough = sym(nlp, "Fever"), sym(nlp, "Cough")
    out = recall.recall([fever, cough], [])
    assert 3 <= len(out) <= 10
    assert [c.score for c in out] == sorted((c.score for c in out), reverse=True)
    assert all(c.id.startswith("DDX:") and c.source.startswith("ddxplus-tristate-v1/") for c in out)
    assert all(set(c.evidence_ids) <= {fever.kg_id, cough.kg_id} for c in out)


def test_uncertain_top_disease_keeps_three_candidates_confident_one_stops(nlp):
    recall, _, _ = build_ddxplus_services(nlp.nodes, KG_FILE, backend="numpy")
    smell, nose = sym(nlp, "Abnormality of the sense of smell"), sym(nlp, "Rhinorrhea")
    assert len(recall.recall([smell, nose], [])) >= 3      # top ~0.64 < 0.8: keep asking
    confident, _, _ = build_ddxplus_services(nlp.nodes, KG_FILE, backend="numpy", confident=0.5)
    assert len(confident.recall([smell, nose], [])) <= 2   # nucleus mass reached: stop rule fires


def test_search_asks_mapped_unasked_symptom_with_ddxplus_wording(services, nlp):
    recall, search, _ = services
    fever, cough = sym(nlp, "Fever"), sym(nlp, "Cough")
    cands = recall.recall([fever, cough], [])
    q = search.select_next(cands, [fever, cough], [], [])
    assert q.kg_name in search.data.evidence_to_name.values()
    assert q.kg_id not in {fever.kg_id, cough.kg_id}
    assert "DDXPlus question:" in q.rationale
    q2 = search.select_next(cands, [fever, cough], [], [q.kg_id])
    assert q2 is None or q2.kg_id != q.kg_id
    limited = DDXPlusSearchService(search.data, nlp.nodes, whitelist={"Rhinorrhea"})
    q3 = limited.select_next(cands, [fever, cough], [], [])
    assert q3 is None or q3.kg_name == "Rhinorrhea"


def test_drugs_use_primekg_names_and_skip_unmapped_diseases(services):
    drugs = services[2]
    out = drugs.recommend([DiseaseCandidate(id="DDX:024", name="Influenza", score=.7),
                           DiseaseCandidate(id="DDX:010", name="Boerhaave", score=.3)])
    assert out and all(d.related_disease.startswith("Influenza") for d in out)
    assert {d.relation for d in out} <= {"indication", "off-label use"}
    labels = {d.related_disease for d in drugs.recommend([DiseaseCandidate(id="x", name="Pneumonia", score=1)])}
    assert labels == {"Pneumonia"}


def test_official_conditions_file_replaces_provisional_table(tmp_path, nlp, model):
    data = DDXPlusData(model, nlp.nodes)
    assert data.table_source == "model-derived-provisional"
    conditions = {d: {"cond-name-eng": d, "symptoms": {"E_91": {}, "E_201": {}}} for d in model.diseases}
    path = tmp_path / "release_conditions.json"
    path.write_text(json.dumps(conditions), encoding="utf-8")
    official = DDXPlusData(model, nlp.nodes, str(path))
    assert official.table_source == "ddxplus-release-conditions"
    assert official.disease_symptoms["Influenza"] == ["Cough", "Fever"]
    french = {d: {"cond-name-eng": d, "symptoms": {"fievre": {}}} for d in model.diseases}
    path.write_text(json.dumps(french), encoding="utf-8")
    with pytest.raises(ValueError, match="E_ evidence codes"):
        DDXPlusData(model, nlp.nodes, str(path))


def test_hard_filter_is_opt_in(nlp, model):
    data = DDXPlusData(model, nlp.nodes)
    soft = DDXPlusRecallService(data, top_k=49, mass=1.01)
    hard = DDXPlusRecallService(data, top_k=49, mass=1.01, hard_filter=True)
    cough, fever = sym(nlp, "Cough"), sym(nlp, "Fever")
    with_fever = {d for d, s in data.disease_symptoms.items() if "Fever" in s}
    assert with_fever & {c.name for c in soft.recall([cough], [fever])}
    assert not with_fever & {c.name for c in hard.recall([cough], [fever])}


def test_colloquial_aliases_reach_the_model(nlp):
    found = {e.kg_name for e in nlp.analyze("I have a sore throat, a runny nose and I feel very tired.").evidence}
    assert {"Pharyngitis", "Rhinorrhea", "Fatigue"} <= found
    assert "drooling" in nlp.aliases and nlp.nodes[nlp.aliases["drooling"][0]].kg_name == "Drooling"


def test_full_session_through_orchestrator(nlp, services):
    recall, search, drugs = services
    data = recall.data
    store = InMemorySessionService()
    chat = ChatbotService(nlp, store, recall, search, drugs, mode="ddxplus", max_rounds=12,
                          dialogue=DialogueAI(vocabulary=nlp.nodes, mention_finder=nlp._mentions))
    try:
        r = chat.start(StartChatRequest(message="I have a fever and a cough."))
        assert r.mode == "ddxplus" and r.candidate_diseases and r.next_symptom
        true = "Influenza"
        seen = set()
        while r.status == "RUNNING" and r.next_symptom:
            assert r.next_symptom.kg_id not in seen
            seen.add(r.next_symptom.kg_id)
            answer = "yes" if r.next_symptom.kg_name in data.disease_symptoms[true] else "no"
            r = chat.answer(AnswerChatRequest(session_id=r.session_id, expected_version=r.version,
                request_id=uuid4().hex, answer=answer,
                symptom=Symptom(kg_id=r.next_symptom.kg_id, kg_name=r.next_symptom.kg_name)))
        assert r.stop_reason in {"candidate_threshold", "no_questions", "max_rounds"}
        assert r.candidate_diseases[0].name == true
    finally:
        store.close()
