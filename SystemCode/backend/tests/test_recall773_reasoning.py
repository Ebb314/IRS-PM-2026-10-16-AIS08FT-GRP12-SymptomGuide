"""REASONING_MODE=recall773: teammate's 773-disease tri-state model behind the chatbot interfaces."""
import csv
import gzip
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.models.chat_models import AnswerChatRequest, DiseaseCandidate, StartChatRequest, Symptom
from app.services.chatbot_service import ChatbotService
from app.services.dialogue_ai import DialogueAI
from app.services.nlp_service import NLPService
from app.services.recall773_reasoning_service import EXTRA_NODES_FILE, MAPPING_DIR, Recall773Data, build_recall773_services
from app.services.ddxplus_runtime import ARTIFACTS_773, load_recall773_model
from app.services.session_service import InMemorySessionService

DATA = Path(__file__).parents[1] / "app/data"
KG_FILE = str(DATA / "kg_reasoning.json.gz")


@pytest.fixture(scope="module")
def nlp():
    return NLPService(DATA / "kg_nodes.csv", use_models=False, extra_vocab_file=str(EXTRA_NODES_FILE))


@pytest.fixture(scope="module")
def services(nlp):
    return build_recall773_services(nlp.nodes, KG_FILE, backend="numpy")


def sym(nlp, name):
    node = next(n for n in nlp.nodes.values() if n.kg_name == name)
    return Symptom(kg_id=node.kg_id, kg_name=node.kg_name)


def test_model_loads_teammate_selection(services):
    model = services[0].data.model
    selected = json.loads((ARTIFACTS_773 / "selection.json").read_text(encoding="utf-8"))["selected_model"]
    assert model.variant == selected and len(model.diseases) == 773 and len(model.features) == 377
    p = model.probabilities(["cough", "fever"])
    assert p.shape == (773,) and abs(p.sum() - 1) < 1e-5


def test_torch_and_numpy_backends_agree_when_torch_is_installed(services):
    pytest.importorskip("torch")
    numpy_model = services[0].data.model
    for variant in ("ce", "ce_pairwise"):
        a, b = load_recall773_model("torch", variant), load_recall773_model("numpy", variant)
        for present, absent in [(["cough", "fever"], ["headache"]), (["toothache"], []), (["diarrhea", "vomiting"], ["fever"])]:
            assert np.allclose(a.logits(present, absent), b.logits(present, absent), atol=1e-5)
    assert numpy_model.backend == "numpy"


def test_extra_vocab_only_in_recall773_mode():
    plain = NLPService(DATA / "kg_nodes.csv", use_models=False, extra_vocab_file="")
    assert not any(k.startswith("DSS_") for k in plain.nodes)


def test_mapping_tables_match_vocabularies(nlp, services):
    model = services[0].data.model
    kg = json.load(gzip.open(KG_FILE, "rt", encoding="utf-8"))
    with open(MAPPING_DIR / "feature_map.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["feature"] for r in rows] == model.features
    for r in rows:
        assert nlp.nodes[r["node_id"]].kg_name == r["node_name"]
        assert r["role"] in {"exact_name", "one_to_one", "group_representative", "model_only_node"}
        if r["represents_head"]:
            assert r["represents_head"] == r["group_head_name"]
    with open(MAPPING_DIR / "disease_map.csv", encoding="utf-8") as f:
        diseases = list(csv.DictReader(f))
    assert [d["disease"] for d in diseases] == model.diseases
    for d in diseases:
        if d["primekg_name"]:
            assert kg["disease_ids"][d["primekg_name"]] == d["primekg_id"]


def test_broad_terms_yes_maps_to_representative_no_rules_out_group(services, nlp):
    data = services[0].data
    chest = sym(nlp, "Chest pain")
    assert data.evidence([chest]) == ["sharp chest pain"]
    assert {"sharp chest pain", "burning chest pain"} <= set(data.absent_evidence([chest]))


def test_model_only_symptoms_are_understood(nlp, services):
    found = {e.kg_name for e in nlp.analyze("I have diarrhea and a toothache.").evidence}
    assert {"Diarrhea", "Toothache"} <= found
    out = services[0].recall([sym(nlp, "Toothache")], [])
    assert out and out[0].id.startswith("DSA:") and out[0].source.startswith("recall773-tristate-v2")


def test_absent_channel_changes_scores_and_unknown_is_not_sent(services, nlp):
    recall = services[0]
    cough, fever = sym(nlp, "Cough"), sym(nlp, "Fever")
    assert recall.recall([], [fever]) == []
    assert [c.score for c in recall.recall([cough], [fever])] != [c.score for c in recall.recall([cough], [])]


def test_search_asks_mapped_unasked_symptom(services, nlp):
    recall, search, _ = services
    pos = [sym(nlp, "Fever"), sym(nlp, "Cough")]
    cands = recall.recall(pos, [])
    q = search.select_next(cands, pos, [], [])
    assert q and q.kg_name in search.data.evidence_to_name.values()
    assert q.kg_id not in {s.kg_id for s in pos}


def test_drugs_use_primekg_names(services):
    data, drugs = services[0].data, services[2]
    name = next(d for d in data.model.diseases if d in data.primekg_name)
    out = drugs.recommend([DiseaseCandidate(id="DSA:0000", name=name, score=1.0)])
    assert all(d.related_disease == data.primekg_name[name] for d in out)


def test_frequency_table_replaces_provisional(tmp_path, nlp, services):
    model = services[0].data.model
    path = tmp_path / "table.csv"
    path.write_text("disease,symptom,frequency\nflu,fever,0.9\nflu,cough,0.8\nflu,skin rash,0.01\n",
                    encoding="utf-8")
    data = Recall773Data(model, nlp.nodes, str(path))
    assert data.table_source == "training-data-frequency"
    assert data.disease_symptoms["flu"] == sorted([data.evidence_to_name["fever"], data.evidence_to_name["cough"]])
    assert data.disease_symptoms["pneumonia"] == services[0].data.disease_symptoms["pneumonia"]   # not in table


def test_full_session_through_orchestrator(nlp, services):
    recall, search, drugs = services
    data = recall.data
    store = InMemorySessionService()
    chat = ChatbotService(nlp, store, recall, search, drugs, mode="recall773", max_rounds=12,
                          dialogue=DialogueAI(vocabulary=nlp.nodes, mention_finder=nlp._mentions))
    try:
        r = chat.start(StartChatRequest(message="I have diarrhea and vomiting."))
        assert r.mode == "recall773" and r.candidate_diseases and r.next_symptom
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
        assert r.candidate_diseases
    finally:
        store.close()
