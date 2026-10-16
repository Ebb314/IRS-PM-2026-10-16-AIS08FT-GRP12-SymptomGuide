"""Rule-based NLP robustness: lay phrases, typos, informal negation, modifier terms, recall773 synonyms."""
from pathlib import Path

import pytest

from app.services.nlp_service import NLPService, _one_edit
from app.services.recall773_reasoning_service import EXTRA_NODES_FILE, build_recall773_services

DATA = Path(__file__).parents[1] / "app/data"


@pytest.fixture(scope="module")
def nlp():
    return NLPService(DATA / "kg_nodes.csv", use_models=False, extra_vocab_file=str(EXTRA_NODES_FILE))


@pytest.fixture(scope="module")
def data(nlp):
    return build_recall773_services(nlp.nodes, str(DATA / "kg_reasoning.json.gz"), backend="numpy")[0].data


def states(nlp, text):
    return {e.kg_name: e.status for e in nlp.analyze(text).evidence}


def test_one_edit_distance():
    assert _one_edit("feaver", "fever") and _one_edit("cogh", "cough") and _one_edit("fevre", "fever")
    assert not _one_edit("fever", "fever") and not _one_edit("couches", "cough")


def test_typos_are_read_with_a_warning_but_real_words_are_not(nlp):
    result = nlp.analyze("I have a feaver and a cogh.")
    assert {e.kg_name for e in result.evidence} == {"Fever", "Cough"}
    assert any("Read 'feaver' as Fever" in w for w in result.warnings)
    assert states(nlp, "I was sitting on the couch, it is thirty degrees and I tried to rest.") == {}


def test_informal_negation(nlp):
    assert states(nlp, "I dont have a headache, but I have nausea.") == {"Headache": "absent", "Nausea": "present"}
    assert states(nlp, "I havent had any fever.") == {"Fever": "absent"}


def test_modifier_terms_are_not_symptoms(nlp):
    found = states(nlp, "I have a severe, sharp headache, mild and chronic.")
    assert found == {"Headache": "present"}


def test_lay_phrases(nlp):
    assert states(nlp, "I've been throwing up and I have the runs.") == {"Vomiting": "present", "Diarrhea": "present"}
    assert states(nlp, "My nose is stuffed up.") == {"Nasal obstruction": "present"}


def test_recall773_synonyms_reach_the_model(nlp, data):
    for text, feature in [("I have a sore throat.", "sore throat"), ("I feel dizzy.", "dizziness"),
                          ("I have a runny nose.", "coryza"), ("I have chronic fatigue.", "fatigue")]:
        ev = nlp.analyze(text).evidence
        assert feature in data.evidence(ev), text
