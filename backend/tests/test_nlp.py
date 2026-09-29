import pytest
from app.services.nlp_service import NLPService

@pytest.fixture(scope="module")
def nlp():
    return NLPService(use_models=False)

@pytest.mark.parametrize("text,expected", [
    ("I have fever and cough.", {"fever":"present","cough":"present"}),
    ("No fever, but I am coughing.", {"fever":"absent","cough":"present"}),
    ("I deny fever and cough.", {"fever":"absent","cough":"absent"}),
    ("No fever. I have chest pain.", {"fever":"absent","chest pain":"present"}),
    ("I'm not sure whether I have fever.", {"fever":"uncertain"}),
    ("Maybe a headache.", {"headache":"uncertain"}),
    ("My father has fever. I am dizzy.", {"dizziness":"present"}),
    ("I used to have headaches last year.", {}),
    ("I had fever last year. Now I have cough.", {"cough":"present"}),
    ("No fever and I have cough.", {"fever":"absent","cough":"present"}),
    ("Not only fever but cough.", {"fever":"present","cough":"present"}),
    ("Fever is absent.", {"fever":"absent"}),
    ("No fever. Actually I have fever.", {"fever":"present"}),
    ("I have fever. I have no fever.", {"fever":"uncertain"}),
    ("My throat hurts and I feel queasy.", {"sore throat":"present","nausea":"present"}),
    ("I am tired and breathless.", {"fatigue":"present","shortness of breath":"present"}),
    ("The weather is lovely.", {}),
])
def test_assertions(nlp, text, expected):
    assert {e.kg_name:e.status for e in nlp.analyze(text).evidence} == expected

def test_spans_and_exclusions(nlp):
    text = "My mother has fever. I have a cough."
    result = nlp.analyze(text, 2)
    assert result.excluded[0].experiencer == "other"
    for item in result.evidence + result.excluded:
        assert text[item.start:item.end] == item.original_text
        assert item.turn == 2

def test_duplicate_and_ambiguous_vocabulary(tmp_path):
    path = tmp_path / "nodes.csv"
    path.write_text("node_id,node_name,node_type,aliases\na,fever,symptom,hot\nb,warmth,symptom,hot\n")
    result = NLPService(path, False).analyze("hot")
    assert not result.evidence and any("Ambiguous" in w for w in result.warnings)
    path.write_text("node_id,node_name,node_type\na,fever,symptom\na,cough,symptom\n")
    with pytest.raises(ValueError):
        NLPService(path, False)

