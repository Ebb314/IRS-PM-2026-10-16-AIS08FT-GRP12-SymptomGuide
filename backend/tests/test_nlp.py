from app.services.nlp_service import NLPService


def test_extracts_positive_and_negative_symptoms_from_kg_vocabulary():
    nlp = NLPService(use_models=False)

    positive, negative = nlp.extract("I have fever and cough, but I do not have chest pain.")

    assert {item.kg_name for item in positive} == {"fever", "cough"}
    assert {item.kg_name for item in negative} == {"chest pain"}


def test_does_not_let_negation_cross_but_boundary():
    nlp = NLPService(use_models=False)

    positive, negative = nlp.extract("I have no fever, but I have cough.")

    assert [item.kg_name for item in positive] == ["cough"]
    assert [item.kg_name for item in negative] == ["fever"]
