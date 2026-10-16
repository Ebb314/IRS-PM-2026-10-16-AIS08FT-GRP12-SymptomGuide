"""Run from any directory after installing requirements.txt."""
import json

from disease_retrieval_tristate import DiseaseRetriever

if __name__ == '__main__':
    retriever = DiseaseRetriever(model_variant='cross_entropy')
    result = retriever.retrieve(
        present_symptoms=['E_201', 'E_66'],
        absent_symptoms=['E_91'],
        topk=5,
    )
    assert len(result['candidates']) == 5
    assert result['known_absent'] == ['E_91']
    print(json.dumps(result, ensure_ascii=False, indent=2))
