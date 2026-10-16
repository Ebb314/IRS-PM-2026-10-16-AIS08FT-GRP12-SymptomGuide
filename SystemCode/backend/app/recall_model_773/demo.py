import json

from disease_retrieval_773_tristate import DiseaseRetriever773Tristate

if __name__=='__main__':
    retriever = DiseaseRetriever773Tristate()
    result = retriever.retrieve(symptom_states={
        'cough':'positive','fever':'negative','shortness of breath':'unknown'},topk=10)
    assert result['status']=='ok'
    assert len(result['candidates'])==10
    assert result['evidence']['negative_symptoms']==['fever']
    print(json.dumps(result,ensure_ascii=False,indent=2))
