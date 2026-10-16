import json
import httpx
import pytest
from app.services.llm_nlp_service import OllamaNLPService
from app.services.nlp_service import NLPService

def mention(**changes):
    return {"kg_id":"DEMO:fever","quote":"fever","status":"absent","experiencer":"self","temporality":"current",**changes}

def make(payload):
    def handler(request):
        sent=json.loads(request.content)
        assert sent["stream"] is False
        assert "properties" in sent["format"]
        return httpx.Response(200,json={"message":{"content":json.dumps(payload)}})
    return OllamaNLPService(NLPService(use_models=False),"test-model",transport=httpx.MockTransport(handler))

def test_structured_llm_output():
    result=make({"mentions":[mention()]}).analyze("No fever",3)
    assert result.engine=="ollama-structured-v1"
    assert result.evidence[0].status=="absent"
    assert result.evidence[0].start==3 and result.evidence[0].turn==3

@pytest.mark.parametrize("payload",[
    {"mentions":[mention(kg_id="invented")]},
    {"mentions":[mention(quote="hallucinated")]},
    {"mentions":[mention(status="diagnosed")]},
    {"mentions":[mention(),mention()]},
    {"diagnosis":"something"},
])
def test_invalid_llm_output_falls_back_visibly(payload):
    result=make(payload).analyze("No fever")
    assert result.engine=="rules-v2"
    assert "rejected" in result.warnings[0]
    assert result.evidence[0].status=="absent"

def test_llm_timeout():
    def handler(request): raise httpx.ReadTimeout("timeout")
    service=OllamaNLPService(NLPService(use_models=False),"test",transport=httpx.MockTransport(handler))
    assert service.analyze("cough").engine=="rules-v2"

def test_llm_does_not_add_family_history_to_current_evidence():
    result=make({"mentions":[mention(experiencer="other",status="present")]}).analyze("My father has fever")
    assert not result.evidence and result.excluded

