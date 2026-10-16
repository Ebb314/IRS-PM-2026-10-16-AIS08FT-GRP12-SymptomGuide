"""HTTP boundary for teammate modules. Never silently fall back to mock results."""
import httpx
from pydantic import TypeAdapter, ValidationError
from app.models.chat_models import DiseaseCandidate, NextSymptom, DrugRecommendation
from app.services.interfaces import RecallService, InformedSearchService, DrugService

class AdapterError(RuntimeError):
    pass

class RemoteGateway:
    def __init__(self, base_url, timeout=3, transport=None):
        self.client = httpx.Client(base_url=base_url.rstrip("/")+"/", timeout=timeout, transport=transport, trust_env=False)
    def call(self, route, payload, output_type):
        try:
            response = self.client.post(route, json=payload)
            response.raise_for_status()
            if len(response.content) > 1_000_000:
                raise ValueError("Oversized adapter response")
            return TypeAdapter(output_type).validate_python(response.json())
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise AdapterError("Teammate module unavailable or returned invalid data. Retry after checking the service.") from exc

class RemoteRecallService(RecallService):
    def __init__(self, gateway): self.gateway = gateway
    def recall(self, positive, negative):
        return self.gateway.call("recall", {"positive": [s.model_dump() for s in positive], "negative": [s.model_dump() for s in negative]}, list[DiseaseCandidate])

class RemoteSearchService(InformedSearchService):
    def __init__(self, gateway): self.gateway = gateway
    def select_next(self, candidates, positive, negative, asked_symptom_ids):
        return self.gateway.call("search", {"candidates": [s.model_dump() for s in candidates],
            "positive": [s.model_dump() for s in positive], "negative": [s.model_dump() for s in negative],
            "asked_symptom_ids": asked_symptom_ids}, NextSymptom | None)

class RemoteDrugService(DrugService):
    def __init__(self, gateway): self.gateway = gateway
    def recommend(self, candidates):
        return self.gateway.call("drug-relations", {"candidates": [s.model_dump() for s in candidates]}, list[DrugRecommendation])

