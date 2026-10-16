"""Fixtures exercise orchestration. Scores and rankings are NOT diagnostic measurements."""
from app.models.chat_models import DiseaseCandidate, NextSymptom
from app.services.interfaces import DrugService, InformedSearchService, RecallService

class MockRecallService(RecallService):
    def recall(self, positive, negative):
        if not positive and not negative:
            return []
        count = len(positive) + len(negative)
        keep = 4 if count <= 2 else 3 if count == 3 else 2
        fixtures = [("D001", "Fixture condition A", .86), ("D002", "Fixture condition B", .78),
                    ("D003", "Fixture condition C", .62), ("D004", "Fixture condition D", .55)]
        return [DiseaseCandidate(id=k, name=n, score=s, evidence_ids=[v.kg_id for v in positive+negative]) for k,n,s in fixtures[:keep]]

class MockInformedSearchService(InformedSearchService):
    _questions = [
        NextSymptom(kg_id="DEMO:breathlessness", kg_name="shortness of breath", heuristic_score=.91, rationale="Mock fixture order: respiratory follow-up."),
        NextSymptom(kg_id="DEMO:chest_pain", kg_name="chest pain", heuristic_score=.84, rationale="Mock fixture order: chest symptom follow-up."),
        NextSymptom(kg_id="DEMO:fatigue", kg_name="fatigue", heuristic_score=.72, rationale="Mock fixture order: general symptom follow-up."),
    ]
    def select_next(self, candidates, positive, negative, asked_symptom_ids):
        resolved = {s.kg_id for s in positive+negative} | set(asked_symptom_ids)
        return next((q for q in self._questions if q.kg_id not in resolved), None)

class MockDrugService(DrugService):
    def recommend(self, candidates):
        # No manufactured medication information. Teammate adapter may return typed KG relations.
        return []

