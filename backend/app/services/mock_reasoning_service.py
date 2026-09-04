from typing import List, Optional

from app.models.chat_models import DiseaseCandidate, DrugRecommendation, NextSymptom, Symptom
from app.services.interfaces import DrugService, InformedSearchService, RecallService


class MockRecallService(RecallService):
    """Deterministic mock. Replace this class with the teammate's adapter."""

    _diseases = [
        ("D001", "Influenza", 0.86),
        ("D002", "Pneumonia", 0.78),
        ("D003", "Common cold", 0.62),
        ("D004", "Bronchitis", 0.55),
    ]

    def recall(self, positive: List[Symptom], negative: List[Symptom]) -> List[DiseaseCandidate]:
        evidence_count = len(positive) + len(negative)
        keep = 4 if evidence_count <= 2 else 3 if evidence_count == 3 else 2
        return [DiseaseCandidate(id=item[0], name=item[1], score=item[2]) for item in self._diseases[:keep]]


class MockInformedSearchService(InformedSearchService):
    _questions = [
        NextSymptom(kg_id="P006", kg_name="shortness of breath", heuristic_score=0.91),
        NextSymptom(kg_id="P003", kg_name="chest pain", heuristic_score=0.84),
        NextSymptom(kg_id="P005", kg_name="fatigue", heuristic_score=0.72),
    ]

    def select_next(
        self,
        candidates: List[DiseaseCandidate],
        positive: List[Symptom],
        negative: List[Symptom],
        asked_symptom_ids: List[str],
    ) -> Optional[NextSymptom]:
        resolved = {s.kg_id for s in positive + negative} | set(asked_symptom_ids)
        return next((question for question in self._questions if question.kg_id not in resolved), None)


class MockDrugService(DrugService):
    def recommend(self, candidates: List[DiseaseCandidate]) -> List[DrugRecommendation]:
        return [
            DrugRecommendation(
                name="Example Drug",
                related_disease=disease.name,
                note="Demonstration only; a clinician must review real treatment recommendations.",
            )
            for disease in candidates
        ]
