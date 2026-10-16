from abc import ABC, abstractmethod
from typing import List, Optional

from app.models.chat_models import DiseaseCandidate, DrugRecommendation, NextSymptom, Symptom


class RecallService(ABC):
    @abstractmethod
    def recall(self, positive: List[Symptom], negative: List[Symptom]) -> List[DiseaseCandidate]:
        """Return diseases using the complete accumulated evidence."""


class InformedSearchService(ABC):
    @abstractmethod
    def select_next(
        self,
        candidates: List[DiseaseCandidate],
        positive: List[Symptom],
        negative: List[Symptom],
        asked_symptom_ids: List[str],
    ) -> Optional[NextSymptom]:
        """Choose the highest-value symptom that has not already been resolved."""


class DrugService(ABC):
    @abstractmethod
    def recommend(self, candidates: List[DiseaseCandidate]) -> List[DrugRecommendation]:
        """Return source-labelled KG drug relations, not individual treatment advice."""
