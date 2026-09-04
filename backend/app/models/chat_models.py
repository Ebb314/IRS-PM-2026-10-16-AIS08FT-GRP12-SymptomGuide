from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Symptom(BaseModel):
    kg_id: str
    kg_name: str


class ExtractedSymptom(Symptom):
    original_text: Optional[str] = None
    confidence: float = 1.0


class DiseaseCandidate(BaseModel):
    id: str
    name: str
    score: float = Field(ge=0, le=1)


class NextSymptom(Symptom):
    heuristic_score: float = Field(ge=0, le=1)


class DrugRecommendation(BaseModel):
    name: str
    related_disease: str
    note: Optional[str] = None


class SessionStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"


class SessionState(BaseModel):
    session_id: str
    positive_symptoms: List[Symptom] = Field(default_factory=list)
    negative_symptoms: List[Symptom] = Field(default_factory=list)
    asked_symptoms: List[str] = Field(default_factory=list)
    candidate_diseases: List[DiseaseCandidate] = Field(default_factory=list)
    last_question: Optional[NextSymptom] = None
    drug_recommendations: List[DrugRecommendation] = Field(default_factory=list)
    round: int = 0
    status: SessionStatus = SessionStatus.RUNNING


class StartChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class AnswerChatRequest(BaseModel):
    session_id: str
    answer: Literal["yes", "no"]
    symptom: Symptom


class ChatResponse(BaseModel):
    session_id: str
    status: SessionStatus
    message: str
    positive_symptoms: List[Symptom]
    negative_symptoms: List[Symptom]
    candidate_diseases: List[DiseaseCandidate]
    next_symptom: Optional[NextSymptom] = None
    drug_recommendations: List[DrugRecommendation] = Field(default_factory=list)
    round: int
