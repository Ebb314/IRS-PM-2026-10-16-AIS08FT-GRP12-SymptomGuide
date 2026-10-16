from enum import Enum
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field

class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class Symptom(Model):
    kg_id: str
    kg_name: str

class ExtractedSymptom(Symptom):
    original_text: str = ""
    confidence: float = 1.0  # match score, not calibrated clinical confidence
    status: Literal["present", "absent", "uncertain"] = "present"
    experiencer: Literal["self", "other"] = "self"
    temporality: Literal["current", "historical"] = "current"
    start: int = 0
    end: int = 0
    turn: int = 0
    source: str = "rules-v2"
    duration_text: str = ""
    severity_text: str = ""

class NLPResult(Model):
    evidence: list[ExtractedSymptom] = Field(default_factory=list)
    excluded: list[ExtractedSymptom] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    engine: str = "rules-v2"
    ai_trace: dict = Field(default_factory=dict)

class DiseaseCandidate(Model):
    id: str
    name: str
    score: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    source: str = "mock-fixture-v1"

class NextSymptom(Symptom):
    heuristic_score: float = Field(ge=0, le=1)
    rationale: str = "Selected by the question adapter."

class DrugRecommendation(Model):
    name: str
    related_disease: str
    note: str = ""
    relation: str = "mock_association"
    source: str = "mock-fixture-v1"

class SessionStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"

class ChatMessage(Model):
    id: str = Field(default_factory=lambda: uuid4().hex)
    sender: Literal["user", "assistant"]
    text: str

class SessionState(Model):
    session_id: str
    dialogue_prompt: Literal['none','description','menu','term'] = 'none'
    clarification_attempts: int = 0
    conversation_memory: dict[str,str] = Field(default_factory=dict)
    pending_evidence: list[ExtractedSymptom] = Field(default_factory=list)
    symptom_question_active: bool = True
    knowledge_focus: Symptom | None = None
    positive_symptoms: list[Symptom] = Field(default_factory=list)
    negative_symptoms: list[Symptom] = Field(default_factory=list)
    uncertain_symptoms: list[Symptom] = Field(default_factory=list)
    evidence: list[ExtractedSymptom] = Field(default_factory=list)
    evidence_history: list[ExtractedSymptom] = Field(default_factory=list)
    excluded_evidence: list[ExtractedSymptom] = Field(default_factory=list)
    asked_symptoms: list[str] = Field(default_factory=list)
    candidate_diseases: list[DiseaseCandidate] = Field(default_factory=list)
    last_question: NextSymptom | None = None
    drug_recommendations: list[DrugRecommendation] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    round: int = 0
    version: int = 0
    status: SessionStatus = SessionStatus.RUNNING
    stop_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    mode: str = "mock"
    nlp_engine: str = "rules-v2"
    ai_trace: dict = Field(default_factory=dict)

class StartChatRequest(Model):
    message: str = Field(min_length=1, max_length=2000)
    request_id: str = Field(default_factory=lambda: uuid4().hex, min_length=8, max_length=100)

class MutationRequest(Model):
    session_id: str = Field(min_length=1, max_length=100)
    request_id: str = Field(min_length=8, max_length=100)
    expected_version: int = Field(ge=1)

class MessageRequest(MutationRequest):
    message: str = Field(min_length=1, max_length=2000)

class AnswerChatRequest(MutationRequest):
    answer: Literal["yes", "no", "unknown"]
    symptom: Symptom

class CorrectionRequest(MutationRequest):
    kg_id: str
    # dismiss applies only to an unconfirmed semantic proposal, never a saved record.
    status: Literal["present", "absent", "uncertain", "dismiss"]

class ChatResponse(SessionState):
    message: str
    next_symptom: NextSymptom | None = None
    nlp_engine: str = "rules-v2"

