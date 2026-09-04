from typing import Iterable, List

from app.config import settings
from app.models.chat_models import (
    AnswerChatRequest,
    ChatResponse,
    SessionState,
    SessionStatus,
    StartChatRequest,
    Symptom,
)
from app.services.interfaces import DrugService, InformedSearchService, RecallService
from app.services.nlp_service import NLPService
from app.services.session_service import InMemorySessionService


class InvalidAnswerError(ValueError):
    pass


class ChatbotService:
    def __init__(
        self,
        nlp: NLPService,
        sessions: InMemorySessionService,
        recall: RecallService,
        search: InformedSearchService,
        drugs: DrugService,
    ) -> None:
        self.nlp = nlp
        self.sessions = sessions
        self.recall = recall
        self.search = search
        self.drugs = drugs

    def start(self, request: StartChatRequest) -> ChatResponse:
        state = self.sessions.create()
        positive, negative = self.nlp.extract(request.message)
        self._merge_evidence(state, positive, negative)
        if not positive and not negative:
            self.sessions.save(state)
            return self._response(state, "I couldn't confidently identify a KG symptom. Please describe your symptoms more specifically.")
        return self._advance(state)

    def answer(self, request: AnswerChatRequest) -> ChatResponse:
        state = self.sessions.get(request.session_id)
        if state.status != SessionStatus.RUNNING or state.last_question is None:
            raise InvalidAnswerError("This session is not waiting for a follow-up answer")
        if request.symptom.kg_id != state.last_question.kg_id:
            raise InvalidAnswerError("The answered symptom does not match the current question")
        if request.symptom.kg_id not in state.asked_symptoms:
            state.asked_symptoms.append(request.symptom.kg_id)
        # Trust the server-side question as the canonical KG entity; clients
        # cannot alter its name while submitting the answer.
        answered_symptom = Symptom(
            kg_id=state.last_question.kg_id,
            kg_name=state.last_question.kg_name,
        )
        if request.answer == "yes":
            self._merge_evidence(state, [answered_symptom], [])
        else:
            self._merge_evidence(state, [], [answered_symptom])
        state.round += 1
        state.last_question = None
        return self._advance(state)

    @staticmethod
    def _merge_evidence(state: SessionState, positive: Iterable[Symptom], negative: Iterable[Symptom]) -> None:
        positive_by_id = {item.kg_id: item for item in state.positive_symptoms}
        negative_by_id = {item.kg_id: item for item in state.negative_symptoms}
        for item in positive:
            negative_by_id.pop(item.kg_id, None)
            positive_by_id[item.kg_id] = Symptom(kg_id=item.kg_id, kg_name=item.kg_name)
        for item in negative:
            positive_by_id.pop(item.kg_id, None)
            negative_by_id[item.kg_id] = Symptom(kg_id=item.kg_id, kg_name=item.kg_name)
        state.positive_symptoms = list(positive_by_id.values())
        state.negative_symptoms = list(negative_by_id.values())

    def _advance(self, state: SessionState) -> ChatResponse:
        # The full accumulated state is always sent to recall.
        state.candidate_diseases = self.recall.recall(state.positive_symptoms, state.negative_symptoms)
        should_finish = (
            len(state.candidate_diseases) <= settings.finish_candidate_threshold
            or state.round >= settings.max_rounds
        )
        if not should_finish:
            state.last_question = self.search.select_next(
                state.candidate_diseases,
                state.positive_symptoms,
                state.negative_symptoms,
                state.asked_symptoms,
            )
            should_finish = state.last_question is None
        if should_finish:
            state.status = SessionStatus.COMPLETE
            state.last_question = None
            state.drug_recommendations = self.drugs.recommend(state.candidate_diseases)
            message = "The mock workflow has completed. These results are for decision-support demonstration only."
        else:
            state.status = SessionStatus.RUNNING
            message = "Are you experiencing {}?".format(state.last_question.kg_name)
        self.sessions.save(state)
        return self._response(state, message)

    @staticmethod
    def _response(state: SessionState, message: str) -> ChatResponse:
        return ChatResponse(
            session_id=state.session_id,
            status=state.status,
            message=message,
            positive_symptoms=state.positive_symptoms,
            negative_symptoms=state.negative_symptoms,
            candidate_diseases=state.candidate_diseases,
            next_symptom=state.last_question,
            drug_recommendations=state.drug_recommendations,
            round=state.round,
        )
