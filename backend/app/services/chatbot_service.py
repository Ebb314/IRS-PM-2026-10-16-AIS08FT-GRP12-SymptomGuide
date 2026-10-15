import hashlib
import json
from time import perf_counter
from app.config import settings
from app.logging_setup import get_logger, names, set_session, user_text
from app.models.chat_models import (
    ChatResponse, ChatMessage, ExtractedSymptom, Symptom, SessionStatus,
)
from app.services.remote_adapters import AdapterError

logger = get_logger("flow")

class InvalidAnswerError(ValueError):
    pass

class ChatbotService:
    def __init__(self, nlp, sessions, recall, search, drugs, mode="mock", max_rounds=None, dialogue=None):
        self.nlp, self.sessions = nlp, sessions
        self.recall, self.search, self.drugs = recall, search, drugs
        self.mode = mode
        self.max_rounds = max_rounds or settings.max_rounds
        # A recall service may define its own stop size (recall773: top-3 differential).
        self.finish_threshold = getattr(recall, "finish_threshold", settings.finish_candidate_threshold)
        from app.services.dialogue_ai import DialogueAI
        self.dialogue = dialogue or DialogueAI()

    def start(self, request):
        return self._execute("start", request)

    def answer(self, request):
        return self._execute("answer", request)

    def message(self, request):
        return self._execute("message", request)

    def correct(self, request):
        return self._execute("correct", request)

    def inspect(self, session_id):
        return self._response(self.sessions.get(session_id))

    def _execute(self, action, request):
        started = perf_counter()
        scope = "__start__" if action == "start" else request.session_id
        fingerprint = hashlib.sha256((action + json.dumps(request.model_dump(), sort_keys=True)).encode()).hexdigest()
        with self.sessions.transaction():
            cached = self.sessions.replay(scope, request.request_id)
            if cached:
                logger.info("%s replayed from cache (duplicate request_id)", action)
                if cached[0] != fingerprint:
                    raise InvalidAnswerError("Request ID was already used for a different payload")
                return ChatResponse.model_validate_json(cached[1])
            state = self.sessions.create() if action == "start" else self.sessions.get(scope)
            set_session(state.session_id)
            self._log_request(action, request, state)
            if action != "start" and state.version != request.expected_version:
                raise InvalidAnswerError("Session changed. Reload the session before submitting again.")
            if state.version >= 100:
                raise InvalidAnswerError("Session reached the 100-message demo limit. Start a new session.")
            state.mode = self.mode
            state.warnings = []
            state.timings_ms = {}
            state.ai_trace = {}
            if action in {"start", "message"}:
                state.messages.append(ChatMessage(sender="user", text=request.message))
                if hasattr(self.dialogue,'understand'):
                    t = perf_counter()
                    observations,analysis=self.dialogue.understand(request.message,state,self.nlp)
                    state.timings_ms['understanding'] = round((perf_counter()-t)*1000,3)
                    advance_message=None
                    if observations:
                        # The agent interprets the latest assertion in context.
                        # A clear denial updates an earlier report even without
                        # magic correction words; history retains the old state.
                        self._merge(state,observations)
                        if state.last_question and any(e.kg_id==state.last_question.kg_id for e in observations):
                            state.asked_symptoms.append(state.last_question.kg_id)
                        if action=='message': state.round+=1
                        advance_message=self._advance(state)
                    elif self._search_driven() and state.status==SessionStatus.RUNNING and not state.pending_evidence:
                        # Small talk / a question / no symptom in this message: keep the interview going.
                        # Re-ask the pending question, or pick the next one if symptoms are on record.
                        # (Not while a suggested match waits for confirmation, and not before any symptom.)
                        if state.last_question is not None:
                            advance_message=self._question_text(state.last_question, again=True)
                        elif state.positive_symptoms:
                            advance_message=self._advance(state)
                        if advance_message:
                            logger.info("no new symptom in this message; %s", "re-asking the pending question"
                                        if state.last_question else "continuing the interview")
                    # kg / ddxplus / recall773: informed search drives the interview, so its question is
                    # asked explicitly with Yes / No / Not sure buttons after the LLM reply.
                    ask_kg=self._search_driven() and state.status==SessionStatus.RUNNING and state.last_question is not None
                    append_kg=self._search_driven() and advance_message is not None
                    state.nlp_engine='conversation-agent-v1'
                    t = perf_counter()
                    message,trace=self.dialogue.reply(request.message,state,analysis,app_will_ask=append_kg)
                    if append_kg:
                        message=message.rstrip()+'\n\n'+advance_message
                    state.timings_ms['reply'] = round((perf_counter()-t)*1000,3)
                    state.ai_trace=trace
                    state.symptom_question_active=ask_kg
                    state.version+=1
                    return self._commit(state,message,scope,request,fingerprint,started)
                intent, intent_trace = self.dialogue.classify(request.message,state)
                logger.info("intent=%s", intent)
                state.ai_trace['routing'] = intent_trace
                state.ai_trace["intent"] = intent
                if intent != "report":
                    message, trace = self.dialogue.conversational(request.message, state, intent)
                    state.ai_trace.update(trace)
                    state.version += 1
                    return self._commit(state, message, scope, request, fingerprint, started)
                handled = False
                simple = request.message.lower().strip(" .!?")
                answers = {"yes":"yes", "yeah":"yes", "no":"no", "nope":"no",
                           "not sure":"unknown", "i don't know":"unknown", "unknown":"unknown"}
                if action == "message" and state.last_question and simple in answers:
                    self._answer(state, answers[simple], request.message)
                    handled = True
                if not handled:
                    t = perf_counter()
                    if hasattr(self.nlp, "analyze_context"):
                        result = self.nlp.analyze_context(request.message, state.version + 1, {
                            "pending": state.last_question.model_dump() if state.last_question else None})
                    else:
                        result = self.nlp.analyze(request.message, state.version + 1)
                    state.ai_trace["nlu"] = result.ai_trace
                    state.nlp_engine = result.engine
                    state.timings_ms["nlp"] = round((perf_counter()-t)*1000,3)
                    state.excluded_evidence.extend(result.excluded)
                    state.warnings.extend(result.warnings)
                    if not result.evidence:
                        if self.dialogue.generator:
                            message,trace=self.dialogue.conversational(request.message,state,'guide')
                            state.ai_trace.update(trace)
                            state.version += 1
                            return self._commit(state,message,scope,request,fingerprint,started)
                        state.knowledge_focus = None
                        state.version += 1
                        message = "Please describe a supported current symptom, or answer the pending question. Your existing evidence has been preserved."
                        return self._commit(state, message, scope, request, fingerprint, started)
                    # Across-turn contradiction without an explicit correction is not silently overwritten.
                    explicit = any(word in request.message.lower() for word in ("actually", "correction", "sorry", "meant", "now", "anymore"))
                    current = {e.kg_id: e for e in state.evidence}
                    for item in result.evidence:
                        old = current.get(item.kg_id)
                        if old and old.status != item.status and old.status != "uncertain" and not explicit:
                            item.status = "uncertain"
                            state.warnings.append(f"Please confirm the change to {item.kg_name} in the evidence panel.")
                    self._merge(state, result.evidence)
                    if state.last_question and any(e.kg_id == state.last_question.kg_id for e in result.evidence):
                        state.asked_symptoms.append(state.last_question.kg_id)
                        state.last_question = None
                    if action == "message":
                        state.round += 1
            elif action == "answer":
                if not state.last_question or not state.symptom_question_active or state.status != SessionStatus.RUNNING:
                    raise InvalidAnswerError("This session is not waiting for a follow-up answer")
                if request.symptom.kg_id != state.last_question.kg_id:
                    raise InvalidAnswerError("The answered symptom does not match the pending question")
                state.messages.append(ChatMessage(sender="user", text=request.answer))
                self._answer(state, request.answer, request.answer)
            else:
                old = next((e for e in state.evidence if e.kg_id == request.kg_id), None)
                proposal = next((e for e in state.pending_evidence if e.kg_id == request.kg_id), None)
                if request.status == 'dismiss':
                    if not proposal:
                        raise InvalidAnswerError("Only an unconfirmed suggestion can be dismissed")
                    state.pending_evidence = [e for e in state.pending_evidence if e.kg_id != request.kg_id]
                    state.messages.append(ChatMessage(sender="user", text=f"Ignore the suggested match: {proposal.kg_name}."))
                    state.version += 1
                    state.ai_trace = {'response_mode':'explicit-record-update'}
                    return self._commit(state, f"Ignored the suggested match for {proposal.kg_name}. Your saved symptom statuses have not changed.", scope, request, fingerprint, started)
                old = old or proposal
                if old is None:
                    raise InvalidAnswerError("Only existing evidence or pending suggestions can be corrected")
                state.messages.append(ChatMessage(sender="user", text=f"Correction: {old.kg_name} is {request.status}."))
                self._merge(state, [old.model_copy(update={"status":request.status, "turn":state.version+1,
                    "source":"user-correction", "original_text":f"Set {request.status}", "start":0, "end":0})])
                state.pending_evidence = [e for e in state.pending_evidence if e.kg_id != request.kg_id]
                state.last_question = None
                if hasattr(self.dialogue, 'understand'):
                    # An explicit UI edit needs no inference. Keep adapters up to date.
                    advance_message = self._advance(state)
                    label = {'present':'Present', 'absent':'Absent', 'uncertain':'Not sure'}[request.status]
                    message = f"Saved: {old.kg_name} — {label}. You can change this again in Symptom log."
                    if self._search_driven():
                        # Informed search drives the interview: continue with its next question (or the result).
                        message += "\n\n" + advance_message
                        state.symptom_question_active = state.status == SessionStatus.RUNNING and state.last_question is not None
                    else:
                        # Random / mock fixtures: do not replace the conversation with a random test question.
                        state.symptom_question_active = False
                    state.version += 1
                    state.ai_trace = {'response_mode':'explicit-record-update'}
                    return self._commit(state, message, scope, request, fingerprint, started)
            state.version += 1
            message = self._advance(state)
            message, trace = self.dialogue.phrase_question(state, message)
            state.ai_trace["generation"] = trace
            return self._commit(state, message, scope, request, fingerprint, started)

    def _log_request(self, action, request, state):
        if action in ("start", "message"):
            logger.info("%s mode=%s round=%d text=%s", action, self.mode, state.round, user_text(request.message))
        elif action == "answer":
            logger.info("answer %s -> %s (round %d)", request.symptom.kg_name, request.answer, state.round)
        else:
            logger.info("correct %s -> %s", request.kg_id, request.status)

    def _search_driven(self):
        return self.mode in ("kg", "ddxplus", "recall773")

    @staticmethod
    def _question_text(question, again=False):
        lead = "To continue: are" if again else "Are"
        return f"{lead} you experiencing {question.kg_name}? You can answer Yes, No, Not sure, or describe it in your own words."

    def _answer(self, state, answer, text):
        question = state.last_question
        if question is None:
            raise InvalidAnswerError("No pending question")
        self._merge(state, [ExtractedSymptom(kg_id=question.kg_id, kg_name=question.kg_name,
            status={"yes":"present","no":"absent","unknown":"uncertain"}[answer],
            original_text=text, turn=state.version+1, source="explicit-answer")])
        state.asked_symptoms.append(question.kg_id)
        state.last_question = None
        state.round += 1

    @staticmethod
    def _merge(state, evidence):
        current = {item.kg_id:item for item in state.evidence}
        for item in evidence:
            current[item.kg_id] = item
            state.knowledge_focus = Symptom(kg_id=item.kg_id,kg_name=item.kg_name)
            state.evidence_history.append(item.model_copy(deep=True))
        state.evidence = list(current.values())
        for field, status in [("positive_symptoms","present"),("negative_symptoms","absent"),("uncertain_symptoms","uncertain")]:
            setattr(state, field, [Symptom(kg_id=e.kg_id, kg_name=e.kg_name) for e in state.evidence if e.status == status])

    def _advance(self, state):
        state.symptom_question_active=True
        state.status = SessionStatus.RUNNING
        state.stop_reason = None
        state.drug_recommendations = []
        state.last_question = None
        logger.info("state round=%d/%d present=%s absent=%s unsure=%s asked=%d", state.round, self.max_rounds,
                    names(state.positive_symptoms), names(state.negative_symptoms),
                    names(state.uncertain_symptoms), len(state.asked_symptoms))
        t = perf_counter()
        state.candidate_diseases = self.recall.recall(state.positive_symptoms, state.negative_symptoms)
        known = {e.kg_id for e in state.evidence if e.status != "uncertain"}
        if any(not set(c.evidence_ids).issubset(known) for c in state.candidate_diseases):
            raise AdapterError("Recall returned evidence IDs not present in the session")
        state.timings_ms["recall"] = round((perf_counter()-t)*1000,3)
        logger.info("recall -> %d candidates (%.1f ms)", len(state.candidate_diseases), state.timings_ms["recall"])
        if not state.candidate_diseases:
            state.stop_reason = "no_candidates"
        elif state.round >= self.max_rounds:
            state.stop_reason = "max_rounds"
        elif len(state.candidate_diseases) <= self.finish_threshold:
            state.stop_reason = "candidate_threshold"
        else:
            t = perf_counter()
            excluded = list(dict.fromkeys(state.asked_symptoms + [e.kg_id for e in state.uncertain_symptoms]))
            state.last_question = self.search.select_next(state.candidate_diseases, state.positive_symptoms, state.negative_symptoms, excluded)
            state.timings_ms["search"] = round((perf_counter()-t)*1000,3)
            if state.last_question:
                q = state.last_question
                node = self.nlp.nodes.get(q.kg_id)
                if node is None or q.kg_id in known or q.kg_id in excluded:
                    raise AdapterError("Search returned an unknown or already answered symptom")
                state.last_question = q.model_copy(update={"kg_name": node.kg_name})
                logger.info("next question: %s (h=%s, %.1f ms)", node.kg_name, getattr(q, "heuristic_score", None),
                            state.timings_ms["search"])
            else:
                state.stop_reason = "no_questions"
        if state.stop_reason:
            state.status = SessionStatus.COMPLETE
            logger.info("STOP reason=%s (round %d/%d, %d candidates, threshold <= %d)", state.stop_reason, state.round,
                        self.max_rounds, len(state.candidate_diseases), self.finish_threshold)
            if state.stop_reason == "candidate_threshold":
                state.drug_recommendations = self.drugs.recommend(state.candidate_diseases)
                logger.info("drugs -> %d recommendations", len(state.drug_recommendations))
                return "Consultation complete. Review the possible conditions; this is not a diagnosis. You can correct your answers or add more information."
            return ("The consultation ended without a clear result (" + state.stop_reason.replace("_", " ") + "). "
                    "Evidence is insufficient for a conclusion; please consider seeing a doctor. You can still correct or add information.")
        return self._question_text(state.last_question)

    def _commit(self, state, message, scope, request, fingerprint, started):
        intent=state.ai_trace.get('intent','report')
        if intent in {'clarification_retry','clarification_stop'}:
            state.clarification_attempts += 1
            if state.dialogue_prompt=='none': state.dialogue_prompt='menu'
        else:
            state.clarification_attempts=0
            state.dialogue_prompt={'guide':'description','repair':'description','clarify':'menu',
                'clarify_term':'term','ask_term':'term'}.get(intent,'none')
        state.timings_ms["total"] = round((perf_counter()-started)*1000,3)
        state.messages.append(ChatMessage(sender="assistant", text=message))
        self.sessions.save(state)
        response = self._response(state)
        self.sessions.record(scope, request.request_id, fingerprint, response)
        # Do not log message content or session tokens (sid is a hash; see app/logging_setup.py).
        logger.info("done status=%s version=%s round=%s question=%s timings_ms=%s", getattr(state.status, "value", state.status),
                    state.version, state.round, state.last_question.kg_name if state.last_question else None, state.timings_ms)
        return response

    def _response(self, state):
        message = next((m.text for m in reversed(state.messages) if m.sender == "assistant"), "")
        return ChatResponse(**state.model_dump(), message=message, next_symptom=state.last_question)


