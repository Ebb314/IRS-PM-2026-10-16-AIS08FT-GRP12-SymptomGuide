"""Agent dialogue + REASONING_MODE=kg: the informed-search question is asked with Yes/No/Not sure buttons."""
import json
from pathlib import Path
from uuid import uuid4

from app.models.chat_models import AnswerChatRequest, StartChatRequest, Symptom
from app.services.chatbot_service import ChatbotService
from app.services.conversation_agent import ConversationAgent
from app.services.kg_reasoning_service import build_kg_services
from app.services.nlp_service import NLPService
from app.services.random_reasoning_service import RandomRecallService, RandomSearchService
from app.services.mock_reasoning_service import MockDrugService
from app.services.session_service import InMemorySessionService

DATA = Path(__file__).parents[1] / "app/data"


class FakeGenerator:
    """Proposes Fever + Cough as observations; replies with a short acknowledgement."""
    name = "fake"

    def __init__(self, nlp):
        self.ids = {n.kg_name: k for k, n in nlp.nodes.items()}
        self.reply_prompts = []

    def generate(self, system, user, max_tokens=220):
        return json.dumps({"user_goal": "report symptoms", "observations": [
            {"concept_id": self.ids["Fever"], "quote": "fever", "status": "present"},
            {"concept_id": self.ids["Cough"], "quote": "cough", "status": "present"}]}), {"model": "fake"}

    def chat(self, messages, max_tokens=260, context_tokens=4096, sampling=True):
        self.reply_prompts.append(messages[0]["content"])
        return "Thanks, I have noted your fever and cough.", {"model": "fake"}


def make(mode):
    nlp = NLPService(DATA / "kg_nodes.csv", use_models=False)
    gen = FakeGenerator(nlp)
    if mode == "kg":
        recall, search, drugs = build_kg_services(str(DATA / "kg_reasoning.json.gz"), nlp.nodes)
    else:
        recall, search, drugs = RandomRecallService(), RandomSearchService(nlp.nodes), MockDrugService()
    chat = ChatbotService(nlp, InMemorySessionService(), recall, search, drugs, mode=mode, max_rounds=12,
                          dialogue=ConversationAgent(gen))
    return chat, gen


def test_kg_mode_agent_asks_search_question_with_buttons():
    chat, gen = make("kg")
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    assert r.next_symptom and r.symptom_question_active is True
    assert r.message.startswith("Thanks, I have noted your fever and cough.")
    assert f"Are you experiencing {r.next_symptom.kg_name}?" in r.message
    assert "do NOT ask any question yourself" in gen.reply_prompts[-1]
    # A button answer goes through the deterministic path: no extra LLM reply call.
    calls = len(gen.reply_prompts)
    r2 = chat.answer(AnswerChatRequest(session_id=r.session_id, expected_version=r.version, request_id=uuid4().hex,
        answer="no", symptom=Symptom(kg_id=r.next_symptom.kg_id, kg_name=r.next_symptom.kg_name)))
    assert len(gen.reply_prompts) == calls
    assert r2.status == "COMPLETE" or (r2.next_symptom and r2.symptom_question_active is True)


def test_random_mode_agent_behaviour_unchanged():
    chat, gen = make("random")
    r = chat.start(StartChatRequest(message="I have a fever and a cough."))
    assert r.symptom_question_active is False
    assert r.message == "Thanks, I have noted your fever and cough."
    assert "do NOT ask any question yourself" not in gen.reply_prompts[-1]
