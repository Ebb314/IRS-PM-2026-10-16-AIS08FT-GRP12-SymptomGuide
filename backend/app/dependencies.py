from functools import lru_cache

from app.services.chatbot_service import ChatbotService
from app.services.mock_reasoning_service import MockDrugService, MockInformedSearchService, MockRecallService
from app.services.nlp_service import NLPService
from app.services.session_service import InMemorySessionService


_sessions = InMemorySessionService()


@lru_cache(maxsize=1)
def get_chatbot_service() -> ChatbotService:
    return ChatbotService(
        nlp=NLPService(),
        sessions=_sessions,
        recall=MockRecallService(),
        search=MockInformedSearchService(),
        drugs=MockDrugService(),
    )
