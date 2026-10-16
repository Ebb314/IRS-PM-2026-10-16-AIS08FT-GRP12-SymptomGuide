import sys
from pathlib import Path
import os
# Legacy regression fixtures remain explicit and independent of application defaults.
os.environ['KG_NODE_FILE']=str(Path(__file__).resolve().parents[1]/'app/data/demo_symptoms.csv')
os.environ['REASONING_MODE']='mock'
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app
from app.dependencies import get_chatbot_service
from app.services.chatbot_service import ChatbotService
from app.services.session_service import InMemorySessionService
from app.services.nlp_service import NLPService
from app.services.mock_reasoning_service import MockRecallService, MockInformedSearchService, MockDrugService

@pytest.fixture
def chatbot():
    store = InMemorySessionService()
    service = ChatbotService(NLPService(use_models=False), store, MockRecallService(), MockInformedSearchService(), MockDrugService())
    yield service
    store.close()

@pytest.fixture
def client(chatbot):
    app.dependency_overrides[get_chatbot_service] = lambda: chatbot
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

