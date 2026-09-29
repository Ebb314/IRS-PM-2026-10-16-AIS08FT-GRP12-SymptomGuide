from functools import lru_cache
from threading import Lock
from app.config import settings
from app.services.chatbot_service import ChatbotService
from app.services.mock_reasoning_service import MockDrugService, MockInformedSearchService, MockRecallService
from app.services.remote_adapters import RemoteGateway, RemoteRecallService, RemoteSearchService, RemoteDrugService
from app.services.nlp_service import NLPService
from app.services.session_service import SQLiteSessionService

_initialization_lock = Lock()

def get_chatbot_service():
    # lru_cache alone permits simultaneous cache misses to load duplicate models.
    with _initialization_lock:
        return _build_chatbot_service()

@lru_cache(maxsize=1)
def _build_chatbot_service():
    if settings.ai_profile not in {"offline", "local"}:
        raise ValueError("AI_PROFILE must be offline or local")
    if settings.ai_profile == "local" and (settings.nlp_engine != "rules" or settings.use_transformer_models):
        raise ValueError("Local profile requires NLP_ENGINE=rules and NLP_USE_TRANSFORMERS=false")
    nlp = NLPService()
    from app.services.dialogue_ai import DialogueAI
    dialogue = DialogueAI(vocabulary=nlp.nodes,mention_finder=nlp._mentions)
    if settings.ai_profile == "local":
        from app.services.model_runtime import LocalGenerator
        from app.services.retrieval_service import load_encoder
        from app.services.context_nlp import ContextNLP
        generator = LocalGenerator(settings.local_model_path)
        encoder = load_encoder()
        nlp = ContextNLP(nlp, generator, encoder)
        if settings.dialogue_engine=='agent':
            from app.services.conversation_agent import ConversationAgent
            dialogue=ConversationAgent(generator,encoder)
        elif settings.dialogue_engine=='legacy':
            dialogue = DialogueAI(generator,encoder,vocabulary=nlp.nodes,mention_finder=nlp._mentions)
        else:
            raise ValueError('DIALOGUE_ENGINE must be agent or legacy')
    if settings.ai_profile != "local" and settings.nlp_engine == "ollama":
        from app.services.llm_nlp_service import OllamaNLPService
        nlp = OllamaNLPService(nlp, settings.ollama_model, settings.ollama_url)
    elif settings.nlp_engine != "rules":
        raise ValueError("NLP_ENGINE must be rules or ollama")
    if settings.reasoning_mode == "mock":
        if not all(k.startswith("DEMO:") for k in nlp.nodes):
            raise ValueError("Mock mode requires the DEMO vocabulary. Use remote mode for teammate data.")
        recall, search, drugs = MockRecallService(), MockInformedSearchService(), MockDrugService()
    elif settings.reasoning_mode == "random":
        from app.services.random_reasoning_service import RandomRecallService,RandomSearchService
        recall,search,drugs=RandomRecallService(settings.random_seed),RandomSearchService(nlp.nodes,settings.random_seed),MockDrugService()
    elif settings.reasoning_mode == "remote":
        gateway = RemoteGateway(settings.teammate_url, settings.adapter_timeout)
        recall, search, drugs = RemoteRecallService(gateway), RemoteSearchService(gateway), RemoteDrugService(gateway)
    else:
        raise ValueError("REASONING_MODE must be mock, random or remote")
    return ChatbotService(nlp, SQLiteSessionService(settings.db_file), recall, search, drugs, mode=settings.reasoning_mode, dialogue=dialogue)

