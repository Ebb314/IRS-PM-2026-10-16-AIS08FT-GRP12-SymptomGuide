from functools import lru_cache
from threading import Lock
from time import perf_counter
from app.config import settings
from app.logging_setup import get_logger, setup_logging
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
    setup_logging()
    started = perf_counter()
    get_logger("startup").info("building services: ai_profile=%s reasoning=%s llm_backend=%s",
                               settings.ai_profile, settings.reasoning_mode, settings.llm_backend)
    if settings.ai_profile not in {"offline", "local"}:
        raise ValueError("AI_PROFILE must be offline or local")
    if settings.ai_profile == "local" and (settings.nlp_engine != "rules" or settings.use_transformer_models):
        raise ValueError("Local profile requires NLP_ENGINE=rules and NLP_USE_TRANSFORMERS=false")
    nlp = NLPService()
    from app.services.dialogue_ai import DialogueAI
    dialogue = DialogueAI(vocabulary=nlp.nodes,mention_finder=nlp._mentions)
    if settings.ai_profile == "local":
        from app.services.context_nlp import ContextNLP
        if settings.llm_backend == "ollama":
            # Ollama-served model (e.g. qwen3:8b); no PyTorch / transformers needed.
            from pathlib import Path
            from app.services.ollama_runtime import OllamaEncoder, OllamaGenerator
            generator = OllamaGenerator(settings.ollama_model or "qwen3:8b", settings.ollama_url,
                                        settings.ollama_timeout, settings.ollama_think).check()
            encoder = (OllamaEncoder(settings.ollama_embed_model, settings.ollama_url, settings.ollama_timeout,
                                     cache_dir=Path(settings.db_file).parent / "embeddings")
                       if settings.ollama_embed_model else None)  # None: BM25-only retrieval
        elif settings.llm_backend == "transformers":
            from app.services.model_runtime import LocalGenerator
            from app.services.retrieval_service import load_encoder
            generator = LocalGenerator(settings.local_model_path)
            encoder = load_encoder()
        else:
            raise ValueError("LLM_BACKEND must be transformers or ollama")
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
    elif settings.reasoning_mode == "kg":
        from app.services.kg_reasoning_service import build_kg_services
        recall, search, drugs = build_kg_services(settings.kg_reasoning_file, nlp.nodes, settings.kg_top_k,
            settings.kg_drugs_per_disease, settings.recall_symptom_whitelist or None)
    elif settings.reasoning_mode == "ddxplus":
        from app.services.ddxplus_reasoning_service import build_ddxplus_services
        recall, search, drugs = build_ddxplus_services(nlp.nodes, settings.kg_reasoning_file, settings.ddxplus_top_k,
            settings.ddxplus_mass, settings.ddxplus_temperature, settings.ddxplus_hard_filter,
            settings.kg_drugs_per_disease, settings.ddxplus_conditions_file or None,
            settings.recall_symptom_whitelist or None, settings.ddxplus_backend, settings.ddxplus_confident)
    elif settings.reasoning_mode == "recall773":
        from app.services.recall773_reasoning_service import build_recall773_services
        recall, search, drugs = build_recall773_services(nlp.nodes, settings.kg_reasoning_file, settings.recall773_top_k,
            settings.recall773_mass, 1.0, settings.kg_drugs_per_disease, settings.disease_symptom_table or None,
            settings.recall_symptom_whitelist or None, settings.recall773_backend, settings.recall773_confident,
            settings.recall773_variant or None)
    else:
        raise ValueError("REASONING_MODE must be mock, random, remote, kg, ddxplus or recall773")
    service = ChatbotService(nlp, SQLiteSessionService(settings.db_file), recall, search, drugs, mode=settings.reasoning_mode, dialogue=dialogue)
    _log_startup(service, perf_counter() - started)
    return service


def _log_startup(service, seconds):
    log = get_logger("startup")
    nodes = service.nlp.nodes
    log.info("vocabulary: %d symptom nodes (%d model-only DSS_ nodes), nlp=%s, dialogue=%s",
             len(nodes), sum(k.startswith("DSS_") for k in nodes), type(service.nlp).__name__, type(service.dialogue).__name__)
    generator = getattr(service.dialogue, "generator", None)
    log.info("llm: %s", generator.name if generator else "none (offline templates)")
    log.info("reasoning: mode=%s recall=%s search=%s drugs=%s max_rounds=%d stop at <= %d candidates",
             service.mode, type(service.recall).__name__, type(service.search).__name__, type(service.drugs).__name__,
             service.max_rounds, settings.finish_candidate_threshold)
    data = getattr(service.recall, "data", None)
    model = getattr(data, "model", None)
    if model is not None:
        log.info("recall model: %s backend=%s diseases=%d symptoms=%d top_k=%s mass=%s confident=%s disease-symptom table=%s",
                 getattr(service.recall, "source", "?"), model.backend, len(model.diseases), len(model.features),
                 service.recall.top_k, service.recall.mass, service.recall.confident, getattr(data, "table_source", "?"))
    log.info("ready in %.1f s", seconds)

