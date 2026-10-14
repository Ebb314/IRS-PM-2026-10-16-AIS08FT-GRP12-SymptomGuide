from dataclasses import dataclass
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Settings:
    app_name: str = "Symptom Guide - Dialogue Lab"
    ai_profile: str = os.getenv("AI_PROFILE", "offline")
    dialogue_engine: str = os.getenv('DIALOGUE_ENGINE','agent')
    local_model_path: str = os.getenv("LOCAL_MODEL_PATH", str(BASE_DIR.parents[1] / "models" / "Qwen3-4B-Instruct-2507"))
    # LLM for AI_PROFILE=local: transformers (bundled Qwen3-4B) or ollama (any model served by Ollama)
    llm_backend: str = os.getenv("LLM_BACKEND", "transformers")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    ollama_think: bool = os.getenv("OLLAMA_THINK", "false").lower() == "true"
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "")
    api_prefix: str = "/api"
    kg_node_file: Path = Path(os.getenv("KG_NODE_FILE", str(BASE_DIR / "data" / "kg_nodes.csv")))
    db_file: str = os.getenv("SESSION_DB", str(BASE_DIR.parent / "runtime" / "sessions-fullkg.sqlite3"))
    finish_candidate_threshold: int = 2
    # kg mode needs more follow-ups: starting from common symptoms, 5 rounds rarely narrow
    # PrimeKG to <= 2 candidates (see docs/KG推理集成记录.md). Other modes keep the contract's 5.
    max_rounds: int = int(os.getenv("MAX_ROUNDS", "12" if os.getenv("REASONING_MODE", "random") == "kg" else "5"))
    use_transformer_models: bool = os.getenv("NLP_USE_TRANSFORMERS", "false").lower() == "true"
    reasoning_mode: str = os.getenv("REASONING_MODE", "random")
    random_seed: int = int(os.getenv("RANDOM_SEED", "20260915"))
    nlp_engine: str = os.getenv("NLP_ENGINE", "rules")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    teammate_url: str = os.getenv("TEAMMATE_API_URL", "http://127.0.0.1:8010")
    adapter_timeout: float = float(os.getenv("ADAPTER_TIMEOUT_SECONDS", "3"))
    # REASONING_MODE=kg: in-process PrimeKG reasoning (app/services/kg_reasoning_service.py)
    kg_reasoning_file: str = os.getenv("KG_REASONING_FILE", str(BASE_DIR / "data" / "kg_reasoning.json.gz"))
    kg_top_k: int = int(os.getenv("KG_TOP_K", "10"))
    kg_drugs_per_disease: int = int(os.getenv("KG_DRUGS_PER_DISEASE", "5"))
    recall_symptom_whitelist: str = os.getenv("RECALL_SYMPTOM_WHITELIST", "")
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")

settings = Settings()


