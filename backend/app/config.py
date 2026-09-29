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
    api_prefix: str = "/api"
    kg_node_file: Path = Path(os.getenv("KG_NODE_FILE", str(BASE_DIR / "data" / "kg_nodes.csv")))
    db_file: str = os.getenv("SESSION_DB", str(BASE_DIR.parent / "runtime" / "sessions-fullkg.sqlite3"))
    finish_candidate_threshold: int = 2
    max_rounds: int = 5
    use_transformer_models: bool = os.getenv("NLP_USE_TRANSFORMERS", "false").lower() == "true"
    reasoning_mode: str = os.getenv("REASONING_MODE", "random")
    random_seed: int = int(os.getenv("RANDOM_SEED", "20260915"))
    nlp_engine: str = os.getenv("NLP_ENGINE", "rules")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    teammate_url: str = os.getenv("TEAMMATE_API_URL", "http://127.0.0.1:8010")
    adapter_timeout: float = float(os.getenv("ADAPTER_TIMEOUT_SECONDS", "3"))
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")

settings = Settings()


