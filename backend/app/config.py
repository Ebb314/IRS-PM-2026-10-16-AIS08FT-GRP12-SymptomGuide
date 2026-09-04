from dataclasses import dataclass
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    app_name: str = "Symptom Triage Chatbot"
    api_prefix: str = "/api"
    kg_node_file: Path = BASE_DIR / "data" / "kg_nodes.csv"
    finish_candidate_threshold: int = int(os.getenv("FINISH_CANDIDATE_THRESHOLD", "2"))
    max_rounds: int = int(os.getenv("MAX_CHAT_ROUNDS", "5"))
    use_transformer_models: bool = os.getenv("NLP_USE_TRANSFORMERS", "false").lower() == "true"
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")


settings = Settings()
