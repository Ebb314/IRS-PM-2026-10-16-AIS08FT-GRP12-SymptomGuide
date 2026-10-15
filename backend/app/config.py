from dataclasses import dataclass
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Settings:
    app_name: str = "Symptom Guide"
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
    # Colloquial phrases for symptom names ("sore throat" -> Pharyngitis). Empty string disables.
    symptom_alias_file: str = os.getenv("SYMPTOM_ALIAS_FILE", str(BASE_DIR / "data" / "symptom_aliases.csv"))
    # Extra symptom nodes appended to the vocabulary (recall773 mode: model-only symptoms).
    extra_vocab_file: str = os.getenv("EXTRA_VOCAB_FILE", str(BASE_DIR / "data" / "recall773" / "extra_symptom_nodes.csv")
                                      if os.getenv("REASONING_MODE", "random") == "recall773" else "")
    db_file: str = os.getenv("SESSION_DB", str(BASE_DIR.parent / "runtime" / "sessions-fullkg.sqlite3"))
    finish_candidate_threshold: int = 2
    # kg mode needs more follow-ups: starting from common symptoms, 5 rounds rarely narrow
    # PrimeKG to <= 2 candidates (see docs/KG推理集成记录.md). Other modes keep the contract's 5.
    max_rounds: int = int(os.getenv("MAX_ROUNDS", "12" if os.getenv("REASONING_MODE", "random") in ("kg", "ddxplus", "recall773") else "5"))
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
    # REASONING_MODE=ddxplus: teammate's DDXPlus recall model (app/services/ddxplus_reasoning_service.py)
    ddxplus_backend: str = os.getenv("DDXPLUS_BACKEND", "auto")  # auto | torch | numpy
    ddxplus_top_k: int = int(os.getenv("DDXPLUS_TOP_K", "10"))
    ddxplus_mass: float = float(os.getenv("DDXPLUS_MASS", "0.95"))
    ddxplus_temperature: float = float(os.getenv("DDXPLUS_TEMPERATURE", "1.0"))
    ddxplus_confident: float = float(os.getenv("DDXPLUS_CONFIDENT", "0.8"))
    ddxplus_hard_filter: bool = os.getenv("DDXPLUS_HARD_FILTER", "false").lower() == "true"
    ddxplus_conditions_file: str = os.getenv("DDXPLUS_CONDITIONS_FILE", "")
    # REASONING_MODE=recall773: teammate's 773-disease model (app/services/recall773_reasoning_service.py)
    recall773_backend: str = os.getenv("RECALL773_BACKEND", "auto")  # auto | torch | numpy
    recall773_variant: str = os.getenv("RECALL773_VARIANT", "")     # empty = teammate's selection.json
    recall773_top_k: int = int(os.getenv("RECALL773_TOP_K", "10"))
    recall773_mass: float = float(os.getenv("RECALL773_MASS", "0.95"))
    recall773_confident: float = float(os.getenv("RECALL773_CONFIDENT", "0.8"))
    # Stop when the top-3 diseases hold this much probability; the result is a top-3 differential.
    recall773_stop_top3_mass: float = float(os.getenv("RECALL773_STOP_TOP3_MASS", "0.9"))
    # Exported by scripts/export_disease_symptom_table.py; used automatically when present.
    disease_symptom_table: str = os.getenv("DISEASE_SYMPTOM_TABLE", str(BASE_DIR / "data" / "recall773" / "disease_symptom_frequency.csv")
                                           if (BASE_DIR / "data" / "recall773" / "disease_symptom_frequency.csv").is_file() else "")
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")

settings = Settings()


