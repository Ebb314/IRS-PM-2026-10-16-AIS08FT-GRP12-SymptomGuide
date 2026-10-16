"""Pipeline logging: one logger per step, every line tagged with the session.

Loggers (filter with LOG_LEVEL / grep):
  chatbot.startup  services built at startup (mode, vocabulary, model, LLM)
  chatbot.flow     each request: action, symptom states, stop decision, timings
  chatbot.nlp      rule-based symptom extraction
  chatbot.agent    LLM understanding (proposals accepted / rejected) and reply
  chatbot.llm      every Ollama call (latency, tokens) and failures
  chatbot.recall   recall: model input, top probabilities, candidate set
  chatbot.search   informed search: heuristic table and chosen symptom
  chatbot.drugs    drug retrieval per disease

Environment:
  LOG_LEVEL      INFO (default) | DEBUG (adds full tables) | WARNING
  LOG_FILE       backend/logs/chatbot.log by default (rotating, 5 MB x 3); empty = console only
  LOG_USER_TEXT  false (default): raw user messages are not logged, only their length
Session IDs are never logged; a short hash (sid=xxxxxxxx) links the lines of one session.
"""
import contextvars
import hashlib
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = "chatbot"
_session = contextvars.ContextVar("chatbot_session", default="-")
_configured = False


def get_logger(step: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{step}")


def short_id(session_id) -> str:
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:8] if session_id else "-"


def set_session(session_id) -> None:
    _session.set(short_id(session_id))


def user_text(text: str) -> str:
    """Raw message only when LOG_USER_TEXT=true; otherwise its length."""
    if os.getenv("LOG_USER_TEXT", "false").lower() == "true":
        return repr(text[:300])
    return f"<{len(text)} chars>"


def fmt_scores(items, n=5) -> str:
    """[(name, score), ...] -> 'a 0.412, b 0.201, ...'"""
    return ", ".join(f"{name} {score:.4f}" for name, score in list(items)[:n]) or "(none)"


def names(symptoms) -> list:
    return [s.kg_name for s in symptoms]


def fmt_states(evidence) -> str:
    """ExtractedSymptom list -> 'present=[..] absent=[..] uncertain=[..]' (empty groups omitted)."""
    groups = {}
    for e in evidence:
        groups.setdefault(e.status, []).append(e.kg_name)
    return " ".join(f"{k}={v}" for k, v in groups.items()) or "(no symptoms)"


class _SessionFilter(logging.Filter):
    def filter(self, record):
        record.sid = _session.get()
        return True


def setup_logging() -> None:
    """Idempotent. Called at startup (dependencies / main); tests that skip it keep pytest's capture."""
    global _configured
    if _configured:
        return
    _configured = True
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s [sid=%(sid)s] %(name)-15s %(message)s", "%H:%M:%S")
    root = logging.getLogger(ROOT)
    root.setLevel(level)
    root.propagate = False  # independent of uvicorn's logging config
    handlers = [logging.StreamHandler(sys.stderr)]
    default_file = Path(__file__).resolve().parents[1] / "logs" / "chatbot.log"
    log_file = os.getenv("LOG_FILE", str(default_file))
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
        handler.addFilter(_SessionFilter())
        root.addHandler(handler)
