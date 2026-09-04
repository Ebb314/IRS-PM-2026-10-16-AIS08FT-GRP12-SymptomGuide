from threading import RLock
from typing import Dict
from uuid import uuid4

from app.models.chat_models import SessionState


class SessionNotFoundError(KeyError):
    pass


class InMemorySessionService:
    """Thread-safe development store; the public API is easy to back with SQLite later."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = RLock()

    def create(self) -> SessionState:
        state = SessionState(session_id=uuid4().hex)
        with self._lock:
            self._sessions[state.session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            return state

    def save(self, state: SessionState) -> None:
        with self._lock:
            self._sessions[state.session_id] = state
