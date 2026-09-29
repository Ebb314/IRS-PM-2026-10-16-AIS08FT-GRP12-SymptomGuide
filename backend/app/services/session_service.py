"""Atomic read/modify/write and replay. SQLite is for the local academic demo."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from uuid import uuid4
from app.models.chat_models import SessionState

class SessionNotFoundError(KeyError):
    pass

class SQLiteSessionService:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=15)
        self._lock = RLock()
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
                scope TEXT, id TEXT, fingerprint TEXT NOT NULL, response TEXT NOT NULL,
                PRIMARY KEY(scope,id));
        """)

    @contextmanager
    def transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def create(self):
        return SessionState(session_id=uuid4().hex)

    def get(self, session_id):
        with self._lock:
            row = self._db.execute("SELECT payload FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise SessionNotFoundError(session_id)
            return SessionState.model_validate_json(row[0])

    def save(self, state):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)", (state.session_id, state.model_dump_json()))

    def replay(self, scope, request_id):
        return self._db.execute("SELECT fingerprint,response FROM requests WHERE scope=? AND id=?", (scope, request_id)).fetchone()

    def record(self, scope, request_id, fingerprint, response):
        self._db.execute("INSERT INTO requests VALUES (?,?,?,?)", (scope, request_id, fingerprint, response.model_dump_json()))

    def delete(self, session_id):
        with self.transaction():
            self.get(session_id)
            self._db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._db.execute("DELETE FROM requests WHERE scope=? OR json_extract(response,'$.session_id')=?", (session_id, session_id))

    def close(self):
        self._db.close()

class InMemorySessionService(SQLiteSessionService):
    def __init__(self):
        super().__init__(":memory:")

