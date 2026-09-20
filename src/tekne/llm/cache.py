"""Content-addressed cache for model calls.

Two jobs.  The obvious one is not paying twice for the same call.  The less
obvious and more important one is *replay determinism*: an extraction run that
cannot be reproduced byte-for-byte cannot be audited, and a technology-tracking
corpus that is rebuilt every quarter needs its old decisions to be explainable.
Keying on the full request (model, system, user, decoding parameters, backend
version) means a cached run replays exactly, and a changed prompt shows up as a
cache miss rather than as a silent behaviour change.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    request     TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS calls_model ON calls(model);
"""


def request_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CallCache:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, request: dict[str, Any], response: Any) -> None:
        if self._conn is None:
            return
        import time

        body = response if isinstance(response, dict) else asdict(response)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls (key, model, request, response, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    model,
                    json.dumps(request, sort_keys=True, default=str),
                    json.dumps(body, default=str),
                    time.time(),
                ),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        if self._conn is None:
            return {"entries": 0}
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        return {"entries": int(n)}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
