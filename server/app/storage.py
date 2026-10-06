"""SQLite storage (PROJECT_CONTEXT Section 2 / context.json — NO Firebase).

Guest mode, local SQLite via the stdlib ``sqlite3`` module. Two tables:

* ``decisions``  — a log of every /api/recognize decision (clip_id, signer_id,
  decision, label, confidence, margin, the full probability vector as JSON, and
  the decision thresholds that were in force), for later review/analysis.
* ``enrollment`` — an index of enrollment clips (signer_id, session_id, sign,
  clip_path, timestamp) used to compute per-sign progress.

The DB file path is configurable via the ``SIGNTALK_DB`` env var and defaults to
``data/signtalk.db`` under the repo root — ``data/`` is gitignored (Section 7),
so the DB is never committed. The directory is created on first use.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Repo/worktree root: this file is at ``<root>/server/app/storage.py``."""
    return Path(__file__).resolve().parents[2]


def default_db_path() -> Path:
    """Default SQLite path: ``data/signtalk.db`` (gitignored) under the root.

    Overridable with the ``SIGNTALK_DB`` environment variable.
    """
    env = os.environ.get("SIGNTALK_DB")
    if env:
        return Path(env)
    return _repo_root() / "data" / "signtalk.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id TEXT NOT NULL,
    signer_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    margin REAL,
    probs_json TEXT,
    thresholds_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS enrollment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    sign TEXT NOT NULL,
    clip_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS confirmations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id TEXT NOT NULL,
    chosen_label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_enroll_signer ON enrollment(signer_id);
CREATE INDEX IF NOT EXISTS idx_decisions_signer ON decisions(signer_id);
CREATE INDEX IF NOT EXISTS idx_confirm_clip ON confirmations(clip_id);
"""


def _now() -> str:
    """UTC ISO-8601 timestamp string."""
    return datetime.now(timezone.utc).isoformat()


class Storage:
    """Thin SQLite wrapper for decisions + enrollment (thread-safe)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock: FastAPI's TestClient / uvicorn may
        # touch the connection from worker threads.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def log_decision(
        self,
        *,
        clip_id: str,
        signer_id: str,
        decision: str,
        label: str | None,
        confidence: float,
        margin: float,
        probs: list[dict[str, Any]] | None,
        thresholds: dict[str, Any] | None,
    ) -> None:
        """Insert one recognition decision into the ``decisions`` log."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO decisions
                   (clip_id, signer_id, decision, label, confidence, margin,
                    probs_json, thresholds_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clip_id,
                    signer_id,
                    decision,
                    label,
                    confidence,
                    margin,
                    json.dumps(probs) if probs is not None else None,
                    json.dumps(thresholds) if thresholds is not None else None,
                    _now(),
                ),
            )
            self._conn.commit()

    def log_confirmation(self, *, clip_id: str, chosen_label: str) -> None:
        """Record a user's confirm-tap for a clip (Section 6.2).

        The chosen candidate becomes a labelled sample for later review
        (``data/feedback`` active-learning flow, Section 8.6) — it is NEVER
        auto-fed into training here.
        """
        with self._lock:
            self._conn.execute(
                """INSERT INTO confirmations (clip_id, chosen_label, created_at)
                   VALUES (?, ?, ?)""",
                (clip_id, chosen_label, _now()),
            )
            self._conn.commit()

    def index_enroll_clip(
        self,
        *,
        signer_id: str,
        session_id: str,
        sign: str,
        clip_path: str,
    ) -> None:
        """Index one saved enrollment clip in the ``enrollment`` table."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO enrollment
                   (signer_id, session_id, sign, clip_path, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (signer_id, session_id, sign, clip_path, _now()),
            )
            self._conn.commit()

    def enroll_progress(self, signer_id: str) -> dict[str, int]:
        """Return a ``{sign: clip_count}`` map for one signer."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT sign, COUNT(*) AS n FROM enrollment
                   WHERE signer_id = ? GROUP BY sign""",
                (signer_id,),
            ).fetchall()
        return {row["sign"]: int(row["n"]) for row in rows}

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()
