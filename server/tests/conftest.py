"""Shared pytest fixtures for the server tests.

Each test gets a FastAPI app backed by an isolated temp SQLite DB and an
isolated enrollment directory, so tests never touch the real ``data/`` tree.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient whose storage + enrollment dir live under a temp path."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("SIGNTALK_DB", str(db_path))

    # Redirect enrollment clips to a temp dir by patching the router's root.
    from server.app.routers import enroll as enroll_router

    enroll_dir = tmp_path / "enroll"
    monkeypatch.setattr(enroll_router, "_enroll_root", lambda: enroll_dir)

    # Build a fresh app so it binds the temp DB from the env var above.
    main = importlib.import_module("server.app.main")
    app = main.create_app()
    # TestClient as a context manager runs the lifespan (opens/closes storage).
    with TestClient(app) as c:
        yield c
