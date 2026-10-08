"""GET /health returns 200 with a status-ok body (Section 6.7)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_vocab_items(client: TestClient) -> None:
    """GET /api/vocab returns the demo signs with the right shape (Section 6.6)."""
    resp = client.get("/api/vocab")
    assert resp.status_code == 200
    items = resp.json()
    assert isinstance(items, list)
    assert len(items) >= 1  # vocabulary tracks the trained model; count may change
    for item in items:
        assert set(item.keys()) == {"label", "category", "demo"}
        assert item["demo"] is True
