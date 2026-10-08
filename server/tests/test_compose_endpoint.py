"""POST /api/compose builds a sentence in en/hi/kn (Section 6.3)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_compose_scripted(client: TestClient) -> None:
    resp = client.post("/api/compose", json={"words": ["hello"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "scripted"
    assert body["verified"] is True
    assert set(body["sentences"].keys()) == {"en", "hi", "kn"}
    assert all(body["sentences"].values())
    assert isinstance(body["latency_ms"], int)


def test_compose_template_fallback(client: TestClient) -> None:
    # A sequence that is NOT in the scripted table, with no LLM key -> template.
    resp = client.post("/api/compose", json={"words": ["zzxx", "qqww"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "template"
    assert body["verified"] is False


def test_compose_requires_words(client: TestClient) -> None:
    resp = client.post("/api/compose", json={"emotion": "neutral"})
    assert resp.status_code == 422
