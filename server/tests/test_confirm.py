"""POST /api/confirm records a user's candidate tap (Section 6.2)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_confirm_stores_choice(client: TestClient) -> None:
    resp = client.post(
        "/api/confirm",
        json={"clip_id": "clip-123", "chosen_label": "water"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"clip_id": "clip-123", "chosen_label": "water", "stored": True}


def test_confirm_requires_fields(client: TestClient) -> None:
    resp = client.post("/api/confirm", json={"clip_id": "x"})
    assert resp.status_code == 422
