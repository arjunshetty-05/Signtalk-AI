"""POST /api/recognize honours the exact Section 6.1 contract shape.

The skeleton has no MediaPipe model bundle wired up, so a synthetic
solid-colour frame batch is treated as no-hands and legitimately returns
``decision == 'reject'`` with ``reject_reason == 'hands_not_visible'``. That is
a PASS: the test asserts the full 6.1 key set and value types, not accuracy.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

# The exact Section 6.1 response keys.
_SIX_ONE_KEYS = {
    "clip_id",
    "decision",
    "label",
    "candidates",
    "confidence",
    "margin",
    "agreement",
    "reject_reason",
    "latency_ms",
}


def _solid_frame_b64(color: tuple[int, int, int], size: int = 64) -> str:
    """Return a base64 JPEG of a solid-colour BGR frame."""
    frame = np.full((size, size, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _assert_six_one_shape(body: dict) -> None:
    assert set(body.keys()) == _SIX_ONE_KEYS
    assert body["decision"] in {"accept", "confirm", "reject"}
    assert isinstance(body["clip_id"], str) and body["clip_id"]
    assert body["label"] is None or isinstance(body["label"], str)
    assert isinstance(body["candidates"], list)
    assert len(body["candidates"]) <= 3
    for cand in body["candidates"]:
        assert set(cand.keys()) == {"label", "p"}
        assert isinstance(cand["label"], str)
        assert isinstance(cand["p"], float)
    assert isinstance(body["confidence"], float)
    assert isinstance(body["margin"], float)
    assert set(body["agreement"].keys()) == {"models", "tta"}
    assert body["reject_reason"] in {
        None,
        "low_quality",
        "hands_not_visible",
        "too_short",
        "too_long",
        "low_confidence",
    }
    assert set(body["latency_ms"].keys()) == {"extract", "models", "fusion", "total"}
    for v in body["latency_ms"].values():
        assert isinstance(v, int)


def test_recognize_json_contract(client: TestClient) -> None:
    """A synthetic JSON frame batch returns 200 with the full 6.1 shape."""
    frames = [_solid_frame_b64((40, 80, 120)) for _ in range(24)]
    resp = client.post(
        "/api/recognize",
        json={"frames": frames, "fps": 20.0, "signer_id": "guest-test"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_six_one_shape(body)
    # No-hands synthetic clip -> reject/hands_not_visible is an acceptable PASS.
    if body["decision"] == "reject":
        assert body["reject_reason"] is not None


def test_recognize_rejects_bad_json(client: TestClient) -> None:
    """Missing required fields -> 422 (Pydantic validation)."""
    resp = client.post("/api/recognize", json={"frames": []})
    assert resp.status_code == 422
