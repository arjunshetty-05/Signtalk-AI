"""Enrollment flow: start -> clip -> progress (Section 6.5).

Asserts a clip file lands under ``data/enroll/<signer>/<session>/`` (redirected
to a temp dir by the conftest fixture) and that the per-sign count increments.
"""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from server.app.routers import enroll as enroll_router


def test_enroll_start_clip_progress(client: TestClient) -> None:
    signer = "guest-enroll"

    # start -> get a session id
    start = client.post("/api/enroll/start", json={"signer_id": signer})
    assert start.status_code == 200, start.text
    session_id = start.json()["session_id"]
    assert session_id

    # clip -> save one clip for sign "hello"
    clip_bytes = b"not-a-real-video-but-non-empty-bytes"
    clip_b64 = base64.b64encode(clip_bytes).decode("ascii")
    resp = client.post(
        "/api/enroll/clip",
        json={
            "signer_id": signer,
            "session_id": session_id,
            "sign": "hello",
            "clip": clip_b64,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1

    # the clip file exists under <enroll_root>/<signer>/<session>/
    enroll_dir = enroll_router._enroll_root() / signer / session_id
    saved = list(enroll_dir.glob("hello_*.webm"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == clip_bytes

    # a second clip for the same sign increments the count
    resp2 = client.post(
        "/api/enroll/clip",
        json={
            "signer_id": signer,
            "session_id": session_id,
            "sign": "hello",
            "clip": clip_b64,
        },
    )
    assert resp2.status_code == 200
    assert resp2.json()["count"] == 2

    # progress -> per-sign counts and total
    prog = client.get(f"/api/enroll/progress/{signer}")
    assert prog.status_code == 200
    pbody = prog.json()
    assert pbody["counts"]["hello"] == 2
    assert pbody["total"] == 2
