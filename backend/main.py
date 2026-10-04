"""
main.py — SignTalk AI / Person A (ML Core), Prompt A1

Standalone FastAPI app exposing /ws/gesture: a WebSocket that accepts
base64-encoded JPEG frames at ~20-30fps, runs them through the MoveNet ->
normalize -> smooth -> buffer -> classify pipeline, and emits stabilized
gesture predictions.

NOTE: this file is the flat, single-module version described in Prompt A1.
Person C's Prompt C1 later reorganizes this logic into pose/ and
websocket/ routers behind auth — see backend/app/ for that restructured
version. This file remains runnable standalone for local dev/testing of
the ML core in isolation.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from classify import classify_sequence
from keypoint_utils import (
    SequenceBuffer,
    TemporalSmoother,
    decode_base64_jpeg,
    extract_keypoints,
    load_movenet,
    normalize_keypoints,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signtalk.main")

app = FastAPI(title="SignTalk AI — Gesture Recognition Core (Person A)")

# Stabilization tuning
STABILIZATION_AGREEMENT_COUNT = 3
COOLDOWN_SECONDS = 1.5
SEQUENCE_LENGTH = 30
SMOOTHING_WINDOW = 4


@app.on_event("startup")
async def startup_event():
    """Load MoveNet once at process startup — never per-request."""
    load_movenet()
    logger.info("SignTalk AI ML core ready.")


class ConnectionState:
    """
    Per-WebSocket-connection state: smoothing, sequence buffering, and
    prediction stabilization (N-consecutive-agreement + cooldown).
    """

    def __init__(self):
        self.smoother = TemporalSmoother(window=SMOOTHING_WINDOW)
        self.buffer = SequenceBuffer(maxlen=SEQUENCE_LENGTH)
        self.movenet_sig = load_movenet()

        self._recent_predictions: deque[str] = deque(maxlen=STABILIZATION_AGREEMENT_COUNT)
        self._last_emitted_label: str | None = None
        self._last_emitted_at: float = 0.0

    def process_frame(self, frame_bgr: np.ndarray) -> dict | None:
        """Runs one frame through the full pipeline. Returns an emission
        dict ({"label", "confidence", "timestamp"}) only when stabilization
        criteria are met, else None."""
        raw_keypoints = extract_keypoints(frame_bgr, self.movenet_sig)
        normalized = normalize_keypoints(raw_keypoints)
        smoothed = self.smoother.smooth(normalized)
        self.buffer.push(smoothed)

        sequence = self.buffer.get_sequence()
        if sequence is None:
            return None

        result = classify_sequence(sequence)
        return self._stabilize(result)

    def _stabilize(self, result: dict) -> dict | None:
        label = result["label"]
        confidence = result["confidence"]
        now = time.time()

        self._recent_predictions.append(label)

        agrees = (
            len(self._recent_predictions) == STABILIZATION_AGREEMENT_COUNT
            and len(set(self._recent_predictions)) == 1
        )
        if not agrees:
            return None

        if label == self._last_emitted_label and (now - self._last_emitted_at) < COOLDOWN_SECONDS:
            return None  # auto-repeat prevention / cooldown

        self._last_emitted_label = label
        self._last_emitted_at = now
        return {"label": label, "confidence": confidence, "timestamp": now}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/gesture")
async def ws_gesture(websocket: WebSocket):
    await websocket.accept()
    state = ConnectionState()
    logger.info("Client connected to /ws/gesture")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                b64_frame = payload["frame"]
            except (json.JSONDecodeError, KeyError, TypeError):
                # Backward-compatible: also accept a raw base64 string frame.
                b64_frame = raw

            try:
                frame_bgr = decode_base64_jpeg(b64_frame)
            except ValueError as exc:
                await websocket.send_text(json.dumps({"error": str(exc)}))
                continue

            emission = state.process_frame(frame_bgr)
            if emission is not None:
                await websocket.send_text(json.dumps(emission))

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/gesture")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
