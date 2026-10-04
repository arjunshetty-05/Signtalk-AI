"""
api/pose/service.py — SignTalk AI / Person C, Prompt C1

Restructured home of the gesture pipeline's per-connection state. Wraps
the top-level keypoint_utils.py + classify.py (Person A's code — reused
as-is, not duplicated) and adds a frame-skip counter so a client under
load can be told to process only every Nth frame.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from classify import classify_sequence
from keypoint_utils import (
    SequenceBuffer,
    TemporalSmoother,
    extract_keypoints,
    load_movenet,
    normalize_keypoints,
)

SEQUENCE_LENGTH = 30
SMOOTHING_WINDOW = 4
STABILIZATION_AGREEMENT_COUNT = 3
COOLDOWN_SECONDS = 1.5
# Below this, a prediction is suppressed entirely (treated as "nothing
# recognized yet") rather than displayed. Re-measured after expanding the
# deployed model from 40 to 262 classes (exp_all262_reg): on a 200-example
# sample, correct whole-clip predictions had median confidence 88.8% and
# wrong ones almost never exceeded 58% (p90 = 57.6%), so 0.55 lets through
# ~0 wrong guesses in that sample while keeping a few more correct ones
# than the old 0.65 did. Note this was measured on whole-clip
# classification (matching /pose/classify-clip's preprocessing) — live
# /ws/gesture streaming uses a sliding window instead, which sees
# incomplete gesture motion more often now that there are 262 candidate
# classes to confuse it with instead of 40, so don't expect this alone to
# fix live-feed responsiveness. Tune if it feels too strict/lenient.
MIN_EMIT_CONFIDENCE = 0.55


class GestureConnectionState:
    """Per-WebSocket-connection pipeline state, with optional frame skipping."""

    def __init__(self, frame_skip: int = 1):
        """frame_skip=1 processes every frame; frame_skip=N processes 1 of
        every N frames (client under load can request a coarser rate)."""
        self.frame_skip = max(1, frame_skip)
        self._frame_counter = 0

        self.smoother = TemporalSmoother(window=SMOOTHING_WINDOW)
        self.buffer = SequenceBuffer(maxlen=SEQUENCE_LENGTH)
        self.movenet_sig = load_movenet()

        self._recent_predictions: deque[str] = deque(maxlen=STABILIZATION_AGREEMENT_COUNT)
        self._last_emitted_label: str | None = None
        self._last_emitted_at: float = 0.0

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[dict | None, float]:
        """Returns (emission_or_None, inference_latency_ms)."""
        self._frame_counter += 1
        if self._frame_counter % self.frame_skip != 0:
            return None, 0.0

        start = time.time()
        raw_keypoints = extract_keypoints(frame_bgr, self.movenet_sig)
        normalized = normalize_keypoints(raw_keypoints)
        smoothed = self.smoother.smooth(normalized)
        self.buffer.push(smoothed)

        sequence = self.buffer.get_sequence()
        if sequence is None:
            return None, (time.time() - start) * 1000

        result = classify_sequence(sequence)
        latency_ms = (time.time() - start) * 1000
        return self._stabilize(result), latency_ms

    def _stabilize(self, result: dict) -> dict | None:
        label, confidence = result["label"], result["confidence"]
        now = time.time()

        if confidence < MIN_EMIT_CONFIDENCE:
            return None

        self._recent_predictions.append(label)
        agrees = (
            len(self._recent_predictions) == STABILIZATION_AGREEMENT_COUNT
            and len(set(self._recent_predictions)) == 1
        )
        if not agrees:
            return None

        if label == self._last_emitted_label and (now - self._last_emitted_at) < COOLDOWN_SECONDS:
            return None

        self._last_emitted_label = label
        self._last_emitted_at = now
        return {"label": label, "confidence": confidence, "timestamp": now}
