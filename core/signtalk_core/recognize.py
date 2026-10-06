"""Clip recognition library (PROJECT_CONTEXT Sections 5.3 / 6.1 / 6.8).

``recognize_clip`` is the single Python entry point used by serving (and demos):
it runs the full record-then-recognise pipeline over a WHOLE clip (never a
sliding window — v2's windows flapped, Section 5.1) and returns the exact
Section 6.1 response dict so the FastAPI ``/api/recognize`` handler is a thin
wrapper.

Pipeline (Section 5.3):

    frames -> extract landmarks -> quality gate -> build features -> model
    forward -> softmax -> top-3 candidates -> decision -> 6.1 dict

Phase-1 decision logic is intentionally trivial: ONE model, no TTA, so
``agreement`` is ``{"models": "1/1", "tta": "1/1"}`` and a clip that passes the
quality gate is accepted on the model's top-1 (ugly / low accuracy is fine for
Gate 1, Section 0.3). A clip that fails the quality gate returns ``decision
"reject"`` with the gate's ``reject_reason`` and a null label.

All tunables (T, F, thresholds) come from ``config/signtalk.yaml`` via
:mod:`signtalk_core.config`; no magic numbers live here.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from signtalk_core.config import Config, load_config
from signtalk_core.landmarks import (
    FrameLandmarks,
    HolisticDetector,
    decode_base64_jpeg,
    extract_landmarks,
)
from signtalk_core.features import build_features
from signtalk_core.quality import quality_check
from signtalk_core.models.pose_tcn import build_model

# Number of top candidates returned in the 6.1 response (Section 6.1: "up to 3").
TOP_K = 3


def _no_hands_frames(frame_count: int) -> list[FrameLandmarks]:
    """Build an all-zero / no-hands landmark sequence of ``frame_count`` frames.

    Used when no MediaPipe model bundle is available (e.g. the Phase-1 skeleton
    smoke test with synthetic solid-colour frames): every frame reports a
    zeroed pose and no hands, so the quality gate legitimately rejects with
    ``hands_not_visible`` and the 6.1 contract shape is still honoured.
    """
    from signtalk_core.landmarks import NUM_POSE_SUBSET

    return [
        FrameLandmarks(
            pose_subset=np.zeros((NUM_POSE_SUBSET, 2), dtype=np.float32),
            left_hand=None,
            right_hand=None,
            left_hand_present=False,
            right_hand_present=False,
        )
        for _ in range(max(frame_count, 1))
    ]


def _extract_or_empty(
    frames: list[str],
    detector: HolisticDetector | None,
) -> list[FrameLandmarks]:
    """Decode base64 frames and extract landmarks, or return a no-hands sequence.

    If a detector is supplied, the real MediaPipe pipeline runs. If not (no
    ``.task`` bundle configured, as in the skeleton smoke test), we decode the
    frames anyway so malformed input still raises, then return a no-hands
    sequence of the same length — which the quality gate rejects as
    ``hands_not_visible`` (a valid contract-shape PASS, Section 6.1).
    """
    if detector is not None:
        decoded = [decode_base64_jpeg(f) for f in frames]
        return extract_landmarks(decoded, detector)
    # No detector: still decode to validate the payload, then no-hands.
    for f in frames:
        decode_base64_jpeg(f)
    return _no_hands_frames(len(frames))


def _vocab_labels(cfg_root: Path) -> list[str]:
    """Load the ordered class labels from ``config/vocabulary.json``.

    The class index order equals the order of the ``signs`` array, so a label is
    recovered as ``labels[class_index]``.
    """
    import json

    path = cfg_root / "config" / "vocabulary.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    signs = data["signs"] if isinstance(data, dict) else data
    return [s["label"] for s in signs]


def _reject_response(
    clip_id: str,
    reject_reason: str,
    latency_ms: dict[str, int],
) -> dict[str, Any]:
    """Assemble a Section 6.1 reject dict (null label, empty candidates)."""
    return {
        "clip_id": clip_id,
        "decision": "reject",
        "label": None,
        "candidates": [],
        "confidence": 0.0,
        "margin": 0.0,
        "agreement": {"models": "1/1", "tta": "1/1"},
        "reject_reason": reject_reason,
        "latency_ms": latency_ms,
    }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis of a 1-D logit vector."""
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def recognize_clip(
    frames: list[str],
    fps: float,
    signer_id: str,
    scenario_id: str | None = None,
    *,
    cfg: Config | None = None,
    detector: HolisticDetector | None = None,
    model: Any | None = None,
    labels: list[str] | None = None,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recognise a whole clip and return the Section 6.1 response dict.

    Args:
        frames: base64 JPEG frames of the clip (one string per frame).
        fps: capture frame rate (used for the quality duration check).
        signer_id: id of the signer (guest id in Phase 1; recorded by callers).
        scenario_id: optional active scenario id (context prior; unused in
            Phase 1 but accepted for the 6.8 contract).
        cfg: optional preloaded config (defaults to ``load_config()``).
        detector: optional MediaPipe holistic detector. If ``None`` the clip is
            treated as no-hands (skeleton mode) and rejected as
            ``hands_not_visible``.
        model: optional preconstructed classifier. Defaults to a random-init
            :class:`PoseGRU` sized to the vocabulary.
        labels: optional ordered class labels. Defaults to the vocabulary file.
        config_root: optional repo root holding ``config/`` (defaults to the
            config module's resolved root).

    Returns:
        A dict shaped EXACTLY like the Section 6.1 ``/api/recognize`` response
        (same keys, same value types), with per-stage ``latency_ms``.

    Raises:
        ValueError: if ``frames`` is empty or ``fps`` is not positive.
    """
    if not frames:
        raise ValueError("frames is empty; nothing to recognise")
    if fps <= 0:
        raise ValueError("fps must be positive")

    if cfg is None:
        cfg = load_config()
    if config_root is None:
        from signtalk_core.config import _repo_root

        root = _repo_root()
    else:
        root = Path(config_root)
    if labels is None:
        labels = _vocab_labels(root)

    clip_id = str(uuid.uuid4())
    latency_ms: dict[str, int] = {"extract": 0, "models": 0, "fusion": 0, "total": 0}
    t_start = time.perf_counter()

    # --- Stage 1: landmark extraction -------------------------------------- #
    t0 = time.perf_counter()
    landmark_seq = _extract_or_empty(frames, detector)
    latency_ms["extract"] = int((time.perf_counter() - t0) * 1000)

    # --- Stage 2: quality gate --------------------------------------------- #
    passed, reject_reason = quality_check(landmark_seq, cfg, fps)
    if not passed:
        latency_ms["total"] = int((time.perf_counter() - t_start) * 1000)
        return _reject_response(clip_id, reject_reason or "low_quality", latency_ms)

    # --- Stage 3: features + model forward --------------------------------- #
    t0 = time.perf_counter()
    feats = build_features(landmark_seq, cfg)  # [T, F] float32
    if model is None:
        model = build_model(num_classes=len(labels))
    x = torch.from_numpy(feats).unsqueeze(0)  # [1, T, F]
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu().numpy()  # [num_classes]
    latency_ms["models"] = int((time.perf_counter() - t0) * 1000)

    # --- Stage 4: softmax + top-k + decision (fusion) ---------------------- #
    t0 = time.perf_counter()
    probs = _softmax_np(logits)
    order = np.argsort(probs)[::-1]
    top = order[:TOP_K]
    candidates = [{"label": labels[int(i)], "p": float(probs[int(i)])} for i in top]
    confidence = float(probs[int(top[0])])
    margin = float(probs[int(top[0])] - probs[int(top[1])]) if len(top) > 1 else confidence

    # Phase-1 trivial decision: a clip that passed the quality gate is accepted
    # on the model's top-1 (accuracy irrelevant for Gate 1, Section 0.3).
    label = labels[int(top[0])]
    decision = "accept"
    latency_ms["fusion"] = int((time.perf_counter() - t0) * 1000)

    latency_ms["total"] = int((time.perf_counter() - t_start) * 1000)
    return {
        "clip_id": clip_id,
        "decision": decision,
        "label": label,
        "candidates": candidates,
        "confidence": confidence,
        "margin": margin,
        "agreement": {"models": "1/1", "tta": "1/1"},
        "reject_reason": None,
        "latency_ms": latency_ms,
    }
