"""Clip recognition library (PROJECT_CONTEXT Sections 5.1 / 5.3 / 6.1 / 6.8).

``recognize_clip`` is the single Python entry point used by serving (and demos):
it runs the full record-then-recognise pipeline over a WHOLE clip (never a
sliding window — v2's windows flapped, Section 5.1) and returns the exact
Section 6.1 response dict so the FastAPI ``/api/recognize`` handler is a thin
wrapper.

Pipeline (Sections 5.1 / 5.3):

    frames -> extract landmarks -> quality gate -> TTA feature views ->
    per-model forward over each view -> per-model temperature calibration ->
    fuse (weighted avg + optional context prior) -> decision engine
    (accept | confirm | reject) -> 6.1 dict

The ensemble (L8), TTA (L9), calibration (L8) and decision engine (L10) are all
driven by :class:`RecognizerBundle`, which serving builds once at startup from
trained checkpoints + fitted temperatures + tuned weights. When no bundle and no
detector are supplied (the Phase-1 skeleton / no ``.task`` model), the function
degrades to a single random-init model and a trivial accept, so the contract
shape is still exercised end to end (Section 0.3).

All tunables (T, F, thresholds, weights) come from ``config/signtalk.yaml`` and
the bundle; no magic numbers live here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from signtalk_core.calibration import apply_temperature
from signtalk_core.config import Config, load_config
from signtalk_core.decision import decide
from signtalk_core.fusion import FusionResult, apply_context_prior, fuse_probabilities
from signtalk_core.landmarks import (
    FrameLandmarks,
    HolisticDetector,
    decode_base64_jpeg,
    extract_landmarks,
)
from signtalk_core.models.pose_tcn import build_model
from signtalk_core.quality import quality_check
from signtalk_core.tta import build_tta_feature_views, tta_agreement_fraction

# Number of top candidates returned in the 6.1 response (Section 6.1: "up to 3").
TOP_K = 3


@dataclass
class RecognizerBundle:
    """A ready-to-serve ensemble: models + per-model calibration + fusion weights.

    Serving constructs this once at startup (from trained ``.pt`` checkpoints,
    fitted temperatures and tuned weights, all produced by the training/tools
    scripts) and passes it to every :func:`recognize_clip` call.

    Attributes:
        models: the ensemble members (each a torch ``nn.Module`` in eval mode).
            All must accept float32 ``[B, T, F]`` and return logits
            ``[B, num_classes]``.
        labels: ordered class labels; ``labels[i]`` is class index ``i``.
        temperatures: per-model temperature (same length/order as ``models``).
            Defaults to 1.0 (no calibration) per model if omitted.
        weights: per-model fusion weights (same length/order). Defaults to equal.
        device: torch device for forward passes.
    """

    models: list[Any]
    labels: list[str]
    temperatures: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("RecognizerBundle needs at least one model")
        if not self.temperatures:
            self.temperatures = [1.0] * len(self.models)
        if not self.weights:
            self.weights = [1.0] * len(self.models)
        if not (len(self.models) == len(self.temperatures) == len(self.weights)):
            raise ValueError("models, temperatures and weights must be the same length")


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
    """Decode base64 frames and extract landmarks, or return a no-hands sequence."""
    if detector is not None:
        decoded = [decode_base64_jpeg(f) for f in frames]
        return extract_landmarks(decoded, detector)
    for f in frames:
        decode_base64_jpeg(f)  # still validate the payload
    return _no_hands_frames(len(frames))


def _vocab_labels(cfg_root: Path) -> list[str]:
    """Load the ordered class labels from ``config/vocabulary.json``."""
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
    num_models: int,
    num_views: int,
) -> dict[str, Any]:
    """Assemble a Section 6.1 reject dict (null label, empty candidates)."""
    return {
        "clip_id": clip_id,
        "decision": "reject",
        "label": None,
        "candidates": [],
        "confidence": 0.0,
        "margin": 0.0,
        "agreement": {"models": f"0/{num_models}", "tta": f"0/{num_views}"},
        "reject_reason": reject_reason,
        "latency_ms": latency_ms,
    }


def _model_probs_over_views(
    model: Any,
    views: list[np.ndarray],
    temperature: float,
    device: str | torch.device,
) -> tuple[np.ndarray, list[int]]:
    """Run one model over every TTA view; return (mean calibrated probs, per-view top1).

    Args:
        model: a torch module mapping ``[B, T, F]`` -> logits ``[B, C]``.
        views: list of float32 ``[T, F]`` feature arrays (TTA views).
        temperature: calibration temperature for this model.
        device: torch device.

    Returns:
        ``(mean_probs[C], per_view_top1)`` where ``mean_probs`` is this model's
        temperature-calibrated probabilities averaged across views.
    """
    batch = torch.from_numpy(np.stack(views, axis=0)).to(device)  # [V, T, F]
    with torch.no_grad():
        logits = model(batch).cpu().numpy()  # [V, C]
    calibrated = apply_temperature(logits, temperature)  # [V, C]
    per_view_top1 = [int(np.argmax(calibrated[v])) for v in range(calibrated.shape[0])]
    return calibrated.mean(axis=0), per_view_top1


def recognize_clip(
    frames: list[str],
    fps: float,
    signer_id: str,
    scenario_id: str | None = None,
    *,
    cfg: Config | None = None,
    detector: HolisticDetector | None = None,
    bundle: RecognizerBundle | None = None,
    context_prior: np.ndarray | None = None,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recognise a whole clip and return the Section 6.1 response dict.

    Args:
        frames: base64 JPEG frames of the clip (one string per frame).
        fps: capture frame rate (used for the quality duration check).
        signer_id: id of the signer (guest id in Phase 1; recorded by callers).
        scenario_id: optional active scenario id (context prior, L11).
        cfg: optional preloaded config (defaults to ``load_config()``).
        detector: optional MediaPipe holistic detector. If ``None`` the clip is
            treated as no-hands (skeleton mode) and rejected ``hands_not_visible``.
        bundle: the serving ensemble (models + calibration + weights). If
            ``None`` a single random-init model is used (Phase-1 fallback).
        context_prior: optional float ``[C]`` prior over the vocabulary (L11),
            applied to the fused probabilities when supplied and
            ``cfg.context_prior.enabled``.
        config_root: optional repo root holding ``config/``.

    Returns:
        A dict shaped EXACTLY like the Section 6.1 ``/api/recognize`` response.

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

    labels = bundle.labels if bundle is not None else _vocab_labels(root)
    num_models = len(bundle.models) if bundle is not None else 1

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
        return _reject_response(
            clip_id, reject_reason or "low_quality", latency_ms, num_models, cfg.tta.views
        )

    # --- Stage 3: TTA views + per-model forward (+ calibration) ------------ #
    t0 = time.perf_counter()
    views = build_tta_feature_views(landmark_seq, cfg)  # list of [T, F]
    num_views = len(views)

    if bundle is None:
        # Phase-1 fallback: one random-init model, temperature 1.0, equal weight.
        model = build_model(num_classes=len(labels))
        mean_probs, per_view_top1 = _model_probs_over_views(model, views, 1.0, "cpu")
        per_model_probs = [mean_probs]
        per_model_view_top1 = [per_view_top1]
        weights = [1.0]
    else:
        per_model_probs = []
        per_model_view_top1 = []
        for model, temp in zip(bundle.models, bundle.temperatures):
            mp, vt = _model_probs_over_views(model, views, temp, bundle.device)
            per_model_probs.append(mp)
            per_model_view_top1.append(vt)
        weights = bundle.weights
    latency_ms["models"] = int((time.perf_counter() - t0) * 1000)

    # --- Stage 4: fusion (+ optional context prior) + decision ------------- #
    t0 = time.perf_counter()
    # TTA agreement: average across models of each model's view agreement.
    tta_frac = float(np.mean([tta_agreement_fraction(v) for v in per_model_view_top1]))
    result = fuse_probabilities(per_model_probs, weights=weights, tta_agreement=tta_frac)

    if context_prior is not None and cfg.context_prior.enabled:
        fused = apply_context_prior(result.probs, context_prior)
        result = FusionResult(
            probs=fused,
            top1=int(np.argmax(fused)),
            top1_p=float(np.max(fused)),
            margin=float(np.sort(fused)[::-1][0] - np.sort(fused)[::-1][1]) if len(fused) > 1 else float(np.max(fused)),
            model_agreement=result.model_agreement,
            num_models=result.num_models,
            tta_agreement=result.tta_agreement,
        )

    probs = result.probs
    order = np.argsort(probs)[::-1]
    top = order[:TOP_K]
    candidates = [{"label": labels[int(i)], "p": float(probs[int(i)])} for i in top]

    outcome = decide(result, cfg.decision)
    label = labels[result.top1] if outcome.decision == "accept" else None
    latency_ms["fusion"] = int((time.perf_counter() - t0) * 1000)
    latency_ms["total"] = int((time.perf_counter() - t_start) * 1000)

    return {
        "clip_id": clip_id,
        "decision": outcome.decision,
        "label": label,
        "candidates": candidates,
        "confidence": result.top1_p,
        "margin": result.margin,
        "agreement": {
            "models": f"{result.model_agreement}/{result.num_models}",
            "tta": f"{int(round(result.tta_agreement * num_views))}/{num_views}",
        },
        "reject_reason": outcome.reject_reason,
        "latency_ms": latency_ms,
    }
