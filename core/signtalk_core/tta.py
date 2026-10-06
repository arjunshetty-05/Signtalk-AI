"""Test-time augmentation (PROJECT_CONTEXT L9 / Section 5.1 step 6).

Classify several slightly different temporal views of the same clip and combine
them. This removes sensitivity to the exact start/end frames and tiny jitter,
and the number of views that agree on the top-1 feeds the decision engine
(``tta_agreement``, L10).

Views are produced by trimming ``cfg.tta.trim_frames`` frames from each end of
the landmark sequence BEFORE features are built, so each view is a genuinely
different temporal crop (not just a re-resample of identical frames). The number
of views is capped at ``cfg.tta.views``.

All functions are pure NumPy over already-extracted landmarks, so TTA is unit
testable without MediaPipe or torch.
"""

from __future__ import annotations

import numpy as np

from signtalk_core.config import Config
from signtalk_core.features import build_features
from signtalk_core.landmarks import FrameLandmarks


def build_tta_feature_views(
    landmark_seq: list[FrameLandmarks],
    cfg: Config,
) -> list[np.ndarray]:
    """Build several temporally-trimmed feature tensors from one clip.

    Args:
        landmark_seq: per-frame landmarks for the whole clip.
        cfg: config supplying ``tta.trim_frames`` and ``tta.views``.

    Returns:
        A list of float32 ``[T, F]`` feature arrays, one per view. The first
        view is always the untrimmed clip. At most ``cfg.tta.views`` views are
        returned; views that would be too short to trim are skipped.
    """
    n = len(landmark_seq)
    trims = sorted(set(int(t) for t in cfg.tta.trim_frames))
    views: list[np.ndarray] = []
    for trim in trims:
        if trim < 0:
            continue
        # Need at least a few frames left after trimming both ends.
        if n - 2 * trim < 2:
            continue
        sub = landmark_seq[trim : n - trim] if trim > 0 else landmark_seq
        views.append(build_features(sub, cfg))
        if len(views) >= cfg.tta.views:
            break
    if not views:  # fallback: at least the untrimmed clip
        views.append(build_features(landmark_seq, cfg))
    return views


def tta_agreement_fraction(per_view_top1: list[int]) -> float:
    """Fraction of TTA views that agree with the majority top-1 label.

    Args:
        per_view_top1: the predicted top-1 class index from each view.

    Returns:
        A float in ``[0, 1]``: (count of the most common label) / (num views).
        Returns ``0.0`` for an empty input.
    """
    if not per_view_top1:
        return 0.0
    values, counts = np.unique(np.asarray(per_view_top1), return_counts=True)
    return float(counts.max()) / float(len(per_view_top1))
