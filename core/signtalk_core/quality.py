"""Clip quality gate (PROJECT_CONTEXT Section 5.3 step 4 / L2 / L10).

Before the models ever run, a clip is checked against the thresholds in
``config/signtalk.yaml`` (read via :mod:`signtalk_core.config` — no magic
numbers). A failed gate yields a ``reject`` decision with a human-readable
reason so the UI can tell the signer exactly what to fix (e.g. "keep both hands
in view").

Reject reasons (the only strings returned):
    ``None``                — passed.
    ``"too_short"``         — clip duration < ``quality.min_seconds``.
    ``"too_long"``          — clip duration > ``quality.max_seconds``.
    ``"hands_not_visible"`` — fraction of frames with >= 1 hand detected is
                              below ``quality.min_hands_visible_pct``.
    ``"low_quality"``       — a clip that is essentially static (total wrist
                              displacement below a tiny motion floor), i.e. no
                              sign was actually performed.
"""

from __future__ import annotations

import numpy as np

from signtalk_core.config import Config
from signtalk_core.landmarks import FrameLandmarks

# Minimum total wrist displacement (in normalised image units, summed over the
# clip) below which a clip is treated as static. This is a sanity floor, not an
# accuracy tunable; it only rejects a frozen/no-motion clip.
_MOTION_FLOOR = 1e-3

_WRIST = 0


def _hands_visible_fraction(landmark_seq: list[FrameLandmarks]) -> float:
    """Fraction of frames in which at least one hand was detected."""
    if not landmark_seq:
        return 0.0
    visible = sum(1 for f in landmark_seq if f.left_hand_present or f.right_hand_present)
    return visible / len(landmark_seq)


def _total_wrist_motion(landmark_seq: list[FrameLandmarks]) -> float:
    """Sum of frame-to-frame wrist displacement across both hand channels.

    Uses whichever hands are present in consecutive frames; a hand that is
    absent in either of two adjacent frames contributes no displacement for
    that step.
    """
    total = 0.0
    for prev, cur in zip(landmark_seq, landmark_seq[1:]):
        for attr in ("left_hand", "right_hand"):
            p = getattr(prev, attr)
            c = getattr(cur, attr)
            if p is not None and c is not None:
                total += float(np.linalg.norm(c[_WRIST] - p[_WRIST]))
    return total


def quality_check(
    landmark_seq: list[FrameLandmarks],
    cfg: Config,
    fps: float,
) -> tuple[bool, str | None]:
    """Check a landmark sequence against the configured quality thresholds.

    Args:
        landmark_seq: per-frame landmarks for the whole clip.
        cfg: loaded config (reads ``quality.*``).
        fps: capture frame rate, used to turn frame count into seconds.

    Returns:
        ``(passed, reject_reason)``. ``passed`` is True iff ``reject_reason`` is
        ``None``. Checks are evaluated in a fixed order (duration → hand
        visibility → motion) and the first failure is returned.

    Raises:
        ValueError: if ``fps`` is not positive.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    n = len(landmark_seq)
    duration_s = n / fps

    if duration_s < cfg.quality.min_seconds:
        return False, "too_short"
    if duration_s > cfg.quality.max_seconds:
        return False, "too_long"

    if _hands_visible_fraction(landmark_seq) < cfg.quality.min_hands_visible_pct:
        return False, "hands_not_visible"

    if _total_wrist_motion(landmark_seq) < _MOTION_FLOOR:
        return False, "low_quality"

    return True, None
