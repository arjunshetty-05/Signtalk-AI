"""Quality-gate tests (PROJECT_CONTEXT Section 5.3 step 4 / L2).

Covers each reject reason — too_short, too_long, hands_not_visible, low_quality
(static clip) — and a passing clip. All thresholds come from
config/signtalk.yaml (min_seconds=0.4, max_seconds=5.0,
min_hands_visible_pct=0.7); the tests build frame counts/fps around those.
"""

from __future__ import annotations

import numpy as np
import pytest

from signtalk_core.config import load_config
from signtalk_core.landmarks import FrameLandmarks
from signtalk_core.quality import quality_check

CFG = load_config()
FPS = 30.0

_POSE = np.array(
    [
        [0.50, 0.20], [0.40, 0.35], [0.60, 0.35], [0.35, 0.50], [0.65, 0.50],
        [0.33, 0.62], [0.67, 0.62], [0.44, 0.75], [0.56, 0.75],
    ],
    dtype=np.float32,
)


def _hand(wrist_x: float, wrist_y: float) -> np.ndarray:
    hand = np.full((21, 2), wrist_x, dtype=np.float32)
    hand[:, 1] = wrist_y
    hand[0] = (wrist_x, wrist_y)
    return hand


def _frame(t: int, hands: bool, moving: bool) -> FrameLandmarks:
    """One frame; optionally with both hands and optional per-frame motion."""
    if not hands:
        return FrameLandmarks(_POSE, None, None, False, False)
    drift = 0.01 * t if moving else 0.0
    left = _hand(0.30, 0.60 + drift)
    right = _hand(0.70, 0.60 + drift)
    return FrameLandmarks(_POSE, left, right, True, True)


def _clip(n_frames: int, hands: bool = True, moving: bool = True) -> list[FrameLandmarks]:
    return [_frame(t, hands, moving) for t in range(n_frames)]


@pytest.mark.unit
def test_passing_clip() -> None:
    # 45 frames @30fps = 1.5s, hands visible every frame, with motion.
    passed, reason = quality_check(_clip(45), CFG, FPS)
    assert passed is True
    assert reason is None


@pytest.mark.unit
def test_too_short() -> None:
    # 9 frames @30fps = 0.3s < min_seconds (0.4).
    passed, reason = quality_check(_clip(9), CFG, FPS)
    assert passed is False
    assert reason == "too_short"


@pytest.mark.unit
def test_too_long() -> None:
    # 160 frames @30fps ≈ 5.33s > max_seconds (5.0).
    passed, reason = quality_check(_clip(160), CFG, FPS)
    assert passed is False
    assert reason == "too_long"


@pytest.mark.unit
def test_hands_not_visible() -> None:
    # 45 frames, only the first 10 have hands => 10/45 ≈ 0.22 < 0.7.
    seq = _clip(10) + _clip(35, hands=False)
    passed, reason = quality_check(seq, CFG, FPS)
    assert passed is False
    assert reason == "hands_not_visible"


@pytest.mark.unit
def test_low_quality_static_clip() -> None:
    # Hands visible every frame but completely static => no motion => low_quality.
    passed, reason = quality_check(_clip(45, moving=False), CFG, FPS)
    assert passed is False
    assert reason == "low_quality"


@pytest.mark.unit
def test_fps_must_be_positive() -> None:
    with pytest.raises(ValueError):
        quality_check(_clip(45), CFG, 0.0)
