"""Handedness-assignment tests (PROJECT_CONTEXT Section 5.3, LOCKED decision).

Hands are assigned to the left/right channel by which side of the body midline
(shoulder-midpoint x) their wrist falls on. MediaPipe's Left/Right labels are
IGNORED. These tests pin that behaviour for two-hand, single-hand, and
label-irrelevance cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from signtalk_core.features import assign_handedness
from signtalk_core.landmarks import FrameLandmarks

# Pose subset with shoulders at x=0.4 and x=0.6 => midline x = 0.5.
_POSE = np.array(
    [
        [0.50, 0.20],  # nose
        [0.40, 0.35],  # L shoulder
        [0.60, 0.35],  # R shoulder
        [0.35, 0.50],  # L elbow
        [0.65, 0.50],  # R elbow
        [0.33, 0.62],  # L wrist
        [0.67, 0.62],  # R wrist
        [0.44, 0.75],  # L hip
        [0.56, 0.75],  # R hip
    ],
    dtype=np.float32,
)


def _hand_at(wrist_x: float, wrist_y: float = 0.6) -> np.ndarray:
    """A [21, 2] hand whose wrist (landmark 0) is at the given x."""
    hand = np.full((21, 2), wrist_x, dtype=np.float32)
    hand[:, 1] = wrist_y
    hand[0] = (wrist_x, wrist_y)  # wrist landmark
    return hand


@pytest.mark.unit
def test_two_hands_straddle_midline() -> None:
    # One hand clearly image-left (x=0.3), one clearly image-right (x=0.7).
    hand_left = _hand_at(0.30)
    hand_right = _hand_at(0.70)
    frame = FrameLandmarks(
        pose_subset=_POSE,
        left_hand=hand_left,
        right_hand=hand_right,
        left_hand_present=True,
        right_hand_present=True,
    )
    left, right = assign_handedness(frame)
    assert left is not None and right is not None
    assert float(left[0, 0]) < 0.5  # left channel wrist is left of midline
    assert float(right[0, 0]) > 0.5


@pytest.mark.unit
def test_label_is_ignored_when_swapped() -> None:
    # Deliberately put the LEFT-of-midline hand into the MediaPipe 'right'
    # channel and vice-versa. Assignment must still follow the midline, not the
    # label/channel it arrived in.
    hand_left_of_midline = _hand_at(0.30)
    hand_right_of_midline = _hand_at(0.70)
    frame = FrameLandmarks(
        pose_subset=_POSE,
        left_hand=hand_right_of_midline,   # swapped on purpose
        right_hand=hand_left_of_midline,   # swapped on purpose
        left_hand_present=True,
        right_hand_present=True,
    )
    left, right = assign_handedness(frame)
    assert float(left[0, 0]) < 0.5
    assert float(right[0, 0]) > 0.5


@pytest.mark.unit
def test_single_hand_left_of_midline() -> None:
    hand = _hand_at(0.30)
    frame = FrameLandmarks(
        pose_subset=_POSE,
        left_hand=hand,
        right_hand=None,
        left_hand_present=True,
        right_hand_present=False,
    )
    left, right = assign_handedness(frame)
    assert left is not None
    assert right is None
    assert float(left[0, 0]) < 0.5


@pytest.mark.unit
def test_single_hand_right_of_midline() -> None:
    # Hand right of midline, arriving via the MediaPipe 'left' channel: must be
    # assigned to the right channel and the left channel left empty.
    hand = _hand_at(0.70)
    frame = FrameLandmarks(
        pose_subset=_POSE,
        left_hand=hand,
        right_hand=None,
        left_hand_present=True,
        right_hand_present=False,
    )
    left, right = assign_handedness(frame)
    assert left is None
    assert right is not None
    assert float(right[0, 0]) > 0.5


@pytest.mark.unit
def test_no_hands() -> None:
    frame = FrameLandmarks(
        pose_subset=_POSE,
        left_hand=None,
        right_hand=None,
        left_hand_present=False,
        right_hand_present=False,
    )
    left, right = assign_handedness(frame)
    assert left is None and right is None
