"""Golden-vector regression test for the feature builder (Section 5.4 / L12).

Loads the committed synthetic landmark clip and the expected ``build_features``
output, re-runs the current feature code, and asserts an EXACT (atol=1e-6) match
and the exact ``[T, F]`` shape. This is the guard that stops any accidental
change to the feature pipeline from silently diverging training and serving.

If this test fails after an INTENTIONAL feature change, regenerate the fixtures
by running ``python core/tests/golden/make_golden.py`` and committing the new
``.npy``/``.npz`` files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signtalk_core.config import load_config
from signtalk_core.features import build_features, feature_dim
from signtalk_core.landmarks import FrameLandmarks

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
EXPECTED_FEATURE_DIM = 374
EXPECTED_T = 32  # == config/signtalk.yaml sequence_length


def _load_sequence(npz_path: Path) -> list[FrameLandmarks]:
    """Rebuild the FrameLandmarks list from the saved ``.npz`` (NaN == missing)."""
    data = np.load(npz_path)
    pose, left, right = data["pose"], data["left"], data["right"]
    seq: list[FrameLandmarks] = []
    for i in range(pose.shape[0]):
        lh = None if np.isnan(left[i]).any() else left[i].astype(np.float32)
        rh = None if np.isnan(right[i]).any() else right[i].astype(np.float32)
        seq.append(
            FrameLandmarks(
                pose_subset=pose[i].astype(np.float32),
                left_hand=lh,
                right_hand=rh,
                left_hand_present=lh is not None,
                right_hand_present=rh is not None,
            )
        )
    return seq


@pytest.mark.unit
def test_feature_dim_is_frozen() -> None:
    assert feature_dim() == EXPECTED_FEATURE_DIM


@pytest.mark.unit
def test_golden_features_match() -> None:
    cfg = load_config()
    seq = _load_sequence(GOLDEN_DIR / "input_landmarks.npz")
    expected = np.load(GOLDEN_DIR / "expected_features.npy")

    result = build_features(seq, cfg)

    assert result.dtype == np.float32
    assert result.shape == (EXPECTED_T, feature_dim())
    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-6)
