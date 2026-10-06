"""Generate the golden-vector fixtures for the feature builder (Section 5.4 / L12).

Run ONCE against the first correct :func:`signtalk_core.features.build_features`
implementation, commit the resulting ``.npy`` files, then only re-run this
script when a feature-code change is *intended* (the golden test will otherwise
fail loudly, preventing silent train/serve divergence).

Usage (from the worktree root, with the venv active)::

    python core/tests/golden/make_golden.py

It writes:
    core/tests/golden/input_landmarks.npz   — the deterministic synthetic clip
    core/tests/golden/expected_features.npy — build_features(...) output [T, F]

The synthetic clip is a tiny, fully deterministic 48-frame sequence of smoothly
varying pose + two-hand landmarks (plus a short right-hand gap to exercise
interpolation). No MediaPipe or camera is involved, so the fixture is stable
across machines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signtalk_core.config import load_config
from signtalk_core.features import build_features
from signtalk_core.landmarks import FrameLandmarks

GOLDEN_DIR = Path(__file__).resolve().parent
N_FRAMES = 48
RIGHT_HAND_GAP = range(10, 13)  # 3 missing right-hand frames (<=3 => interpolated)


def _synthetic_sequence() -> list[FrameLandmarks]:
    """Build a deterministic 48-frame landmark sequence.

    Pose and both hands drift smoothly with the frame index so velocities are
    non-trivial. The two hands straddle the body midline (left hand to image
    left, right hand to image right) and a 3-frame right-hand gap is injected to
    exercise short-gap interpolation.
    """
    seq: list[FrameLandmarks] = []
    for t in range(N_FRAMES):
        phase = t / N_FRAMES

        # Pose subset [9, 2]: nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip.
        pose = np.array(
            [
                [0.50, 0.20 + 0.02 * phase],  # nose
                [0.40, 0.35],                 # L shoulder (image-left of midline)
                [0.60, 0.35],                 # R shoulder (image-right)
                [0.35, 0.50 + 0.05 * phase],  # L elbow
                [0.65, 0.50 + 0.05 * phase],  # R elbow
                [0.33, 0.62 + 0.08 * phase],  # L wrist
                [0.67, 0.62 + 0.08 * phase],  # R wrist
                [0.44, 0.75],                 # L hip
                [0.56, 0.75],                 # R hip
            ],
            dtype=np.float32,
        )

        # Left-channel hand (image-left: wrist x < midline 0.5).
        left_base = np.array([0.30, 0.60], dtype=np.float32)
        left_hand = _hand_cloud(left_base, phase, spread=0.04)

        # Right-channel hand (image-right: wrist x > midline).
        right_base = np.array([0.70, 0.60], dtype=np.float32)
        right_hand = _hand_cloud(right_base, phase, spread=0.04)

        if t in RIGHT_HAND_GAP:
            right_hand = None

        seq.append(
            FrameLandmarks(
                pose_subset=pose,
                left_hand=left_hand,
                right_hand=right_hand,
                left_hand_present=left_hand is not None,
                right_hand_present=right_hand is not None,
            )
        )
    return seq


def _hand_cloud(base: np.ndarray, phase: float, spread: float) -> np.ndarray:
    """Deterministic [21, 2] hand: wrist at ``base`` + a fixed landmark fan."""
    idx = np.arange(21, dtype=np.float32)
    # A smooth, deterministic fan of 21 landmarks around the wrist.
    xs = base[0] + spread * np.cos(idx * 0.3 + phase)
    ys = base[1] + spread * np.sin(idx * 0.3 + phase)
    hand = np.stack([xs, ys], axis=1).astype(np.float32)
    hand[0] = base  # landmark 0 = wrist exactly at base (midline check depends on it)
    return hand


def _save_sequence(seq: list[FrameLandmarks], path: Path) -> None:
    """Serialise the landmark sequence to a single ``.npz`` (loadable in tests)."""
    pose = np.stack([f.pose_subset for f in seq], axis=0)  # [T, 9, 2]
    hand_nan = np.full((21, 2), np.nan, dtype=np.float32)
    left = np.stack(
        [f.left_hand if f.left_hand is not None else hand_nan for f in seq], axis=0
    )
    right = np.stack(
        [f.right_hand if f.right_hand is not None else hand_nan for f in seq], axis=0
    )
    np.savez(path, pose=pose, left=left, right=right)


def main() -> None:
    cfg = load_config()
    seq = _synthetic_sequence()
    _save_sequence(seq, GOLDEN_DIR / "input_landmarks.npz")

    features = build_features(seq, cfg)
    np.save(GOLDEN_DIR / "expected_features.npy", features)

    print(f"Wrote golden fixtures to {GOLDEN_DIR}")
    print(f"  input_landmarks.npz   : {N_FRAMES} frames")
    print(f"  expected_features.npy : shape {features.shape}, dtype {features.dtype}")


if __name__ == "__main__":
    main()
