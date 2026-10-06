"""Feature builder — the SINGLE source of truth for features (Section 5.4).

Training and serving BOTH import this module (Rule 12.3), so the features a
model sees at inference are byte-for-byte the ones it was trained on. Any change
here is caught by the golden-vector test (``core/tests/test_features_golden.py``).

Pipeline order (fixed by PROJECT_CONTEXT Section 5.4; do not reorder):

    landmarks → resolve handedness by body midline → interpolate short hand
    gaps (<= 3 frames) → moving-average smooth (window 3) → build per-frame
    features → resample to ``cfg.sequence_length`` frames (linear).

Per-frame feature vector (all 2D, x/y only — depth hurt accuracy, Section 4.3):

    pose subset body-relative      9 x 2 = 18
    left hand body-relative       21 x 2 = 42
    right hand body-relative      21 x 2 = 42
    left hand hand-local          21 x 2 = 42
    right hand hand-local         21 x 2 = 42
    --------------------------------------- position block = 186
    velocity (Δ of position block)        = 186
    hand-presence masks (left, right)     =   2
    --------------------------------------- F = 374

``feature_dim()`` returns 374 and is frozen by the golden fixtures.

Normalisation (Section 5.4):
  * Origin  = shoulder midpoint (pose subset indices 1 and 2 = L/R shoulder).
  * Scale   = shoulder width (distance between the two shoulders).
  * Hand-local = each hand relative to its own wrist (landmark 0), scaled by
    palm size = distance(wrist=0, middle-finger MCP=9).

Handedness (LOCKED decision, Section 5.3 / context.json): assign each detected
hand to the ``left``/``right`` channel by which side of the shoulder-midpoint x
its wrist falls on. MediaPipe's Left/Right labels are IGNORED (they flip under
mirroring).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from signtalk_core.config import Config
from signtalk_core.landmarks import (
    FrameLandmarks,
    NUM_HAND_LANDMARKS,
    NUM_POSE_SUBSET,
)

# Pose-subset local indices (position within POSE_SUBSET_INDICES, not the raw
# MediaPipe indices): the subset is ordered (nose, L-shoulder, R-shoulder, ...).
_LEFT_SHOULDER = 1
_RIGHT_SHOULDER = 2

# Hand landmark indices used for hand-local scaling.
_WRIST = 0
_MIDDLE_MCP = 9

# Max gap length (in frames) that short-gap interpolation will fill (Section 5.4).
MAX_INTERP_GAP = 3

_EPS = 1e-6

# Feature-block sizes (see module docstring).
_POSE_DIMS = NUM_POSE_SUBSET * 2            # 18
_HAND_DIMS = NUM_HAND_LANDMARKS * 2         # 42
_POSITION_DIMS = _POSE_DIMS + 4 * _HAND_DIMS  # 18 + 4*42 = 186
_PRESENCE_DIMS = 2
_FEATURE_DIM = _POSITION_DIMS * 2 + _PRESENCE_DIMS  # 186*2 + 2 = 374


def feature_dim() -> int:
    """Return the exact per-frame feature dimensionality ``F`` (frozen = 374)."""
    return _FEATURE_DIM


# --------------------------------------------------------------------------- #
# Handedness
# --------------------------------------------------------------------------- #
def _midline_x(pose_subset: np.ndarray) -> float:
    """Shoulder-midpoint x = midline between the two shoulders (Section 5.3)."""
    return float((pose_subset[_LEFT_SHOULDER, 0] + pose_subset[_RIGHT_SHOULDER, 0]) / 2.0)


def assign_handedness(frame: FrameLandmarks) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Assign detected hands to (left_channel, right_channel) by body midline.

    "left"/"right" here mean image-left / image-right of the body midline (the
    shoulder-midpoint x), NOT MediaPipe's semantic labels, which are ignored.

    Args:
        frame: one frame's raw landmarks.

    Returns:
        ``(left, right)`` where each is a float32 ``[21, 2]`` hand array or
        ``None`` if no hand occupies that side this frame. If two detected hands
        fall on the same side, the one with the smaller wrist x is placed left
        and the other right (deterministic tie-break).
    """
    midline = _midline_x(frame.pose_subset)
    hands: list[np.ndarray] = []
    if frame.left_hand is not None:
        hands.append(frame.left_hand)
    if frame.right_hand is not None:
        hands.append(frame.right_hand)

    if not hands:
        return None, None

    if len(hands) == 1:
        wrist_x = float(hands[0][_WRIST, 0])
        if wrist_x < midline:
            return hands[0].astype(np.float32), None
        return None, hands[0].astype(np.float32)

    # Two hands detected: order by wrist x so the leftmost goes to the left
    # channel. This also resolves the (rare) same-side case deterministically.
    hands.sort(key=lambda h: float(h[_WRIST, 0]))
    return hands[0].astype(np.float32), hands[1].astype(np.float32)


@dataclass
class _ResolvedFrame:
    """A frame after handedness resolution (channels fixed by body midline)."""

    pose_subset: np.ndarray            # [9, 2]
    left_hand: np.ndarray | None       # [21, 2] or None
    right_hand: np.ndarray | None      # [21, 2] or None
    left_present: bool
    right_present: bool


def _resolve_sequence(landmark_seq: list[FrameLandmarks]) -> list[_ResolvedFrame]:
    """Apply :func:`assign_handedness` to every frame."""
    resolved: list[_ResolvedFrame] = []
    for f in landmark_seq:
        left, right = assign_handedness(f)
        resolved.append(
            _ResolvedFrame(
                pose_subset=f.pose_subset.astype(np.float32),
                left_hand=left,
                right_hand=right,
                left_present=left is not None,
                right_present=right is not None,
            )
        )
    return resolved


# --------------------------------------------------------------------------- #
# Short-gap interpolation
# --------------------------------------------------------------------------- #
def _interpolate_channel(
    hands: list[np.ndarray | None],
) -> tuple[list[np.ndarray | None], list[bool]]:
    """Linearly fill short (<= 3 frame) gaps of a single hand channel.

    A "gap" is a maximal run of ``None`` frames bounded on BOTH sides by a
    present frame. Gaps of length <= :data:`MAX_INTERP_GAP` are filled by linear
    interpolation between the surrounding present frames. Longer gaps, and
    leading/trailing ``None`` runs (no two-sided anchor), stay ``None``.

    Args:
        hands: per-frame ``[21, 2]`` arrays or ``None``.

    Returns:
        ``(filled, present)`` — ``filled`` has interpolated arrays where a short
        gap was closed; ``present`` is the resulting per-frame presence mask
        (True where a hand is now available, interpolated or original).
    """
    n = len(hands)
    filled: list[np.ndarray | None] = [h.copy() if h is not None else None for h in hands]

    i = 0
    while i < n:
        if filled[i] is not None:
            i += 1
            continue
        # Start of a None-run; find its end.
        start = i
        while i < n and filled[i] is None:
            i += 1
        end = i  # first present index after the run (or n)
        gap_len = end - start
        has_left_anchor = start - 1 >= 0 and filled[start - 1] is not None
        has_right_anchor = end < n and filled[end] is not None
        if has_left_anchor and has_right_anchor and gap_len <= MAX_INTERP_GAP:
            left_val = filled[start - 1]
            right_val = filled[end]
            for k in range(gap_len):
                t = (k + 1) / (gap_len + 1)
                filled[start + k] = ((1.0 - t) * left_val + t * right_val).astype(np.float32)

    present = [h is not None for h in filled]
    return filled, present


def _interpolate_short_gaps(resolved: list[_ResolvedFrame]) -> list[_ResolvedFrame]:
    """Interpolate short gaps independently for the left and right channels."""
    left_filled, left_present = _interpolate_channel([r.left_hand for r in resolved])
    right_filled, right_present = _interpolate_channel([r.right_hand for r in resolved])
    out: list[_ResolvedFrame] = []
    for i, r in enumerate(resolved):
        out.append(
            replace(
                r,
                left_hand=left_filled[i],
                right_hand=right_filled[i],
                left_present=left_present[i],
                right_present=right_present[i],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0, edge-truncated (window<=current).

    Args:
        arr: float32 ``[T, ...]``.
        window: averaging window (>= 1). ``window=1`` is a no-op.

    Returns:
        float32 array of the same shape as ``arr``.
    """
    if window <= 1 or arr.shape[0] <= 1:
        return arr.astype(np.float32)
    n = arr.shape[0]
    half = window // 2
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return out


def _smooth_sequence(resolved: list[_ResolvedFrame], window: int) -> list[_ResolvedFrame]:
    """Moving-average smooth pose + each present hand channel (Section 5.4).

    Smoothing is applied to the normalised-input coordinates (before building
    body/hand-local features). Missing-hand frames (``None``) are left missing;
    only runs of consecutive present frames are averaged, so a smoothing window
    never pulls a hand's position toward frames where it was absent.
    """
    n = len(resolved)
    pose_stack = np.stack([r.pose_subset for r in resolved], axis=0)  # [T, 9, 2]
    pose_sm = _moving_average(pose_stack, window)

    left_sm = _smooth_channel([r.left_hand for r in resolved], window)
    right_sm = _smooth_channel([r.right_hand for r in resolved], window)

    out: list[_ResolvedFrame] = []
    for i in range(n):
        out.append(
            replace(
                resolved[i],
                pose_subset=pose_sm[i],
                left_hand=left_sm[i],
                right_hand=right_sm[i],
            )
        )
    return out


def _smooth_channel(hands: list[np.ndarray | None], window: int) -> list[np.ndarray | None]:
    """Moving-average smooth a hand channel within maximal present runs."""
    n = len(hands)
    out: list[np.ndarray | None] = [None] * n
    i = 0
    while i < n:
        if hands[i] is None:
            i += 1
            continue
        start = i
        while i < n and hands[i] is not None:
            i += 1
        run = np.stack(hands[start:i], axis=0)  # [L, 21, 2]
        sm = _moving_average(run, window)
        for k in range(sm.shape[0]):
            out[start + k] = sm[k]
    return out


# --------------------------------------------------------------------------- #
# Per-frame feature construction
# --------------------------------------------------------------------------- #
def _body_relative(points: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    """Translate by ``origin`` and divide by ``scale`` (body-relative coords)."""
    return ((points - origin) / scale).astype(np.float32)


def _hand_local(hand: np.ndarray) -> np.ndarray:
    """Hand-local coordinates: relative to the wrist, scaled by palm size.

    Palm size = distance(wrist landmark 0, middle-finger MCP landmark 9).
    """
    wrist = hand[_WRIST]
    palm = float(np.linalg.norm(hand[_MIDDLE_MCP] - wrist))
    if palm < _EPS:
        palm = 1.0
    return ((hand - wrist) / palm).astype(np.float32)


def _frame_positions(frame: _ResolvedFrame) -> np.ndarray:
    """Build the 186-dim per-frame POSITION block (no velocity/presence yet).

    Layout (flattened): pose body-rel (18), left-hand body-rel (42),
    right-hand body-rel (42), left-hand local (42), right-hand local (42).
    """
    pose = frame.pose_subset
    origin = (pose[_LEFT_SHOULDER] + pose[_RIGHT_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(pose[_LEFT_SHOULDER] - pose[_RIGHT_SHOULDER]))
    if scale < _EPS:
        scale = 1.0

    pose_rel = _body_relative(pose, origin, scale).reshape(-1)  # 18

    def hand_blocks(hand: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if hand is None:
            return (
                np.zeros(_HAND_DIMS, dtype=np.float32),
                np.zeros(_HAND_DIMS, dtype=np.float32),
            )
        body_rel = _body_relative(hand, origin, scale).reshape(-1)  # 42
        local = _hand_local(hand).reshape(-1)  # 42
        return body_rel, local

    left_body, left_local = hand_blocks(frame.left_hand)
    right_body, right_local = hand_blocks(frame.right_hand)

    return np.concatenate(
        [pose_rel, left_body, right_body, left_local, right_local]
    ).astype(np.float32)


def _build_per_frame(resolved: list[_ResolvedFrame]) -> np.ndarray:
    """Build the full per-frame feature matrix ``[T, F]`` (F = 374).

    Velocity = frame-to-frame diff of the position block (first frame = 0).
    Presence masks (2 floats) are appended last.
    """
    positions = np.stack([_frame_positions(f) for f in resolved], axis=0)  # [T, 186]
    velocity = np.zeros_like(positions)
    if positions.shape[0] > 1:
        velocity[1:] = np.diff(positions, axis=0)

    presence = np.array(
        [[1.0 if f.left_present else 0.0, 1.0 if f.right_present else 0.0] for f in resolved],
        dtype=np.float32,
    )  # [T, 2]

    return np.concatenate([positions, velocity, presence], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Temporal resampling
# --------------------------------------------------------------------------- #
def _resample_time(feats: np.ndarray, target_t: int) -> np.ndarray:
    """Linearly resample ``[T, F]`` to ``[target_t, F]`` along the time axis.

    A single input frame is tiled; otherwise each output frame is a linear
    interpolation of the two nearest input frames.
    """
    t_in = feats.shape[0]
    if t_in == target_t:
        return feats.astype(np.float32)
    if t_in == 0:
        return np.zeros((target_t, feats.shape[1]), dtype=np.float32)
    if t_in == 1:
        return np.repeat(feats, target_t, axis=0).astype(np.float32)

    src = np.linspace(0.0, t_in - 1, num=target_t)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t_in - 1)
    frac = (src - lo).astype(np.float32)[:, None]
    out = (1.0 - frac) * feats[lo] + frac * feats[hi]
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_features(landmark_seq: list[FrameLandmarks], cfg: Config) -> np.ndarray:
    """Build the model-ready feature tensor from a landmark sequence.

    Pipeline (Section 5.4): resolve handedness → interpolate short gaps →
    moving-average smooth → per-frame features → resample to
    ``cfg.sequence_length`` frames.

    Args:
        landmark_seq: per-frame landmarks (e.g. from
            :func:`signtalk_core.landmarks.extract_landmarks`). Must be
            non-empty.
        cfg: loaded config (reads ``sequence_length`` and ``smoothing_window``;
            no magic numbers here).

    Returns:
        float32 ``[T, F]`` where ``T == cfg.sequence_length`` and
        ``F == feature_dim()`` (374).

    Raises:
        ValueError: if ``landmark_seq`` is empty.
    """
    if not landmark_seq:
        raise ValueError("landmark_seq is empty; nothing to build features from")

    resolved = _resolve_sequence(landmark_seq)
    resolved = _interpolate_short_gaps(resolved)
    resolved = _smooth_sequence(resolved, cfg.smoothing_window)
    per_frame = _build_per_frame(resolved)  # [T_in, 374]
    out = _resample_time(per_frame, cfg.sequence_length)  # [T, 374]
    return out.astype(np.float32)
