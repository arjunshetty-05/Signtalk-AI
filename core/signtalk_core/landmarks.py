"""Per-frame landmark extraction (PROJECT_CONTEXT Sections 5.3 / L5).

This module turns raw camera input (base64 JPEG frames or a video file) into a
sequence of :class:`FrameLandmarks` — the compact, 2D (x, y) landmark set that
the feature builder (:mod:`signtalk_core.features`) consumes. It is used by both
training (offline, from stored clips) and serving (online, from uploaded
frames), so this is the single source of landmark extraction (Rule 12.3).

Landmark backend: MediaPipe Tasks ``HolisticLandmarker`` (the legacy
``mp.solutions.*`` API is unsupported — PROJECT_CONTEXT Section 4.4). The
combined Holistic task IS available in mediapipe 0.10.14, so separate
Pose + Hand landmarkers are not needed. The extractor requires a downloaded
``.task`` model bundle; the path is passed in by the caller (serving/training
code), which keeps this module free of network/model-download side effects and
lets the pure-NumPy feature/quality code be unit-tested without any model file.

Coordinate convention: all landmark arrays are float32, normalised image
coordinates in ``[0, 1]`` with x = horizontal (left→right in the raw,
un-mirrored frame) and y = vertical (top→bottom). We never mirror the uploaded
frames (PROJECT_CONTEXT Section 5.3); handedness is resolved later by body
midline in :mod:`signtalk_core.features`, NOT by MediaPipe's Left/Right labels.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

# MediaPipe pose landmark indices kept as the "pose subset" (Section 5.3):
# 0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips.
POSE_SUBSET_INDICES: tuple[int, ...] = (0, 11, 12, 13, 14, 15, 16, 23, 24)
NUM_POSE_SUBSET = len(POSE_SUBSET_INDICES)  # 9
NUM_HAND_LANDMARKS = 21


@dataclass(frozen=True)
class FrameLandmarks:
    """2D landmarks for a single frame.

    Attributes:
        pose_subset: float32 ``[9, 2]`` — (x, y) of the pose subset landmarks
            in normalised image coordinates. Always present (zeros if no pose
            was detected).
        left_hand: float32 ``[21, 2]`` of the hand assigned to the image-left,
            or ``None`` if that hand was not detected this frame.
        right_hand: float32 ``[21, 2]`` of the image-right hand, or ``None``.
        left_hand_present: True iff ``left_hand`` is not None.
        right_hand_present: True iff ``right_hand`` is not None.

    NOTE: ``left_hand`` / ``right_hand`` here are keyed by the raw MediaPipe
    channels only as a transport convenience. The accuracy-critical handedness
    decision (by body midline) happens in :mod:`signtalk_core.features`; callers
    must not treat these as semantically "the signer's left/right hand".
    """

    pose_subset: np.ndarray
    left_hand: np.ndarray | None
    right_hand: np.ndarray | None
    left_hand_present: bool
    right_hand_present: bool


class HolisticDetector(Protocol):
    """Minimal structural interface for a per-frame holistic detector.

    MediaPipe's ``HolisticLandmarker`` satisfies this via its ``detect`` method.
    Declaring it as a Protocol lets callers (and tests) inject a fake detector
    without importing MediaPipe or downloading a model bundle.
    """

    def detect(self, image: "object") -> "object":  # pragma: no cover - protocol
        ...


def decode_base64_jpeg(b64_string: str) -> np.ndarray:
    """Decode a base64 JPEG (optionally ``data:`` prefixed) to a BGR frame.

    Args:
        b64_string: base64 payload, with or without a ``data:image/...;base64,``
            prefix.

    Returns:
        BGR ``uint8`` ndarray of shape ``[H, W, 3]`` (OpenCV convention).

    Raises:
        ValueError: on malformed base64 or undecodable JPEG bytes.
    """
    if "," in b64_string and b64_string.strip().startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_string)
    except Exception as exc:  # noqa: BLE001 - normalise to ValueError
        raise ValueError(f"Invalid base64 payload: {exc}") from exc

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode JPEG frame")
    return image


def read_video_frames(video_path: str | Path) -> list[np.ndarray]:
    """Decode every frame of a WebM/MP4 clip to BGR arrays via OpenCV.

    Args:
        video_path: path to the video file.

    Returns:
        List of BGR ``uint8`` frames in capture order (may be empty).

    Raises:
        ValueError: if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _landmarks_to_xy(landmark_list, count: int) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmark list to a float32 ``[count, 2]``.

    Only (x, y) are kept (2D only — depth hurt accuracy, Section 4.3).
    """
    out = np.zeros((count, 2), dtype=np.float32)
    for i, lm in enumerate(landmark_list):
        if i >= count:
            break
        out[i, 0] = float(lm.x)
        out[i, 1] = float(lm.y)
    return out


def _mp_result_to_frame(result) -> FrameLandmarks:
    """Translate a HolisticLandmarkerResult into a :class:`FrameLandmarks`."""
    pose = getattr(result, "pose_landmarks", None)
    if pose:
        full_pose = _landmarks_to_xy(pose, 33)
        pose_subset = full_pose[list(POSE_SUBSET_INDICES)].astype(np.float32)
    else:
        pose_subset = np.zeros((NUM_POSE_SUBSET, 2), dtype=np.float32)

    left_raw = getattr(result, "left_hand_landmarks", None)
    right_raw = getattr(result, "right_hand_landmarks", None)
    left_hand = _landmarks_to_xy(left_raw, NUM_HAND_LANDMARKS) if left_raw else None
    right_hand = _landmarks_to_xy(right_raw, NUM_HAND_LANDMARKS) if right_raw else None

    return FrameLandmarks(
        pose_subset=pose_subset,
        left_hand=left_hand,
        right_hand=right_hand,
        left_hand_present=left_hand is not None,
        right_hand_present=right_hand is not None,
    )


def extract_landmarks(
    frames_bgr: list[np.ndarray],
    detector: HolisticDetector,
) -> list[FrameLandmarks]:
    """Run the holistic detector over a list of BGR frames.

    Args:
        frames_bgr: BGR ``uint8`` frames (e.g. from :func:`read_video_frames`
            or :func:`decode_base64_jpeg`).
        detector: a MediaPipe ``HolisticLandmarker`` (or any object with a
            ``detect(mp.Image) -> HolisticLandmarkerResult`` method).

    Returns:
        One :class:`FrameLandmarks` per input frame, in order.
    """
    import mediapipe as mp  # local import: keep MediaPipe optional for pure tests

    out: list[FrameLandmarks] = []
    for frame_bgr in frames_bgr:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        out.append(_mp_result_to_frame(result))
    return out


def create_holistic_detector(model_path: str | Path):
    """Create a MediaPipe Tasks ``HolisticLandmarker`` in IMAGE mode.

    Args:
        model_path: path to the ``holistic_landmarker.task`` model bundle.
            (Downloaded out-of-band by serving/training setup; not fetched here.)

    Returns:
        A configured ``HolisticLandmarker`` ready for per-frame ``detect`` calls.

    Raises:
        FileNotFoundError: if the model bundle is missing.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Holistic model bundle not found: {model_path}")

    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def extract_from_base64_frames(
    b64_frames: list[str],
    detector: HolisticDetector,
) -> list[FrameLandmarks]:
    """Decode base64 JPEG frames and extract landmarks (serving entry point)."""
    frames = [decode_base64_jpeg(f) for f in b64_frames]
    return extract_landmarks(frames, detector)


def extract_from_video(
    video_path: str | Path,
    detector: HolisticDetector,
) -> list[FrameLandmarks]:
    """Decode a video file and extract landmarks (training entry point)."""
    frames = read_video_frames(video_path)
    return extract_landmarks(frames, detector)
