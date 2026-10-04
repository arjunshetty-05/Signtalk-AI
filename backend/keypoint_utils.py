"""
keypoint_utils.py — SignTalk AI / Person A (ML Core)

Pose extraction, normalization, temporal smoothing, and sequence buffering
built around MoveNet Thunder (17 COCO-style body keypoints). Every function
here is reused as-is by main.py (live inference), offline_inference.py
(on-device TFLite), and record_samples.py (dataset collection) — do not
duplicate this logic elsewhere.

Keypoint index reference (COCO-17, MoveNet order):
0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
"""

from __future__ import annotations

import base64
import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

logger = logging.getLogger("signtalk.keypoint_utils")

NUM_KEYPOINTS = 17
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
MOVENET_INPUT_SIZE = 256  # thunder variant
CONFIDENCE_THRESHOLD = 0.3
MOVENET_TFHUB_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"

# ---------------------------------------------------------------------------
# Model loading — module-level singleton, loaded once at process startup.
# ---------------------------------------------------------------------------
_movenet_module = None
_movenet_signature = None


def load_movenet():
    """
    Loads MoveNet Thunder from TF Hub exactly once per process and caches it
    at module scope. Safe to call repeatedly — subsequent calls are no-ops
    that return the cached signature. Call this once at FastAPI startup,
    never per-request/per-frame.
    """
    global _movenet_module, _movenet_signature
    if _movenet_signature is not None:
        return _movenet_signature

    logger.info("Loading MoveNet Thunder from TF Hub (%s)...", MOVENET_TFHUB_URL)
    _movenet_module = hub.load(MOVENET_TFHUB_URL)
    _movenet_signature = _movenet_module.signatures["serving_default"]
    logger.info("MoveNet Thunder loaded.")
    return _movenet_signature


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------
def decode_base64_jpeg(b64_string: str) -> np.ndarray:
    """
    Decodes a base64-encoded JPEG (with or without a data URI prefix like
    'data:image/jpeg;base64,') into a BGR np.ndarray (OpenCV convention).
    Raises ValueError on malformed input.
    """
    if "," in b64_string and b64_string.strip().startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_string)
    except Exception as exc:
        raise ValueError(f"Invalid base64 payload: {exc}") from exc

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode JPEG frame")
    return image


# ---------------------------------------------------------------------------
# Keypoint extraction
# ---------------------------------------------------------------------------
def extract_keypoints(image_bgr: np.ndarray, movenet_signature=None) -> np.ndarray:
    """
    Runs MoveNet Thunder on a single BGR frame.

    Args:
        image_bgr: raw camera frame, shape (H, W, 3), BGR (OpenCV convention).
        movenet_signature: the loaded serving signature (from load_movenet()).
            If None, load_movenet() is called (and cached) automatically.

    Returns:
        np.ndarray of shape (17, 3): (y, x, confidence) per keypoint, all
        normalized to [0, 1] relative to the (padded/resized) input image.
    """
    if movenet_signature is None:
        movenet_signature = load_movenet()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(
        tf.expand_dims(image_rgb, axis=0), MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE
    )
    img = tf.cast(img, dtype=tf.int32)

    outputs = movenet_signature(img)
    keypoints = outputs["output_0"].numpy()  # shape (1, 1, 17, 3)
    return keypoints[0, 0, :, :]  # (17, 3) -> (y, x, score)


# ---------------------------------------------------------------------------
# Normalization — camera-distance invariant
# ---------------------------------------------------------------------------
def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalizes raw (17, 3) MoveNet keypoints [(y, x, score), ...] into a
    scale/translation-invariant (17, 2) array of (x, y) coordinates.

    Origin: the shoulder midpoint (avg of left/right shoulder).
    Scale: the bounding-box diagonal spanning all keypoints with
        confidence >= CONFIDENCE_THRESHOLD. If fewer than 4 keypoints are
        confident, falls back to shoulder-to-hip distance as the scale.
        If even that is unavailable, scale defaults to 1.0 (no-op) to avoid
        divide-by-zero.

    Returns:
        np.ndarray shape (17, 2), dtype float32.
    """
    yx = keypoints[:, :2]  # (17, 2) as (y, x)
    scores = keypoints[:, 2]
    xy = yx[:, ::-1].astype(np.float32)  # convert to (x, y)

    confident_mask = scores >= CONFIDENCE_THRESHOLD

    left_sh, right_sh = xy[LEFT_SHOULDER], xy[RIGHT_SHOULDER]
    shoulders_confident = confident_mask[LEFT_SHOULDER] and confident_mask[RIGHT_SHOULDER]
    if shoulders_confident:
        origin = (left_sh + right_sh) / 2.0
    else:
        # fall back to centroid of confident points, or image center
        if confident_mask.any():
            origin = xy[confident_mask].mean(axis=0)
        else:
            origin = np.array([0.5, 0.5], dtype=np.float32)

    num_confident = int(confident_mask.sum())
    if num_confident >= 4:
        pts = xy[confident_mask]
        bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
        scale = float(np.linalg.norm(bbox_max - bbox_min))
    else:
        left_hip, right_hip = xy[LEFT_HIP], xy[RIGHT_HIP]
        hips_confident = confident_mask[LEFT_HIP] and confident_mask[RIGHT_HIP]
        if shoulders_confident and hips_confident:
            hip_mid = (left_hip + right_hip) / 2.0
            scale = float(np.linalg.norm(hip_mid - origin))
        else:
            scale = 0.0

    if scale < 1e-6:
        scale = 1.0

    normalized = (xy - origin) / scale
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------
class TemporalSmoother:
    """
    Moving-average smoother over the last `window` normalized keypoint
    frames, each of shape (17, 2). Reduces per-frame jitter before frames
    enter the sequence buffer.
    """

    def __init__(self, window: int = 4):
        self.window = window
        self._frames: deque[np.ndarray] = deque(maxlen=window)

    def smooth(self, normalized_frame: np.ndarray) -> np.ndarray:
        self._frames.append(normalized_frame)
        stacked = np.stack(list(self._frames), axis=0)  # (n, 17, 2)
        return stacked.mean(axis=0).astype(np.float32)

    def reset(self) -> None:
        self._frames.clear()


# ---------------------------------------------------------------------------
# Sequence buffering
# ---------------------------------------------------------------------------
class SequenceBuffer:
    """
    Maintains a sliding window of the last `maxlen` smoothed, normalized
    keypoint frames (each (17, 2)) for one connection/session. Used to
    build the (30, 17, 2) input consumed by classify_sequence().
    """

    def __init__(self, maxlen: int = 30):
        self.maxlen = maxlen
        self._buffer: deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, frame: np.ndarray) -> None:
        self._buffer.append(frame)

    def is_full(self) -> bool:
        return len(self._buffer) == self.maxlen

    def get_sequence(self) -> Optional[np.ndarray]:
        """Returns (maxlen, 17, 2) array once full, else None."""
        if not self.is_full():
            return None
        return np.stack(list(self._buffer), axis=0).astype(np.float32)

    def reset(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
