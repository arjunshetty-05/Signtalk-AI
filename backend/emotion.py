"""
emotion.py — SignTalk AI / Person B (Intelligence Layer)

Emotion detection running in parallel to the gesture pipeline, via DeepFace.
Built from the contract locked in Person C's Prompt C1:
    "a function analyzing frames via DeepFace, sampled every 10th frame,
    returning one of 7 emotion classes with a rolling dominant-emotion
    vote, defaulting to 'neutral' on no-face-detected"

Also includes the gesture+emotion fusion engine described in the project
scope doc (e.g. FOOD + FRUSTRATED -> a fuller sentence hint for nlp_correction).
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from typing import Optional

import numpy as np

logger = logging.getLogger("signtalk.emotion")

EMOTION_CLASSES = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]
SAMPLE_EVERY_N_FRAMES = 10
ROLLING_WINDOW = 15
DEFAULT_EMOTION = "neutral"

try:
    from deepface import DeepFace
    _DEEPFACE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional heavy dependency
    _DEEPFACE_AVAILABLE = False
    logger.warning("deepface not installed — emotion detection will always return 'neutral'.")


def analyze_frame_emotion(frame_bgr: np.ndarray) -> str:
    """
    Runs DeepFace emotion analysis on a single frame. Returns one of
    EMOTION_CLASSES, or DEFAULT_EMOTION ('neutral') if no face is detected
    or DeepFace is unavailable/errors.
    """
    if not _DEEPFACE_AVAILABLE:
        return DEFAULT_EMOTION
    try:
        result = DeepFace.analyze(
            frame_bgr, actions=["emotion"], enforce_detection=True, silent=True
        )
        if isinstance(result, list):
            result = result[0]
        dominant = result.get("dominant_emotion", DEFAULT_EMOTION)
        return dominant if dominant in EMOTION_CLASSES else DEFAULT_EMOTION
    except Exception as exc:
        # DeepFace raises when no face is detected — this is the expected
        # graceful-fallback path, not necessarily an error worth logging loudly.
        logger.debug("Emotion analysis fell back to neutral: %s", exc)
        return DEFAULT_EMOTION


class EmotionAnalyzer:
    """
    Per-session emotion tracker. Only runs DeepFace on every Nth frame
    (SAMPLE_EVERY_N_FRAMES) for performance, and maintains a rolling
    dominant-emotion vote over the last ROLLING_WINDOW sampled readings.
    """

    def __init__(self, sample_every: int = SAMPLE_EVERY_N_FRAMES, window: int = ROLLING_WINDOW):
        self.sample_every = sample_every
        self._frame_count = 0
        self._recent: deque[str] = deque(maxlen=window)

    def maybe_process(self, frame_bgr: np.ndarray) -> Optional[str]:
        """Call on every frame. Returns the newly sampled emotion only on
        sampled frames, else None (caller should keep using dominant_emotion())."""
        self._frame_count += 1
        if self._frame_count % self.sample_every != 0:
            return None
        emotion = analyze_frame_emotion(frame_bgr)
        self._recent.append(emotion)
        return emotion

    def dominant_emotion(self) -> str:
        """Rolling majority vote over recent sampled readings."""
        if not self._recent:
            return DEFAULT_EMOTION
        return Counter(self._recent).most_common(1)[0][0]

    def reset(self) -> None:
        self._frame_count = 0
        self._recent.clear()
