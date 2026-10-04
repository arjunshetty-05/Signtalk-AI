"""
api/pose/router.py — SignTalk AI

POST /pose/classify-clip — batch classification of an ENTIRE pre-recorded
clip, not a live sliding window. Mirrors dataset_tools/preprocess_include.py's
video_to_sequence(): every frame is keypoint-extracted, normalized, and
smoothed, then the whole sequence is uniformly resampled to SEQUENCE_LENGTH
frames via np.linspace — matching exactly how the training data was built.

Why this exists: /ws/gesture's live pipeline uses a continuously-sliding
30-frame trailing window (see GestureConnectionState.buffer). For a short
clip, that is NOT equivalent to how the model was trained/evaluated — while
the window is still filling or sliding past the clip's end, it holds
partial/transitional motion that can transiently resemble other signs
before ever representing the complete, whole-clip gesture. Confirmed live:
replaying a reference clip through /ws/gesture produced a flapping sequence
of wrong labels (e.g. "tall", "warm", "old") for a clip that classifies
correctly (~99% confidence) when given the full-clip resampling below — the
same preprocessing used to build the training set. The "Play demo clip"
frontend feature calls this endpoint instead of streaming through
/ws/gesture, so it gets the same accuracy verified offline.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.auth.dependencies import CurrentUser, get_current_user
from api.core.config import settings
from api.core.limiter import limiter
from api.core.offline_mode import get_offline_mode
from api.pose.service import SEQUENCE_LENGTH, SMOOTHING_WINDOW
from classify import classify_sequence
from keypoint_utils import (
    TemporalSmoother,
    decode_base64_jpeg,
    extract_keypoints,
    load_movenet,
    normalize_keypoints,
)
import nlp_correction

logger = logging.getLogger("signtalk.pose")
router = APIRouter(prefix="/pose", tags=["pose"])
_executor = ThreadPoolExecutor(max_workers=2)


class ClassifyClipRequest(BaseModel):
    frames: list[str] = Field(
        ..., min_length=2, description="Base64 JPEG frames (data URL or raw), in chronological order"
    )


class ClassifyClipResponse(BaseModel):
    label: str
    confidence: float
    sentence: str
    source: str
    low_confidence: bool


def _classify_clip_sync(frames_b64: list[str]) -> dict:
    movenet_sig = load_movenet()
    smoother = TemporalSmoother(window=SMOOTHING_WINDOW)
    smoothed = []
    for b64 in frames_b64:
        frame_bgr = decode_base64_jpeg(b64)
        raw_kp = extract_keypoints(frame_bgr, movenet_sig)
        normalized = normalize_keypoints(raw_kp)
        smoothed.append(smoother.smooth(normalized))

    indices = np.linspace(0, len(smoothed) - 1, SEQUENCE_LENGTH).round().astype(int)
    sequence = np.stack([smoothed[i] for i in indices], axis=0).astype(np.float32)
    return classify_sequence(sequence)


@router.post("/classify-clip", response_model=ClassifyClipResponse)
@limiter.limit(settings.RATE_LIMIT_PREDICTION)
async def classify_clip(
    request: Request,
    body: ClassifyClipRequest,
    user: CurrentUser = Depends(get_current_user),
    offline_mode: bool = Depends(get_offline_mode),
):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _classify_clip_sync, body.frames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "diag classify-clip: label=%s confidence=%.3f n_frames=%d",
        result["label"], result["confidence"], len(body.frames),
    )

    nlp_result = await loop.run_in_executor(
        _executor,
        nlp_correction.correct_sentence,
        [result["label"]],
        "neutral",
        [],
        offline_mode,
    )

    try:
        from api.firebase.firebase_client import log_emotion, save_conversation
        from api.socket_manager import broadcast_new_conversation

        entry = {
            "user_id": user.uid,
            "session_id": user.uid,
            "gesture_tokens": [result["label"]],
            "corrected_sentence": nlp_result["sentence"],
            "language": "en",
            "emotion": "neutral",
        }
        save_conversation(entry)
        log_emotion(user.uid, "neutral", 1.0)
        await broadcast_new_conversation(user.uid, entry)
    except Exception:
        logger.warning("Firestore/Socket.IO side-effects skipped for /pose/classify-clip", exc_info=False)

    return ClassifyClipResponse(
        label=result["label"],
        confidence=result["confidence"],
        sentence=nlp_result["sentence"],
        source=nlp_result["source"],
        low_confidence=nlp_result["low_confidence"],
    )
