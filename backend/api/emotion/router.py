"""
api/emotion/router.py — SignTalk AI / Person C, Prompt C1

REST wrapper around Person B's emotion.py (DeepFace-based emotion analysis),
mainly for standalone testing — the live gesture pipeline uses
emotion.EmotionAnalyzer directly inside the /ws/gesture handler.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from api.auth.dependencies import CurrentUser, get_current_user
from api.core.config import settings
from api.core.limiter import limiter
from keypoint_utils import decode_base64_jpeg
import emotion as emotion_module

router = APIRouter(prefix="/emotion", tags=["emotion"])
_executor = ThreadPoolExecutor(max_workers=2)


class EmotionAnalyzeRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded JPEG frame")


class EmotionAnalyzeResponse(BaseModel):
    emotion: str = Field(..., description="One of the 7 supported emotion classes")


@router.post("/analyze", response_model=EmotionAnalyzeResponse)
@limiter.limit(settings.RATE_LIMIT_PREDICTION)
async def analyze_emotion(
    request: Request,
    body: EmotionAnalyzeRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Analyzes a single base64 JPEG frame and returns the dominant emotion.
    Runs DeepFace in a thread pool so it never blocks the event loop."""
    frame_bgr = decode_base64_jpeg(body.frame)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, emotion_module.analyze_frame_emotion, frame_bgr)
    return EmotionAnalyzeResponse(emotion=result)
