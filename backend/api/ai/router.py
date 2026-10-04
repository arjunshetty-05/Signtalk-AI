"""
api/ai/router.py — SignTalk AI / Person C, Prompt C1 (wired further in C2/C3)

POST /predict — raw gesture tokens (+ emotion + conversation history) ->
grammatically corrected sentence, via Person B's nlp_correction.correct_sentence().
Also aliased as POST /ai/fuse for the fusion-engine naming used in Person C's
Prompt C2 handoff notes.

C3 wires the offline-mode dependency into this route: when offline, it skips
Gemini entirely and goes straight to the Flan-T5-Small fallback.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from api.auth.dependencies import CurrentUser, get_current_user
from api.core.config import settings
from api.core.limiter import limiter
from api.core.offline_mode import get_offline_mode
import nlp_correction

logger = logging.getLogger("signtalk.ai")
router = APIRouter(prefix="/ai", tags=["ai"])
_executor = ThreadPoolExecutor(max_workers=2)


class CorrectSentenceRequest(BaseModel):
    gesture_tokens: list[str] = Field(..., description="Raw recognized sign tokens, in order")
    emotion: str = Field("neutral", description="Current dominant emotion from Person B's emotion module")
    conversation_history: list[str] = Field(default_factory=list, description="Recent prior corrected sentences")


class CorrectSentenceResponse(BaseModel):
    sentence: str
    source: str = Field(..., description='"gemini" or "flan-t5"')
    low_confidence: bool


@router.post("/predict", response_model=CorrectSentenceResponse)
@router.post("/fuse", response_model=CorrectSentenceResponse)
@limiter.limit(settings.RATE_LIMIT_PREDICTION)
async def predict_sentence(
    request: Request,
    body: CorrectSentenceRequest,
    user: CurrentUser = Depends(get_current_user),
    offline_mode: bool = Depends(get_offline_mode),
):
    """Converts raw gesture tokens (+ emotion context) into a corrected,
    grammatical sentence. Runs the (possibly network-bound) Gemini/Flan-T5
    call in a thread pool so it never blocks the event loop. Also persists
    the result to Firestore (conversations/emotion_logs/gesture_logs) and
    broadcasts it over Socket.IO to the user's other connected devices."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        nlp_correction.correct_sentence,
        body.gesture_tokens,
        body.emotion,
        body.conversation_history,
        offline_mode,  # offline_mode=True -> skip Gemini, go straight to Flan-T5-Small
    )

    try:
        from api.firebase.firebase_client import log_emotion, save_conversation

        entry = {
            "user_id": user.uid,
            "session_id": user.uid,
            "gesture_tokens": body.gesture_tokens,
            "corrected_sentence": result["sentence"],
            "emotion": body.emotion,
            "language": "en",
        }
        save_conversation(entry)
        log_emotion(user.uid, body.emotion, 1.0)

        from api.socket_manager import broadcast_new_conversation

        await broadcast_new_conversation(user.uid, entry)
    except Exception:
        logger.warning("Firestore/Socket.IO side-effects skipped (not configured yet)", exc_info=False)

    return CorrectSentenceResponse(**result)
