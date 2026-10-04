"""
api/translation/router.py — SignTalk AI / Person C, Prompt C1 (wired further in C3)

POST /translate — thin REST wrapper around Person B's
translation.translate_text(). C3 wires the offline-mode dependency in: when
offline, only the phrasebook is checked (network + Firestore cache skipped).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from api.auth.dependencies import CurrentUser, get_current_user
from api.core.config import settings
from api.core.limiter import limiter
from api.core.offline_mode import get_offline_mode
import translation as translation_module

router = APIRouter(tags=["translation"])
_executor = ThreadPoolExecutor(max_workers=2)


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = Field(..., description='"en" | "hi" | "kn"')
    offline: bool = False


class TranslateResponse(BaseModel):
    translated_text: str


@router.post("/translate", response_model=TranslateResponse)
@limiter.limit(settings.RATE_LIMIT_PREDICTION)
async def translate(
    request: Request,
    body: TranslateRequest,
    user: CurrentUser = Depends(get_current_user),
    offline_mode: bool = Depends(get_offline_mode),
):
    """Translates text into the target language, checking the offline
    phrasebook, then in-memory cache, then Firestore cache, then falling
    back to the Google Translate API on a full miss."""
    effective_offline = offline_mode or body.offline  # shared dependency is authoritative; body.offline kept for back-compat
    loop = asyncio.get_event_loop()
    translated = await loop.run_in_executor(
        _executor, translation_module.translate_text, body.text, body.target_lang, effective_offline
    )
    return TranslateResponse(translated_text=translated)
