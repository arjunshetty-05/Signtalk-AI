"""
api/speech/router.py — SignTalk AI / Person C, Prompt C1 (wired further in C3)

REST endpoints for speech: POST /text-to-speech and POST /speech-to-text.
The streaming /ws/speech WebSocket route itself lives in api/websocket/router.py
and reuses speech.py's StreamingTranscriber.

C3 wires the offline-mode dependency into /text-to-speech: when offline, it
forces mode="offline" (Coqui) regardless of what the client requested.
"""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.auth.dependencies import CurrentUser, get_current_user
from api.core.config import settings
from api.core.limiter import limiter
from api.core.offline_mode import get_offline_mode
import speech as speech_module
import tts as tts_module

router = APIRouter(tags=["speech"])
_executor = ThreadPoolExecutor(max_workers=2)


class TextToSpeechRequest(BaseModel):
    text: str
    lang: str = Field("en", description='"en" | "hi" | "kn"')
    mode: str = Field("online", description='"online" | "offline"')


class SpeechToTextRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded 16-bit PCM mono audio, 16kHz")


class SpeechToTextResponse(BaseModel):
    text: str


@router.post("/text-to-speech")
@limiter.limit(settings.RATE_LIMIT_PREDICTION)
async def text_to_speech(
    request: Request,
    body: TextToSpeechRequest,
    user: CurrentUser = Depends(get_current_user),
    offline_mode: bool = Depends(get_offline_mode),
):
    """Synthesizes speech audio for the given text/lang/mode and returns raw
    audio bytes (audio/mpeg for gTTS, audio/wav for Coqui)."""
    mode = "offline" if offline_mode else body.mode  # offline_mode=True forces Coqui, ignoring client's requested mode
    loop = asyncio.get_event_loop()
    audio_bytes = await loop.run_in_executor(
        _executor, tts_module.synthesize_speech, body.text, body.lang, mode
    )
    media_type = "audio/mpeg" if mode == "online" else "audio/wav"
    return Response(content=audio_bytes, media_type=media_type)


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
@limiter.limit(settings.RATE_LIMIT_PREDICTION)
async def speech_to_text(
    request: Request,
    body: SpeechToTextRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Non-streaming speech-to-text over a full audio clip (use /ws/speech
    for live streaming transcription instead)."""
    import numpy as np

    pcm_bytes = base64.b64decode(body.audio_base64)
    audio_f32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(_executor, speech_module._transcribe, audio_f32)
    return SpeechToTextResponse(text=text)
