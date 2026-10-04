"""
speech.py — SignTalk AI / Person B (Intelligence Layer)

Speech-to-text via Whisper Small (offline, multilingual). Exposes a
/ws/speech WebSocket route streaming partial + final transcripts as
{"text": str, "is_final": bool}, using RMS-based silence detection to
decide when a partial utterance becomes final.

Client protocol: binary WebSocket frames of 16-bit PCM mono audio,
16kHz sample rate, sent in small chunks (e.g. 100-250ms each).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("signtalk.speech")

router = APIRouter()

WHISPER_MODEL_SIZE = "small"
SAMPLE_RATE = 16000
SILENCE_RMS_THRESHOLD = 0.01
SILENCE_CHUNKS_FOR_FINAL = 6  # ~ SILENCE_CHUNKS_FOR_FINAL * chunk duration of quiet -> finalize
PARTIAL_EMIT_EVERY_SECONDS = 1.0
MIN_AUDIO_SECONDS_TO_TRANSCRIBE = 0.3


@lru_cache(maxsize=1)
def _load_whisper_model():
    import whisper

    logger.info("Loading Whisper %s model...", WHISPER_MODEL_SIZE)
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    logger.info("Whisper %s loaded.", WHISPER_MODEL_SIZE)
    return model


def _transcribe(audio_f32: np.ndarray) -> str:
    if audio_f32.size < int(MIN_AUDIO_SECONDS_TO_TRANSCRIBE * SAMPLE_RATE):
        return ""
    try:
        model = _load_whisper_model()
        result = model.transcribe(audio_f32, fp16=False)
        return (result.get("text") or "").strip()
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        return ""


class StreamingTranscriber:
    """Per-connection audio buffer + silence detector driving partial/final
    transcript emission."""

    def __init__(self):
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._silence_streak = 0
        self._last_partial_at = 0.0

    def push_chunk(self, pcm_bytes: bytes) -> np.ndarray:
        chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, chunk])
        return chunk

    def is_silent(self, chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return True
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        return rms < SILENCE_RMS_THRESHOLD

    def should_emit_final(self, chunk: np.ndarray) -> bool:
        if self.is_silent(chunk):
            self._silence_streak += 1
        else:
            self._silence_streak = 0
        return self._silence_streak >= SILENCE_CHUNKS_FOR_FINAL and self._buffer.size > 0

    def reset(self) -> None:
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._silence_streak = 0

    @property
    def buffer(self) -> np.ndarray:
        return self._buffer


@router.websocket("/ws/speech")
async def ws_speech(websocket: WebSocket):
    await websocket.accept()
    transcriber = StreamingTranscriber()
    logger.info("Client connected to /ws/speech")

    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()
            chunk = transcriber.push_chunk(pcm_bytes)

            if transcriber.should_emit_final(chunk):
                final_text = _transcribe(transcriber.buffer)
                if final_text:
                    await websocket.send_json({"text": final_text, "is_final": True})
                transcriber.reset()
            else:
                # Emit a partial transcript from the buffer so far (cheap,
                # approximate — full accuracy comes at finalization).
                partial_text = _transcribe(transcriber.buffer)
                if partial_text:
                    await websocket.send_json({"text": partial_text, "is_final": False})

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/speech")
