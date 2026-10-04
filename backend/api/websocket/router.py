"""
api/websocket/router.py — SignTalk AI / Person C, Prompt C1

/ws/gesture and /ws/speech, restructured behind Firebase-JWT auth (passed as
a `?token=` query param, since browser WebSocket clients can't set custom
headers) and wired to Person A's pose pipeline (api/pose/service.py) and
Person B's emotion + speech modules.

Also emits {"type": "corrected_sentence", ...} once enough gesture tokens
accumulate, matching the contract Person D's frontend/mobile prompts were
built against.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api.auth.dependencies import CurrentUser
from api.core.metrics import metrics
from api.pose.service import GestureConnectionState
from keypoint_utils import decode_base64_jpeg
import emotion as emotion_module
import nlp_correction
import speech as speech_module

logger = logging.getLogger("signtalk.websocket")
router = APIRouter()

GESTURE_TOKENS_BEFORE_CORRECTION = 3


async def _authenticate_ws(websocket: WebSocket, token: str | None) -> CurrentUser | None:
    if not token:
        await websocket.close(code=4401, reason="Missing auth token")
        return None
    try:
        from api.auth.dependencies import _ensure_firebase_initialized
        from firebase_admin import auth as firebase_auth

        _ensure_firebase_initialized()
        decoded = firebase_auth.verify_id_token(token)
        return CurrentUser(uid=decoded.get("uid") or decoded.get("user_id"), email=decoded.get("email"), claims=decoded)
    except Exception as exc:
        logger.warning("WebSocket auth failed: %s", exc)
        await websocket.close(code=4401, reason="Invalid auth token")
        return None


@router.websocket("/ws/gesture")
async def ws_gesture(websocket: WebSocket, token: str | None = Query(default=None), frame_skip: int = Query(default=1)):
    """Accepts base64 JPEG frames, runs the gesture + emotion pipeline, and
    emits stabilized {"label", "confidence", "timestamp"} predictions plus
    periodic {"type": "corrected_sentence", ...} events."""
    await websocket.accept()
    user = await _authenticate_ws(websocket, token)
    if user is None:
        return

    metrics.connection_opened()
    state = GestureConnectionState(frame_skip=frame_skip)
    emotion_analyzer = emotion_module.EmotionAnalyzer()
    conversation_memory = nlp_correction.ConversationMemory()
    accumulated_tokens: list[str] = []
    logger.info("User %s connected to /ws/gesture", user.uid)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                b64_frame = payload["frame"]
            except (json.JSONDecodeError, KeyError, TypeError):
                b64_frame = raw

            try:
                frame_bgr = decode_base64_jpeg(b64_frame)
            except ValueError as exc:
                await websocket.send_text(json.dumps({"error": str(exc)}))
                continue

            emotion_analyzer.maybe_process(frame_bgr)
            emission, latency_ms = state.process_frame(frame_bgr)

            if emission is not None:
                metrics.record_prediction(latency_ms)
                await websocket.send_text(json.dumps(emission))

                try:
                    from api.firebase.firebase_client import log_gesture

                    log_gesture(user.uid, emission["label"], emission["confidence"], latency_ms)
                except Exception:
                    pass  # Firestore optional in dev — logging is best-effort

                accumulated_tokens.append(emission["label"])
                if len(accumulated_tokens) >= GESTURE_TOKENS_BEFORE_CORRECTION:
                    dominant_emotion = emotion_analyzer.dominant_emotion()
                    result = nlp_correction.correct_sentence(
                        accumulated_tokens, dominant_emotion, conversation_memory.get()
                    )
                    conversation_memory.add(result["sentence"])
                    accumulated_tokens.clear()
                    await websocket.send_text(json.dumps({
                        "type": "corrected_sentence",
                        "sentence": result["sentence"],
                        "source": result["source"],
                        "low_confidence": result["low_confidence"],
                    }))

    except WebSocketDisconnect:
        logger.info("User %s disconnected from /ws/gesture", user.uid)
    finally:
        metrics.connection_closed()


@router.websocket("/ws/speech")
async def ws_speech(websocket: WebSocket, token: str | None = Query(default=None)):
    """Streams partial + final Whisper Small transcripts as
    {"text": str, "is_final": bool}, using RMS-based silence detection."""
    await websocket.accept()
    user = await _authenticate_ws(websocket, token)
    if user is None:
        return

    metrics.connection_opened()
    transcriber = speech_module.StreamingTranscriber()
    logger.info("User %s connected to /ws/speech", user.uid)

    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()
            chunk = transcriber.push_chunk(pcm_bytes)

            if transcriber.should_emit_final(chunk):
                final_text = speech_module._transcribe(transcriber.buffer)
                if final_text:
                    await websocket.send_json({"text": final_text, "is_final": True})
                transcriber.reset()
            else:
                partial_text = speech_module._transcribe(transcriber.buffer)
                if partial_text:
                    await websocket.send_json({"text": partial_text, "is_final": False})

    except WebSocketDisconnect:
        logger.info("User %s disconnected from /ws/speech", user.uid)
    finally:
        metrics.connection_closed()
