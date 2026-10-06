"""Direction B — speech to live captions (Section 6.4): WS /ws/speech.

The hearing person speaks; the browser streams audio chunks over this
WebSocket and the server returns transcript messages
``{"type":"transcript","text":str,"is_final":bool,"lang":str}`` for live
captions + the conversation log (Direction B, Section 2.4 / 5.12).

Transcription uses Whisper (``faster-whisper`` preferred, else
``openai-whisper``), loaded lazily and ONCE, with the model size from
``WHISPER_MODEL`` (small/base/tiny). Whisper is a heavy, OPTIONAL dependency: if
neither package is installed the socket still accepts and replies with a single
``is_final`` message explaining STT is unavailable, so the app degrades
gracefully (the browser Web Speech API is the client-side fallback, Section
5.12).

Protocol (client -> server):
    * binary frames  = raw audio chunks (webm/opus or wav bytes) to buffer
    * text  "flush"  = transcribe the buffer so far, emit a final transcript
    * text  "lang:<code>" = set the spoken language hint (e.g. "lang:en")
    * close          = end the session
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("signtalk.server.speech")

router = APIRouter(tags=["speech"])

# Lazily-initialised singleton transcriber (loaded on first socket).
_TRANSCRIBER = None
_TRANSCRIBER_KIND = None  # "faster" | "openai" | "unavailable"


def _load_transcriber():
    """Load Whisper once; record which backend (if any) is available."""
    global _TRANSCRIBER, _TRANSCRIBER_KIND
    if _TRANSCRIBER_KIND is not None:
        return
    model_size = os.environ.get("WHISPER_MODEL", "small")
    try:
        from faster_whisper import WhisperModel

        _TRANSCRIBER = WhisperModel(model_size, device="auto", compute_type="int8")
        _TRANSCRIBER_KIND = "faster"
        logger.info("STT: faster-whisper (%s)", model_size)
        return
    except Exception:  # noqa: BLE001 - try the other backend
        pass
    try:
        import whisper

        _TRANSCRIBER = whisper.load_model(model_size)
        _TRANSCRIBER_KIND = "openai"
        logger.info("STT: openai-whisper (%s)", model_size)
        return
    except Exception:  # noqa: BLE001
        _TRANSCRIBER_KIND = "unavailable"
        logger.warning("STT: no Whisper backend installed -> transcription disabled")


def _transcribe(audio_bytes: bytes, lang: str) -> str:
    """Transcribe buffered audio bytes to text using whichever backend loaded."""
    if _TRANSCRIBER_KIND in (None, "unavailable") or not audio_bytes:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        if _TRANSCRIBER_KIND == "faster":
            segments, _ = _TRANSCRIBER.transcribe(str(tmp_path), language=lang or None)
            return " ".join(seg.text.strip() for seg in segments).strip()
        # openai-whisper
        result = _TRANSCRIBER.transcribe(str(tmp_path), language=lang or None)
        return str(result.get("text", "")).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("transcription failed: %s", exc)
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)


@router.websocket("/ws/speech")
async def speech(ws: WebSocket) -> None:
    """Live speech-to-text WebSocket (Section 6.4)."""
    await ws.accept()
    _load_transcriber()

    if _TRANSCRIBER_KIND == "unavailable":
        await ws.send_json(
            {
                "type": "transcript",
                "text": "Speech-to-text is not available on this server "
                "(install faster-whisper). Use the browser's own dictation.",
                "is_final": True,
                "lang": "en",
            }
        )
        await ws.close()
        return

    buffer = bytearray()
    lang = "en"
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                buffer.extend(data)
                continue
            text = (msg.get("text") or "").strip()
            if text.startswith("lang:"):
                lang = text.split(":", 1)[1].strip() or "en"
            elif text == "flush":
                transcript = _transcribe(bytes(buffer), lang)
                buffer.clear()
                await ws.send_json(
                    {"type": "transcript", "text": transcript, "is_final": True, "lang": lang}
                )
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("speech socket error: %s", exc)
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
