"""Direction B — speech to live captions (Section 6.4): WS /ws/speech.

The hearing person speaks; the browser streams audio chunks over this
WebSocket and the server returns transcript messages
``{"type":"transcript","text":str,"is_final":bool,"lang":str}`` for live
captions + the conversation log (Direction B, Section 2.4 / 5.12).

Transcription uses Whisper (``faster-whisper`` preferred, else
``openai-whisper``), loaded lazily and ONCE, with the model size from
``WHISPER_MODEL`` (small/base/medium/tiny/large-v3). Audio is transcribed in
the selected language, and output without that language's expected writing
system is rejected instead of being shown as a wrong-language caption. Whisper
is a heavy, OPTIONAL dependency: if
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

import asyncio
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("signtalk.server.speech")

router = APIRouter(tags=["speech"])

# Lazily-initialised singleton transcriber (loaded on first socket).
_TRANSCRIBER = None
_TRANSCRIBER_KIND = None  # "faster" | "openai" | "unavailable"
_TRANSCRIBER_LOCK = threading.Lock()
_SUPPORTED_LANGUAGES = {"en", "hi", "kn"}
_EXPECTED_SCRIPT_RANGES = {
    "en": ((ord("A"), ord("Z")), (ord("a"), ord("z"))),
    "hi": ((0x0900, 0x097F),),
    "kn": ((0x0C80, 0x0CFF),),
}


class TranscriptLanguageMismatch(ValueError):
    """Raised when transcription output is not in the selected writing system."""


def _validate_transcript_language(text: str, lang: str) -> str:
    """Reject non-empty transcripts that have no characters in the chosen script."""
    if not text or lang not in _EXPECTED_SCRIPT_RANGES:
        return text
    ranges = _EXPECTED_SCRIPT_RANGES[lang]
    if not any(start <= ord(char) <= end for char in text for start, end in ranges):
        language_name = {"en": "English", "hi": "Hindi", "kn": "Kannada"}[lang]
        raise TranscriptLanguageMismatch(
            f"Could not recognize {language_name} speech confidently. "
            f"Check that {language_name} is selected and try again."
        )
    return text


def _load_transcriber():
    """Load Whisper once; record which backend (if any) is available."""
    global _TRANSCRIBER, _TRANSCRIBER_KIND
    with _TRANSCRIBER_LOCK:
        if _TRANSCRIBER_KIND is not None:
            return
        model_size = os.environ.get("WHISPER_MODEL", "small")
        try:
            from faster_whisper import WhisperModel

            _TRANSCRIBER = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
            _TRANSCRIBER_KIND = "faster"
            logger.info("STT: faster-whisper (%s, CPU)", model_size)
            return
        except Exception:  # noqa: BLE001 - try the other backend
            logger.exception("STT: faster-whisper failed to load")
        try:
            import whisper

            _TRANSCRIBER = whisper.load_model(model_size)
            _TRANSCRIBER_KIND = "openai"
            logger.info("STT: openai-whisper (%s)", model_size)
            return
        except Exception:  # noqa: BLE001
            _TRANSCRIBER_KIND = "unavailable"
            logger.exception("STT: no Whisper backend could be loaded")


def _transcribe(audio_bytes: bytes, lang: str) -> str:
    """Transcribe buffered audio bytes to text using whichever backend loaded."""
    if _TRANSCRIBER_KIND in (None, "unavailable") or not audio_bytes:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        if _TRANSCRIBER_KIND == "faster":
            started_at = time.monotonic()
            segments, _ = _TRANSCRIBER.transcribe(
                str(tmp_path),
                language=lang or None,
                # Greedy decoding is substantially faster for short live
                # caption chunks; keep the same model size to preserve quality.
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.info(
                "speech transcription completed: language=%s duration=%.1fs characters=%d",
                lang,
                time.monotonic() - started_at,
                len(text),
            )
            return _validate_transcript_language(text, lang)
        # openai-whisper
        result = _TRANSCRIBER.transcribe(str(tmp_path), language=lang or None)
        text = str(result.get("text", "")).strip()
        return _validate_transcript_language(text, lang)
    except TranscriptLanguageMismatch:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("transcription failed: %s", exc)
        raise RuntimeError("speech transcription failed") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.websocket("/ws/speech")
async def speech(ws: WebSocket) -> None:
    """Live speech-to-text WebSocket (Section 6.4)."""
    await ws.accept()
    await asyncio.to_thread(_load_transcriber)

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
                requested_lang = text.split(":", 1)[1].strip() or "en"
                if requested_lang not in _SUPPORTED_LANGUAGES:
                    await ws.send_json(
                        {
                            "type": "error",
                            "text": "Choose English, Hindi, or Kannada before recording.",
                        }
                    )
                    continue
                lang = requested_lang
                logger.info("speech session language selected: %s", lang)
            elif text == "flush":
                try:
                    transcript = await asyncio.to_thread(_transcribe, bytes(buffer), lang)
                except TranscriptLanguageMismatch as exc:
                    buffer.clear()
                    await ws.send_json({"type": "error", "text": str(exc)})
                    continue
                except Exception:  # noqa: BLE001
                    logger.exception("speech transcription failed")
                    buffer.clear()
                    await ws.send_json(
                        {
                            "type": "error",
                            "text": "Could not transcribe this audio. Please try again.",
                        }
                    )
                    continue
                buffer.clear()
                if not transcript:
                    language_name = {"en": "English", "hi": "Hindi", "kn": "Kannada"}[lang]
                    await ws.send_json(
                        {
                            "type": "no_speech",
                            "text": (
                                f"No {language_name} speech recognized. "
                                "Speak clearly and try again."
                            ),
                            "lang": lang,
                        }
                    )
                    continue
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
