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

Live captions: as audio accumulates, the server auto-transcribes the buffer so
far every ``INTERIM_BYTES`` and emits an INTERIM (``is_final: false``) transcript
so a caption appears without waiting for a manual flush. ``flush`` emits a FINAL
(``is_final: true``) transcript and clears the buffer, committing that turn to
the conversation log. This matches the web client, which shows interim text in
the live-caption area and only appends finals to the log.

Protocol (client -> server):
    * binary frames  = raw audio chunks (webm/opus or wav bytes) to buffer
    * text  "flush"  = transcribe the buffer, emit a FINAL transcript, clear it
    * text  "lang:<code>" = set the spoken language hint (e.g. "lang:en")
    * close          = end the session
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("signtalk.server.speech")

router = APIRouter(tags=["speech"])

# Emit an interim transcript once the buffer grows by at least this many bytes
# since the last interim (keeps Whisper calls bounded while captions stay live).
INTERIM_BYTES = 48_000

# A transcriber is a callable ``(audio_bytes, lang) -> text``. The default is
# the lazily-loaded Whisper singleton; tests inject a fake via
# ``set_transcriber_factory`` so the real protocol path is exercised without a
# heavy Whisper install or real audio.
Transcriber = Callable[[bytes, str], str]

_TRANSCRIBER_FACTORY: Optional[Callable[[], Optional[Transcriber]]] = None

# Lazily-initialised Whisper singleton.
_WHISPER = None
_WHISPER_KIND = None  # "faster" | "openai" | "unavailable"


def set_transcriber_factory(factory: Optional[Callable[[], Optional[Transcriber]]]) -> None:
    """Override how a transcriber is obtained (used by tests).

    ``factory() -> Transcriber`` enables STT with that callable; ``factory() ->
    None`` forces the unavailable/graceful-degradation path. Pass ``None`` to
    restore the default Whisper-backed behaviour.
    """
    global _TRANSCRIBER_FACTORY
    _TRANSCRIBER_FACTORY = factory


def _load_whisper() -> None:
    """Load Whisper once; record which backend (if any) is available."""
    global _WHISPER, _WHISPER_KIND
    if _WHISPER_KIND is not None:
        return
    model_size = os.environ.get("WHISPER_MODEL", "small")
    # Default to CPU: it needs no CUDA runtime DLLs (``device="auto"`` fails with
    # "cublas64_12.dll not found" on machines without the CUDA toolkit on PATH)
    # and int8 on CPU is fast enough for small/base. Override with
    # ``WHISPER_DEVICE=cuda`` + ``WHISPER_COMPUTE=float16`` on a GPU box whose
    # CUDA runtime is installed.
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("WHISPER_COMPUTE", "int8")
    try:
        from faster_whisper import WhisperModel

        _WHISPER = WhisperModel(model_size, device=device, compute_type=compute_type)
        _WHISPER_KIND = "faster"
        logger.info("STT: faster-whisper (%s, device=%s, %s)", model_size, device, compute_type)
        return
    except Exception:  # noqa: BLE001 - try the other backend
        pass
    try:
        import whisper

        _WHISPER = whisper.load_model(model_size)
        _WHISPER_KIND = "openai"
        logger.info("STT: openai-whisper (%s)", model_size)
        return
    except Exception:  # noqa: BLE001
        _WHISPER_KIND = "unavailable"
        logger.warning("STT: no Whisper backend installed -> transcription disabled")


def _whisper_transcribe(audio_bytes: bytes, lang: str) -> str:
    """Transcribe buffered audio bytes to text using whichever backend loaded."""
    if _WHISPER_KIND in (None, "unavailable") or not audio_bytes:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        if _WHISPER_KIND == "faster":
            segments, _ = _WHISPER.transcribe(str(tmp_path), language=lang or None)
            return " ".join(seg.text.strip() for seg in segments).strip()
        # openai-whisper
        result = _WHISPER.transcribe(str(tmp_path), language=lang or None)
        return str(result.get("text", "")).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("transcription failed: %s", exc)
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)


def _resolve_transcriber() -> Optional[Transcriber]:
    """Return the active transcriber callable, or ``None`` if STT is unavailable.

    Honours a test-injected factory first; otherwise lazily loads Whisper and
    returns its transcribe function (or ``None`` when no backend is installed).
    """
    if _TRANSCRIBER_FACTORY is not None:
        return _TRANSCRIBER_FACTORY()
    _load_whisper()
    if _WHISPER_KIND == "unavailable":
        return None
    return _whisper_transcribe


@router.websocket("/ws/speech")
async def speech(ws: WebSocket) -> None:
    """Live speech-to-text WebSocket (Section 6.4)."""
    await ws.accept()
    transcriber = _resolve_transcriber()

    if transcriber is None:
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
    bytes_since_interim = 0
    lang = "en"
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                buffer.extend(data)
                bytes_since_interim += len(data)
                # Auto-emit a live (interim) caption as audio accumulates.
                if bytes_since_interim >= INTERIM_BYTES:
                    bytes_since_interim = 0
                    interim = transcriber(bytes(buffer), lang)
                    if interim:
                        await ws.send_json(
                            {"type": "transcript", "text": interim, "is_final": False, "lang": lang}
                        )
                continue
            text = (msg.get("text") or "").strip()
            if text.startswith("lang:"):
                lang = text.split(":", 1)[1].strip() or "en"
            elif text == "flush":
                transcript = transcriber(bytes(buffer), lang)
                buffer.clear()
                bytes_since_interim = 0
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
