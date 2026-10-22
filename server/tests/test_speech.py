"""WS /ws/speech contract (Section 6.4).

Two paths are covered:

* **Graceful degradation** — with no Whisper backend the socket still accepts
  and sends a single final message explaining STT is unavailable (the browser
  Web Speech API is the client-side fallback).
* **Real transcription protocol** — a FAKE transcriber is injected (so no heavy
  Whisper install / real audio is needed) to verify the live protocol end to
  end: a ``lang:`` hint is honoured, enough buffered audio auto-emits an INTERIM
  (``is_final: false``) caption, and ``flush`` emits a FINAL (``is_final:
  true``) transcript and clears the buffer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app.routers import speech as speech_router


def test_speech_ws_graceful_without_whisper(client: TestClient) -> None:
    # Force the unavailable path regardless of what's installed locally.
    speech_router.set_transcriber_factory(lambda: None)
    try:
        with client.websocket_connect("/ws/speech") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "transcript"
            assert msg["is_final"] is True
            assert msg["lang"] == "en"
            assert isinstance(msg["text"], str) and msg["text"]
    finally:
        speech_router.set_transcriber_factory(None)


def test_speech_ws_flush_emits_final(client: TestClient) -> None:
    """A ``flush`` returns a FINAL transcript tagged with the chosen language."""
    speech_router.set_transcriber_factory(
        lambda: (lambda audio, lang: f"heard {len(audio)} bytes in {lang}")
    )
    try:
        with client.websocket_connect("/ws/speech") as ws:
            ws.send_text("lang:hi")
            ws.send_bytes(b"\x00" * 10)
            ws.send_text("flush")
            msg = ws.receive_json()
            assert msg["type"] == "transcript"
            assert msg["is_final"] is True
            assert msg["lang"] == "hi"
            assert msg["text"] == "heard 10 bytes in hi"
    finally:
        speech_router.set_transcriber_factory(None)


def test_speech_ws_emits_interim_caption(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enough buffered audio auto-emits an INTERIM (non-final) live caption."""
    # Make the interim threshold tiny so one chunk triggers it.
    monkeypatch.setattr(speech_router, "INTERIM_BYTES", 4)
    speech_router.set_transcriber_factory(lambda: (lambda audio, lang: "live words"))
    try:
        with client.websocket_connect("/ws/speech") as ws:
            ws.send_bytes(b"\x01" * 8)  # >= INTERIM_BYTES -> interim fires
            msg = ws.receive_json()
            assert msg["type"] == "transcript"
            assert msg["is_final"] is False
            assert msg["text"] == "live words"
    finally:
        speech_router.set_transcriber_factory(None)
