"""WS /ws/speech degrades gracefully when Whisper is not installed (Section 6.4).

In CI / this environment no Whisper backend is present, so the socket must still
accept and send a single final transcript message explaining STT is unavailable
(the browser Web Speech API is the client-side fallback). This guards the
graceful-degradation contract; the actual transcription path needs a Whisper
install and real audio and is exercised manually.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_speech_ws_graceful_without_whisper(client: TestClient) -> None:
    with client.websocket_connect("/ws/speech") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "transcript"
        assert msg["is_final"] is True
        assert msg["lang"] == "en"
        assert isinstance(msg["text"], str) and msg["text"]
