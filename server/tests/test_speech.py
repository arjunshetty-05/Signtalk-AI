"""WS /ws/speech degrades gracefully when Whisper is not installed (Section 6.4).

In CI / this environment no Whisper backend is present, so the socket must still
accept and send a single final transcript message explaining STT is unavailable
(the browser Web Speech API is the client-side fallback). This guards the
graceful-degradation contract; the actual transcription path needs a Whisper
install and real audio and is exercised manually.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient

from server.app.routers import speech as speech_router


def test_speech_ws_graceful_without_whisper(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(speech_router, "_TRANSCRIBER_KIND", "unavailable")
    monkeypatch.setattr(speech_router, "_load_transcriber", lambda: None)
    with client.websocket_connect("/ws/speech") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "transcript"
        assert msg["is_final"] is True
        assert msg["lang"] == "en"
        assert isinstance(msg["text"], str) and msg["text"]


def test_faster_whisper_uses_cpu_when_loading(monkeypatch) -> None:
    calls = {}

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type, cpu_threads):
            calls.update(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setattr(speech_router, "_TRANSCRIBER", None)
    monkeypatch.setattr(speech_router, "_TRANSCRIBER_KIND", None)
    monkeypatch.setenv("WHISPER_MODEL", "tiny")

    speech_router._load_transcriber()

    assert calls == {
        "model_size": "tiny",
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 4,
    }
    assert speech_router._TRANSCRIBER_KIND == "faster"


def test_speech_ws_transcribes_off_event_loop(client: TestClient, monkeypatch) -> None:
    class FakeTranscriber:
        def transcribe(
            self,
            path,
            *,
            language,
            beam_size,
            condition_on_previous_text,
            vad_filter,
            vad_parameters,
        ):
            assert beam_size == 5
            assert condition_on_previous_text is False
            assert vad_filter is True
            assert vad_parameters == {"min_silence_duration_ms": 500, "speech_pad_ms": 200}
            return [SimpleNamespace(text=" test transcript ")], None

    monkeypatch.setattr(speech_router, "_TRANSCRIBER", FakeTranscriber())
    monkeypatch.setattr(speech_router, "_TRANSCRIBER_KIND", "faster")
    monkeypatch.setattr(speech_router, "_load_transcriber", lambda: None)

    with client.websocket_connect("/ws/speech") as ws:
        ws.send_bytes(b"test audio")
        ws.send_text("flush")
        msg = ws.receive_json()

    assert msg == {
        "type": "transcript",
        "text": "test transcript",
        "is_final": True,
        "lang": "en",
    }


def test_speech_ws_passes_kannada_language_and_rejects_hindi_script(
    client: TestClient, monkeypatch
) -> None:
    calls = []

    class FakeTranscriber:
        def transcribe(
            self,
            path,
            *,
            language,
            beam_size,
            condition_on_previous_text,
            vad_filter,
            vad_parameters,
        ):
            calls.append(language)
            return [SimpleNamespace(text="यह हिंदी पाठ है।")], None

    monkeypatch.setattr(speech_router, "_TRANSCRIBER", FakeTranscriber())
    monkeypatch.setattr(speech_router, "_TRANSCRIBER_KIND", "faster")
    monkeypatch.setattr(speech_router, "_load_transcriber", lambda: None)

    with client.websocket_connect("/ws/speech") as ws:
        ws.send_text("lang:kn")
        ws.send_bytes(b"test audio")
        ws.send_text("flush")
        msg = ws.receive_json()

    assert calls == ["kn"]
    assert msg["type"] == "error"
    assert "Kannada" in msg["text"]


def test_speech_ws_rejects_unsupported_language(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(speech_router, "_TRANSCRIBER_KIND", "faster")
    monkeypatch.setattr(speech_router, "_load_transcriber", lambda: None)

    with client.websocket_connect("/ws/speech") as ws:
        ws.send_text("lang:ta")
        msg = ws.receive_json()

    assert msg == {
        "type": "error",
        "text": "Choose English, Hindi, or Kannada before recording.",
    }


def test_speech_ws_reports_empty_kannada_transcription(
    client: TestClient, monkeypatch
) -> None:
    class EmptyTranscriber:
        def transcribe(
            self,
            path,
            *,
            language,
            beam_size,
            condition_on_previous_text,
            vad_filter,
            vad_parameters,
        ):
            assert language == "kn"
            return [], None

    monkeypatch.setattr(speech_router, "_TRANSCRIBER", EmptyTranscriber())
    monkeypatch.setattr(speech_router, "_TRANSCRIBER_KIND", "faster")
    monkeypatch.setattr(speech_router, "_load_transcriber", lambda: None)

    with client.websocket_connect("/ws/speech") as ws:
        ws.send_text("lang:kn")
        ws.send_bytes(b"test audio")
        ws.send_text("flush")
        msg = ws.receive_json()

    assert msg == {
        "type": "no_speech",
        "text": "No Kannada speech recognized. Speak clearly and try again.",
        "lang": "kn",
    }


def test_speech_ws_keeps_empty_segment_informational(client: TestClient, monkeypatch) -> None:
    class EmptyTranscriber:
        def transcribe(
            self,
            path,
            *,
            language,
            beam_size,
            condition_on_previous_text,
            vad_filter,
            vad_parameters,
        ):
            return [], None

    monkeypatch.setattr(speech_router, "_TRANSCRIBER", EmptyTranscriber())
    monkeypatch.setattr(speech_router, "_TRANSCRIBER_KIND", "faster")
    monkeypatch.setattr(speech_router, "_load_transcriber", lambda: None)

    with client.websocket_connect("/ws/speech") as ws:
        ws.send_text("lang:en")
        ws.send_bytes(b"silent segment")
        ws.send_text("flush")
        msg = ws.receive_json()

    assert msg["type"] == "no_speech"
    assert msg["lang"] == "en"


def test_validate_transcript_requires_selected_script() -> None:
    assert speech_router._validate_transcript_language("ನಮಸ್ಕಾರ", "kn") == "ನಮಸ್ಕಾರ"
    assert speech_router._validate_transcript_language("नमस्ते", "hi") == "नमस्ते"
    try:
        speech_router._validate_transcript_language("नमस्ते", "kn")
    except speech_router.TranscriptLanguageMismatch as exc:
        assert "Kannada" in str(exc)
    else:
        raise AssertionError("Hindi-script output must not be shown as Kannada")
