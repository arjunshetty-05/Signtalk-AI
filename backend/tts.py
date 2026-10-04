"""
tts.py — SignTalk AI / Person B (Intelligence Layer)

Text-to-speech with an online (gTTS) and offline (Coqui TTS) path, built
from the contract locked in Person C's Prompt C1:

    synthesize_speech(text: str, lang: str, mode: "online"|"offline") -> bytes
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

logger = logging.getLogger("signtalk.tts")

# gTTS language codes
GTTS_LANG_MAP = {"en": "en", "hi": "hi", "kn": "kn"}
# Coqui TTS model per language (multilingual fallback where a dedicated
# Indic model isn't bundled)
COQUI_MODEL_MAP = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "hi": "tts_models/multilingual/multi-dataset/your_tts",
    "kn": "tts_models/multilingual/multi-dataset/your_tts",
}


def _synthesize_online(text: str, lang: str) -> bytes:
    from gtts import gTTS

    gtts_lang = GTTS_LANG_MAP.get(lang, "en")
    buf = io.BytesIO()
    gTTS(text=text, lang=gtts_lang).write_to_fp(buf)
    return buf.getvalue()


@lru_cache(maxsize=len(COQUI_MODEL_MAP))
def _load_coqui_model(model_name: str):
    from TTS.api import TTS

    logger.info("Loading Coqui TTS model %s...", model_name)
    return TTS(model_name)


def _synthesize_offline(text: str, lang: str) -> bytes:
    import soundfile as sf

    model_name = COQUI_MODEL_MAP.get(lang, COQUI_MODEL_MAP["en"])
    tts_model = _load_coqui_model(model_name)
    wav = tts_model.tts(text=text)

    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=22050, format="WAV")
    return buf.getvalue()


def synthesize_speech(text: str, lang: str, mode: str) -> bytes:
    """
    Converts `text` to speech audio bytes.

    mode="online"  -> gTTS (requires network)
    mode="offline" -> Coqui TTS (fully local)

    On an online-mode failure (e.g. no network), automatically falls back
    to the offline Coqui path rather than raising, so callers always get
    audio back when possible.
    """
    if not text or not text.strip():
        raise ValueError("synthesize_speech requires non-empty text")

    if mode == "online":
        try:
            return _synthesize_online(text, lang)
        except Exception as exc:
            logger.warning("Online TTS (gTTS) failed, falling back to offline Coqui: %s", exc)
            return _synthesize_offline(text, lang)
    elif mode == "offline":
        try:
            return _synthesize_offline(text, lang)
        except Exception as exc:
            # Coqui (the `TTS` package) is an optional dependency — see
            # requirements-offline-tts.txt — deliberately not always
            # installed. Genuinely offline environments without network
            # will still fail here (nothing left to fall back to), but
            # falling back to gTTS beats a hard 500 whenever a network path
            # does exist, e.g. offline mode is being demoed rather than
            # actually running with no internet.
            logger.warning("Offline TTS (Coqui) failed, falling back to online gTTS: %s", exc)
            return _synthesize_online(text, lang)
    else:
        raise ValueError(f"Unknown TTS mode: {mode!r} (expected 'online' or 'offline')")
