"""
translation.py — SignTalk AI / Person B (Intelligence Layer)

Multilingual (English/Hindi/Kannada) translation pipeline, built from the
contract locked in Person C's Prompt C1:

    translate_text(text: str, target_lang: str, offline: bool) -> str

Lookup order: offline phrasebook -> in-memory cache -> Firestore cache ->
Google Translate API (fallback on cache miss). When offline=True, only the
phrasebook is checked (network/Firestore are skipped entirely).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

logger = logging.getLogger("signtalk.translation")

SUPPORTED_LANGS = {"en", "hi", "kn"}
PHRASEBOOK_PATH = os.path.join(os.path.dirname(__file__), "phrasebook.json")

_in_memory_cache: dict[str, str] = {}


def _load_phrasebook() -> dict:
    if not os.path.exists(PHRASEBOOK_PATH):
        logger.warning("phrasebook.json not found at %s", PHRASEBOOK_PATH)
        return {}
    with open(PHRASEBOOK_PATH, encoding="utf-8") as f:
        return json.load(f)


_phrasebook = _load_phrasebook()


def detect_language(text: str) -> str:
    """Lightweight auto language detection across en/hi/kn using Unicode
    script ranges (Devanagari for Hindi, Kannada script for Kannada, else
    assumed English). Good enough for routing; not a full langid model."""
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "hi"
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"
    return "en"


def cache_key(text: str, target_lang: str) -> str:
    digest = hashlib.sha256(f"{text}|{target_lang}".encode("utf-8")).hexdigest()
    return digest


def _check_phrasebook(text: str, target_lang: str) -> str | None:
    entry = _phrasebook.get(text.strip().lower())
    if entry:
        return entry.get(target_lang)
    return None


def _check_memory_cache(key: str) -> str | None:
    return _in_memory_cache.get(key)


def _check_firestore_cache(key: str) -> str | None:
    try:
        from api.firebase.firebase_client import get_translation_cache

        return get_translation_cache(key)
    except Exception:
        return None  # Firebase layer (Person C, Prompt C2) may not exist/be configured yet


def _write_caches(key: str, translated_text: str) -> None:
    _in_memory_cache[key] = translated_text
    try:
        from api.firebase.firebase_client import set_translation_cache

        set_translation_cache(key, translated_text)
    except Exception:
        pass  # Firestore optional — in-memory cache still applies


def _call_google_translate(text: str, target_lang: str) -> str:
    api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_TRANSLATE_API_KEY not set — returning original text unchanged.")
        return text
    try:
        # google.cloud.translate_v2.Client() authenticates via Application
        # Default Credentials (a service account / gcloud login) and never
        # actually uses an API key — that's a different auth mechanism than
        # what GOOGLE_TRANSLATE_API_KEY implies. The v2 REST endpoint is what
        # actually accepts a plain API key, so call that directly instead.
        import requests

        response = requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            params={"key": api_key},
            json={"q": text, "target": target_lang, "format": "text"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["data"]["translations"][0]["translatedText"]
    except Exception as exc:
        logger.error("Google Translate API call failed: %s", exc)
        return text


def translate_text(text: str, target_lang: str, offline: bool = False) -> str:
    """
    Translates `text` into `target_lang` ('en'|'hi'|'kn').

    Lookup order (online):
        1. offline phrasebook (exact-phrase match)
        2. in-memory cache
        3. Firestore cache
        4. Google Translate API (result written back to both caches)

    Lookup order (offline=True): phrasebook only, else returns original text.
    """
    if target_lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported target_lang: {target_lang!r} (expected one of {SUPPORTED_LANGS})")

    phrase_hit = _check_phrasebook(text, target_lang)
    if phrase_hit:
        return phrase_hit

    if offline:
        logger.info("Offline mode: no phrasebook match for %r — returning original text.", text)
        return text

    key = cache_key(text, target_lang)

    memory_hit = _check_memory_cache(key)
    if memory_hit:
        return memory_hit

    firestore_hit = _check_firestore_cache(key)
    if firestore_hit:
        _in_memory_cache[key] = firestore_hit
        return firestore_hit

    translated = _call_google_translate(text, target_lang)
    _write_caches(key, translated)
    return translated
