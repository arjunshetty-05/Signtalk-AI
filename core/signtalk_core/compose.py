"""Sentence layer — words -> grammatical sentence in en/hi/kn (Section 5.10).

Determinism first (PROJECT_CONTEXT Section 5.10, order of preference):

    1. SCRIPTED TABLE  (config/sentences.json): a known word sequence maps to a
       pre-verified sentence in en/hi/kn. No LLM, no hallucination risk. This is
       what the demo relies on. source = "scripted", verified = True.
    2. LLM             (optional provider): one call returning strict JSON
       {en, hi, kn, tone}; validated and SAFETY-CHECKED (every signed word, or a
       known synonym, must appear in the English sentence) — else discarded and
       we fall through. source = "llm", verified = False.
    3. TEMPLATE        fallback: capitalise/join the words; translate each via
       the offline phrasebook (config/phrasebook.json). Always available,
       offline. source = "template", verified = False.

This module is pure Python + a pluggable :class:`LLMProvider` (so the LLM, and
all its network/latency risk, lives behind an interface and can be swapped for
``none``/ollama). It never imports a network client directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

LANGS = ("en", "hi", "kn")


@dataclass(frozen=True)
class ComposeResult:
    """Section 6.3 response payload (minus latency, added by the caller).

    Attributes:
        sentences: ``{"en": str, "hi": str, "kn": str}``.
        source: ``"scripted"`` | ``"llm"`` | ``"template"``.
        verified: True only for scripted (native-speaker-verified) sentences.
    """

    sentences: dict[str, str]
    source: str
    verified: bool


class LLMProvider(Protocol):
    """Minimal interface for an optional sentence LLM.

    An implementation takes the signed words (+ optional context) and returns a
    dict with ``en``/``hi``/``kn`` (and optionally ``tone``), or raises/returns
    ``None`` on any failure — the caller then falls back to the template. The
    4 s timeout and retry policy live inside the implementation (Section 5.10).
    """

    def compose(
        self, words: list[str], history: list[str], scenario_id: str | None
    ) -> dict | None:  # pragma: no cover - protocol
        ...


class NullProvider:
    """The ``LLM_PROVIDER=none`` provider: always declines, forcing fallback."""

    def compose(self, words, history, scenario_id):  # noqa: D401, ANN001
        return None


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def _config_root(config_root: str | Path | None) -> Path:
    if config_root is not None:
        return Path(config_root)
    from signtalk_core.config import _repo_root

    return _repo_root()


def load_sentences(config_root: str | Path | None = None) -> dict[str, dict]:
    """Load the scripted sentence table (``config/sentences.json``)."""
    root = _config_root(config_root)
    with open(root / "config" / "sentences.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("sentences", {})


def load_phrasebook(config_root: str | Path | None = None) -> dict[str, dict]:
    """Load the offline phrasebook (``config/phrasebook.json``)."""
    root = _config_root(config_root)
    with open(root / "config" / "phrasebook.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("words", {})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm_key(words: list[str]) -> str:
    """Scripted-table key: lower-cased, single-spaced word sequence."""
    return " ".join(w.strip().lower() for w in words if w.strip())


def _template_sentence(words: list[str], phrasebook: dict[str, dict]) -> dict[str, str]:
    """Build the template-fallback sentence in all three languages.

    English is the capitalised, punctuated join of the words. Hindi/Kannada are
    built by looking each word up in the phrasebook (falling back to the English
    word when a translation is missing, so the sentence is never empty).
    """
    clean = [w.strip() for w in words if w.strip()]
    if not clean:
        return {"en": "", "hi": "", "kn": ""}

    def join(lang: str) -> str:
        parts = []
        for w in clean:
            entry = phrasebook.get(w.lower())
            parts.append(entry.get(lang, w) if entry else w)
        s = " ".join(parts)
        return s[0].upper() + s[1:] + "." if lang == "en" and s else s

    return {"en": join("en"), "hi": join("hi"), "kn": join("kn")}


_WORD_RE = re.compile(r"[a-z]+")


def llm_output_is_safe(words: list[str], english: str, synonyms: dict[str, list[str]] | None = None) -> bool:
    """Safety check (Section 5.10): every signed word must appear in the EN output.

    A signed word is satisfied if the word itself OR any of its configured
    synonyms appears as a token in the English sentence. This blocks the LLM
    from inventing or dropping meaning.

    Args:
        words: the signed words.
        english: the LLM's English sentence.
        synonyms: optional ``{word: [synonym, ...]}`` map.

    Returns:
        True iff every signed word (or a synonym) is present in ``english``.
    """
    tokens = set(_WORD_RE.findall(english.lower()))
    syn = synonyms or {}
    for w in words:
        wl = w.strip().lower()
        if not wl:
            continue
        candidates = {wl, *(s.lower() for s in syn.get(wl, []))}
        # Multi-word signs (e.g. "thank you"): satisfied if ALL their tokens appear.
        ok = any(all(tok in tokens for tok in cand.split()) for cand in candidates)
        if not ok:
            return False
    return True


def _validate_llm_json(obj: object) -> dict[str, str] | None:
    """Return a clean ``{en,hi,kn}`` dict if ``obj`` is a valid LLM payload."""
    if not isinstance(obj, dict):
        return None
    out = {}
    for lang in LANGS:
        val = obj.get(lang)
        if not isinstance(val, str) or not val.strip():
            return None
        out[lang] = val.strip()
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def compose_sentence(
    words: list[str],
    *,
    history: list[str] | None = None,
    scenario_id: str | None = None,
    provider: LLMProvider | None = None,
    synonyms: dict[str, list[str]] | None = None,
    config_root: str | Path | None = None,
) -> ComposeResult:
    """Compose a sentence from signed words (Section 5.10 order of preference).

    Args:
        words: the accumulated signed words, in order.
        history: prior conversation turns (passed to the LLM for context).
        scenario_id: optional active scenario id (passed to the LLM).
        provider: optional LLM provider; ``None`` (or NullProvider) skips step 2.
        synonyms: optional synonym map for the LLM safety check.
        config_root: optional repo root holding ``config/``.

    Returns:
        A :class:`ComposeResult`.
    """
    history = history or []
    sentences = load_sentences(config_root)
    phrasebook = load_phrasebook(config_root)

    # 1) scripted table -------------------------------------------------------
    key = _norm_key(words)
    if key and key in sentences:
        entry = sentences[key]
        picked = {lang: str(entry.get(lang, "")) for lang in LANGS}
        if all(picked.values()):
            return ComposeResult(sentences=picked, source="scripted", verified=True)

    # 2) LLM (optional) -------------------------------------------------------
    if provider is not None and not isinstance(provider, NullProvider):
        try:
            raw = provider.compose(words, history, scenario_id)
        except Exception:  # noqa: BLE001 - any provider failure -> fall through
            raw = None
        validated = _validate_llm_json(raw)
        if validated is not None and llm_output_is_safe(words, validated["en"], synonyms):
            return ComposeResult(sentences=validated, source="llm", verified=False)

    # 3) template fallback ----------------------------------------------------
    return ComposeResult(
        sentences=_template_sentence(words, phrasebook), source="template", verified=False
    )
