"""LLM provider construction for the sentence layer (Section 5.10 / 13).

Builds the :class:`signtalk_core.compose.LLMProvider` the compose endpoint uses,
chosen by the ``LLM_PROVIDER`` env var:

    none    -> NullProvider (always falls back to scripted/template; the default
               and the demo-safe path with no key)
    gemini  -> GeminiProvider (one strict-JSON call, 4 s timeout, one retry;
               model name from GEMINI_MODEL so a retired model is swappable
               without code changes — Section 4.5)
    ollama  -> [VERIFY] not implemented yet; falls back to NullProvider

Network/SDK imports stay here (lazy), never in ``signtalk_core``.
"""

from __future__ import annotations

import json
import logging
import os
import re

from signtalk_core.compose import NullProvider

logger = logging.getLogger("signtalk.server.providers")

_PROMPT = (
    "You convert a sequence of recognised sign-language WORDS into one short, "
    "natural sentence. Use ONLY the meaning of the given words — add no new "
    "facts, names, numbers or objects. Keep it short.\n"
    "Return STRICT JSON only, no prose, with exactly these keys: "
    '{{"en": "...", "hi": "...", "kn": "...", "tone": "neutral"}}\n'
    "en = English, hi = Hindi (Devanagari), kn = Kannada script.\n"
    "Signed words: {words}\n"
    "Conversation so far (for context, may be empty): {history}\n"
)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response (tolerates code fences)."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"\{.*\}", text, re.DOTALL)
    if not fence:
        return None
    try:
        return json.loads(fence.group(0))
    except json.JSONDecodeError:
        return None


class GeminiProvider:
    """Gemini sentence provider: one strict-JSON call with a timeout + 1 retry."""

    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        self._api_key = api_key
        self._model_name = model
        self._timeout_s = timeout_s

    def compose(self, words, history, scenario_id):  # noqa: ANN001
        """Return the parsed JSON dict, or None on any failure (-> fallback)."""
        try:
            import google.generativeai as genai
        except ImportError:
            logger.warning("google-generativeai not installed; LLM disabled")
            return None

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model_name)
        prompt = _PROMPT.format(words=", ".join(words), history=" | ".join(history or []))

        for attempt in (1, 2):  # one retry (Section 5.10)
            try:
                resp = model.generate_content(
                    prompt,
                    request_options={"timeout": self._timeout_s},
                    generation_config={"response_mime_type": "application/json"},
                )
                parsed = _extract_json(resp.text or "")
                if parsed is not None:
                    return parsed
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini attempt %d failed: %s", attempt, exc)
        return None


def build_provider():
    """Construct the configured LLM provider (defaults to NullProvider)."""
    provider = os.environ.get("LLM_PROVIDER", "none").strip().lower()
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty -> none")
            return NullProvider()
        model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip()
        timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "4"))
        logger.info("LLM provider: gemini (model=%s, timeout=%ss)", model, timeout)
        return GeminiProvider(key, model, timeout)
    if provider == "ollama":
        logger.warning("LLM_PROVIDER=ollama not implemented yet -> none")
        return NullProvider()
    logger.info("LLM provider: none (scripted/template only)")
    return NullProvider()
