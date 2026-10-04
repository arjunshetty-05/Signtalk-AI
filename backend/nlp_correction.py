"""
nlp_correction.py — SignTalk AI / Person B (Intelligence Layer)

Raw sign-token sequences -> grammatical sentence correction. Built from the
contract locked in Person C's Prompt C1:

    correct_sentence(gesture_tokens: list[str], emotion: str,
                      conversation_history: list[str]) -> {
        "sentence": str, "source": "gemini"|"flan-t5", "low_confidence": bool
    }

Primary path: Gemini 2.0 Flash. Falls back to a local Flan-T5-Small model on
API failure/timeout — so the pipeline degrades gracefully rather than
blocking. Low-confidence is flagged via a simple word-overlap ratio check
between the gesture tokens and the produced sentence.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

logger = logging.getLogger("signtalk.nlp_correction")

GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_TIMEOUT_SECONDS = 6
FLAN_T5_MODEL_NAME = "google/flan-t5-small"
LOW_CONFIDENCE_OVERLAP_THRESHOLD = 0.4
MAX_HISTORY_TURNS = 5

_STOPWORDS = {"a", "an", "the", "is", "are", "to", "of", "and", "i", "you"}


def _clean_tokens(gesture_tokens: list[str]) -> list[str]:
    """Basic token cleanup: strip whitespace/punctuation, uppercase-normalize
    for comparison but keep original case for prompting."""
    return [t.strip() for t in gesture_tokens if t and t.strip()]


def _build_prompt(gesture_tokens: list[str], emotion: str, conversation_history: list[str]) -> str:
    history_snippet = "\n".join(conversation_history[-MAX_HISTORY_TURNS:]) if conversation_history else "(none)"
    tokens_str = " ".join(gesture_tokens)
    return (
        "You are a sign-language sentence corrector. Convert the following raw "
        "sign tokens into a single, natural, grammatically correct sentence in "
        "the same intent. Consider the signer's current emotional state as tone "
        "context, but do not mention the emotion explicitly.\n\n"
        f"Sign tokens: {tokens_str}\n"
        f"Signer emotion: {emotion}\n"
        f"Recent conversation:\n{history_snippet}\n\n"
        "Respond with ONLY the corrected sentence, no explanation."
    )


def _call_gemini(prompt: str) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — skipping Gemini, using offline fallback.")
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            prompt,
            request_options={"timeout": GEMINI_TIMEOUT_SECONDS},
        )
        text = (response.text or "").strip()
        return text or None
    except Exception as exc:
        logger.warning("Gemini call failed, falling back to Flan-T5-Small: %s", exc)
        return None


@lru_cache(maxsize=1)
def _load_flan_t5():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_T5_MODEL_NAME)
    return tokenizer, model


def _call_flan_t5(prompt: str) -> str:
    try:
        tokenizer, model = _load_flan_t5()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=64)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return text
    except Exception as exc:
        logger.error("Flan-T5-Small fallback also failed: %s", exc)
        # last-resort: naive join so the pipeline never hard-fails
        return ""


def _word_overlap_ratio(gesture_tokens: list[str], sentence: str) -> float:
    """Fraction of (non-stopword) gesture tokens that appear, case-insensitively,
    somewhere in the produced sentence. Low overlap suggests the model may have
    drifted/hallucinated away from the actual signed content."""
    meaningful = [t.lower() for t in gesture_tokens if t.lower() not in _STOPWORDS]
    if not meaningful:
        return 1.0
    sentence_words = set(re.findall(r"[a-z']+", sentence.lower()))
    matched = sum(1 for t in meaningful if t.lower() in sentence_words)
    return matched / len(meaningful)


def correct_sentence(
    gesture_tokens: list[str],
    emotion: str,
    conversation_history: list[str],
    force_offline: bool = False,
) -> dict:
    """
    Converts raw gesture tokens into a grammatical sentence.

    force_offline: when True (set by Person C's core.offline_mode
    dependency), skips the Gemini call entirely and goes straight to the
    Flan-T5-Small fallback — used for the airplane-mode demo.

    Returns: {"sentence": str, "source": "gemini"|"flan-t5", "low_confidence": bool}
    """
    tokens = _clean_tokens(gesture_tokens)
    if not tokens:
        return {"sentence": "", "source": "flan-t5", "low_confidence": True}

    prompt = _build_prompt(tokens, emotion, conversation_history)

    sentence = None if force_offline else _call_gemini(prompt)
    source = "gemini"
    if sentence is None:
        sentence = _call_flan_t5(prompt)
        source = "flan-t5"

    if not sentence:
        sentence = " ".join(tokens).capitalize() + "."
        source = "flan-t5"

    overlap = _word_overlap_ratio(tokens, sentence)
    low_confidence = overlap < LOW_CONFIDENCE_OVERLAP_THRESHOLD

    return {"sentence": sentence, "source": source, "low_confidence": low_confidence}


class ConversationMemory:
    """Simple multi-turn conversation memory for context-aware correction."""

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self._history: list[str] = []

    def add(self, sentence: str) -> None:
        self._history.append(sentence)
        self._history = self._history[-self.max_turns:]

    def get(self) -> list[str]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
