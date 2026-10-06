"""Unit tests for the sentence layer (signtalk_core.compose, Section 5.10)."""

from __future__ import annotations

import pytest

from signtalk_core.compose import (
    NullProvider,
    compose_sentence,
    llm_output_is_safe,
)


@pytest.mark.unit
def test_scripted_table_hit_is_verified():
    # "hello" is in the seeded scripted table -> deterministic, verified.
    r = compose_sentence(["hello"])
    assert r.source == "scripted"
    assert r.verified is True
    assert r.sentences["en"] and r.sentences["hi"] and r.sentences["kn"]


@pytest.mark.unit
def test_template_fallback_for_unknown_sequence():
    # A sequence not in the table and no LLM -> template fallback, unverified.
    r = compose_sentence(["water", "more"], provider=NullProvider())
    assert r.source == "template"
    assert r.verified is False
    assert r.sentences["en"].lower().startswith("water")
    # Phrasebook provides hi/kn for both words.
    assert r.sentences["hi"] and r.sentences["kn"]


@pytest.mark.unit
def test_empty_words_gives_empty_template():
    r = compose_sentence([], provider=NullProvider())
    assert r.source == "template"
    assert r.sentences == {"en": "", "hi": "", "kn": ""}


@pytest.mark.unit
def test_llm_used_when_safe_and_not_scripted():
    class GoodLLM:
        def compose(self, words, history, scenario_id):
            return {"en": "I want more water.", "hi": "x", "kn": "y", "tone": "neutral"}

    r = compose_sentence(["water", "more"], provider=GoodLLM())
    assert r.source == "llm"
    assert r.verified is False
    assert r.sentences["en"] == "I want more water."


@pytest.mark.unit
def test_llm_rejected_when_unsafe_falls_back_to_template():
    # LLM drops the signed word "water" -> safety check fails -> template.
    class DropsWordLLM:
        def compose(self, words, history, scenario_id):
            return {"en": "I am fine.", "hi": "x", "kn": "y"}

    r = compose_sentence(["water", "more"], provider=DropsWordLLM())
    assert r.source == "template"


@pytest.mark.unit
def test_llm_exception_falls_back():
    class BrokenLLM:
        def compose(self, words, history, scenario_id):
            raise RuntimeError("network down")

    r = compose_sentence(["water"], provider=BrokenLLM())
    # "water" IS scripted, so this is scripted; use a non-scripted word instead.
    assert r.source in {"scripted", "template"}
    r2 = compose_sentence(["more"], provider=BrokenLLM())
    assert r2.source == "template"


@pytest.mark.unit
def test_safety_check_handles_multiword_signs_and_synonyms():
    assert llm_output_is_safe(["thank you"], "Thank you very much.")
    assert not llm_output_is_safe(["thank you"], "You are welcome.")
    # synonym satisfies the check
    assert llm_output_is_safe(["help"], "Please assist me.", {"help": ["assist"]})
