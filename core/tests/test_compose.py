"""Unit tests for the sentence layer (signtalk_core.compose, Section 5.10).

These tests are CONFIG-INDEPENDENT: each builds its own tiny ``config/``
(sentences.json + phrasebook.json) in a temp dir and passes it as
``config_root``, so they test the compose LOGIC (scripted -> LLM -> template
routing + the safety check) regardless of the shipped vocabulary.
"""

from __future__ import annotations

import json

import pytest

from signtalk_core.compose import (
    NullProvider,
    compose_sentence,
    llm_output_is_safe,
)


@pytest.fixture()
def cfg_root(tmp_path):
    """A temp repo root with a known scripted table + phrasebook."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "sentences.json").write_text(
        json.dumps(
            {
                "sentences": {
                    "hello": {"en": "Hello.", "hi": "नमस्ते।", "kn": "ನಮಸ್ಕಾರ."}
                }
            }
        ),
        encoding="utf-8",
    )
    (cfg / "phrasebook.json").write_text(
        json.dumps(
            {
                "words": {
                    "water": {"en": "water", "hi": "पानी", "kn": "ನೀರು"},
                    "more": {"en": "more", "hi": "और", "kn": "ಇನ್ನಷ್ಟು"},
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.unit
def test_scripted_table_hit_is_verified(cfg_root):
    r = compose_sentence(["hello"], config_root=cfg_root)
    assert r.source == "scripted"
    assert r.verified is True
    assert r.sentences["en"] and r.sentences["hi"] and r.sentences["kn"]


@pytest.mark.unit
def test_template_fallback_for_unknown_sequence(cfg_root):
    r = compose_sentence(["water", "more"], provider=NullProvider(), config_root=cfg_root)
    assert r.source == "template"
    assert r.verified is False
    assert r.sentences["en"].lower().startswith("water")
    assert r.sentences["hi"] and r.sentences["kn"]  # phrasebook provides both


@pytest.mark.unit
def test_empty_words_gives_empty_template(cfg_root):
    r = compose_sentence([], provider=NullProvider(), config_root=cfg_root)
    assert r.source == "template"
    assert r.sentences == {"en": "", "hi": "", "kn": ""}


@pytest.mark.unit
def test_llm_used_when_safe_and_not_scripted(cfg_root):
    class GoodLLM:
        def compose(self, words, history, scenario_id):
            return {"en": "I want more water.", "hi": "x", "kn": "y", "tone": "neutral"}

    r = compose_sentence(["water", "more"], provider=GoodLLM(), config_root=cfg_root)
    assert r.source == "llm"
    assert r.verified is False
    assert r.sentences["en"] == "I want more water."


@pytest.mark.unit
def test_llm_rejected_when_unsafe_falls_back_to_template(cfg_root):
    # LLM drops the signed word "water" -> safety check fails -> template.
    class DropsWordLLM:
        def compose(self, words, history, scenario_id):
            return {"en": "I am fine.", "hi": "x", "kn": "y"}

    r = compose_sentence(["water", "more"], provider=DropsWordLLM(), config_root=cfg_root)
    assert r.source == "template"


@pytest.mark.unit
def test_llm_exception_falls_back(cfg_root):
    class BrokenLLM:
        def compose(self, words, history, scenario_id):
            raise RuntimeError("network down")

    # "more" is not scripted here -> template after the LLM raises.
    r = compose_sentence(["more"], provider=BrokenLLM(), config_root=cfg_root)
    assert r.source == "template"


@pytest.mark.unit
def test_safety_check_handles_multiword_signs_and_synonyms():
    assert llm_output_is_safe(["thank you"], "Thank you very much.")
    assert not llm_output_is_safe(["thank you"], "You are welcome.")
    assert llm_output_is_safe(["help"], "Please assist me.", {"help": ["assist"]})
