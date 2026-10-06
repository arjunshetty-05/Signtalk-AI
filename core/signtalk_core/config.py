"""Typed accessor for config/signtalk.yaml — the single source of tunables.

PROJECT_CONTEXT Sections 6.9, 12.4 and 13: every tunable lives in one YAML
file and callers read it only through this module, so there are no magic
numbers anywhere else in the codebase.

Usage::

    from signtalk_core.config import load_config
    cfg = load_config()                 # finds config/signtalk.yaml
    cfg.sequence_length                 # -> 32
    cfg.quality.min_hands_visible_pct   # -> 0.7
    cfg.decision.t_accept               # -> 0.85

No ML logic lives here; this is pure configuration loading/validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    """Return the worktree/repo root (parent of the ``config/`` directory).

    This file lives at ``<root>/core/signtalk_core/config.py`` so the root is
    three parents up.
    """
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """Return the default path to ``config/signtalk.yaml`` under the repo root."""
    return _repo_root() / "config" / "signtalk.yaml"


@dataclass(frozen=True)
class QualityConfig:
    """Clip quality-gate thresholds (PROJECT_CONTEXT Section 5.3 / L2)."""

    min_hands_visible_pct: float
    min_seconds: float
    max_seconds: float


@dataclass(frozen=True)
class TtaConfig:
    """Test-time augmentation settings (PROJECT_CONTEXT L9)."""

    views: int
    trim_frames: list[int]


@dataclass(frozen=True)
class DecisionConfig:
    """Accept / confirm / reject thresholds (PROJECT_CONTEXT L10).

    NOTE: these are PLACEHOLDER values in Phase 0; tools/pick_thresholds.py
    replaces them later with values computed from the validation
    risk-coverage curve. ``min_model_agreement`` is either the string
    ``"all"`` or an integer ``k`` meaning "k of n models must agree".
    """

    t_accept: float
    m_accept: float
    min_model_agreement: str | int
    tta_agreement: float
    t_confirm: float


@dataclass(frozen=True)
class SentenceConfig:
    """Sentence-buffer behaviour (PROJECT_CONTEXT Section 5.10)."""

    idle_seconds: float
    max_words: int
    mode: str  # "strict" | "fast"


@dataclass(frozen=True)
class ContextPriorConfig:
    """Context/script prior settings (PROJECT_CONTEXT Section 5.9 / L11)."""

    enabled: bool
    scenario: str | None


@dataclass(frozen=True)
class Config:
    """Top-level typed view over config/signtalk.yaml."""

    sequence_length: int
    rgb_frames: int
    smoothing_window: int
    quality: QualityConfig
    tta: TtaConfig
    decision: DecisionConfig
    sentence: SentenceConfig
    context_prior: ContextPriorConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        """Build a Config from a parsed YAML mapping, validating required keys."""
        try:
            quality = QualityConfig(
                min_hands_visible_pct=float(raw["quality"]["min_hands_visible_pct"]),
                min_seconds=float(raw["quality"]["min_seconds"]),
                max_seconds=float(raw["quality"]["max_seconds"]),
            )
            tta = TtaConfig(
                views=int(raw["tta"]["views"]),
                trim_frames=[int(v) for v in raw["tta"]["trim_frames"]],
            )
            decision = DecisionConfig(
                t_accept=float(raw["decision"]["t_accept"]),
                m_accept=float(raw["decision"]["m_accept"]),
                min_model_agreement=raw["decision"]["min_model_agreement"],
                tta_agreement=float(raw["decision"]["tta_agreement"]),
                t_confirm=float(raw["decision"]["t_confirm"]),
            )
            sentence = SentenceConfig(
                idle_seconds=float(raw["sentence"]["idle_seconds"]),
                max_words=int(raw["sentence"]["max_words"]),
                mode=str(raw["sentence"]["mode"]),
            )
            context_prior = ContextPriorConfig(
                enabled=bool(raw["context_prior"]["enabled"]),
                scenario=raw["context_prior"]["scenario"],
            )
            return cls(
                sequence_length=int(raw["sequence_length"]),
                rgb_frames=int(raw["rgb_frames"]),
                smoothing_window=int(raw["smoothing_window"]),
                quality=quality,
                tta=tta,
                decision=decision,
                sentence=sentence,
                context_prior=context_prior,
            )
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"config/signtalk.yaml is missing required key: {exc}") from exc


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the SignTalk config.

    Args:
        path: optional path to a signtalk.yaml file. Defaults to
            ``config/signtalk.yaml`` under the repo root.

    Returns:
        A frozen, typed :class:`Config`.
    """
    cfg_path = Path(path) if path is not None else default_config_path()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} did not parse to a mapping")
    return Config.from_dict(raw)
