"""Decision engine — ACCEPT / CONFIRM / REJECT (PROJECT_CONTEXT L10).

This is what converts a high *raw* accuracy into 100% *accepted* accuracy: when
the ensemble is confident and everyone agrees, the word is accepted; when it is
only probably-right, the user is asked to tap one of the top-3 ("confirm"); when
it is unsure, the system refuses to guess ("reject"). Uncertain cases become a
human tap instead of a wrong word on screen.

Rules (L10), all thresholds read from ``config/signtalk.yaml`` via
:class:`signtalk_core.config.DecisionConfig` (no magic numbers):

    ACCEPT  if fused top-1 prob   >= t_accept
                AND margin         >= m_accept
                AND model agreement satisfies min_model_agreement
                AND tta_agreement  >= tta_agreement
                (quality gate is checked earlier, in recognize_clip)
    CONFIRM if not accepted but fused top-1 >= t_confirm  (offer top-3 chips)
    REJECT  otherwise  (reason "low_confidence")

``min_model_agreement`` is either ``"all"`` (every model must agree on the
top-1) or an integer ``k`` meaning "at least k of n models agree".

Pure function over a :class:`signtalk_core.fusion.FusionResult`; no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

from signtalk_core.config import DecisionConfig
from signtalk_core.fusion import FusionResult


@dataclass(frozen=True)
class Decision:
    """Outcome of the decision engine.

    Attributes:
        decision: ``"accept"`` | ``"confirm"`` | ``"reject"``.
        reject_reason: ``"low_confidence"`` when rejected here, else ``None``.
            (Quality-gate reject reasons are set earlier in the pipeline.)
    """

    decision: str
    reject_reason: str | None


def _model_agreement_ok(result: FusionResult, min_agreement: str | int) -> bool:
    """True if the ensemble top-1 agreement meets ``min_model_agreement``."""
    if isinstance(min_agreement, str):
        if min_agreement.strip().lower() == "all":
            return result.model_agreement >= result.num_models
        # Unknown string -> be strict (require all) rather than silently loose.
        return result.model_agreement >= result.num_models
    k = int(min_agreement)
    return result.model_agreement >= k


def decide(result: FusionResult, cfg: DecisionConfig) -> Decision:
    """Apply the accept / confirm / reject rules to a fusion result.

    Args:
        result: the fused ensemble result (probabilities + agreement diagnostics).
        cfg: decision thresholds from config.

    Returns:
        A :class:`Decision`.
    """
    accept = (
        result.top1_p >= cfg.t_accept
        and result.margin >= cfg.m_accept
        and _model_agreement_ok(result, cfg.min_model_agreement)
        and result.tta_agreement >= cfg.tta_agreement
    )
    if accept:
        return Decision(decision="accept", reject_reason=None)

    if result.top1_p >= cfg.t_confirm:
        return Decision(decision="confirm", reject_reason=None)

    return Decision(decision="reject", reject_reason="low_confidence")
