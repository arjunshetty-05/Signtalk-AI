"""Unit tests for fusion + decision engine (L8/L10)."""

from __future__ import annotations

import numpy as np
import pytest

from signtalk_core.config import DecisionConfig
from signtalk_core.decision import decide
from signtalk_core.fusion import apply_context_prior, fuse_probabilities
from signtalk_core.tta import tta_agreement_fraction


def _decision_cfg(**over) -> DecisionConfig:
    base = dict(
        t_accept=0.85, m_accept=0.30, min_model_agreement="all",
        tta_agreement=0.8, t_confirm=0.40,
    )
    base.update(over)
    return DecisionConfig(**base)


@pytest.mark.unit
def test_fuse_equal_weights_averages():
    a = np.array([0.8, 0.1, 0.1])
    b = np.array([0.6, 0.3, 0.1])
    r = fuse_probabilities([a, b])
    assert r.top1 == 0
    assert r.num_models == 2
    assert r.model_agreement == 2            # both models' top-1 is class 0
    assert np.isclose(r.probs.sum(), 1.0)
    assert np.isclose(r.probs[0], 0.7)       # (0.8 + 0.6) / 2


@pytest.mark.unit
def test_fuse_detects_disagreement():
    a = np.array([0.7, 0.2, 0.1])  # top-1 = 0
    b = np.array([0.2, 0.7, 0.1])  # top-1 = 1
    r = fuse_probabilities([a, b])
    assert r.model_agreement == 1            # only one model agrees with fused top-1


@pytest.mark.unit
def test_decide_accept_when_confident_and_unanimous():
    r = fuse_probabilities([np.array([0.95, 0.03, 0.02]), np.array([0.92, 0.05, 0.03])],
                           tta_agreement=1.0)
    out = decide(r, _decision_cfg())
    assert out.decision == "accept"
    assert out.reject_reason is None


@pytest.mark.unit
def test_decide_confirm_on_middling_confidence():
    # top-1 ~0.5: below t_accept (0.85) but above t_confirm (0.40) -> confirm.
    r = fuse_probabilities([np.array([0.5, 0.4, 0.1])], tta_agreement=1.0)
    out = decide(r, _decision_cfg())
    assert out.decision == "confirm"


@pytest.mark.unit
def test_decide_reject_when_unsure():
    r = fuse_probabilities([np.array([0.34, 0.33, 0.33])], tta_agreement=1.0)
    out = decide(r, _decision_cfg())
    assert out.decision == "reject"
    assert out.reject_reason == "low_confidence"


@pytest.mark.unit
def test_decide_reject_when_margin_too_small():
    # High top-1 but a near-tie second place -> margin fails -> not accept.
    r = fuse_probabilities([np.array([0.5, 0.49, 0.01])], tta_agreement=1.0)
    out = decide(r, _decision_cfg(t_accept=0.4))
    assert out.decision in {"confirm", "reject"}
    assert out.decision != "accept"


@pytest.mark.unit
def test_decide_blocks_accept_on_model_disagreement():
    a = np.array([0.9, 0.08, 0.02])
    b = np.array([0.1, 0.88, 0.02])  # disagrees
    r = fuse_probabilities([a, b], tta_agreement=1.0)
    # fused top-1 may still be confident, but models disagree -> no accept.
    out = decide(r, _decision_cfg(t_accept=0.4, m_accept=0.0))
    assert out.decision != "accept"


@pytest.mark.unit
def test_context_prior_renormalises_and_can_flip():
    probs = np.array([0.5, 0.3, 0.2])
    prior = np.array([0.0, 1.0, 1.0])  # class 0 disallowed by context
    post = apply_context_prior(probs, prior)
    assert np.isclose(post.sum(), 1.0)
    assert post[0] == 0.0
    assert np.argmax(post) == 1


@pytest.mark.unit
def test_context_prior_defensive_on_all_zero():
    probs = np.array([0.5, 0.5])
    post = apply_context_prior(probs, np.array([0.0, 0.0]))
    # Must not return an all-zero distribution.
    assert np.isclose(post.sum(), 1.0)


@pytest.mark.unit
def test_tta_agreement_fraction():
    assert tta_agreement_fraction([3, 3, 3, 1, 2]) == pytest.approx(3 / 5)
    assert tta_agreement_fraction([]) == 0.0
    assert tta_agreement_fraction([7]) == 1.0
