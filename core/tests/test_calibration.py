"""Unit tests for temperature-scaling calibration (signtalk_core.calibration)."""

from __future__ import annotations

import numpy as np
import pytest

from signtalk_core.calibration import apply_temperature, fit_temperature, softmax


@pytest.mark.unit
def test_softmax_sums_to_one():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p > 0)


@pytest.mark.unit
def test_apply_temperature_preserves_argmax():
    logits = np.array([[2.0, 1.0, 0.1]])
    cold = apply_temperature(logits, 0.5)
    hot = apply_temperature(logits, 5.0)
    # Temperature never changes which class is top-1.
    assert np.argmax(cold) == np.argmax(logits)
    assert np.argmax(hot) == np.argmax(logits)
    # Hotter temperature -> less peaky (lower max probability).
    assert hot.max() < cold.max()


@pytest.mark.unit
def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_temperature(np.array([1.0, 2.0]), 0.0)


@pytest.mark.unit
def test_fit_temperature_softens_overconfident_logits():
    # Build over-confident logits that are often WRONG, so NLL wants T>1.
    rng = np.random.default_rng(0)
    n, c = 300, 5
    labels = rng.integers(0, c, size=n)
    logits = np.full((n, c), -5.0)
    # Put a huge logit on a (frequently wrong) predicted class.
    pred = rng.integers(0, c, size=n)
    logits[np.arange(n), pred] = 10.0
    t = fit_temperature(logits, labels)
    assert t > 1.0  # calibration should soften the over-confidence
