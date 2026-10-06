"""Unit tests for risk-coverage threshold selection (L10 / Section 1.3)."""

from __future__ import annotations

import numpy as np
import pytest

from signtalk_core.risk_coverage import pick_thresholds


@pytest.mark.unit
def test_zero_error_threshold_separates_confident_correct_from_wrong():
    # Build a validation set where confident predictions are correct and
    # low-confidence ones are wrong -> a threshold exists giving 0 accepted error.
    rows = []
    labels = []
    # 20 confident-correct (class 0): high top-1, big margin
    for _ in range(20):
        rows.append([0.95, 0.03, 0.02])
        labels.append(0)
    # 10 unsure-wrong: top-1 barely ahead, and WRONG (true label 1)
    for _ in range(10):
        rows.append([0.4, 0.35, 0.25])
        labels.append(1)
    probs = np.array(rows)
    ys = np.array(labels)

    choice = pick_thresholds(probs, ys)
    assert choice.zero_error is True
    assert choice.accepted_accuracy == pytest.approx(1.0)
    # It should accept the 20 confident-correct and exclude the 10 unsure-wrong.
    assert choice.n_accepted == 20
    assert choice.coverage == pytest.approx(20 / 30)


@pytest.mark.unit
def test_fallback_when_no_zero_error_possible():
    # Every item is confidently WRONG at every threshold -> no zero-error pair.
    rows = [[0.99, 0.005, 0.005] for _ in range(10)]
    ys = np.ones(10, dtype=int)  # true class is 1, model always says 0
    choice = pick_thresholds(np.array(rows), ys)
    assert choice.zero_error is False
    assert choice.accepted_accuracy < 1.0


@pytest.mark.unit
def test_shape_validation():
    with pytest.raises(ValueError):
        pick_thresholds(np.zeros((0, 3)), np.array([]))
    with pytest.raises(ValueError):
        pick_thresholds(np.zeros((3, 3)), np.array([0, 1]))
