"""Risk-coverage threshold selection (PROJECT_CONTEXT L10 / Section 1.3 / 5.7).

The decision engine (``signtalk_core.decision``) accepts a prediction only when
the fused top-1 probability and margin clear thresholds ``t_accept`` /
``m_accept``. Those thresholds must NOT be guessed or tuned on test data — they
are chosen from the VALIDATION risk-coverage curve (Section 1.6): the loosest
thresholds that still give ZERO accepted errors, maximising coverage.

This module is the pure, reusable core of that search. ``tools/pick_thresholds.py``
feeds it validation (fused_probs, true_label) pairs and writes the chosen values
back into ``config/signtalk.yaml`` plus a report.

Definitions (Section 1.3):
    * accepted = the engine auto-accepts (top1_p >= t_accept AND margin >= m_accept)
    * accepted accuracy = fraction of accepted items that are correct
    * coverage = fraction of ALL items that are accepted
The goal: the (t_accept, m_accept) with coverage as high as possible subject to
accepted accuracy == 1.0 (no wrong word is ever spoken).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdChoice:
    """Chosen thresholds and the validation metrics behind them.

    Attributes:
        t_accept: chosen top-1 probability threshold.
        m_accept: chosen margin threshold.
        coverage: fraction of validation items accepted at this threshold.
        accepted_accuracy: accuracy among accepted items (1.0 by construction
            when ``zero_error`` is True).
        n_accepted: number of accepted validation items.
        n_total: total validation items.
        zero_error: whether a zero-accepted-error threshold was found at all.
    """

    t_accept: float
    m_accept: float
    coverage: float
    accepted_accuracy: float
    n_accepted: int
    n_total: int
    zero_error: bool


def _top1_and_margin(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_label, top1_p, margin) for each row of a ``[N, C]`` prob matrix."""
    order = np.argsort(probs, axis=1)
    pred = order[:, -1]
    top1_p = probs[np.arange(len(probs)), order[:, -1]]
    if probs.shape[1] > 1:
        top2_p = probs[np.arange(len(probs)), order[:, -2]]
    else:
        top2_p = np.zeros(len(probs))
    return pred, top1_p, top1_p - top2_p


def pick_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    t_grid: np.ndarray | None = None,
    m_grid: np.ndarray | None = None,
) -> ThresholdChoice:
    """Pick (t_accept, m_accept) maximising coverage at zero accepted error.

    Searches a grid of thresholds; among all (t, m) that produce ZERO wrong
    accepts on validation, returns the pair with the highest coverage (ties
    broken toward the lower t, then lower m, i.e. the loosest). If no pair gives
    zero error (e.g. two classes are genuinely indistinguishable at every
    threshold), falls back to the pair with the best accepted accuracy and marks
    ``zero_error=False`` so the caller can warn.

    Args:
        val_probs: float ``[N, C]`` fused probabilities on the validation set.
        val_labels: int ``[N]`` true class indices.
        t_grid: candidate ``t_accept`` values (default 0.50..0.99 step 0.01).
        m_grid: candidate ``m_accept`` values (default 0.00..0.60 step 0.02).

    Returns:
        A :class:`ThresholdChoice`.

    Raises:
        ValueError: on shape mismatch or empty input.
    """
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    if val_probs.ndim != 2 or val_probs.shape[0] == 0:
        raise ValueError("val_probs must be a non-empty [N, C] array")
    if val_probs.shape[0] != val_labels.shape[0]:
        raise ValueError("val_probs and val_labels disagree on N")

    if t_grid is None:
        t_grid = np.round(np.arange(0.50, 0.991, 0.01), 3)
    if m_grid is None:
        m_grid = np.round(np.arange(0.00, 0.601, 0.02), 3)

    pred, top1_p, margin = _top1_and_margin(val_probs)
    correct = pred == val_labels
    n_total = len(val_labels)

    best_zero: ThresholdChoice | None = None
    best_any: ThresholdChoice | None = None

    for t in t_grid:
        for m in m_grid:
            accepted = (top1_p >= t) & (margin >= m)
            n_acc = int(accepted.sum())
            if n_acc == 0:
                continue
            acc_correct = int(correct[accepted].sum())
            acc_accuracy = acc_correct / n_acc
            coverage = n_acc / n_total
            choice = ThresholdChoice(
                t_accept=float(t),
                m_accept=float(m),
                coverage=coverage,
                accepted_accuracy=acc_accuracy,
                n_accepted=n_acc,
                n_total=n_total,
                zero_error=acc_accuracy >= 1.0,
            )
            # Track the best zero-error choice (max coverage, loosest ties).
            if acc_accuracy >= 1.0:
                if best_zero is None or coverage > best_zero.coverage:
                    best_zero = choice
            # Track the best overall by accuracy then coverage, as a fallback.
            if best_any is None or (acc_accuracy, coverage) > (
                best_any.accepted_accuracy,
                best_any.coverage,
            ):
                best_any = choice

    if best_zero is not None:
        return best_zero
    # No zero-error threshold existed; return the best-accuracy fallback.
    assert best_any is not None  # at least one (t, m) accepted >=1 item
    return best_any
