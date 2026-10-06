"""Probability calibration — temperature scaling (PROJECT_CONTEXT 5.6 / L8).

A model's raw softmax is usually over-confident, so "90% sure" is not right 90%
of the time. Temperature scaling divides the logits by a single learned scalar
``T`` before softmax, which fixes the confidences without changing which class
is top-1 (so accuracy is unchanged — only the probabilities are calibrated).

``T`` is fit on VALIDATION data (never test — Section 1.6) by minimising the
negative log-likelihood. Each ensemble member gets its own ``T`` so their
calibrated probabilities are comparable before fusion.

Pure NumPy: no torch dependency, so this is cheap to unit test and to apply at
serving time.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax over ``axis``.

    Args:
        logits: float array of any shape.
        axis: axis to normalise over.

    Returns:
        Probabilities of the same shape as ``logits``.
    """
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling then softmax.

    Args:
        logits: float32 ``[N, C]`` (or ``[C]``) raw logits.
        temperature: positive scalar ``T``; ``T>1`` softens, ``T<1`` sharpens.

    Returns:
        Calibrated probabilities, same shape as ``logits``.

    Raises:
        ValueError: if ``temperature`` is not positive.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood of ``labels`` under temperature-scaled probs."""
    probs = apply_temperature(logits, temperature)
    n = labels.shape[0]
    picked = probs[np.arange(n), labels]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


def fit_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 10.0,
    steps: int = 200,
) -> float:
    """Fit a single temperature on validation data by grid + local search.

    A simple, dependency-free optimiser: scan a log-spaced grid for the NLL
    minimum, then refine once around it. This is accurate enough for
    calibration (the NLL vs T curve is smooth and unimodal) and avoids pulling
    in scipy.

    Args:
        val_logits: float ``[N, C]`` validation logits.
        val_labels: int ``[N]`` ground-truth class indices.
        t_min, t_max: search bounds for ``T``.
        steps: grid resolution.

    Returns:
        The fitted temperature ``T`` (a positive float).
    """
    val_logits = np.asarray(val_logits, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    if val_logits.ndim != 2:
        raise ValueError("val_logits must be [N, C]")
    if val_logits.shape[0] != val_labels.shape[0]:
        raise ValueError("logits and labels disagree on N")

    grid = np.geomspace(t_min, t_max, steps)
    nlls = [_nll(val_logits, val_labels, float(t)) for t in grid]
    best_i = int(np.argmin(nlls))

    # Local refinement between the two neighbours of the grid minimum.
    lo = grid[max(best_i - 1, 0)]
    hi = grid[min(best_i + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, steps)
    fine_nlls = [_nll(val_logits, val_labels, float(t)) for t in fine]
    return float(fine[int(np.argmin(fine_nlls))])
