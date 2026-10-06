"""Ensemble fusion (PROJECT_CONTEXT 5.6 / L8) + optional context prior (L11).

Combines the calibrated probability vectors of several models (and several TTA
views) into one fused distribution, and reports the diagnostics the decision
engine needs: fused probabilities, per-model top-1 agreement, TTA agreement and
the margin.

Fusion is a weighted average of calibrated probabilities (Section 5.6). Weights
are tuned on validation and stored in config; equal weights are the default.
The optional context prior (L11) multiplies the fused probabilities by a prior
over the vocabulary and renormalises — OFF by default and always logged so
results can be reported with it on AND off.

Pure NumPy; no torch. The per-model forward passes happen in the caller
(serving/eval), which hands this module the already-computed probability arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusionResult:
    """Everything the decision engine (L10) needs from fusion.

    Attributes:
        probs: fused probability vector, float64 ``[C]`` (sums to 1).
        top1: index of the highest fused probability.
        top1_p: the fused probability of ``top1``.
        margin: ``top1_p - top2_p`` (0 if only one class).
        model_agreement: number of models whose own top-1 equals ``top1``.
        num_models: total number of models fused.
        tta_agreement: fraction of TTA views agreeing with their majority top-1.
    """

    probs: np.ndarray
    top1: int
    top1_p: float
    margin: float
    model_agreement: int
    num_models: int
    tta_agreement: float


def fuse_probabilities(
    per_model_probs: list[np.ndarray],
    weights: list[float] | None = None,
    tta_agreement: float = 1.0,
) -> FusionResult:
    """Fuse per-model probability vectors by a weighted average.

    Args:
        per_model_probs: list of float ``[C]`` calibrated probability vectors,
            one per model. Each view's probabilities should already be averaged
            into its model's vector by the caller before fusion, OR each entry
            may itself be a per-view mean — either way each entry is one ``[C]``
            distribution.
        weights: optional per-model weights (need not sum to 1; they are
            normalised). Defaults to equal weights.
        tta_agreement: the TTA agreement fraction to carry into the result
            (computed by :func:`signtalk_core.tta.tta_agreement_fraction`).

    Returns:
        A :class:`FusionResult`.

    Raises:
        ValueError: if ``per_model_probs`` is empty or shapes disagree.
    """
    if not per_model_probs:
        raise ValueError("per_model_probs is empty")
    mats = [np.asarray(p, dtype=np.float64).ravel() for p in per_model_probs]
    c = mats[0].shape[0]
    for m in mats:
        if m.shape[0] != c:
            raise ValueError("per-model probability vectors differ in length")

    n = len(mats)
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        if len(weights) != n:
            raise ValueError("weights length must match number of models")
        w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    fused = np.zeros(c, dtype=np.float64)
    for weight, m in zip(w, mats):
        fused += weight * m
    fused = fused / fused.sum()  # renormalise against float drift

    top1 = int(np.argmax(fused))
    order = np.argsort(fused)[::-1]
    top1_p = float(fused[order[0]])
    top2_p = float(fused[order[1]]) if c > 1 else 0.0
    margin = top1_p - top2_p

    per_model_top1 = [int(np.argmax(m)) for m in mats]
    model_agreement = sum(1 for t in per_model_top1 if t == top1)

    return FusionResult(
        probs=fused,
        top1=top1,
        top1_p=top1_p,
        margin=margin,
        model_agreement=model_agreement,
        num_models=n,
        tta_agreement=float(tta_agreement),
    )


def apply_context_prior(
    probs: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    """Multiply fused probabilities by a context prior and renormalise (L11).

    Args:
        probs: float ``[C]`` fused probabilities.
        prior: float ``[C]`` non-negative prior weights over the vocabulary
            (e.g. allowed-next-sign soft weights from scenarios.json).

    Returns:
        The re-normalised float64 ``[C]`` posterior. If the prior zeroes out
        every class that had mass, the original ``probs`` are returned unchanged
        (defensive: never return an all-zero distribution).
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    prior = np.asarray(prior, dtype=np.float64).ravel()
    if prior.shape != probs.shape:
        raise ValueError("prior and probs must have the same shape")
    posterior = probs * prior
    s = posterior.sum()
    if s <= 0:
        return probs
    return posterior / s
