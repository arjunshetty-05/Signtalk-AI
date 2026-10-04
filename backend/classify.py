"""
classify.py — SignTalk AI / Person A (ML Core)

Single source of truth for classify_sequence(), extracted out of main.py so
both the flat Prompt-A1 app and Person C's restructured app/pose/service.py
import the exact same function (no drift between the two).

Contract (locked team-wide, see Person C's Prompt C1):
    classify_sequence(sequence: np.ndarray) -> {"label": str, "confidence": float}
    Input shape: (30, 17, 2), dtype float32.
"""

from __future__ import annotations

import os

import numpy as np

SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17

_MODEL_PATH_ENV = "SIGNTALK_BILSTM_SAVEDMODEL"
_LABELS_PATH_ENV = "SIGNTALK_LABELS_JSON"

_model = None
_labels: dict[str, str] | None = None


def _try_load_trained_model():
    """Lazily loads a trained SavedModel + labels.json if the environment
    points at one. Falls back silently to the placeholder if not configured
    or if loading fails — this is what makes swapping in the trained model
    a zero-code-change operation (just set the env vars)."""
    global _model, _labels
    if _model is not None:
        return

    model_path = os.environ.get(_MODEL_PATH_ENV)
    labels_path = os.environ.get(_LABELS_PATH_ENV)
    if not model_path or not labels_path or not os.path.exists(model_path):
        return

    try:
        import json
        import tensorflow as tf

        _model = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
        with open(labels_path) as f:
            _labels = json.load(f)
    except Exception:
        _model = None
        _labels = None


def classify_sequence(sequence: np.ndarray) -> dict:
    """
    Input:  np.ndarray, shape (30, 17, 2), dtype float32 — a full sequence
            buffer of smoothed, normalized keypoints.
    Output: dict with exactly two keys: {"label": str, "confidence": float}

    Uses the trained BiLSTM SavedModel if SIGNTALK_BILSTM_SAVEDMODEL and
    SIGNTALK_LABELS_JSON env vars point at one; otherwise returns the
    documented PLACEHOLDER stub so the rest of the pipeline is fully
    runnable before training completes.
    """
    assert sequence.shape == (SEQUENCE_LENGTH, NUM_KEYPOINTS, 2), (
        f"classify_sequence expected shape (30, 17, 2), got {sequence.shape}"
    )

    _try_load_trained_model()
    if _model is None or _labels is None:
        return {"label": "PLACEHOLDER", "confidence": 0.0}

    flat = sequence.reshape(1, SEQUENCE_LENGTH, NUM_KEYPOINTS * 2).astype(np.float32)
    output = _model(flat)
    probs = list(output.values())[0].numpy()[0] if isinstance(output, dict) else output.numpy()[0]
    class_idx = int(np.argmax(probs))
    return {"label": _labels.get(str(class_idx), f"class_{class_idx}"), "confidence": float(probs[class_idx])}
