"""Build the serving recognizer (detector + ensemble bundle) at startup.

Keeps model/detector loading out of the request path and out of
``signtalk_core`` (which stays free of filesystem/model-download side effects).
Everything here is driven by environment variables so the same server binary
runs in two modes:

  * **Skeleton mode (default, no artifacts):** no ``.task`` bundle and no
    trained checkpoints -> returns ``(None, None)``. ``recognize_clip`` then
    runs its Phase-1 fallback (no-detector reject / single random model), so the
    contract shape still works with nothing installed.
  * **Trained mode:** once you have run the training scripts and downloaded the
    MediaPipe bundle, point the env vars at them and the server loads a real
    :class:`signtalk_core.recognize.RecognizerBundle` (ensemble + per-model
    temperatures + fusion weights) and a MediaPipe detector.

Environment variables (all optional):
    HOLISTIC_MODEL_PATH   path to models/holistic_landmarker.task
    RECOGNIZER_MANIFEST   path to a JSON manifest describing the ensemble:
        {
          "labels": ["hello", "thank you", ...],
          "members": [
            {"arch": "gru",         "checkpoint": "models/m1_gru.pt",
             "temperature": 1.3, "weight": 0.5},
            {"arch": "transformer", "checkpoint": "models/m2_tf.pt",
             "temperature": 0.9, "weight": 0.5}
          ]
        }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("signtalk.server.recognizer")


def load_detector() -> Any | None:
    """Create the MediaPipe Pose+Hand detector if both bundles are present.

    Paths come from ``POSE_MODEL_PATH`` / ``HAND_MODEL_PATH`` (defaulting to
    ``models/pose_landmarker.task`` and ``models/hand_landmarker.task``). If
    either is missing, returns None and the server runs in skeleton mode.
    """
    pose_path = os.environ.get("POSE_MODEL_PATH", "models/pose_landmarker.task")
    hand_path = os.environ.get("HAND_MODEL_PATH", "models/hand_landmarker.task")
    if not (Path(pose_path).is_file() and Path(hand_path).is_file()):
        logger.info(
            "pose/hand bundles not found (%s, %s) -> no detector (skeleton mode)",
            pose_path,
            hand_path,
        )
        return None
    from signtalk_core.landmarks import create_holistic_detector

    logger.info("loading MediaPipe Pose+Hand detector from %s + %s", pose_path, hand_path)
    return create_holistic_detector(pose_path, hand_path)


def load_bundle() -> Any | None:
    """Build a RecognizerBundle from the manifest if one is configured."""
    manifest_path = os.environ.get("RECOGNIZER_MANIFEST")
    if not manifest_path:
        logger.info("no RECOGNIZER_MANIFEST set -> single random-init fallback model")
        return None
    mp = Path(manifest_path)
    if not mp.is_file():
        logger.warning("RECOGNIZER_MANIFEST=%s does not exist -> fallback model", manifest_path)
        return None

    import torch

    from signtalk_core.models.pose_tcn import build_model
    from signtalk_core.models.pose_transformer import build_transformer
    from signtalk_core.recognize import RecognizerBundle

    with open(mp, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    labels = spec["labels"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, temps, weights = [], [], []
    for member in spec["members"]:
        arch = member["arch"]
        ckpt = member.get("checkpoint")
        if arch == "gru":
            model = build_model(len(labels), checkpoint_path=ckpt, device=device)
        elif arch == "transformer":
            model = build_transformer(len(labels), checkpoint_path=ckpt, device=device)
        else:
            raise ValueError(f"unknown arch in manifest: {arch!r}")
        models.append(model)
        temps.append(float(member.get("temperature", 1.0)))
        weights.append(float(member.get("weight", 1.0)))

    logger.info("loaded ensemble: %d members, device=%s", len(models), device)
    return RecognizerBundle(
        models=models, labels=labels, temperatures=temps, weights=weights, device=device
    )
