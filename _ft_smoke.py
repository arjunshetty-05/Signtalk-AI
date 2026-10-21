"""Throwaway smoke test for finetune_enrolled internals (deleted after run).

Exercises session_split + finetune_member against the REAL trained GRU
checkpoint using synthetic feature items, so we verify warm-start loading, the
train loop, and session-aware splitting without needing recorded clips.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from signtalk_core.features import feature_dim
from training.finetune_enrolled import _FeatureItem, finetune_member, session_split

T = 32
F = feature_dim()

with open("models/ensemble.json", "r", encoding="utf-8") as fh:
    manifest = json.load(fh)
num_classes = len(manifest["labels"])
print("num_classes", num_classes, "feature_dim", F)

rng = np.random.default_rng(0)
# Two sessions, a few clips per class in each -> should trigger session-holdout.
items: list[_FeatureItem] = []
for session in ("s1", "s2"):
    for cls in range(num_classes):
        for _ in range(2):
            feats = rng.standard_normal((T, F)).astype("float32")
            items.append(_FeatureItem(feats, cls, session))

train, val, mode = session_split(items, seed=0)
print("split_mode:", mode, "train", len(train), "val", len(val))
assert "session-holdout" in mode, mode
assert train and val

# Single-session fallback path.
single = [it for it in items if it.session == "s1"]
_, _, mode2 = session_split(single, seed=0)
print("single-session mode:", mode2)
assert "stratified-fallback" in mode2, mode2

# Warm-start + fine-tune the real GRU checkpoint for 2 epochs.
gru_ckpt = Path("runs/m1_gru/best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
rep = finetune_member(
    "gru", gru_ckpt, num_classes, train, val, Path("runs/_smoke_ft/gru.pt"),
    epochs=2, lr=2e-4, weight_decay=1e-4, label_smoothing=0.1,
    patience=5, seed=0, device=device,
)
print("member report:", json.dumps(rep, indent=2))
assert Path(rep["checkpoint"]).is_file(), "checkpoint not written"
assert rep["warm_start_from"] == str(gru_ckpt)
print("SMOKE PASS")
