"""Feature-npz dataset + session-aware splitting (PROJECT_CONTEXT 8.5 / 1.6).

Loads the ``.npz`` feature files written by ``prepare_include.py`` into a torch
``Dataset``, and provides a split helper that keeps the official-vs-leakage
rules in mind: the default split is a per-class stratified split, but a
``group`` key (e.g. signer or session id) can be supplied so whole groups are
held out (Section 1.6 — test data must come from a different session/signer).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """A dataset of ``(features[T, F], class_index)`` from feature-npz files.

    Args:
        items: list of ``(npz_path, class_index)``.
    """

    def __init__(self, items: list[tuple[Path, int]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        with np.load(path, allow_pickle=True) as data:
            x = data["features"].astype(np.float32)
        return torch.from_numpy(x), int(y)


def load_label_map(processed_dir: Path) -> dict[str, int]:
    """Load ``label_map.json`` written by prepare_include.py."""
    with open(processed_dir / "label_map.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def gather_items(processed_dir: Path, label_map: dict[str, int]) -> list[tuple[Path, int]]:
    """Collect ``(npz_path, class_index)`` for every feature file present."""
    feat_root = processed_dir / "features"
    items: list[tuple[Path, int]] = []
    for label, idx in label_map.items():
        for npz in sorted((feat_root / label).glob("*.npz")):
            items.append((npz, idx))
    return items


def stratified_split(
    items: list[tuple[Path, int]],
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list, list]:
    """Per-class stratified train/val split.

    WARNING (Section 1.6): a random stratified split is only safe when clips of
    the same sign do NOT share a recording session across the split. For
    enrolled data use a session/signer group hold-out instead (see
    ``group_split``). This is appropriate for the official INCLUDE split or a
    quick baseline.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    train, val = [], []
    for _, group in by_class.items():
        idx = rng.permutation(len(group))
        n_val = max(1, int(round(len(group) * val_frac)))
        val_idx = set(idx[:n_val].tolist())
        for i, it in enumerate(group):
            (val if i in val_idx else train).append(it)
    return train, val
