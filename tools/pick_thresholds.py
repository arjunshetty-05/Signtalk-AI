"""Pick decision thresholds from the validation risk-coverage curve (L10 / 5.7).

Runs the trained ensemble over a VALIDATION split, fuses calibrated
probabilities exactly as serving does, then calls
:func:`signtalk_core.risk_coverage.pick_thresholds` to choose the loosest
``t_accept`` / ``m_accept`` that give zero accepted errors (maximising coverage).
The chosen values are written back into ``config/signtalk.yaml`` and a human
report is written next to the run.

Run on YOUR machine after training (Phase 2) and BEFORE Gate 3. Never run it on
the test split — thresholds are tuned on validation only (Section 1.6).

Example::

    python tools/pick_thresholds.py \
        --processed-dir data/processed/include \
        --manifest models/ensemble.json \
        --report runs/thresholds_report.json \
        --write-config

The manifest is the same JSON the server loads (see server/app/recognizer.py):
a ``labels`` list and ``members`` with arch/checkpoint/temperature/weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running as a plain script: ensure the repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from signtalk_core.calibration import apply_temperature  # noqa: E402
from signtalk_core.config import default_config_path  # noqa: E402
from signtalk_core.fusion import fuse_probabilities  # noqa: E402
from signtalk_core.models.pose_tcn import build_model  # noqa: E402
from signtalk_core.models.pose_transformer import build_transformer  # noqa: E402
from signtalk_core.risk_coverage import pick_thresholds  # noqa: E402
from training.dataset import (  # noqa: E402
    FeatureDataset,
    gather_items,
    load_label_map,
    stratified_split,
)


def _build_members(manifest: dict, device: str):
    labels = manifest["labels"]
    members = []
    for m in manifest["members"]:
        if m["arch"] == "gru":
            model = build_model(len(labels), checkpoint_path=m.get("checkpoint"), device=device)
        elif m["arch"] == "transformer":
            model = build_transformer(len(labels), checkpoint_path=m.get("checkpoint"), device=device)
        else:
            raise ValueError(f"unknown arch: {m['arch']!r}")
        members.append((model, float(m.get("temperature", 1.0)), float(m.get("weight", 1.0))))
    return labels, members


@torch.no_grad()
def _fused_val_probs(members, items, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (fused_probs [N, C], labels [N]) over a validation item list."""
    loader = FeatureDataset(items)
    fused_rows, label_rows = [], []
    for i in range(len(loader)):
        x, y = loader[i]
        xb = x.unsqueeze(0).to(device)  # [1, T, F]
        per_model = []
        weights = []
        for model, temp, weight in members:
            logits = model(xb).cpu().numpy()[0]  # [C]
            per_model.append(apply_temperature(logits, temp))
            weights.append(weight)
        fused = fuse_probabilities(per_model, weights=weights)
        fused_rows.append(fused.probs)
        label_rows.append(y)
    return np.stack(fused_rows), np.asarray(label_rows, dtype=np.int64)


def _write_config_thresholds(t_accept: float, m_accept: float) -> Path:
    """Patch decision.t_accept / decision.m_accept in config/signtalk.yaml."""
    path = default_config_path()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["decision"]["t_accept"] = round(float(t_accept), 3)
    raw["decision"]["m_accept"] = round(float(m_accept), 3)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False, allow_unicode=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pick decision thresholds from validation")
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path, help="ensemble JSON (recognizer manifest)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", type=Path, default=Path("runs/thresholds_report.json"))
    ap.add_argument("--write-config", action="store_true", help="patch config/signtalk.yaml")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    labels, members = _build_members(manifest, device)

    label_map = load_label_map(args.processed_dir)
    items = gather_items(args.processed_dir, label_map)
    if not items:
        print("[fail] no feature .npz found — run prepare_include.py first")
        return 1
    _, val_items = stratified_split(items, args.val_frac, args.seed)
    print(f"[info] validation items={len(val_items)} classes={len(labels)} device={device}")

    probs, ys = _fused_val_probs(members, val_items, device)
    choice = pick_thresholds(probs, ys)

    report = {
        "t_accept": choice.t_accept,
        "m_accept": choice.m_accept,
        "coverage": round(choice.coverage, 4),
        "accepted_accuracy": round(choice.accepted_accuracy, 4),
        "n_accepted": choice.n_accepted,
        "n_total": choice.n_total,
        "zero_accepted_error": choice.zero_error,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "note": (
            "Thresholds chosen on the VALIDATION split only (Section 1.6). Verify "
            "on the untouched TEST session before quoting a number. If "
            "zero_accepted_error is false, no threshold gave 0 wrong accepts — "
            "inspect the confusion matrix and drop confusable signs (L1)."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    if not choice.zero_error:
        print("[warn] NO zero-error threshold found — see note; do not claim 100% accepted.")
    if args.write_config:
        p = _write_config_thresholds(choice.t_accept, choice.m_accept)
        print(f"[ok  ] wrote t_accept/m_accept into {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
