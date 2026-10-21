"""Offline evaluation harness (PROJECT_CONTEXT Section 9 / Gate 6).

Runs the trained ensemble over a held-out split using the SAME calibration +
fusion path as serving (never re-implemented — Section 12), and reports the
paper-grade metric set:

* top-1 / top-5 accuracy (micro),
* macro-F1 and a per-class precision/recall/F1 table,
* the full confusion matrix,
* decision-layer coverage + accepted-accuracy at the configured thresholds
  (so the risk-coverage story is reproducible), and
* an optional open-set wrong-accept check: feed an UNKNOWN set (signs the model
  was NOT trained on, or idle motion) and report how many get *accepted* — the
  target is zero (Section 9.5).

It writes a machine-readable ``runs/evaluation.json`` and a human-readable
``docs/EVALUATION.md`` (Gate 6 asset). Every number is traceable to the split
seed, the manifest, and the config thresholds recorded in the report.

HONESTY (Section 1.6 / 0.7): a stratified split that shares a signer across
train and val is a *pipeline-validation* number, NOT a generalisation claim.
This tool records which split mode was used and labels leaky numbers as such;
it NEVER fabricates an accuracy figure. For the defensible number, point
``--processed-dir`` at the official INCLUDE test split or an enrolled
session/signer hold-out and pass ``--split official`` so no train/val mixing
happens.

Example::

    python tools/evaluate.py \
        --processed-dir data/processed/include \
        --manifest models/ensemble.json \
        --split stratified --val-frac 0.2 --seed 0 \
        --report runs/evaluation.json --markdown docs/EVALUATION.md
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
from training.dataset import (  # noqa: E402
    FeatureDataset,
    gather_items,
    load_label_map,
    stratified_split,
)


# --------------------------------------------------------------------------- #
# Model loading (mirrors tools/pick_thresholds.py / server/app/recognizer.py)
# --------------------------------------------------------------------------- #
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
def _fused_probs(members, items, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (fused_probs [N, C], labels [N]) over an item list.

    Uses the identical calibrate -> weighted-fuse path as serving so the
    reported numbers match what the running system would produce.
    """
    loader = FeatureDataset(items)
    fused_rows, label_rows = [], []
    for i in range(len(loader)):
        x, y = loader[i]
        xb = x.unsqueeze(0).to(device)  # [1, T, F]
        per_model, weights = [], []
        for model, temp, weight in members:
            logits = model(xb).cpu().numpy()[0]  # [C]
            per_model.append(apply_temperature(logits, temp))
            weights.append(weight)
        fused = fuse_probabilities(per_model, weights=weights)
        fused_rows.append(fused.probs)
        label_rows.append(y)
    return np.stack(fused_rows), np.asarray(label_rows, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Metrics (pure NumPy)
# --------------------------------------------------------------------------- #
def _top_k_accuracy(probs: np.ndarray, ys: np.ndarray, k: int) -> float:
    if probs.shape[0] == 0:
        return 0.0
    k = min(k, probs.shape[1])
    topk = np.argsort(probs, axis=1)[:, -k:]  # [N, k]
    hits = np.any(topk == ys[:, None], axis=1)
    return float(np.mean(hits))


def _confusion_matrix(preds: np.ndarray, ys: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(ys, preds):
        cm[int(t), int(p)] += 1
    return cm


def _per_class_prf(cm: np.ndarray) -> list[dict]:
    """Precision/recall/F1 and support per class from a confusion matrix."""
    rows = []
    n = cm.shape[0]
    for c in range(n):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "class_index": c,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )
    return rows


def _macro_f1(per_class: list[dict]) -> float:
    vals = [r["f1"] for r in per_class if r["support"] > 0]
    return float(np.mean(vals)) if vals else 0.0


def _decision_coverage(
    probs: np.ndarray,
    ys: np.ndarray,
    t_accept: float,
    m_accept: float,
) -> dict:
    """Coverage + accepted-accuracy at the configured accept thresholds.

    A clip is *accepted* when fused top-1 prob >= t_accept AND margin >=
    m_accept (the accept branch of the decision engine). Reports how often that
    happens (coverage) and how accurate the accepted subset is.
    """
    if probs.shape[0] == 0:
        return {"coverage": 0.0, "accepted_accuracy": 0.0, "n_accepted": 0, "n_total": 0}
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top1_p = probs[np.arange(len(probs)), top1]
    top2_p = probs[np.arange(len(probs)), order[:, -2]] if probs.shape[1] > 1 else np.zeros(len(probs))
    margin = top1_p - top2_p
    accepted = (top1_p >= t_accept) & (margin >= m_accept)
    n_accepted = int(accepted.sum())
    if n_accepted:
        acc = float(np.mean(top1[accepted] == ys[accepted]))
    else:
        acc = 0.0
    return {
        "coverage": round(n_accepted / len(probs), 4),
        "accepted_accuracy": round(acc, 4),
        "n_accepted": n_accepted,
        "n_total": int(len(probs)),
    }


def _open_set_wrong_accept(probs: np.ndarray, t_accept: float, m_accept: float) -> dict:
    """Fraction of UNKNOWN clips wrongly *accepted* (target: 0 — Section 9.5).

    Every clip here is out-of-vocabulary, so ANY accept is a wrong-accept.
    """
    if probs.shape[0] == 0:
        return {"n_total": 0, "n_wrongly_accepted": 0, "wrong_accept_rate": 0.0}
    order = np.argsort(probs, axis=1)
    top1_p = probs[np.arange(len(probs)), order[:, -1]]
    top2_p = probs[np.arange(len(probs)), order[:, -2]] if probs.shape[1] > 1 else np.zeros(len(probs))
    margin = top1_p - top2_p
    accepted = (top1_p >= t_accept) & (margin >= m_accept)
    n = int(accepted.sum())
    return {
        "n_total": int(len(probs)),
        "n_wrongly_accepted": n,
        "wrong_accept_rate": round(n / len(probs), 4),
    }


# --------------------------------------------------------------------------- #
# Config / thresholds
# --------------------------------------------------------------------------- #
def _load_thresholds() -> tuple[float, float]:
    with open(default_config_path(), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    dec = cfg.get("decision", {})
    return float(dec.get("t_accept", 0.85)), float(dec.get("m_accept", 0.3))


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _render_markdown(report: dict, labels: list[str]) -> str:
    leaky = report["split"]["mode"] == "stratified"
    lines: list[str] = []
    lines.append("# SignTalk AI — Evaluation (Gate 6)")
    lines.append("")
    lines.append(
        "Generated by `tools/evaluate.py`. Every number below is reproducible "
        "from the split seed, manifest, and thresholds recorded in "
        "`runs/evaluation.json`. Numbers are measured, never fabricated "
        "(PROJECT_CONTEXT Section 0.7)."
    )
    lines.append("")

    # Honesty banner.
    if leaky:
        lines.append("> ⚠️ **Pipeline-validation numbers, not a generalisation claim.**")
        lines.append(">")
        lines.append(
            "> These use a **random stratified split** that likely shares a "
            "signer across train and val (Section 1.6), so they prove the "
            "pipeline learns — they are NOT a defensible accuracy figure. For "
            "the honest number, re-run with the official INCLUDE test split or "
            "an enrolled session/signer hold-out and `--split official`."
        )
    else:
        lines.append(
            "> These use an **official / held-out split** (`--split official`): "
            "no train/val mixing was done by this tool. Treat this as the "
            "defensible number for the evaluated set (state n and bound below)."
        )
    lines.append("")

    # Provenance.
    s = report["split"]
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Manifest: `{report['manifest']}`")
    lines.append(f"- Members: {report['members']}")
    lines.append(f"- Split mode: `{s['mode']}` (val_frac={s['val_frac']}, seed={s['seed']})")
    lines.append(f"- Evaluated clips (n): **{report['n_eval']}**, classes: **{report['num_classes']}**")
    lines.append(f"- Device: {report['device']}")
    lines.append(f"- Thresholds (from config): t_accept={report['thresholds']['t_accept']}, "
                 f"m_accept={report['thresholds']['m_accept']}")
    lines.append("")

    # Headline.
    o = report["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Top-1 accuracy | {o['top1']:.4f} |")
    lines.append(f"| Top-5 accuracy | {o['top5']:.4f} |")
    lines.append(f"| Macro-F1 | {o['macro_f1']:.4f} |")
    lines.append("")

    # Decision layer.
    d = report["decision"]
    lines.append("## Decision layer (accept branch)")
    lines.append("")
    lines.append(
        "Coverage = fraction of clips the system would *speak* (accept); "
        "accepted-accuracy = correctness within that accepted subset."
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Coverage | {d['coverage']:.4f} ({d['n_accepted']}/{d['n_total']}) |")
    lines.append(f"| Accepted accuracy | {d['accepted_accuracy']:.4f} |")
    lines.append("")

    # Open set.
    if report.get("open_set"):
        os_ = report["open_set"]
        lines.append("## Open-set wrong-accept (Section 9.5)")
        lines.append("")
        lines.append(
            "Out-of-vocabulary clips — every accept here is wrong. Target: 0."
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| UNKNOWN clips | {os_['n_total']} |")
        lines.append(f"| Wrongly accepted | {os_['n_wrongly_accepted']} |")
        lines.append(f"| Wrong-accept rate | {os_['wrong_accept_rate']:.4f} |")
        lines.append("")

    # Per-class table.
    lines.append("## Per-class")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for row in report["per_class"]:
        name = labels[row["class_index"]] if row["class_index"] < len(labels) else str(row["class_index"])
        lines.append(
            f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['support']} |"
        )
    lines.append("")

    # Confusion matrix.
    cm = np.asarray(report["confusion_matrix"], dtype=np.int64)
    lines.append("## Confusion matrix")
    lines.append("")
    lines.append("Rows = true class, columns = predicted class.")
    lines.append("")
    header = "| true \\ pred | " + " | ".join(str(i) for i in range(cm.shape[1])) + " |"
    sep = "|---" * (cm.shape[1] + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for i in range(cm.shape[0]):
        name = labels[i] if i < len(labels) else str(i)
        lines.append(f"| {i} {name} | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
    lines.append("")
    lines.append("Column index legend: " + ", ".join(f"{i}={labels[i]}" for i in range(len(labels))))
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline evaluation (Gate 6)")
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path, help="ensemble JSON (recognizer manifest)")
    ap.add_argument(
        "--split",
        choices=["stratified", "official"],
        default="stratified",
        help="stratified = take the val slice of a random split (leaky, pipeline-validation). "
        "official = evaluate EVERY clip in --processed-dir as-is (point it at a held-out set).",
    )
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--unknown-dir",
        type=Path,
        default=None,
        help="optional processed-dir of OUT-OF-VOCAB clips for the open-set check",
    )
    ap.add_argument("--report", type=Path, default=Path("runs/evaluation.json"))
    ap.add_argument("--markdown", type=Path, default=Path("docs/EVALUATION.md"))
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    labels, members = _build_members(manifest, device)
    num_classes = len(labels)

    label_map = load_label_map(args.processed_dir)
    items = gather_items(args.processed_dir, label_map)
    if not items:
        print("[fail] no feature .npz found — run prepare_include.py first")
        return 1

    if args.split == "stratified":
        _, eval_items = stratified_split(items, args.val_frac, args.seed)
    else:  # official: evaluate every clip present, no mixing
        eval_items = items

    print(f"[info] eval items={len(eval_items)} classes={num_classes} "
          f"mode={args.split} device={device}")

    probs, ys = _fused_probs(members, eval_items, device)
    preds = np.argmax(probs, axis=1)

    cm = _confusion_matrix(preds, ys, num_classes)
    per_class = _per_class_prf(cm)
    t_accept, m_accept = _load_thresholds()

    report = {
        "manifest": str(args.manifest),
        "members": [m["arch"] for m in manifest["members"]],
        "device": device,
        "num_classes": num_classes,
        "n_eval": int(len(eval_items)),
        "split": {"mode": args.split, "val_frac": args.val_frac, "seed": args.seed},
        "thresholds": {"t_accept": t_accept, "m_accept": m_accept},
        "overall": {
            "top1": round(_top_k_accuracy(probs, ys, 1), 4),
            "top5": round(_top_k_accuracy(probs, ys, 5), 4),
            "macro_f1": round(_macro_f1(per_class), 4),
        },
        "decision": _decision_coverage(probs, ys, t_accept, m_accept),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    # Optional open-set check.
    if args.unknown_dir is not None:
        u_map = load_label_map(args.unknown_dir)
        u_items = gather_items(args.unknown_dir, u_map)
        if u_items:
            u_probs, _ = _fused_probs(members, u_items, device)
            report["open_set"] = _open_set_wrong_accept(u_probs, t_accept, m_accept)
        else:
            print("[warn] --unknown-dir has no feature .npz; skipping open-set check")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    md = _render_markdown(report, labels)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    with open(args.markdown, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(json.dumps(report["overall"], indent=2))
    print(f"[ok  ] wrote {args.report} and {args.markdown}")
    if report["split"]["mode"] == "stratified":
        print("[warn] stratified split — pipeline-validation only, NOT a generalisation number (Section 1.6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
