"""Train one ensemble member (PROJECT_CONTEXT 5.5 / Section 10 Phase 2).

Trains M1 (``gru``) or M2 (``transformer``) on the preprocessed INCLUDE feature
``.npz`` files, with the Section 5.5 recipe: label smoothing 0.1, dropout (in
the model), weight decay, early stopping on validation, fixed seed. Saves the
best checkpoint (``.pt``) plus a ``metrics.json`` recording the real validation
top-1/top-5 — the number you paste into ``docs/PROGRESS.md`` for Gate 2.

Runs on YOUR machine / GPU (RTX 4050 is plenty for these small models). CUDA is
used automatically when available. Example::

    python training/train.py \
        --processed-dir data/processed/include \
        --arch gru --epochs 60 --seed 0 \
        --out runs/m1_gru

Train several seeds / both archs to build the ensemble; fuse later with
``tools/make_report.py`` / the serving bundle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signtalk_core.models.pose_tcn import PoseGRU
from signtalk_core.models.pose_transformer import PoseTransformer
from training.dataset import FeatureDataset, gather_items, load_label_map, stratified_split


def build_arch(arch: str, num_classes: int) -> nn.Module:
    if arch == "gru":
        return PoseGRU(num_classes=num_classes)
    if arch == "transformer":
        return PoseTransformer(num_classes=num_classes)
    raise ValueError(f"unknown arch: {arch!r} (expected 'gru' or 'transformer')")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (top1, top5, all_logits, all_labels) over a loader."""
    model.eval()
    logits_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).cpu().numpy()
        logits_all.append(logits)
        labels_all.append(y.numpy())
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    top1 = float(np.mean(np.argmax(logits, axis=1) == labels))
    k = min(5, logits.shape[1])
    topk = np.argsort(logits, axis=1)[:, -k:]
    top5 = float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))
    return top1, top5, logits, labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train one ensemble member")
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--arch", choices=["gru", "transformer"], default="gru")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=12, help="early-stopping patience")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    label_map = load_label_map(args.processed_dir)
    items = gather_items(args.processed_dir, label_map)
    if not items:
        print("[fail] no feature .npz files found — run prepare_include.py first")
        return 1
    train_items, val_items = stratified_split(items, args.val_frac, args.seed)
    print(f"[info] classes={len(label_map)} train={len(train_items)} val={len(val_items)}")

    train_loader = DataLoader(FeatureDataset(train_items), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_items), batch_size=args.batch_size)

    model = build_arch(args.arch, len(label_map)).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    args.out.mkdir(parents=True, exist_ok=True)
    best_top1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss) * x.size(0)
        train_loss = running / max(len(train_items), 1)
        top1, top5, _, _ = evaluate(model, val_loader, device)
        print(f"[e{epoch:03d}] loss={train_loss:.4f} val_top1={top1:.4f} val_top5={top5:.4f}")

        if top1 > best_top1:
            best_top1, best_epoch = top1, epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.out / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[info] early stop at epoch {epoch} (best {best_top1:.4f} @ e{best_epoch})")
                break

    elapsed = time.perf_counter() - t0
    metrics = {
        "arch": args.arch,
        "seed": args.seed,
        "num_classes": len(label_map),
        "best_val_top1": best_top1,
        "best_epoch": best_epoch,
        "train_clips": len(train_items),
        "val_clips": len(val_items),
        "train_seconds": round(elapsed, 1),
        "note": "stratified split — see Section 1.6; use session/signer hold-out for the honest number",
    }
    with open(args.out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    # Persist the label order used so serving/eval match the checkpoint.
    with open(args.out / "label_map.json", "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2, ensure_ascii=False)
    print(f"[done] best val_top1={best_top1:.4f}; checkpoint {args.out/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
