"""Fine-tune the ensemble on enrolled signer clips (PROJECT_CONTEXT L7 / Phase 4).

Enrollment is the single biggest accuracy lever (HANDOVER Section 8b): the
INCLUDE-trained models never saw your signers / camera / lighting, so a live
sign looks unlike anything they learned. This script closes that domain gap by
warm-starting each trained ensemble member from its checkpoint and fine-tuning
it on the clips your signers recorded in the app's Enrollment tab.

It reuses the SAME feature path as serving and INCLUDE preprocessing
(``extract_from_video`` -> ``quality_check`` -> ``build_features``) so there is
no train/serve skew (Rule 12.3), and the SAME label index order as the ensemble
manifest so the fine-tuned checkpoints drop straight back into
``models/ensemble.json`` with no head reshuffle.

Honesty (Section 1.6): enrolled clips are split by **session** — the latest
session is held out as validation — so the reported number is a
session-independent figure, not a leaky one. With only one session present it
falls back to a per-clip stratified split and SAYS SO in the report (that number
is weaker). Signs recorded in enrollment but absent from the trained vocabulary
are skipped with a warning (growing the vocabulary needs retraining, not
fine-tuning a fixed head).

Enrollment layout written by server/app/routers/enroll.py::

    data/enroll/<signer>/<session>/<sign>_<uuid>.webm

and indexed in SQLite (table ``enrollment``: signer_id, session_id, sign,
clip_path). This script reads the SQLite index when available and falls back to
walking the filesystem.

Run on YOUR machine after recording enrollment sessions::

    python training/finetune_enrolled.py \
        --manifest models/ensemble.json \
        --enroll-root data/enroll \
        --out-dir runs/finetune \
        --report runs/finetune_report.json

Then point ``models/ensemble.json`` members at the new checkpoints under
``runs/finetune/`` (or pass ``--write-manifest`` to patch it in place) and
re-run ``tools/pick_thresholds.py`` + ``tools/evaluate.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow running as a plain script as well as a module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from signtalk_core.config import load_config  # noqa: E402
from signtalk_core.features import build_features  # noqa: E402
from signtalk_core.landmarks import create_holistic_detector, extract_from_video  # noqa: E402
from signtalk_core.models.pose_tcn import PoseGRU  # noqa: E402
from signtalk_core.models.pose_transformer import PoseTransformer  # noqa: E402
from signtalk_core.quality import quality_check  # noqa: E402
from server.app.storage import default_db_path  # noqa: E402

VIDEO_EXTS = {".webm", ".mp4", ".mov", ".avi", ".mkv"}
# Clip filenames are "<sign>_<uuid>.<ext>"; recover the sign from the stem.
_UUID_SUFFIX_RE = re.compile(r"_[0-9a-fA-F]{32}$")


# --------------------------------------------------------------------------- #
# Enrollment clip discovery
# --------------------------------------------------------------------------- #
class EnrollClip:
    """One enrollment clip: its path, sign label, and session id (for grouping)."""

    __slots__ = ("path", "sign", "session")

    def __init__(self, path: Path, sign: str, session: str) -> None:
        self.path = path
        self.sign = sign
        self.session = session


def _sign_from_filename(stem: str) -> str:
    """Recover the sign label from a ``<sign>_<uuid>`` clip stem."""
    return _UUID_SUFFIX_RE.sub("", stem)


def discover_from_sqlite(db_path: Path) -> list[EnrollClip]:
    """Read the enrollment index from SQLite, if the DB + table exist."""
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT sign, session_id, clip_path FROM enrollment"
        ).fetchall()
    except sqlite3.OperationalError:
        return []  # table not created yet
    finally:
        conn.close()
    clips: list[EnrollClip] = []
    for sign, session_id, clip_path in rows:
        p = Path(clip_path)
        if p.is_file():
            clips.append(EnrollClip(p, str(sign), str(session_id)))
    return clips


def discover_from_fs(enroll_root: Path) -> list[EnrollClip]:
    """Walk ``data/enroll/<signer>/<session>/<sign>_<uuid>.<ext>``."""
    clips: list[EnrollClip] = []
    if not enroll_root.is_dir():
        return clips
    for signer_dir in sorted(p for p in enroll_root.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in signer_dir.iterdir() if p.is_dir()):
            for vid in sorted(session_dir.iterdir()):
                if vid.is_file() and vid.suffix.lower() in VIDEO_EXTS:
                    clips.append(
                        EnrollClip(vid, _sign_from_filename(vid.stem), session_dir.name)
                    )
    return clips


# --------------------------------------------------------------------------- #
# Feature extraction (same path as serving / prepare_include)
# --------------------------------------------------------------------------- #
class _FeatureItem:
    __slots__ = ("features", "label_idx", "session")

    def __init__(self, features: np.ndarray, label_idx: int, session: str) -> None:
        self.features = features
        self.label_idx = label_idx
        self.session = session


def extract_features(
    clips: list[EnrollClip],
    label_to_idx: dict[str, int],
    pose_model: Path,
    hand_model: Path,
    fps: float,
) -> tuple[list[_FeatureItem], dict[str, int]]:
    """Run the shared extractor over each clip; keep only in-vocabulary signs.

    Returns the usable feature items and a ``{reason: count}`` skip tally.
    """
    cfg = load_config()
    detector = create_holistic_detector(pose_model, hand_model)
    items: list[_FeatureItem] = []
    skips: dict[str, int] = defaultdict(int)
    for clip in clips:
        if clip.sign not in label_to_idx:
            skips["sign_not_in_trained_vocab"] += 1
            continue
        try:
            seq = extract_from_video(clip.path, detector)
        except Exception as exc:  # noqa: BLE001
            skips[f"extract_error:{type(exc).__name__}"] += 1
            continue
        passed, reason = quality_check(seq, cfg, fps)
        if not passed:
            skips[f"quality:{reason or 'low'}"] += 1
            continue
        feats = build_features(seq, cfg).astype(np.float32)
        items.append(_FeatureItem(feats, label_to_idx[clip.sign], clip.session))
    return items, dict(skips)


# --------------------------------------------------------------------------- #
# Splitting (session-aware, honest)
# --------------------------------------------------------------------------- #
def session_split(
    items: list[_FeatureItem], seed: int
) -> tuple[list[_FeatureItem], list[_FeatureItem], str]:
    """Hold out the latest session for validation (session-independent number).

    Falls back to a per-class stratified split when only one session exists,
    and reports which mode was used so a weaker number is never mislabelled.
    """
    sessions = sorted({it.session for it in items})
    if len(sessions) >= 2:
        holdout = sessions[-1]
        train = [it for it in items if it.session != holdout]
        val = [it for it in items if it.session == holdout]
        if train and val:
            return train, val, f"session-holdout (val session={holdout!r})"
    # Fallback: stratified by label.
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[_FeatureItem]] = defaultdict(list)
    for it in items:
        by_class[it.label_idx].append(it)
    train, val = [], []
    for group in by_class.values():
        idx = rng.permutation(len(group))
        n_val = max(1, int(round(len(group) * 0.2)))
        val_idx = set(idx[:n_val].tolist())
        for i, it in enumerate(group):
            (val if i in val_idx else train).append(it)
    return train, val, "stratified-fallback (single session — WEAKER, not session-independent)"


class _InMemoryDataset(Dataset):
    def __init__(self, items: list[_FeatureItem]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        return torch.from_numpy(it.features), int(it.label_idx)


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #
def _build_warm_model(arch: str, num_classes: int, checkpoint: Path | None, device: str) -> nn.Module:
    if arch == "gru":
        model = PoseGRU(num_classes=num_classes)
    elif arch == "transformer":
        model = PoseTransformer(num_classes=num_classes)
    else:
        raise ValueError(f"unknown arch: {arch!r}")
    if checkpoint is not None and checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def _val_top1(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        pred = torch.argmax(model(x), dim=1).cpu().numpy()
        correct += int(np.sum(pred == y.numpy()))
        total += len(y)
    return correct / total if total else 0.0


def finetune_member(
    arch: str,
    checkpoint: Path | None,
    num_classes: int,
    train_items: list[_FeatureItem],
    val_items: list[_FeatureItem],
    out_path: Path,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    patience: int,
    seed: int,
    device: str,
) -> dict:
    """Warm-start from ``checkpoint`` and fine-tune; save best val checkpoint."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = _build_warm_model(arch, num_classes, checkpoint, device)
    warm_top1 = _val_top1(model, DataLoader(_InMemoryDataset(val_items), batch_size=32), device) if val_items else 0.0

    train_loader = DataLoader(_InMemoryDataset(train_items), batch_size=min(32, max(1, len(train_items))), shuffle=True)
    val_loader = DataLoader(_InMemoryDataset(val_items), batch_size=32) if val_items else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_top1, best_epoch, no_improve = -1.0, -1, 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        top1 = _val_top1(model, val_loader, device) if val_loader else 0.0
        print(f"[{arch}][e{epoch:03d}] val_top1={top1:.4f}")
        if val_loader is None:
            torch.save(model.state_dict(), out_path)  # no val: keep last
            best_top1, best_epoch = top1, epoch
            continue
        if top1 > best_top1:
            best_top1, best_epoch, no_improve = top1, epoch, 0
            torch.save(model.state_dict(), out_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{arch}] early stop @ e{epoch} (best {best_top1:.4f} @ e{best_epoch})")
                break
    if best_epoch < 0:  # never saved (e.g. 0 epochs) — save current
        torch.save(model.state_dict(), out_path)
    return {
        "arch": arch,
        "warm_start_from": str(checkpoint) if checkpoint else None,
        "val_top1_before_finetune": round(float(warm_top1), 4),
        "val_top1_after_finetune": round(float(best_top1), 4),
        "best_epoch": best_epoch,
        "checkpoint": str(out_path),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the ensemble on enrolled clips (L7)")
    ap.add_argument("--manifest", required=True, type=Path, help="ensemble JSON (labels + members)")
    ap.add_argument("--enroll-root", type=Path, default=Path("data/enroll"))
    ap.add_argument("--db", type=Path, default=None, help="SQLite path (default: data/signtalk.db)")
    ap.add_argument("--pose-model", type=Path, default=Path("models/pose_landmarker.task"))
    ap.add_argument("--hand-model", type=Path, default=Path("models/hand_landmarker.task"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4, help="low LR for fine-tuning a warm start")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/finetune"))
    ap.add_argument("--report", type=Path, default=Path("runs/finetune_report.json"))
    ap.add_argument(
        "--write-manifest",
        action="store_true",
        help="patch the --manifest members' checkpoints to the fine-tuned ones",
    )
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    labels = manifest["labels"]
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    num_classes = len(labels)

    # Discover clips: SQLite index first, filesystem fallback (and union).
    db_path = args.db or default_db_path()
    clips = discover_from_sqlite(db_path)
    source = "sqlite"
    if not clips:
        clips = discover_from_fs(args.enroll_root)
        source = "filesystem"
    if not clips:
        print(
            f"[fail] no enrollment clips found (looked in SQLite {db_path} and "
            f"{args.enroll_root}). Record clips in the app's Enrollment tab first."
        )
        return 1
    print(f"[info] {len(clips)} enrollment clips discovered via {source}")

    items, skips = extract_features(clips, label_to_idx, args.pose_model, args.hand_model, args.fps)
    if skips:
        print(f"[info] skipped: {skips}")
    if not items:
        print("[fail] no usable clips after feature extraction (all skipped — see tally above).")
        return 1

    # Per-sign usable counts, to flag thin coverage.
    per_sign: dict[str, int] = defaultdict(int)
    for it in items:
        per_sign[labels[it.label_idx]] += 1

    train_items, val_items, split_mode = session_split(items, args.seed)
    print(f"[info] usable={len(items)} train={len(train_items)} val={len(val_items)} split={split_mode}")

    member_reports = []
    new_checkpoints = []
    for m in manifest["members"]:
        arch = m["arch"]
        ckpt = Path(m["checkpoint"]) if m.get("checkpoint") else None
        out_path = args.out_dir / f"{arch}_finetuned.pt"
        rep = finetune_member(
            arch, ckpt, num_classes, train_items, val_items, out_path,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing, patience=args.patience,
            seed=args.seed, device=device,
        )
        member_reports.append(rep)
        new_checkpoints.append((m, str(out_path)))

    report = {
        "source": source,
        "db_path": str(db_path),
        "enroll_root": str(args.enroll_root),
        "num_classes": num_classes,
        "clips_discovered": len(clips),
        "clips_usable": len(items),
        "skipped": skips,
        "per_sign_usable": dict(sorted(per_sign.items())),
        "split_mode": split_mode,
        "train_clips": len(train_items),
        "val_clips": len(val_items),
        "device": device,
        "lr": args.lr,
        "members": member_reports,
        "note": (
            "Session-holdout gives a session-independent number; a "
            "stratified-fallback (single session) number is weaker and is "
            "labelled as such. Re-run tools/pick_thresholds.py and "
            "tools/evaluate.py on the fine-tuned manifest before quoting any "
            "figure (Section 1.6)."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps({"split_mode": split_mode, "members": member_reports}, indent=2))
    print(f"[ok  ] wrote {args.report}; checkpoints in {args.out_dir}")

    if args.write_manifest:
        for m, path in new_checkpoints:
            m["checkpoint"] = path
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        print(f"[ok  ] patched {args.manifest} members -> fine-tuned checkpoints")

    if "single session" in split_mode:
        print("[warn] single-session split — number is NOT session-independent; record another session (R L7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
