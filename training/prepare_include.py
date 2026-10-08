"""Preprocess INCLUDE videos into feature tensors (PROJECT_CONTEXT 8.2).

Turns a directory of INCLUDE ``.mp4`` sign clips into per-clip ``.npz`` feature
files using the SAME extractor + feature builder the server uses (Rule 12.3 —
no train/serve skew). Every failed or skipped clip is logged with a reason to a
CSV so the old "4,257 vs ~4,292" mystery can never recur (Section 8.2).

This runs on YOUR machine (it needs the downloaded INCLUDE videos and the
MediaPipe ``.task`` bundle); it is not run in CI. Expected layout of the input,
one folder per sign label::

    <include_root>/
        hello/     001.mp4 002.mp4 ...
        thank you/ ...

Output::

    <out_dir>/
        features/<label>/<clip_stem>.npz   # arr "features" float32 [T, F], "label" str
        manifest.csv                        # clip, label, status, reason, n_frames, hands_pct
        label_map.json                      # {label: index} in sorted order

Run from the repo root, e.g.::

    python training/prepare_include.py \
        --include-root data/raw/include \
        --out-dir data/processed/include \
        --model models/holistic_landmarker.task \
        --vocab-only            # keep only labels present in config/vocabulary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from signtalk_core.config import load_config
from signtalk_core.features import build_features
from signtalk_core.landmarks import create_holistic_detector, extract_from_video
from signtalk_core.quality import quality_check

import re

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# INCLUDE sign folders are named like "48. Hello" / "55. Thank you": a number
# prefix + the real label. Strip the "<n>. " prefix so the class label is the
# clean sign name (e.g. "Hello"), matching what the app's vocabulary expects.
_LABEL_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def clean_label(folder_name: str) -> str:
    """Turn an INCLUDE sign folder name into a clean label.

    "48. Hello" -> "Hello"; "55. Thank you" -> "Thank you". A folder with no
    number prefix is returned trimmed and unchanged.
    """
    return _LABEL_PREFIX_RE.sub("", folder_name).strip()


def _vocab_labels() -> list[str]:
    import json as _json

    root = Path(__file__).resolve().parents[1]
    with open(root / "config" / "vocabulary.json", "r", encoding="utf-8") as fh:
        data = _json.load(fh)
    signs = data["signs"] if isinstance(data, dict) else data
    return [s["label"] for s in signs]


def _dir_has_videos(d: Path) -> bool:
    """True if a directory directly contains at least one video file."""
    return any(c.is_file() and c.suffix.lower() in VIDEO_EXTS for c in d.iterdir())


def discover_clips(include_root: Path) -> list[tuple[str, Path]]:
    """Return ``(clean_label, video_path)`` for every INCLUDE clip found.

    Robust to the real INCLUDE layout, which nests one level deeper than a flat
    label tree:

        <root>/<Category>/<NN. Sign>/<clip>.MOV      (e.g. Greetings/48. Hello/)
        <root>/<NN. Sign>/<clip>.MOV                 (flat, also supported)

    A "sign folder" is any directory that DIRECTLY contains video files; its
    (prefix-stripped) name is the label. We walk the whole tree so category
    wrappers, Adults/Kids subfolders, etc. are all handled, and clips from
    folders that clean to the same label are merged.
    """
    clips: list[tuple[str, Path]] = []
    for d in sorted(p for p in include_root.rglob("*") if p.is_dir()):
        if not _dir_has_videos(d):
            continue
        label = clean_label(d.name)
        if not label:
            continue
        for vid in sorted(d.iterdir()):
            if vid.is_file() and vid.suffix.lower() in VIDEO_EXTS:
                clips.append((label, vid))
    return clips


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess INCLUDE videos -> feature .npz")
    ap.add_argument("--include-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path, help="holistic_landmarker.task")
    ap.add_argument("--fps", type=float, default=30.0, help="assumed clip fps for quality")
    ap.add_argument("--vocab-only", action="store_true", help="keep only vocabulary labels")
    args = ap.parse_args(argv)

    cfg = load_config()
    detector = create_holistic_detector(args.model)
    keep = set(_vocab_labels()) if args.vocab_only else None

    clips = discover_clips(args.include_root)
    if keep is not None:
        clips = [(lbl, p) for (lbl, p) in clips if lbl in keep]
    print(f"[info] {len(clips)} clips to process")

    feat_root = args.out_dir / "features"
    feat_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"

    labels_seen: set[str] = set()
    n_ok = n_fail = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip", "label", "status", "reason", "n_frames", "hands_pct"])
        for label, vid in clips:
            try:
                seq = extract_from_video(vid, detector)
            except Exception as exc:  # noqa: BLE001 - log and continue
                writer.writerow([str(vid), label, "error", f"extract:{exc}", 0, 0.0])
                n_fail += 1
                continue
            n_frames = len(seq)
            hands_pct = (
                float(np.mean([(f.left_hand_present or f.right_hand_present) for f in seq]))
                if seq
                else 0.0
            )
            passed, reason = quality_check(seq, cfg, args.fps)
            if not passed:
                writer.writerow([str(vid), label, "skipped", reason or "low_quality", n_frames, round(hands_pct, 3)])
                n_fail += 1
                continue
            feats = build_features(seq, cfg)  # [T, F]
            out = feat_root / label / (vid.stem + ".npz")
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, features=feats.astype(np.float32), label=label)
            writer.writerow([str(vid), label, "ok", "", n_frames, round(hands_pct, 3)])
            labels_seen.add(label)
            n_ok += 1

    label_map = {lbl: i for i, lbl in enumerate(sorted(labels_seen))}
    with open(args.out_dir / "label_map.json", "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2, ensure_ascii=False)

    print(f"[done] ok={n_ok} failed/skipped={n_fail}; {len(label_map)} labels")
    print(f"[done] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
