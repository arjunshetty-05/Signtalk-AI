"""
preprocess_include.py — SignTalk AI / dataset_tools

Converts extracted INCLUDE dataset video clips into the (30, 17, 2)
keypoint-sequence .npy + labels.csv format train_bilstm.py expects — the
preprocessing step download_include.py's docstring flags as missing.

INCLUDE's on-disk layout (verified against a real, full category
download — Seasons, all 85 clips — not guessed from the paper):
    <include_dir>/<Category>/<idx>. <Word>/<video>.MOV|.MP4
e.g. Seasons/61. Summer/MVI_4565.MOV — the class label is the "<Word>"
part of the second-level folder name (the leading "<idx>. " is a global
sign index across all 263 INCLUDE words, stripped here). A handful of
folders use "Ex." instead of a number (e.g. "Ex. Monsoon") for signs
outside the official 263-word list — also stripped.

IMPORTANT — signer_id is a best-effort placeholder, not real signer
identity: INCLUDE's file/folder structure does not encode which physical
signer recorded which clip anywhere (no such field in the zip we
inspected). train_bilstm.py splits train/val BY SIGNER specifically to
prevent identity leakage; that protection does NOT apply to samples this
script produces; --signer_id_mode controls how the placeholder is built,
but neither mode is verified against ground truth. If INCLUDE publishes
real per-video signer metadata elsewhere (the dataset's Google Drive links
mentioned alongside the official train/test split CSVs might), use that
instead and pass --signer_id_mode=none to skip the placeholder.

Usage:
    python dataset_tools/preprocess_include.py --include_dir data/raw/include
    python dataset_tools/preprocess_include.py --include_dir data/raw/include --max_videos 20   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keypoint_utils import (  # noqa: E402
    TemporalSmoother,
    extract_keypoints,
    load_movenet,
    normalize_keypoints,
)

SEQUENCE_LENGTH = 30
VIDEO_EXTENSIONS = {".mov", ".mp4"}
WORD_FOLDER_RE = re.compile(r"^(?:\d+|Ex)\.\s*(.+)$")
LABELS_CSV_HEADER = ["filename", "class", "signer_id"]
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def parse_args():
    p = argparse.ArgumentParser(description="Convert INCLUDE video clips into (30,17,2) keypoint sequences")
    p.add_argument("--include_dir", required=True, help="Root of extracted INCLUDE category folders")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                    help="Output dataset root — writes <data_dir>/sequences/*.npy and appends <data_dir>/labels.csv")
    p.add_argument("--signer_id_mode", choices=["per_video", "per_session", "none"], default="per_video",
                    help="per_video: unique id per clip (signer-split degenerates to random-split for these "
                         "rows — safest, no false grouping). per_session: groups clips by near-consecutive "
                         "camera file numbers within a word folder as a HEURISTIC proxy for 'same recording "
                         "session' — unverified, may be wrong. none: skip rows needing real signer_id "
                         "(nothing will be skipped since INCLUDE never provides one; this errors instead, "
                         "as a guard against silently mislabeling).")
    p.add_argument("--max_videos", type=int, default=None, help="Cap number of videos processed (smoke testing)")
    p.add_argument("--smoother_window", type=int, default=4, help="TemporalSmoother window, same default as record_samples.py")
    return p.parse_args()


def find_videos(include_dir: str) -> list[str]:
    pattern = os.path.join(include_dir, "*", "*", "*")
    return sorted(
        path for path in glob.glob(pattern)
        if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS
    )


def parse_label(video_path: str) -> tuple[str, str]:
    """Returns (category, word) from <include_dir>/<Category>/<idx>. <Word>/<video>."""
    word_folder = os.path.basename(os.path.dirname(video_path))
    category = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
    m = WORD_FOLDER_RE.match(word_folder)
    word = m.group(1) if m else word_folder
    return category, word


def _camera_number(filename: str) -> int | None:
    m = re.search(r"(\d+)", os.path.splitext(filename)[0])
    return int(m.group(1)) if m else None


def assign_session_ids(videos: list[str], gap_threshold: int = 5) -> dict[str, str]:
    """Heuristic per_session grouping: within each (category, word) folder,
    clips are bucketed by runs of near-consecutive camera file numbers
    (gap <= gap_threshold), on the theory that one recording session
    produces a contiguous run of camera-assigned numbers. UNVERIFIED against
    any ground-truth signer label — see module docstring."""
    by_folder: dict[str, list[str]] = {}
    for v in videos:
        by_folder.setdefault(os.path.dirname(v), []).append(v)

    session_id = {}
    for folder, vids in by_folder.items():
        numbered = sorted(((_camera_number(os.path.basename(v)), v) for v in vids), key=lambda t: (t[0] is None, t[0]))
        category, word = parse_label(numbered[0][1])
        session = 0
        prev_num = None
        for num, v in numbered:
            if prev_num is not None and num is not None and num - prev_num > gap_threshold:
                session += 1
            session_id[v] = f"{category}_{word}_sess{session}"
            prev_num = num if num is not None else prev_num
    return session_id


def video_to_sequence(video_path: str, movenet_sig, smoother_window: int) -> np.ndarray | None:
    """Extracts every frame's normalized keypoints, smooths them, then
    uniformly resamples to exactly SEQUENCE_LENGTH frames — mirrors
    record_samples.py's per-frame smooth-then-buffer behavior, but over a
    full pre-recorded clip instead of a live sliding window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    smoother = TemporalSmoother(window=smoother_window)
    smoothed_frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_kp = extract_keypoints(frame, movenet_sig)
            normalized = normalize_keypoints(raw_kp)
            smoothed_frames.append(smoother.smooth(normalized))
    finally:
        cap.release()

    if len(smoothed_frames) < 2:
        return None

    indices = np.linspace(0, len(smoothed_frames) - 1, SEQUENCE_LENGTH).round().astype(int)
    return np.stack([smoothed_frames[i] for i in indices], axis=0).astype(np.float32)


def ensure_labels_csv(labels_csv_path: str) -> None:
    if not os.path.exists(labels_csv_path):
        os.makedirs(os.path.dirname(labels_csv_path), exist_ok=True)
        with open(labels_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(LABELS_CSV_HEADER)


def main():
    args = parse_args()
    if args.signer_id_mode == "none":
        raise SystemExit(
            "INCLUDE never provides real signer_id — --signer_id_mode=none has nothing valid to write. "
            "Use per_video or per_session (both are placeholders, see module docstring), or source real "
            "signer metadata yourself and write labels.csv directly."
        )

    videos = find_videos(args.include_dir)
    if not videos:
        raise SystemExit(f"No .MOV/.MP4 files found under {args.include_dir}/*/*/*")
    if args.max_videos:
        videos = videos[: args.max_videos]
    print(f"Found {len(videos)} video(s) to process.")

    session_ids = assign_session_ids(videos) if args.signer_id_mode == "per_session" else {}

    sequences_dir = os.path.join(args.data_dir, "sequences")
    labels_csv_path = os.path.join(args.data_dir, "labels.csv")
    os.makedirs(sequences_dir, exist_ok=True)
    ensure_labels_csv(labels_csv_path)

    movenet_sig = load_movenet()

    processed, skipped = 0, 0
    with open(labels_csv_path, "a", newline="") as out_f:
        writer = csv.writer(out_f)
        for i, video_path in enumerate(videos, 1):
            category, word = parse_label(video_path)
            stem = os.path.splitext(os.path.basename(video_path))[0]
            out_filename = f"include_{category}_{word}_{stem}.npy".replace(" ", "_")
            out_path = os.path.join(sequences_dir, out_filename)

            print(f"[{i}/{len(videos)}] {video_path} -> {out_filename}", end=" ")
            if os.path.exists(out_path):
                print("(already processed, skipping)")
                continue

            sequence = video_to_sequence(video_path, movenet_sig, args.smoother_window)
            if sequence is None:
                print("[warn] unreadable/too short, skipping")
                skipped += 1
                continue

            np.save(out_path, sequence)
            if args.signer_id_mode == "per_session":
                signer_id = session_ids[video_path]
            else:
                signer_id = f"include_{stem}"
            writer.writerow([out_filename, word, signer_id])
            print("ok")
            processed += 1

    print(f"Done. Processed {processed}, skipped {skipped}. Labels appended to {labels_csv_path}")


if __name__ == "__main__":
    main()
