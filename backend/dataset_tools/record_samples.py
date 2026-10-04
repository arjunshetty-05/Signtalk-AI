"""
record_samples.py — SignTalk AI / Person A (ML Core), Prompt A3

CLI tool for recording (30, 17, 2) keypoint-sequence samples straight into
the dataset format train_bilstm.py expects. Reuses keypoint_utils.py's
extraction + normalization — does not reimplement pose logic.

Usage:
    python dataset_tools/record_samples.py --class HELLO --signer_id s01
    python dataset_tools/record_samples.py --class HELLO --signer_id s01 --batch 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keypoint_utils import (  # noqa: E402
    SequenceBuffer,
    TemporalSmoother,
    extract_keypoints,
    load_movenet,
    normalize_keypoints,
)

SEQUENCE_LENGTH = 30
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LABELS_CSV_HEADER = ["filename", "class", "signer_id"]


def parse_args():
    p = argparse.ArgumentParser(description="Record a labeled gesture sample from webcam")
    p.add_argument("--class", dest="class_label", required=True, help="Sign class label")
    p.add_argument("--signer_id", required=True, help="Signer/session identifier")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Root dataset directory")
    p.add_argument("--fps", type=int, default=25, help="Target capture FPS")
    p.add_argument("--batch", type=int, default=1, help="Record N consecutive samples")
    p.add_argument("--pause", type=float, default=2.0, help="Seconds paused between batch samples")
    p.add_argument("--camera_index", type=int, default=0)
    return p.parse_args()


def ensure_labels_csv(labels_csv_path: str) -> None:
    if not os.path.exists(labels_csv_path):
        os.makedirs(os.path.dirname(labels_csv_path), exist_ok=True)
        with open(labels_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(LABELS_CSV_HEADER)


def append_label_row(labels_csv_path: str, filename: str, class_label: str, signer_id: str) -> None:
    with open(labels_csv_path, "a", newline="") as f:
        csv.writer(f).writerow([filename, class_label, signer_id])


def record_one_sample(cap, movenet_sig, fps: int) -> np.ndarray:
    """Records a fixed-length (SEQUENCE_LENGTH, 17, 2) clip with a live
    on-screen countdown + recording indicator."""
    smoother = TemporalSmoother(window=4)
    buffer = SequenceBuffer(maxlen=SEQUENCE_LENGTH)
    frame_delay = 1.0 / fps

    # Countdown before recording starts
    for count in (3, 2, 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Webcam read failed during countdown")
        display = frame.copy()
        cv2.putText(display, f"Starting in {count}...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow("SignTalk AI — Record Sample", display)
        cv2.waitKey(1)
        time.sleep(1.0)

    print("Recording...")
    frames_captured = 0
    while frames_captured < SEQUENCE_LENGTH:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Webcam read failed during recording")

        raw_kp = extract_keypoints(frame, movenet_sig)
        normalized = normalize_keypoints(raw_kp)
        smoothed = smoother.smooth(normalized)
        buffer.push(smoothed)
        frames_captured += 1

        display = frame.copy()
        cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)  # recording indicator
        cv2.putText(display, f"REC {frames_captured}/{SEQUENCE_LENGTH}", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("SignTalk AI — Record Sample", display)
        cv2.waitKey(1)

        elapsed = time.time() - t0
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    sequence = buffer.get_sequence()
    if sequence is None:
        raise RuntimeError("Buffer did not fill to full sequence length")
    return sequence


def main():
    args = parse_args()
    data_dir = args.data_dir
    sequences_dir = os.path.join(data_dir, "sequences")
    labels_csv_path = os.path.join(data_dir, "labels.csv")
    os.makedirs(sequences_dir, exist_ok=True)
    ensure_labels_csv(labels_csv_path)

    movenet_sig = load_movenet()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    try:
        for i in range(args.batch):
            print(f"Sample {i + 1}/{args.batch} — class={args.class_label} signer={args.signer_id}")
            sequence = record_one_sample(cap, movenet_sig, args.fps)

            timestamp = int(time.time() * 1000)
            filename = f"{args.class_label}_{args.signer_id}_{timestamp}.npy"
            filepath = os.path.join(sequences_dir, filename)
            np.save(filepath, sequence)
            append_label_row(labels_csv_path, filename, args.class_label, args.signer_id)
            print(f"Saved {filepath}")

            if args.batch > 1 and i < args.batch - 1:
                print(f"Pausing {args.pause}s before next sample...")
                time.sleep(args.pause)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
