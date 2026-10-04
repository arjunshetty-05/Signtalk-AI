"""Guided webcam recorder for a small personal sign-language dataset.

Records each requested label into the same keypoint format used by
record_samples.py and train_bilstm.py. The signer still performs every sign;
the script automates camera setup, countdowns, pauses, and label bookkeeping.
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
from dataset_tools.record_samples import (  # noqa: E402
    LABELS_CSV_HEADER,
    append_label_row,
    ensure_labels_csv,
    record_one_sample,
)
from keypoint_utils import load_movenet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Record a guided personal sign dataset")
    parser.add_argument(
        "--classes",
        default="hello,thank_you,yes,no,help",
        help="Comma-separated labels to record",
    )
    parser.add_argument("--signer_id", default="my_camera")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "my_webcam"))
    parser.add_argument("--samples", type=int, default=40, help="Samples per class")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--yes", action="store_true", help="Start without asking before each class")
    return parser.parse_args()


def main():
    args = parse_args()
    labels = [label.strip() for label in args.classes.split(",") if label.strip()]
    if not labels or args.samples < 1:
        raise SystemExit("Provide at least one class and one sample per class.")

    sequences_dir = os.path.join(args.data_dir, "sequences")
    labels_csv_path = os.path.join(args.data_dir, "labels.csv")
    os.makedirs(sequences_dir, exist_ok=True)
    ensure_labels_csv(labels_csv_path)

    print(f"Dataset: {args.data_dir}")
    print(f"Classes: {', '.join(labels)}")
    print(f"Samples per class: {args.samples}")
    if not args.yes:
        input("Position yourself, then press Enter to open the camera...")

    movenet_sig = load_movenet()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    try:
        for class_index, label in enumerate(labels, 1):
            print(f"\nClass {class_index}/{len(labels)}: {label}")
            if not args.yes:
                input(f"Press Enter, then perform only the '{label}' sign...")

            for sample_index in range(args.samples):
                print(f"Recording {sample_index + 1}/{args.samples} for {label}")
                sequence = record_one_sample(cap, movenet_sig, args.fps)
                timestamp = int(time.time() * 1000)
                filename = f"{label}_{args.signer_id}_{timestamp}.npy"
                np.save(os.path.join(sequences_dir, filename), sequence)
                append_label_row(labels_csv_path, filename, label, args.signer_id)
                if sample_index + 1 < args.samples:
                    time.sleep(args.pause)

        print(f"\nDone. Recorded {len(labels) * args.samples} samples.")
        print(f"Labels file: {labels_csv_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()