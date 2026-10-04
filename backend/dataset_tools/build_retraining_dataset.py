"""
build_retraining_dataset.py — SignTalk AI / Person A (ML Core), Prompt A3

Folds reviewed/approved corrections from the feedback queue back into the
main training dataset, in train_bilstm.py's expected format (appends to
labels.csv rather than overwriting it).

Usage:
    python dataset_tools/build_retraining_dataset.py --data_dir data --approved-only
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_tools.feedback_queue import (  # noqa: E402
    APPROVED_CSV_NAME,
    CORRECTIONS_CSV_NAME,
    PENDING_REVIEW_DIRNAME,
    load_approved_filenames,
)

LABELS_HEADER = ["filename", "class", "signer_id"]


def parse_args():
    p = argparse.ArgumentParser(description="Merge approved corrections into the main dataset")
    p.add_argument("--data_dir", required=True, help="Root dataset directory (has sequences/, labels.csv, corrections.csv)")
    p.add_argument("--approved-only", action="store_true",
                    help="Only merge corrections whose filename appears in approved.csv")
    p.add_argument("--default_signer_id", default="corrected",
                    help="signer_id recorded for merged correction samples")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    sequences_dir = os.path.join(data_dir, "sequences")
    pending_dir = os.path.join(data_dir, PENDING_REVIEW_DIRNAME)
    corrections_csv_path = os.path.join(data_dir, CORRECTIONS_CSV_NAME)
    labels_csv_path = os.path.join(data_dir, "labels.csv")

    if not os.path.exists(corrections_csv_path):
        print(f"No corrections.csv found at {corrections_csv_path} — nothing to merge.")
        return

    os.makedirs(sequences_dir, exist_ok=True)
    if not os.path.exists(labels_csv_path):
        with open(labels_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(LABELS_HEADER)

    approved = load_approved_filenames(data_dir) if args.approved_only else None

    merged, skipped = 0, 0
    with open(corrections_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(labels_csv_path, "a", newline="") as out_f:
        writer = csv.writer(out_f)
        for row in rows:
            filename = row["filename"]
            correct_label = row["correct_label"]

            if args.approved_only and filename not in approved:
                skipped += 1
                continue

            src_path = os.path.join(pending_dir, filename)
            dst_path = os.path.join(sequences_dir, filename)
            if not os.path.exists(src_path):
                print(f"[warn] missing pending file, skipping: {src_path}")
                skipped += 1
                continue

            shutil.copy2(src_path, dst_path)
            writer.writerow([filename, correct_label, args.default_signer_id])
            merged += 1

    print(f"Merged {merged} corrections into {labels_csv_path} ({skipped} skipped).")
    if args.approved_only and not os.path.exists(os.path.join(data_dir, APPROVED_CSV_NAME)):
        print(f"[note] --approved-only was set but no {APPROVED_CSV_NAME} exists yet — nothing was eligible.")


if __name__ == "__main__":
    main()
