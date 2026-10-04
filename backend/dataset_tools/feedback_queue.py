"""
feedback_queue.py — SignTalk AI / Person A (ML Core), Prompt A3

Active-learning feedback loop: misclassified sequences get logged here so
they can later be reviewed and folded back into the training set. This
module is intentionally dependency-free of FastAPI/HTTP — Person C's
backend calls record_correction() directly from its own
/analytics/feedback route, and the CLI tools use it standalone too.

corrections.csv columns: filename, predicted_label, correct_label,
                          session_id, timestamp
approved.csv columns:    filename  (one filename per row = approved for
                          retraining; simple allowlist, hand- or
                          UI-curated)
"""

from __future__ import annotations

import csv
import os
import time
import uuid

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
PENDING_REVIEW_DIRNAME = "pending_review"
CORRECTIONS_CSV_NAME = "corrections.csv"
APPROVED_CSV_NAME = "approved.csv"

CORRECTIONS_HEADER = ["filename", "predicted_label", "correct_label", "session_id", "timestamp"]
APPROVED_HEADER = ["filename"]


def _ensure_csv(path: str, header: list[str]) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def record_correction(
    session_id: str,
    predicted_label: str,
    correct_label: str,
    sequence: np.ndarray,
    data_dir: str = DEFAULT_DATA_DIR,
) -> str:
    """
    Saves a misclassified (30, 17, 2) keypoint sequence into pending_review/
    and appends a row to corrections.csv.

    Returns the saved .npy filename.
    """
    if sequence.shape != (30, 17, 2):
        raise ValueError(f"Expected sequence shape (30, 17, 2), got {sequence.shape}")

    pending_dir = os.path.join(data_dir, PENDING_REVIEW_DIRNAME)
    os.makedirs(pending_dir, exist_ok=True)

    timestamp = time.time()
    filename = f"correction_{correct_label}_{session_id}_{uuid.uuid4().hex[:8]}.npy"
    filepath = os.path.join(pending_dir, filename)
    np.save(filepath, sequence)

    corrections_csv_path = os.path.join(data_dir, CORRECTIONS_CSV_NAME)
    _ensure_csv(corrections_csv_path, CORRECTIONS_HEADER)
    with open(corrections_csv_path, "a", newline="") as f:
        csv.writer(f).writerow([filename, predicted_label, correct_label, session_id, timestamp])

    return filename


def load_approved_filenames(data_dir: str = DEFAULT_DATA_DIR) -> set[str]:
    """Reads approved.csv into a set of approved filenames. Returns an
    empty set if the file doesn't exist yet."""
    approved_csv_path = os.path.join(data_dir, APPROVED_CSV_NAME)
    if not os.path.exists(approved_csv_path):
        return set()
    with open(approved_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"] for row in reader}


def approve_correction(filename: str, data_dir: str = DEFAULT_DATA_DIR) -> None:
    """Appends a filename to approved.csv (idempotent — skips duplicates)."""
    approved_csv_path = os.path.join(data_dir, APPROVED_CSV_NAME)
    _ensure_csv(approved_csv_path, APPROVED_HEADER)
    if filename in load_approved_filenames(data_dir):
        return
    with open(approved_csv_path, "a", newline="") as f:
        csv.writer(f).writerow([filename])
