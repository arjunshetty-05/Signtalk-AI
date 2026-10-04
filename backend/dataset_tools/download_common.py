"""
download_common.py — SignTalk AI / dataset_tools

Shared helpers for the download_*.py dataset-acquisition scripts. Not a
CLI itself — imported by download_include.py, download_wlasl.py,
download_emotion.py, download_indic_nlp.py, download_indic_tts.py.
"""

from __future__ import annotations

import os
import subprocess
import time

import requests


def default_output_dir(dataset_name: str) -> str:
    """
    Colab/Kaggle-aware default download location — prefers the notebook
    environment's local disk (fast, ephemeral, no re-upload from a laptop)
    over this repo. Falls back to backend/data/raw/<name> (gitignored)
    when run locally.
    """
    if os.path.isdir("/content"):
        return f"/content/datasets/{dataset_name}"
    if os.path.isdir("/kaggle/working"):
        return f"/kaggle/working/datasets/{dataset_name}"
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(backend_dir, "data", "raw", dataset_name)


def _download_attempt(url: str, dest_path: str, tmp_path: str, chunk_size: int) -> None:
    """One resume-aware streaming attempt. Raises on any connection error —
    the partial .part file is left in place either way, so the caller can
    just retry."""
    resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
        resp.raise_for_status()
        resumed = resume_from > 0 and resp.status_code == 206
        if resume_from > 0 and not resumed:
            print(f"Server doesn't support resume for {os.path.basename(dest_path)} — restarting from scratch.")
            resume_from = 0

        content_length = int(resp.headers.get("content-length", 0))
        total = (resume_from + content_length) if content_length else 0
        written = resume_from

        mode = "ab" if resumed else "wb"
        with open(tmp_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r{os.path.basename(dest_path)}: {written / 1e6:,.0f}MB / {total / 1e6:,.0f}MB ({pct:.1f}%)", end="")
                else:
                    print(f"\r{os.path.basename(dest_path)}: {written / 1e6:,.0f}MB", end="")
    print()

    final_size = os.path.getsize(tmp_path)
    if total and final_size != total:
        raise requests.exceptions.ConnectionError(
            f"Download incomplete for {dest_path}: got {final_size} bytes, expected {total}"
        )


def download_file(url: str, dest_path: str, chunk_size: int = 1 << 20, max_retries: int = 10) -> str:
    """
    Streams `url` to `dest_path` with progress printed to stdout, resuming
    an interrupted download via HTTP Range rather than restarting.

    dest_path only ever exists once the transfer is verified complete
    (size matches the server's Content-Length) — a run that gets cut off
    mid-stream (dropped connection, closed laptop lid, etc.) leaves a
    `dest_path + ".part"` file behind instead, so a truncated download can
    never be mistaken for a finished one on the next run. If the server
    doesn't support Range (falls back to a 200 instead of 206), the partial
    file is discarded and the download restarts from scratch.

    Retries up to `max_retries` times on connection errors, resuming from
    wherever the .part file left off each time — some hosts (Zenodo, in
    practice) drop long-lived connections every minute or two on large
    files, so a single-shot attempt routinely fails partway through even
    though the file itself downloads fine in pieces.
    """
    if os.path.exists(dest_path):
        print(f"Already downloaded, skipping: {dest_path}")
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"

    for attempt in range(1, max_retries + 1):
        try:
            _download_attempt(url, dest_path, tmp_path, chunk_size)
            break
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise
            resumed_at = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            print(f"[retry {attempt}/{max_retries}] {exc} — resuming from {resumed_at / 1e6:,.0f}MB in 3s...")
            time.sleep(3)

    os.replace(tmp_path, dest_path)
    return dest_path


def run(cmd: list[str], **kwargs) -> None:
    """Thin subprocess wrapper — prints the command before running it, and
    raises on non-zero exit rather than continuing silently on failure."""
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def git_clone(repo_url: str, dest_dir: str) -> str:
    """Clones repo_url into dest_dir, or pulls latest if already cloned."""
    if os.path.isdir(os.path.join(dest_dir, ".git")):
        print(f"Repo already cloned at {dest_dir}, pulling latest...")
        run(["git", "-C", dest_dir, "pull"])
    else:
        parent = os.path.dirname(os.path.abspath(dest_dir))
        os.makedirs(parent, exist_ok=True)
        run(["git", "clone", repo_url, dest_dir])
    return dest_dir
