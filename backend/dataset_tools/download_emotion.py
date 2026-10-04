"""
download_emotion.py — SignTalk AI / dataset_tools

RAF-DB requires a signed license agreement — there is no direct-download
URL, so this script cannot fetch it automatically. It prints the request
steps and, once you have a link emailed to you, unzips it from a local
path or URL you pass in.

FER2013 is the zero-wait fallback (weaker accuracy, but instantly
downloadable via kagglehub/Kaggle CLI) — the same role INCLUDE plays as an
open fallback elsewhere.

Neither dataset is currently wired into a training script in this repo:
emotion.py calls DeepFace's pretrained model at inference time and has no
fine-tuning path. These downloads are acquisition-only, for whenever that
training script gets written.

Usage (Colab/Kaggle):
    !python dataset_tools/download_emotion.py --fer2013
    !python dataset_tools/download_emotion.py --raf_db_archive /content/drive/MyDrive/RAF-DB.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_common import default_output_dir, download_file  # noqa: E402

RAF_DB_REQUEST_URL = "http://www.whdeng.cn/RAF/model1.html"
FER2013_KAGGLE_ID = "msambare/fer2013"


def parse_args():
    p = argparse.ArgumentParser(description="Acquire emotion-recognition datasets (RAF-DB gated, FER2013 open)")
    p.add_argument("--output_dir", default=None, help="Defaults to a Colab/Kaggle-aware path")
    p.add_argument("--fer2013", action="store_true", help="Download FER2013 via kagglehub (needs Kaggle credentials)")
    p.add_argument("--raf_db_archive", default=None,
                    help="Local path or URL to a RAF-DB zip you already obtained via the license request "
                         "flow — unzips it into <output_dir>/raf_db")
    return p.parse_args()


def _print_raf_db_instructions() -> None:
    print("RAF-DB is not directly downloadable — it requires a signed license agreement:")
    print(f"  1. Go to {RAF_DB_REQUEST_URL}")
    print("  2. Fill out the request form (academic use, your institution)")
    print("  3. Wait for an email with the download link (usually faster than similarly gated datasets)")
    print("  4. Re-run this script with --raf_db_archive <path-or-url-they-emailed-you>")


def _fetch_fer2013(output_dir: str) -> None:
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed. Either:")
        print("  pip install kagglehub")
        print(f"  or: kaggle datasets download -d {FER2013_KAGGLE_ID} -p {output_dir} --unzip")
        return
    path = kagglehub.dataset_download(FER2013_KAGGLE_ID)
    print(f"FER2013 cached by kagglehub at: {path}")


def _fetch_raf_db(archive: str, output_dir: str) -> None:
    dest_dir = os.path.join(output_dir, "raf_db")
    os.makedirs(dest_dir, exist_ok=True)
    if archive.startswith("http://") or archive.startswith("https://"):
        zip_path = os.path.join(output_dir, "raf_db.zip")
        download_file(archive, zip_path)
        archive = zip_path
    print(f"Extracting {archive} to {dest_dir}...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    print(f"RAF-DB extracted to {dest_dir}")


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir("emotion")
    os.makedirs(output_dir, exist_ok=True)

    if not args.fer2013 and not args.raf_db_archive:
        _print_raf_db_instructions()
        print()
        print("No action taken — pass --fer2013 for the instant fallback, "
              "or --raf_db_archive once you have RAF-DB's emailed link.")
        return

    if args.raf_db_archive:
        _fetch_raf_db(args.raf_db_archive, output_dir)

    if args.fer2013:
        _fetch_fer2013(output_dir)


if __name__ == "__main__":
    main()
