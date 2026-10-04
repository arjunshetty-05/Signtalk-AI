"""
download_include.py — SignTalk AI / dataset_tools

Downloads the INCLUDE Indian Sign Language dataset (Zenodo, open — no
license request needed) and unzips it.

There is NOT a single INCLUDE.zip (an earlier version of this script
assumed there was, and 404'd — verified against
https://zenodo.org/api/records/4010759 directly). The dataset is 44 files:
42 per-category zip parts (~50GB total across all 15 word categories),
plus README.md and download_data.sh. Downloading everything is rarely what
you want for a first pass — use --categories to pick a subset, or --list
to see sizes before committing.

Only feeds train_bilstm.py's dataset AFTER preprocessing: INCLUDE ships raw
video clips, not (30, 17, 2) keypoint sequences. See preprocess_include.py
for the video -> sequence conversion step.

Usage:
    python dataset_tools/download_include.py --list
    python dataset_tools/download_include.py --categories Seasons
    python dataset_tools/download_include.py --categories Greetings Colours
    python dataset_tools/download_include.py --categories all   # ~50GB, hours
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_common import default_output_dir, download_file  # noqa: E402

ZENODO_RECORD = "4010759"
ZENODO_FILES_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

# category -> number of split zip parts (Category_XofN.zip), from the
# record's own file listing (44 files: 42 category parts + README +
# download_data.sh).
CATEGORY_PARTS = {
    "Adjectives": 8,
    "Animals": 2,
    "Clothes": 2,
    "Colours": 2,
    "Days_and_Time": 3,
    "Electronics": 2,
    "Greetings": 2,
    "Home": 4,
    "Jobs": 2,
    "Means_of_Transportation": 2,
    "People": 5,
    "Places": 4,
    "Pronouns": 2,
    "Seasons": 1,
    "Society": 3,
}


def parse_args():
    p = argparse.ArgumentParser(description="Download + unzip INCLUDE ISL dataset categories")
    p.add_argument("--output_dir", default=None,
                    help="Defaults to a Colab/Kaggle-aware path (see download_common.default_output_dir)")
    p.add_argument("--categories", nargs="+", default=None,
                    help=f"Category names to download, or 'all' (~50GB). One of: {sorted(CATEGORY_PARTS)}")
    p.add_argument("--keep_zip", action="store_true", help="Don't delete zip parts after extracting")
    p.add_argument("--list", action="store_true", help="Print categories + part counts and exit, no download")
    return p.parse_args()


def _category_urls(category: str) -> list[tuple[str, str]]:
    """Returns [(filename, url), ...] for every part of `category`."""
    n = CATEGORY_PARTS[category]
    return [
        (f"{category}_{i}of{n}.zip", f"{ZENODO_FILES_BASE}/{category}_{i}of{n}.zip/content")
        for i in range(1, n + 1)
    ]


def main():
    args = parse_args()

    if args.list or not args.categories:
        print("INCLUDE categories (name: parts):")
        for name, n in CATEGORY_PARTS.items():
            print(f"  {name}: {n} part(s)")
        if not args.categories:
            print("\nPass --categories <name> [<name> ...] or --categories all to download.")
        return

    categories = list(CATEGORY_PARTS) if args.categories == ["all"] else args.categories
    unknown = [c for c in categories if c not in CATEGORY_PARTS]
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}. Valid: {sorted(CATEGORY_PARTS)}")

    output_dir = args.output_dir or default_output_dir("include")
    os.makedirs(output_dir, exist_ok=True)

    for category in categories:
        for filename, url in _category_urls(category):
            zip_path = os.path.join(output_dir, filename)
            download_file(url, zip_path)

            print(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(output_dir)

            if not args.keep_zip:
                os.remove(zip_path)

    print(f"Done. INCLUDE categories {categories} extracted to: {output_dir}")


if __name__ == "__main__":
    main()
