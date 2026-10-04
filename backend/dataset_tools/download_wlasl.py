"""
download_wlasl.py — SignTalk AI / dataset_tools

Clones the WLASL repo and, optionally, runs its own start_kit/
video_downloader.py to pull each clip from its original source URL. WLASL
is not a single archive — cloning only gets you the metadata JSON and the
downloader script; the actual video fetch is a separate, long-running step
that this script can kick off but doesn't reimplement (it's WLASL's own
third-party code).

Known dataset issue (not a bug here): some source videos have gone offline
since WLASL was published, so expect a small percentage of missing clips
after running the downloader — that's normal for this dataset.

Like INCLUDE, only feeds train_bilstm.py after a video -> (30, 17, 2)
keypoint preprocessing step that doesn't exist yet in this repo.

Usage (Colab/Kaggle):
    !python dataset_tools/download_wlasl.py
    !python dataset_tools/download_wlasl.py --run_downloader   # also runs video_downloader.py (slow, hours)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_common import default_output_dir, git_clone, run  # noqa: E402

WLASL_REPO_URL = "https://github.com/dxli94/WLASL.git"


def parse_args():
    p = argparse.ArgumentParser(description="Clone WLASL + optionally run its video downloader")
    p.add_argument("--output_dir", default=None, help="Defaults to a Colab/Kaggle-aware path")
    p.add_argument("--run_downloader", action="store_true",
                    help="Also run start_kit/video_downloader.py after cloning (slow — can take hours, "
                         "and some clips will be missing since source videos go offline over time)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir("wlasl")
    repo_dir = os.path.join(output_dir, "WLASL")

    git_clone(WLASL_REPO_URL, repo_dir)

    downloader = os.path.join(repo_dir, "start_kit", "video_downloader.py")
    if args.run_downloader:
        if not os.path.exists(downloader):
            raise FileNotFoundError(
                f"video_downloader.py not found at {downloader} — WLASL repo layout may have changed"
            )
        run([sys.executable, "video_downloader.py"], cwd=os.path.dirname(downloader))
    else:
        print(f"Cloned WLASL to {repo_dir}.")
        print("Next step (not run automatically — this is slow, expect hours):")
        print(f"  cd {os.path.dirname(downloader)} && python video_downloader.py")
        print("Or re-run this script with --run_downloader.")


if __name__ == "__main__":
    main()
