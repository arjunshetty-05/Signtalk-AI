"""
download_indic_tts.py — SignTalk AI / dataset_tools

Clones AI4Bharat's Indic-TTS repo. Pretrained checkpoints aren't at a fixed
URL — they're linked from that repo's README and change over time, so this
script does not guess a checkpoint URL; check the cloned README yourself
and pass the specific one you need via --checkpoint_url.

Not currently wired into tts.py: the offline TTS path there
(COQUI_MODEL_MAP in tts.py) uses Coqui TTS's own pretrained multilingual
model ("tts_models/multilingual/multi-dataset/your_tts"), auto-downloaded
by the TTS library itself — it does not reference AI4Bharat's Indic-TTS at
all today. This script is for a future swap to dedicated Hindi/Kannada
voices, not something the live pipeline currently loads.

Usage (Colab/Kaggle):
    !python dataset_tools/download_indic_tts.py
    !python dataset_tools/download_indic_tts.py --checkpoint_url https://.../hi_male.pth
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_common import default_output_dir, download_file, git_clone, run  # noqa: E402

INDIC_TTS_REPO_URL = "https://github.com/AI4Bharat/Indic-TTS.git"


def parse_args():
    p = argparse.ArgumentParser(description="Clone AI4Bharat Indic-TTS + optionally fetch a checkpoint")
    p.add_argument("--output_dir", default=None, help="Defaults to a Colab/Kaggle-aware path")
    p.add_argument("--install_requirements", action="store_true",
                    help="pip install -r requirements.txt inside the cloned repo")
    p.add_argument("--checkpoint_url", default=None,
                    help="Direct URL to a checkpoint file linked from the repo's README (check it yourself — "
                         "there is no single stable URL)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir("indic_tts")
    repo_dir = os.path.join(output_dir, "Indic-TTS")

    git_clone(INDIC_TTS_REPO_URL, repo_dir)

    if args.install_requirements:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo_dir)

    if args.checkpoint_url:
        checkpoints_dir = os.path.join(repo_dir, "checkpoints")
        filename = args.checkpoint_url.rstrip("/").rsplit("/", 1)[-1]
        download_file(args.checkpoint_url, os.path.join(checkpoints_dir, filename))
    else:
        print(f"Cloned to {repo_dir}.")
        print(f"Pretrained checkpoints are linked from its README — open {repo_dir}/README.md, "
              "find the link you need, and re-run with --checkpoint_url <url>.")


if __name__ == "__main__":
    main()
