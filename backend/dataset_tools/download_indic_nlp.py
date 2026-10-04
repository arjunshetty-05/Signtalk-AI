"""
download_indic_nlp.py — SignTalk AI / dataset_tools

Pulls Indic-language translation/speech-translation corpora from Hugging
Face straight into the notebook's HF cache — no manual upload step.

"BhasaAnuvaad" here is AI4Bharat's NPTEL speech-translation corpus
(ai4bharat/NPTEL, config indic2en) — verified to exist on Hugging Face with
that config and per-language splits. A repo literally named
ai4bharat/BhasaAnuvaad returned 401/gated when checked while writing this
script, so don't assume that id works without checking huggingface.co
yourself first.

Neither this nor Samanantar is currently wired into a training script —
translation.py calls Google Translate / an offline phrasebook, and
nlp_correction.py calls Gemini / a pretrained Flan-T5-Small. These are
acquisition-only, for whenever a custom translation fine-tune is planned.

Usage (Colab/Kaggle):
    !pip install datasets
    !python dataset_tools/download_indic_nlp.py --nptel --split hindi
    !python dataset_tools/download_indic_nlp.py --samanantar --lang hi
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_common import default_output_dir  # noqa: E402

NPTEL_SPLITS = [
    "assamese", "bengali", "gujarati", "hindi", "kannada",
    "malayalam", "marathi", "tamil", "telugu",
]


def parse_args():
    p = argparse.ArgumentParser(description="Download Indic NLP datasets from Hugging Face")
    p.add_argument("--output_dir", default=None, help="Defaults to a Colab/Kaggle-aware path")
    p.add_argument("--nptel", action="store_true", help="Download the NPTEL (\"BhasaAnuvaad\") indic2en corpus")
    p.add_argument("--split", default="hindi", choices=NPTEL_SPLITS, help="NPTEL indic2en split")
    p.add_argument("--samanantar", action="store_true", help="Download Samanantar")
    p.add_argument("--lang", default="hi", help="Samanantar language config, e.g. hi, kn, ta")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.nptel and not args.samanantar:
        raise SystemExit("Pass --nptel and/or --samanantar")

    from datasets import load_dataset

    output_dir = args.output_dir or default_output_dir("indic_nlp")
    os.makedirs(output_dir, exist_ok=True)

    if args.nptel:
        print(f"Loading ai4bharat/NPTEL indic2en split={args.split}...")
        data = load_dataset("ai4bharat/NPTEL", "indic2en", split=args.split)
        dest = os.path.join(output_dir, f"nptel_{args.split}")
        data.save_to_disk(dest)
        print(f"Saved {len(data)} rows to {dest}")

    if args.samanantar:
        print(f"Loading ai4bharat/samanantar config={args.lang}...")
        data = load_dataset("ai4bharat/samanantar", args.lang)
        dest = os.path.join(output_dir, f"samanantar_{args.lang}")
        data.save_to_disk(dest)
        print(f"Saved samanantar[{args.lang}] to {dest}")


if __name__ == "__main__":
    main()
