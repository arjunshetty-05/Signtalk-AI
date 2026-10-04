"""
dataset_stats.py — SignTalk AI / Person A (ML Core), Prompt A3

Reads labels.csv and reports dataset health: total/per-class/per-signer
counts, a class-balance bar chart, and a flag for under-represented classes.

Usage:
    python dataset_tools/dataset_stats.py --labels_csv data/labels.csv --min_samples 20
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Report dataset statistics from labels.csv")
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--output_dir", default=None, help="Defaults to labels_csv's directory")
    p.add_argument("--min_samples", type=int, default=20,
                    help="Classes with fewer samples than this are flagged")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.labels_csv))
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    required_cols = {"filename", "class", "signer_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing required columns: {missing}")

    total = len(df)
    per_class = df["class"].value_counts().sort_index()
    per_signer = df["signer_id"].value_counts().sort_index()

    print(f"Total samples: {total}")
    print(f"Unique classes: {df['class'].nunique()}")
    print(f"Unique signers: {df['signer_id'].nunique()}")
    print("\nPer-class counts:")
    print(per_class.to_string())
    print("\nPer-signer counts:")
    print(per_signer.to_string())

    under_represented = per_class[per_class < args.min_samples]
    if len(under_represented) > 0:
        print(f"\n[FLAG] Classes below min_samples={args.min_samples}:")
        print(under_represented.to_string())
    else:
        print(f"\nAll classes meet the minimum of {args.min_samples} samples.")

    # Class-balance bar chart
    plt.figure(figsize=(max(6, len(per_class) * 0.5), 5))
    colors = ["#e74c3c" if c < args.min_samples else "#2ecc71" for c in per_class.values]
    plt.bar(per_class.index.astype(str), per_class.values, color=colors)
    plt.axhline(args.min_samples, color="gray", linestyle="--", label=f"min={args.min_samples}")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Sample count")
    plt.title("Class balance")
    plt.legend()
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "class_balance.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nSaved class-balance chart to {chart_path}")


if __name__ == "__main__":
    main()
