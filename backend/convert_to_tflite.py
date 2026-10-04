"""
convert_to_tflite.py — SignTalk AI / Person A (ML Core), Prompt A2

Converts a trained BiLSTM SavedModel into a float16-quantized .tflite file.

MoveNet Thunder .tflite:
    train_bilstm.py handles the BiLSTM export. MoveNet Thunder's TFLite
    build is NOT trained/converted here — it should be sourced directly
    from TF Hub's pre-converted lite model:
        https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4
    Download it once and cache it locally at backend/models/movenet.tflite
    (see download_movenet_tflite() below), rather than converting the
    full TF-Hub SavedModel yourself.

Usage:
    python convert_to_tflite.py --saved_model runs/exp1/saved_model \
        --output runs/exp1/bilstm.tflite
    python convert_to_tflite.py --download_movenet
"""

from __future__ import annotations

import argparse
import os
import urllib.request

import tensorflow as tf

MOVENET_TFLITE_URL = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
)
DEFAULT_MOVENET_CACHE_PATH = os.path.join(os.path.dirname(__file__), "models", "movenet.tflite")


def convert_bilstm_to_tflite(saved_model_dir: str, output_path: str) -> str:
    """Converts a BiLSTM SavedModel dir to a float16-quantized .tflite file."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote float16 BiLSTM TFLite model to {output_path}")
    return output_path


def download_movenet_tflite(cache_path: str = DEFAULT_MOVENET_CACHE_PATH) -> str:
    """Downloads MoveNet Thunder's pre-converted float16 TFLite build from
    TF Hub and caches it locally. Safe to call repeatedly (skips if cached)."""
    if os.path.exists(cache_path):
        print(f"MoveNet TFLite already cached at {cache_path}")
        return cache_path

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Downloading MoveNet Thunder TFLite from {MOVENET_TFLITE_URL} ...")
    urllib.request.urlretrieve(MOVENET_TFLITE_URL, cache_path)
    print(f"Cached MoveNet TFLite at {cache_path}")
    return cache_path


def parse_args():
    p = argparse.ArgumentParser(description="Convert SignTalk AI models to TFLite")
    p.add_argument("--saved_model", help="Path to BiLSTM SavedModel directory")
    p.add_argument("--output", help="Output .tflite path for the BiLSTM model")
    p.add_argument("--download_movenet", action="store_true",
                    help="Download and cache MoveNet Thunder's TFLite build")
    return p.parse_args()


def main():
    args = parse_args()
    if args.download_movenet:
        download_movenet_tflite()
    if args.saved_model and args.output:
        convert_bilstm_to_tflite(args.saved_model, args.output)
    if not args.download_movenet and not (args.saved_model and args.output):
        print("Nothing to do — pass --download_movenet and/or --saved_model + --output.")


if __name__ == "__main__":
    main()
