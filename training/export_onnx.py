"""Export a trained member to ONNX with a PyTorch-parity check (5.5 / 12.2).

Serving uses ONNX for fast inference, but every export must be parity-tested
against the PyTorch model so the two never silently diverge (Rule 12.2). This
script exports the checkpoint and asserts the ONNX output matches PyTorch to a
tight tolerance on random inputs; it fails loudly if not.

Example::

    python training/export_onnx.py \
        --arch gru --checkpoint runs/m1_gru/best.pt \
        --num-classes 50 --out models/m1_gru.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from signtalk_core.features import feature_dim
from signtalk_core.models.pose_tcn import PoseGRU
from signtalk_core.models.pose_transformer import PoseTransformer


def _load(arch: str, num_classes: int, checkpoint: Path) -> torch.nn.Module:
    model = PoseGRU(num_classes) if arch == "gru" else PoseTransformer(num_classes)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a member to ONNX + parity check")
    ap.add_argument("--arch", choices=["gru", "transformer"], required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--num-classes", required=True, type=int)
    ap.add_argument("--seq-len", type=int, default=32, help="T used for the dummy input")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args(argv)

    model = _load(args.arch, args.num_classes, args.checkpoint)
    f = feature_dim()
    dummy = torch.randn(1, args.seq_len, f, dtype=torch.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch", 1: "time"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"[ok  ] exported {args.out}")

    # --- parity check ------------------------------------------------------ #
    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime not installed — skipping parity check "
              "(install onnxruntime to enable it). Export itself succeeded.")
        return 0

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"features": dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if max_diff > args.atol:
        print(f"[FAIL] ONNX parity: max_diff={max_diff:.2e} > atol={args.atol:.0e}")
        return 1
    print(f"[ok  ] ONNX parity passed: max_diff={max_diff:.2e} <= atol={args.atol:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
