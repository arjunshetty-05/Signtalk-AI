"""Tiny pose-sequence classifier (PROJECT_CONTEXT Sections 5.5 / M1).

Phase-1 "walking skeleton" model: a small 2-layer bidirectional GRU over the
per-frame feature sequence produced by :mod:`signtalk_core.features`. It is kept
deliberately tiny so it trains and runs on the project's RTX 4050 (~6 GB VRAM)
target; random-init weights are acceptable for Gate 1 (accuracy is irrelevant
for the skeleton — PROJECT_CONTEXT Section 0.3).

The module is shaped so that an RGB-crop branch (M3, currently skipped) can be
added later as a SEPARATE ensemble member: fusion happens over probability
vectors, not inside this model, so a second model's softmax output can simply be
averaged alongside this one without touching this file. Do NOT build RGB now.

Tensor-shape conventions (see docstrings):
    x: float32 ``[B, T, F]``  (batch, time, per-frame features = feature_dim())
    logits: float32 ``[B, num_classes]``
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from signtalk_core.features import feature_dim


class PoseGRU(nn.Module):
    """A small 2-layer BiGRU classifier over pose/hand feature sequences.

    Args:
        num_classes: number of output sign classes (len of the vocabulary).
        feature_dim: per-frame feature dimensionality ``F``. Defaults to
            :func:`signtalk_core.features.feature_dim` so training and serving
            share one definition (Rule 12.3).
        hidden_size: GRU hidden size per direction. Kept small for the 6 GB
            target.
        num_layers: number of stacked GRU layers.
        dropout: dropout applied between GRU layers and before the classifier.

    Shapes:
        input  x: float32 ``[B, T, F]``
        output   : float32 ``[B, num_classes]`` (raw logits)
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = feature_dim(),
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # 2 * hidden_size because the GRU is bidirectional.
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Args:
            x: float32 ``[B, T, F]`` feature sequence.

        Returns:
            float32 ``[B, num_classes]`` raw logits (apply softmax downstream).
        """
        if x.dim() != 3:
            raise ValueError(f"expected x of shape [B, T, F], got {tuple(x.shape)}")
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: model expects F={self.feature_dim}, "
                f"got {x.shape[-1]}"
            )
        # outputs: [B, T, 2*hidden]; we take the last timestep of both directions.
        outputs, _ = self.gru(x)
        pooled = outputs[:, -1, :]  # [B, 2*hidden]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_model(
    num_classes: int,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> PoseGRU:
    """Construct the pose model, loading weights from a checkpoint if present.

    This is the single construction entry point used by serving (and, later,
    training). If ``checkpoint_path`` exists it is loaded; otherwise the model
    runs with random-init weights, which is acceptable for the Phase-1 skeleton
    (PROJECT_CONTEXT Section 0.3). Kept generic so an RGB branch can later be
    built as a separate ensemble member and fused at the probability level.

    Args:
        num_classes: number of sign classes.
        checkpoint_path: optional path to a ``.pt`` state_dict. Missing/``None``
            -> random init.
        device: torch device for the model.

    Returns:
        A :class:`PoseGRU` in eval mode on ``device``.
    """
    model = PoseGRU(num_classes=num_classes)
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if ckpt.is_file():
            state = torch.load(ckpt, map_location=device)
            # Accept either a bare state_dict or a {"state_dict": ...} wrapper.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
