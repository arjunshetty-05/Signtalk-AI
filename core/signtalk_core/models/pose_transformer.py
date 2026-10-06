"""Second ensemble member M2 — a small Transformer encoder (Section 5.5 / M2).

A diverse second model for the ensemble (PROJECT_CONTEXT L8: different models
make different mistakes, so averaging their calibrated probabilities cancels
many errors). Where :class:`signtalk_core.models.pose_tcn.PoseGRU` is recurrent,
this is a small self-attention encoder over the same per-frame feature sequence,
so the two members are architecturally diverse while sharing one feature
definition (Rule 12.3).

Kept deliberately small (Section 5.5 suggests ~4 layers, d_model 128, 4 heads)
so it trains and runs on the RTX 4050 (~6 GB) target. Pretrained initialisation
(OpenHands, L6) is a later, optional step — random init is fine to start.

Tensor-shape conventions:
    x: float32 ``[B, T, F]``  (batch, time, per-frame features = feature_dim())
    logits: float32 ``[B, num_classes]``
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from signtalk_core.features import feature_dim


class _PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding added to the input.

    Shapes: input/output float32 ``[B, T, d_model]``.
    """

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class PoseTransformer(nn.Module):
    """A small Transformer-encoder classifier over pose/hand feature sequences.

    Args:
        num_classes: number of output sign classes.
        feature_dim: per-frame feature dimensionality ``F`` (defaults to the
            shared :func:`signtalk_core.features.feature_dim`).
        d_model: internal embedding width.
        nhead: number of attention heads.
        num_layers: number of stacked encoder layers.
        dim_feedforward: feed-forward width inside each encoder layer.
        dropout: dropout rate.

    Shapes:
        input  x: float32 ``[B, T, F]``
        output   : float32 ``[B, num_classes]`` (raw logits)
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = feature_dim(),
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

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
        h = self.input_proj(x)             # [B, T, d_model]
        h = self.pos_enc(h)
        h = self.encoder(h)                # [B, T, d_model]
        pooled = self.norm(h.mean(dim=1))  # mean-pool over time -> [B, d_model]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_transformer(
    num_classes: int,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> PoseTransformer:
    """Construct the Transformer model, loading a checkpoint if present.

    Mirrors :func:`signtalk_core.models.pose_tcn.build_model` so the two
    ensemble members are constructed and loaded identically.

    Args:
        num_classes: number of sign classes.
        checkpoint_path: optional ``.pt`` state_dict path; missing/``None`` ->
            random init.
        device: torch device.

    Returns:
        A :class:`PoseTransformer` in eval mode on ``device``.
    """
    model = PoseTransformer(num_classes=num_classes)
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if ckpt.is_file():
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
