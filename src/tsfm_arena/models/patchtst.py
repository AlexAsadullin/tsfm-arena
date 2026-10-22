"""A small PatchTST-style model (Nie et al. 2023) with a point head and monotone quantile heads.

Each series is normalised by its own context mean and standard deviation (instance normalisation), cut
into patches, embedded, passed through a Transformer encoder and decoded into ``1 + Q`` outputs for the
cumulative ``horizon``-bar return: a point score and ``Q`` quantiles that are monotone by construction
(the first quantile plus a cumulative sum of softplus increments).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor, nn

from tsfm_arena.data.panel import FloatArray, ReturnPanel
from tsfm_arena.models.base import Forecast

_EPS = 1e-8  # numerical guard against zero context volatility


class PatchTSTConfig(BaseModel):
    context_length: int
    patch_len: int
    stride: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    horizon: int
    levels: list[float]

    @property
    def n_patches(self) -> int:
        return (self.context_length - self.patch_len) // self.stride + 1


class PatchTSTNet(nn.Module):
    def __init__(self, cfg: PatchTSTConfig) -> None:
        super().__init__()
        if cfg.patch_len > cfg.context_length:
            raise ValueError("patch_len must not exceed context_length")
        self.cfg = cfg
        self.embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.position = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(cfg.n_patches * cfg.d_model, 1 + len(cfg.levels))

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Map ``(B, L)`` returns (NaN = missing) to a point score ``(B,)`` and quantiles ``(B, Q)``,
        both in return units of the cumulative horizon return."""
        mask = ~torch.isnan(context)
        filled = torch.nan_to_num(context)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        loc = filled.sum(dim=1, keepdim=True) / count
        var = (((filled - loc) * mask) ** 2).sum(dim=1, keepdim=True) / count
        scale = var.sqrt() + _EPS
        normed = ((filled - loc) / scale) * mask
        patches = normed.unfold(1, self.cfg.patch_len, self.cfg.stride)
        hidden = self.encoder(self.embed(patches) + self.position)
        out = self.head(hidden.flatten(1))
        point = out[:, 0]
        quant = torch.cat([out[:, 1:2], nn.functional.softplus(out[:, 2:])], dim=1).cumsum(dim=1)
        h = self.cfg.horizon
        shift, stretch = h * loc, scale * float(np.sqrt(h))
        return shift[:, 0] + stretch[:, 0] * point, shift + stretch * quant


class PatchTSTForecaster:
    """Adapter that serves a trained :class:`PatchTSTNet` through the ``Forecaster`` protocol."""

    needs_fit = False

    def __init__(self, name: str, net: PatchTSTNet, meta: dict[str, Any] | None = None) -> None:
        self.name = name
        self.net = net.eval()
        self.context_length = net.cfg.context_length
        self.meta = meta or {}

    def fit(self, history: ReturnPanel, horizon: int) -> None:
        """Trained once, offline; nothing to fit at evaluation time."""

    def predict(self, context: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast:
        cfg = self.net.cfg
        if horizon != cfg.horizon:
            raise ValueError(f"{self.name} was trained for horizon {cfg.horizon}, not {horizon}")
        if list(levels) != cfg.levels:
            raise ValueError(f"{self.name} was trained for quantile levels {cfg.levels}")
        window = _left_pad(context, cfg.context_length)
        with torch.no_grad():
            point, quant = self.net(torch.tensor(window, dtype=torch.float32))
        return Forecast(point.double().numpy(), quant.double().numpy(), tuple(levels))


def _left_pad(context: FloatArray, length: int) -> FloatArray:
    if context.shape[1] >= length:
        return context[:, -length:]
    pad = np.full((context.shape[0], length - context.shape[1]), np.nan)
    return np.concatenate([pad, context], axis=1)


def save_checkpoint(path: Path, net: PatchTSTNet, meta: dict[str, Any]) -> Path:
    """Save weights, the architecture and metadata (plain types only, so loading can be weights-only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "config": net.cfg.model_dump(), "meta": meta}, path)
    return path


def load_checkpoint(path: Path) -> tuple[PatchTSTNet, dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    net = PatchTSTNet(PatchTSTConfig(**blob["config"]))
    net.load_state_dict(blob["state_dict"])
    meta: dict[str, Any] = dict(blob["meta"])
    return net, meta
