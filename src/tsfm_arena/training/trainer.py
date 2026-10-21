"""Training of the PatchTST-style model: from-scratch pretraining and rank-tuning on finance data.

Both use the same loop. Each step draws a batch of dates; every date contributes its whole cross-section,
so rank losses see complete cross-sections. The objective is::

    quantile_weight * pinball + [point_loss] + rank_weight * rank_loss(point scores per date)

Training data is strictly limited to bars at or before the model's declared pretraining cutoff, and the
checkpoint records the last bar that any target touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch

from tsfm_arena.data.panel import FloatArray, IntArray, ReturnPanel
from tsfm_arena.models.patchtst import PatchTSTConfig, PatchTSTNet
from tsfm_arena.models.spec import TrainingSpec
from tsfm_arena.settings import TrainingSettings
from tsfm_arena.timeutil import iso, to_ns
from tsfm_arena.training import losses


@dataclass(frozen=True)
class SampleSet:
    """``contexts`` ``(D, N, L)``, ``targets`` ``(D, N)`` and the origin timestamps ``(D,)``."""

    contexts: FloatArray
    targets: FloatArray
    origin_ts: IntArray
    target_end_ts: int


def build_samples(panel: ReturnPanel, context_length: int, horizon: int) -> SampleSet:
    """Every origin ``t`` with a full context and a target fully inside ``panel``."""
    last_origin = panel.n_periods - 1 - horizon
    origins = np.arange(context_length - 1, last_origin + 1)
    if origins.size == 0:
        raise ValueError(
            f"panel of {panel.n_periods} bars is too short for context {context_length} + horizon {horizon}"
        )
    windows = np.lib.stride_tricks.sliding_window_view(panel.returns, context_length, axis=0)
    contexts = windows[origins - context_length + 1]  # (D, N, L)
    targets = panel.forward_returns(horizon)[origins]
    return SampleSet(
        np.ascontiguousarray(contexts),
        targets,
        panel.ts[origins],
        int(panel.ts[last_origin + horizon]),
    )


@dataclass
class TrainResult:
    net: PatchTSTNet
    meta: dict[str, Any]
    epoch_losses: list[float] = field(default_factory=list)


def new_net(settings: TrainingSettings, horizon: int, levels: list[float]) -> PatchTSTNet:
    arch = settings.patchtst
    return PatchTSTNet(PatchTSTConfig(**arch.model_dump(), horizon=horizon, levels=levels))


def train_patchtst(
    panel: ReturnPanel,
    cutoff: datetime,
    spec: TrainingSpec,
    settings: TrainingSettings,
    horizon: int,
    levels: list[float],
    seed: int,
    init: PatchTSTNet | None = None,
) -> TrainResult:
    """Train (or continue training ``init``) on ``panel`` restricted to ``ts <= cutoff``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = init if init is not None else new_net(settings, horizon, levels)
    if net.cfg.horizon != horizon or net.cfg.levels != levels:
        raise ValueError("initial network was trained for a different horizon or quantile levels")
    known = panel.upto(to_ns(cutoff))
    samples = build_samples(known, net.cfg.context_length, horizon)
    contexts = torch.as_tensor(samples.contexts, dtype=torch.float32)
    targets = torch.as_tensor(samples.targets, dtype=torch.float32)
    valid = ~torch.isnan(targets) & ((~torch.isnan(contexts)).sum(dim=2) > 1)
    targets = torch.nan_to_num(targets)
    rank_fn = None if spec.rank_loss == "none" else losses.rank_loss(spec.rank_loss)

    optimiser = torch.optim.AdamW(
        net.parameters(), lr=spec.learning_rate or settings.learning_rate, weight_decay=settings.weight_decay
    )
    epochs = spec.epochs or settings.epochs
    n_dates, n_assets = targets.shape
    result = TrainResult(net, {})
    net.train()
    for _ in range(epochs):
        order = rng.permutation(n_dates)
        total = 0.0
        for lo in range(0, n_dates, settings.dates_per_batch):
            idx = torch.as_tensor(order[lo : lo + settings.dates_per_batch])
            ctx, tgt, mask = contexts[idx], targets[idx], valid[idx]
            point, quant = net(ctx.reshape(-1, ctx.shape[-1]))
            flat_t, flat_m = tgt.reshape(-1), mask.reshape(-1)
            loss = spec.quantile_weight * losses.pinball_loss(quant, flat_t, levels, flat_m)
            if spec.point_loss == "mse":
                loss = loss + losses.mse_loss(point, flat_t, flat_m)
            if rank_fn is not None:
                scores = point.reshape(-1, n_assets)
                loss = loss + spec.rank_weight * rank_fn(scores, tgt, mask, settings.rank_temperature)
            optimiser.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            total += float(loss.detach()) * len(idx)
        result.epoch_losses.append(total / n_dates)
    net.eval()
    result.meta = {
        "train_start": iso(int(known.ts[0])),
        "train_end": iso(samples.target_end_ts),
        "declared_cutoff": cutoff.isoformat(),
        "n_dates": int(n_dates),
        "n_assets": int(n_assets),
        "epochs": int(epochs),
        "seed": int(seed),
        "objective": spec.model_dump(mode="json"),
        "final_loss": result.epoch_losses[-1],
    }
    return result
