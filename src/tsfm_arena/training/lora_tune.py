"""Rank-tuning of Chronos-Bolt through LoRA adapters.

The frozen Chronos-Bolt model produces quantiles for each future step from a batch of contexts. The
ranking score of an asset is the forecast median of its cumulative ``horizon``-bar return; the loss is
``rank_weight * rank_loss(scores per date) + quantile_weight * pinball(cumulative quantiles)``, where the
cumulative quantiles are the per-step quantiles summed over the horizon (exact for ``horizon = 1``).
Only the LoRA matrices receive gradients.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import torch

from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.models.spec import TrainingSpec
from tsfm_arena.settings import TrainingSettings
from tsfm_arena.timeutil import iso, to_ns
from tsfm_arena.training import losses
from tsfm_arena.training.trainer import build_samples


def lora_rank_tune(
    model: Any,
    panel: ReturnPanel,
    cutoff: datetime,
    spec: TrainingSpec,
    settings: TrainingSettings,
    horizon: int,
    context_length: int,
    seed: int,
) -> dict[str, Any]:
    """Train the LoRA parameters of ``model`` (already passed through ``add_lora``) in place."""
    if spec.rank_loss == "none":
        raise ValueError("LoRA rank-tuning needs a rank loss")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lora_cfg = settings.lora
    known = panel.upto(to_ns(cutoff))
    samples = build_samples(known, context_length, horizon)
    contexts = torch.as_tensor(samples.contexts, dtype=torch.float32)
    targets = torch.as_tensor(samples.targets, dtype=torch.float32)
    valid = ~torch.isnan(targets) & ((~torch.isnan(contexts)).sum(dim=2) > 1)
    targets = torch.nan_to_num(targets)

    levels: list[float] = [float(q) for q in model.chronos_config.quantiles]
    median = levels.index(0.5)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters; call add_lora first")
    optimiser = torch.optim.AdamW(trainable, lr=spec.learning_rate or lora_cfg.learning_rate)
    rank_fn = losses.rank_loss(spec.rank_loss)
    epochs = spec.epochs or lora_cfg.epochs
    n_dates, n_assets = targets.shape
    epoch_losses: list[float] = []
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n_dates)
        total = 0.0
        for lo in range(0, n_dates, lora_cfg.dates_per_batch):
            idx = torch.as_tensor(order[lo : lo + lora_cfg.dates_per_batch])
            ctx, tgt, mask = contexts[idx], targets[idx], valid[idx]
            preds = model(context=ctx.reshape(-1, ctx.shape[-1])).quantile_preds  # (B, Q, H_model)
            cumulative = preds[:, :, :horizon].sum(dim=2)  # (B, Q)
            scores = cumulative[:, median].reshape(-1, n_assets)
            loss = spec.rank_weight * rank_fn(scores, tgt, mask, settings.rank_temperature)
            if spec.quantile_weight > 0:
                loss = loss + spec.quantile_weight * losses.pinball_loss(
                    cumulative, tgt.reshape(-1), levels, mask.reshape(-1)
                )
            optimiser.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            total += float(loss.detach()) * len(idx)
        epoch_losses.append(total / n_dates)
    model.eval()
    return {
        "train_start": iso(int(known.ts[0])),
        "train_end": iso(samples.target_end_ts),
        "declared_cutoff": cutoff.isoformat(),
        "n_dates": int(n_dates),
        "epochs": int(epochs),
        "seed": int(seed),
        "objective": spec.model_dump(mode="json"),
        "epoch_losses": epoch_losses,
    }
