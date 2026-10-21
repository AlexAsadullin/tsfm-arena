"""Training losses. Rank losses act on one cross-section at a time.

Rank losses take ``scores`` and ``targets`` of shape ``(G, N)``: ``G`` cross-sections (dates) of ``N``
assets, with a boolean ``mask`` marking valid entries. Each loss is averaged over the groups that have at
least two valid assets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor

RankLossName = Literal["none", "pairwise", "listnet", "spearman"]
PointLossName = Literal["none", "mse"]
RankLoss = Callable[[Tensor, Tensor, Tensor, float], Tensor]

_EPS = 1e-8  # numerical guard for divisions by a standard deviation


def pinball_loss(pred: Tensor, target: Tensor, levels: Sequence[float], mask: Tensor) -> Tensor:
    """Mean quantile (pinball) loss ``max(q (y - x), (q - 1)(y - x))`` over levels and valid targets.

    ``pred`` is ``(B, Q)``, ``target`` and ``mask`` are ``(B,)``.
    """
    q = torch.tensor(list(levels), dtype=pred.dtype, device=pred.device)
    diff = target[:, None] - pred
    loss = torch.maximum(q * diff, (q - 1.0) * diff).mean(dim=1)
    return _masked_mean(loss, mask)


def mse_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return _masked_mean((pred - target) ** 2, mask)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _group_mean(per_group: Tensor, valid_groups: Tensor) -> Tensor:
    weight = valid_groups.to(per_group.dtype)
    return (per_group * weight).sum() / weight.sum().clamp_min(1.0)


def pairwise_logistic_loss(scores: Tensor, targets: Tensor, mask: Tensor, temperature: float) -> Tensor:
    """RankNet loss (Burges et al. 2005): mean of ``log(1 + exp(-(s_i - s_j) / T))`` over pairs with
    ``y_i > y_j``, so the loss is ``log 2`` when all scores are equal."""
    diff_s = (scores[:, :, None] - scores[:, None, :]) / temperature
    better = (targets[:, :, None] > targets[:, None, :]) & mask[:, :, None] & mask[:, None, :]
    pair_loss = torch.nn.functional.softplus(-diff_s) * better
    n_pairs = better.sum(dim=(1, 2))
    per_group = pair_loss.sum(dim=(1, 2)) / n_pairs.clamp_min(1)
    return _group_mean(per_group, n_pairs > 0)


def _masked_zscore(x: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(x.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * weight).sum(dim=1, keepdim=True) / count
    var = (((x - mean) * weight) ** 2).sum(dim=1, keepdim=True) / count
    return (x - mean) / (var.sqrt() + _EPS) * weight


def listnet_loss(scores: Tensor, targets: Tensor, mask: Tensor, temperature: float) -> Tensor:
    """ListNet top-one loss (Cao et al. 2007): cross-entropy between ``softmax(z(y))`` and
    ``softmax(s / T)``, where ``z`` is the cross-sectional z-score of the realised returns."""
    neg_inf = torch.finfo(scores.dtype).min
    target_logits = _masked_zscore(targets, mask).masked_fill(~mask, neg_inf)
    score_logits = (scores / temperature).masked_fill(~mask, neg_inf)
    p_target = torch.softmax(target_logits, dim=1)
    log_p_score = torch.log_softmax(score_logits, dim=1)
    per_group = -(p_target * log_p_score * mask).sum(dim=1)
    return _group_mean(per_group, mask.sum(dim=1) > 1)


def soft_ranks(x: Tensor, mask: Tensor, temperature: float) -> Tensor:
    """Differentiable ranks: ``r_i = sum_j sigmoid((x_i - x_j) / T)`` over valid ``j``.

    As ``T -> 0`` this tends to the 0-based rank plus 0.5 (the self-comparison).
    """
    pair = torch.sigmoid((x[:, :, None] - x[:, None, :]) / temperature)
    return (pair * mask[:, None, :]).sum(dim=2)


def soft_spearman_loss(scores: Tensor, targets: Tensor, mask: Tensor, temperature: float) -> Tensor:
    """Negative Pearson correlation between soft ranks of the scores and exact ranks of the targets,
    a differentiable surrogate of the Spearman rank correlation (as in Blondel et al. 2020)."""
    hard = targets.masked_fill(~mask, torch.finfo(targets.dtype).max).argsort(dim=1).argsort(dim=1)
    rank_s = _masked_zscore(soft_ranks(scores, mask, temperature), mask)
    rank_y = _masked_zscore(hard.to(scores.dtype), mask)
    count = mask.sum(dim=1).to(scores.dtype)
    corr = (rank_s * rank_y).sum(dim=1) / count.clamp_min(1.0)
    return _group_mean(-corr, mask.sum(dim=1) > 2)


RANK_LOSSES: dict[str, RankLoss] = {
    "pairwise": pairwise_logistic_loss,
    "listnet": listnet_loss,
    "spearman": soft_spearman_loss,
}


def rank_loss(name: RankLossName) -> RankLoss:
    """Look up a rank loss by name; ``"none"`` is not a loss and raises ``KeyError``."""
    return RANK_LOSSES[name]
