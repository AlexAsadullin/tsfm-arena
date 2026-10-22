"""Stationary bootstrap over origins (Politis & Romano 1994) for confidence intervals.

Blocks of random length (geometric with mean ``mean_block``) are drawn with wrap-around, so the resamples
keep the short-range dependence of per-origin metrics such as daily IC or portfolio returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tsfm_arena.data.panel import FloatArray


def stationary_indices(
    n: int, mean_block: float, n_resamples: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """``(n_resamples, n)`` resampled indices."""
    p = 1.0 / mean_block
    out = np.empty((n_resamples, n), dtype=np.int64)
    starts = rng.integers(0, n, size=(n_resamples, n))
    new_block = rng.random((n_resamples, n)) < p
    out[:, 0] = starts[:, 0]
    for t in range(1, n):
        out[:, t] = np.where(new_block[:, t], starts[:, t], (out[:, t - 1] + 1) % n)
    return out


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def bootstrap_ci(
    values: FloatArray,
    statistic: Callable[[FloatArray], float],
    mean_block: float,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> Interval:
    """Percentile interval of ``statistic`` over stationary-bootstrap resamples of ``values`` (axis 0)."""
    if values.shape[0] == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    clean = values[~np.isnan(values).reshape(values.shape[0], -1).any(axis=1)]
    estimate = statistic(clean) if clean.shape[0] else float("nan")
    if clean.shape[0] < 3:
        return Interval(estimate, float("nan"), float("nan"))
    idx = stationary_indices(clean.shape[0], mean_block, n_resamples, np.random.default_rng(seed))
    stats = np.array([statistic(clean[row]) for row in idx])
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return Interval(estimate, float("nan"), float("nan"))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return Interval(estimate, float(low), float(high))
