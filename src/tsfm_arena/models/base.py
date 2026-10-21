"""The single forecasting interface every adapter implements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm

from tsfm_arena.data.panel import FloatArray, ReturnPanel


class ModelUnavailableError(RuntimeError):
    """The model cannot be built here (weights not downloaded, downloads disabled, missing extra)."""


class CutoffViolationError(RuntimeError):
    """A checkpoint was trained on data later than the model's declared pretraining cutoff."""


@dataclass(frozen=True)
class Forecast:
    """Forecast of the cumulative log return over ``horizon`` bars for each of ``N`` assets.

    ``mean`` has shape ``(N,)`` and is also the ranking score; ``quantiles`` has shape ``(N, Q)`` for the
    strictly increasing ``levels``. Crossing quantiles are sorted on construction (a monotone rearrangement,
    Chernozhukov et al. 2010), which never increases the pinball loss.
    """

    mean: FloatArray
    quantiles: FloatArray
    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.quantiles.shape != (self.mean.shape[0], len(self.levels)):
            raise ValueError(f"bad forecast shapes: mean {self.mean.shape}, quantiles {self.quantiles.shape}")
        object.__setattr__(self, "quantiles", np.sort(self.quantiles, axis=1))


@runtime_checkable
class Forecaster(Protocol):
    """Point + quantile forecaster over a cross-section.

    ``context`` is ``(N, L)`` past log returns ending at the origin (NaN = missing, left-padded).
    ``fit`` receives the full history known at the origin; models trained once (TSFMs, checkpoints)
    implement it as a no-op and set ``needs_fit = False``.
    """

    name: str
    context_length: int
    needs_fit: bool

    def fit(self, history: ReturnPanel, horizon: int) -> None: ...

    def predict(self, context: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast: ...


def gaussian_forecast(mean: FloatArray, sigma: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast:
    """Forecast with i.i.d. Gaussian bars: quantile ``q`` is ``mean + sigma * sqrt(h) * z_q``."""
    z = norm.ppf(np.asarray(levels, dtype=np.float64))
    quantiles = mean[:, None] + (sigma * np.sqrt(horizon))[:, None] * z[None, :]
    return Forecast(mean, quantiles, tuple(levels))


def nan_std(context: FloatArray, floor: float) -> FloatArray:
    """Per-row standard deviation ignoring NaN, never below ``floor`` (empty rows get ``floor``)."""
    counts = np.sum(~np.isnan(context), axis=1)
    safe = np.where(np.isnan(context), 0.0, context)
    mean = safe.sum(axis=1) / np.maximum(counts, 1)
    var = np.where(np.isnan(context), 0.0, (context - mean[:, None]) ** 2).sum(axis=1)
    std = np.sqrt(var / np.maximum(counts - 1, 1))
    return np.maximum(np.where(counts > 1, std, floor), floor)


def nan_mean(context: FloatArray) -> FloatArray:
    """Per-row mean ignoring NaN (0 for empty rows)."""
    counts = np.sum(~np.isnan(context), axis=1)
    total: FloatArray = np.where(np.isnan(context), 0.0, context).sum(axis=1)
    return np.asarray(total / np.maximum(counts, 1), dtype=np.float64)
