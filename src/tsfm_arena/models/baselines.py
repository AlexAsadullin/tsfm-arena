"""Strong simple baselines: random walk, historical mean and a pooled ridge AR on lagged returns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tsfm_arena.data.panel import FloatArray, ReturnPanel
from tsfm_arena.models.base import Forecast, gaussian_forecast, nan_mean, nan_std

# Numerical guard against zero volatility in flat or empty contexts; not a modelling choice.
_VOL_FLOOR = 1e-12


class RandomWalk:
    """Log-price random walk: zero expected return, Gaussian quantiles from the context volatility."""

    needs_fit = False

    def __init__(self, name: str, context_length: int) -> None:
        self.name = name
        self.context_length = context_length

    def fit(self, history: ReturnPanel, horizon: int) -> None:
        """Nothing to fit."""

    def predict(self, context: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast:
        sigma = nan_std(context, _VOL_FLOOR)
        return gaussian_forecast(np.zeros(context.shape[0]), sigma, horizon, levels)


class HistoricalMean:
    """Expected return = ``horizon`` x the mean of the context; Gaussian quantiles."""

    needs_fit = False

    def __init__(self, name: str, context_length: int) -> None:
        self.name = name
        self.context_length = context_length

    def fit(self, history: ReturnPanel, horizon: int) -> None:
        """Nothing to fit."""

    def predict(self, context: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast:
        sigma = nan_std(context, _VOL_FLOOR)
        return gaussian_forecast(horizon * nan_mean(context), sigma, horizon, levels)


def lagged_design(
    returns: FloatArray, lags: int, horizon: int, scale: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Pooled design for every asset and every row with a full lag window and a realised target.

    Features are the last ``lags`` returns divided by the asset's ``scale``; the target is the next
    ``horizon``-bar cumulative return divided by ``scale``. Rows with any NaN are dropped.
    """
    t_total, _ = returns.shape
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    for t in range(lags - 1, t_total - horizon):
        x = returns[t - lags + 1 : t + 1][::-1].T / scale[:, None]
        y = returns[t + 1 : t + 1 + horizon].sum(axis=0) / scale
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.empty((0, lags)), np.empty(0)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    keep = ~(np.isnan(x_all).any(axis=1) | np.isnan(y_all))
    return x_all[keep], y_all[keep]


def ridge_fit(x: FloatArray, y: FloatArray, alpha: float) -> tuple[float, FloatArray]:
    """Ridge regression with an unpenalised intercept: ``argmin |y - b0 - X b|^2 + alpha |b|^2``."""
    x_mean, y_mean = x.mean(axis=0), float(y.mean())
    xc, yc = x - x_mean, y - y_mean
    gram = xc.T @ xc + alpha * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, xc.T @ yc)
    return y_mean - float(x_mean @ coef), coef


class ARRidge:
    """Pooled cross-sectional AR model fitted with ridge on volatility-scaled lagged returns.

    Refitted by the arena on the most recent ``train_window`` bars known at the origin. Quantiles are
    empirical quantiles of the in-sample standardised residuals, rescaled by each asset's volatility.
    """

    needs_fit = True

    def __init__(self, name: str, lags: int, alpha: float, train_window: int) -> None:
        self.name = name
        self.lags = lags
        self.alpha = alpha
        self.train_window = train_window
        self.context_length = max(lags, train_window)
        self._intercept = 0.0
        self._coef: FloatArray | None = None
        self._residuals: FloatArray = np.zeros(1)
        self._horizon: int | None = None

    def fit(self, history: ReturnPanel, horizon: int) -> None:
        window = history.returns[-self.train_window :]
        scale = nan_std(window, _VOL_FLOOR)
        x, y = lagged_design(window, self.lags, horizon, scale)
        if x.shape[0] <= self.lags:
            raise ValueError(f"{self.name}: {x.shape[0]} samples are too few to fit {self.lags} lags")
        self._intercept, self._coef = ridge_fit(x, y, self.alpha)
        self._residuals = y - (self._intercept + x @ self._coef)
        self._horizon = horizon

    def predict(self, context: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast:
        if self._coef is None or self._horizon != horizon:
            raise RuntimeError(f"{self.name} must be fitted for horizon {horizon} before predicting")
        scale = nan_std(context, _VOL_FLOOR)
        lagged = np.nan_to_num(context[:, -self.lags :][:, ::-1]) / scale[:, None]
        mean_std = self._intercept + lagged @ self._coef
        resid_q = np.quantile(self._residuals, np.asarray(levels))
        quantiles = (mean_std[:, None] + resid_q[None, :]) * scale[:, None]
        return Forecast(mean_std * scale, quantiles, tuple(levels))
