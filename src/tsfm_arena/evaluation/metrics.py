"""Forecast and decision metrics. Pure numpy functions; NaN marks a missing value everywhere."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew

from tsfm_arena.data.panel import FloatArray

EULER_GAMMA = 0.5772156649015329


def mae(pred: FloatArray, actual: FloatArray) -> float:
    """Mean absolute error over pairs where both values exist."""
    ok = ~(np.isnan(pred) | np.isnan(actual))
    return float(np.mean(np.abs(pred[ok] - actual[ok]))) if ok.any() else float("nan")


def pinball(quantile: FloatArray, actual: FloatArray, level: float) -> FloatArray:
    """Quantile loss ``max(q (y - x), (q - 1)(y - x))`` elementwise."""
    diff = actual - quantile
    return np.maximum(level * diff, (level - 1.0) * diff)


def crps_from_quantiles(quantiles: FloatArray, actual: FloatArray, levels: Sequence[float]) -> float:
    """Quantile approximation of the continuous ranked probability score.

    ``CRPS(F, y) = 2 * integral_0^1 pinball_q(F^-1(q), y) dq`` (Gneiting & Raftery 2007, eq. 21 rewritten
    with the quantile score), approximated by the average over the ``Q`` levels times 2. ``quantiles`` is
    ``(..., Q)`` and ``actual`` is ``(...)``.
    """
    losses = np.stack(
        [pinball(quantiles[..., j], actual, q) for j, q in enumerate(levels)],
        axis=-1,
    )
    per_point = 2.0 * losses.mean(axis=-1)
    ok = ~np.isnan(per_point)
    return float(per_point[ok].mean()) if ok.any() else float("nan")


def spearman(x: FloatArray, y: FloatArray) -> float:
    """Spearman rank correlation of one cross-section, average ranks for ties; NaN if < 3 pairs."""
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def rank_ic_series(forecasts: FloatArray, realised: FloatArray) -> FloatArray:
    """Rank information coefficient per origin: Spearman of forecasts vs realised across assets."""
    return np.array([spearman(f, r) for f, r in zip(forecasts, realised, strict=True)])


@dataclass(frozen=True)
class ICSummary:
    mean: float
    std: float
    icir: float  # mean / std, per origin (not annualised)
    t_stat: float  # mean / (std / sqrt(n)); i.i.d. approximation
    hit_rate: float  # share of origins with IC > 0
    n: int


def ic_summary(ics: FloatArray) -> ICSummary:
    valid = ics[~np.isnan(ics)]
    n = int(valid.size)
    if n < 2:
        return ICSummary(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), n)
    mean, std = float(valid.mean()), float(valid.std(ddof=1))
    icir = mean / std if std > 0 else float("nan")
    return ICSummary(mean, std, icir, icir * np.sqrt(n), float((valid > 0).mean()), n)


def long_short_weights(scores: FloatArray, fraction: float, gross: float) -> FloatArray:
    """Dollar-neutral weights: ``+gross/2`` spread equally over the top ``fraction`` of scores,
    ``-gross/2`` over the bottom ``fraction``; at least one asset per leg; NaN scores get zero weight."""
    weights = np.zeros_like(scores)
    ok = np.flatnonzero(~np.isnan(scores))
    if ok.size < 2:
        return weights
    k = max(1, int(np.floor(fraction * ok.size)))
    order = ok[np.argsort(scores[ok], kind="stable")]
    weights[order[-k:]] = gross / 2.0 / k
    weights[order[:k]] = -gross / 2.0 / k
    return weights


@dataclass(frozen=True)
class LongShortResult:
    gross_returns: FloatArray
    net_returns: FloatArray
    turnover: FloatArray  # one-way: sum |w_t - w_{t-1}|


def long_short_returns(weights: FloatArray, realised_log: FloatArray, cost_bps: float) -> LongShortResult:
    """Per-origin portfolio returns. Log returns are converted to simple returns (``exp(r) - 1``) because
    portfolio returns aggregate simple returns. Costs are ``cost_bps / 1e4`` times turnover, with the
    first period paying to build the book from cash; missing realised returns count as zero."""
    simple = np.nan_to_num(np.expm1(realised_log))
    gross = (weights * simple).sum(axis=1)
    previous = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    turnover = np.abs(weights - previous).sum(axis=1)
    net = gross - cost_bps / 1e4 * turnover
    return LongShortResult(gross, net, turnover)


def sharpe(returns: FloatArray, periods_per_year: float) -> float:
    """Annualised Sharpe ratio ``mean / std * sqrt(periods_per_year)`` (zero risk-free rate)."""
    r = returns[~np.isnan(returns)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def probabilistic_sharpe(returns: FloatArray, benchmark_sr: float) -> float:
    """PSR (Bailey & Lopez de Prado 2012): ``P[SR > SR*]`` for a per-period benchmark ``SR*``::

        PSR = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2) )

    with per-period ``SR``, skewness ``g3`` and (non-excess) kurtosis ``g4`` of the returns.
    """
    r = returns[~np.isnan(returns)]
    if r.size < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - benchmark_sr) * np.sqrt(r.size - 1) / np.sqrt(denom)))


def expected_max_sharpe(n_trials: float, sr_variance: float) -> float:
    """Expected maximum of ``N`` per-period Sharpe estimates under the null (Bailey & Lopez de Prado 2014)::

        SR0 = sqrt(V) * ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))

    It is 0 for a single trial.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(returns: FloatArray, n_trials: float, sr_variance: float) -> float:
    """DSR = PSR with the benchmark set to the expected maximum Sharpe of ``n_trials`` trials."""
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sr_variance))


def per_period_sharpe(returns: FloatArray) -> float:
    r = returns[~np.isnan(returns)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1))


@dataclass(frozen=True)
class Dispersion:
    ratio: float  # mean over origins of cs-std(forecast) / cs-std(realised)
    collapse_rate: float  # share of origins with ratio below the threshold


def forecast_dispersion(forecasts: FloatArray, realised: FloatArray, threshold: float) -> Dispersion:
    """Cross-sectional forecast collapse: how much of the realised cross-sectional spread the forecasts
    reproduce. A ratio near 0 means the model predicts nearly the same value for every asset."""
    f_std = np.nanstd(forecasts, axis=1)
    r_std = np.nanstd(realised, axis=1)
    ok = r_std > 0
    if not ok.any():
        return Dispersion(float("nan"), float("nan"))
    ratio = f_std[ok] / r_std[ok]
    return Dispersion(float(ratio.mean()), float((ratio < threshold).mean()))
