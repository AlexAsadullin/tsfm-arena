"""Synthetic cross-section with a planted, weak and decaying signal.

Model, for asset ``i`` and bar ``t + 1``::

    s[i, t]   = cross-sectional z-score of mean(r[i, t-k+1 : t+1])        (momentum signal, known at t)
    rho[t]    = signal_ic * 0.5 ** (t / signal_halflife)                   (planted IC, decaying)
    r[i, t+1] = beta[i] * f[t+1] + vol[i] * (rho[t] * s[i, t] + sqrt(1 - rho[t]^2) * e[i, t+1])

with ``f ~ N(0, market_vol^2)``, ``e`` unit-variance Student-t, ``vol ~ U(vol_low, vol_high)`` and
``beta ~ N(1, beta_std^2)``. Conditional on the past, the correlation between ``s`` and the
idiosyncratic standardised return is ``rho[t]``, so ``rho`` is an upper bound on the attainable IC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsfm_arena.data.panel import FloatArray, ReturnPanel
from tsfm_arena.settings import SyntheticSettings
from tsfm_arena.timeutil import to_ns


@dataclass(frozen=True)
class SyntheticCrossSection:
    panel: ReturnPanel
    signal: FloatArray  # (T, N): s[t] predicts returns[t + 1]
    planted_ic: FloatArray  # (T,): rho[t]


def _cs_zscore(x: FloatArray) -> FloatArray:
    centred = x - x.mean()
    std = centred.std()
    return centred / std if std > 0 else np.zeros_like(x)


def generate_cross_section(cfg: SyntheticSettings) -> SyntheticCrossSection:
    """Generate the panel deterministically from ``cfg.seed``."""
    rng = np.random.default_rng(cfg.seed)
    n, t_total, k = cfg.n_assets, cfg.n_periods, cfg.momentum_window
    vol = rng.uniform(cfg.vol_low, cfg.vol_high, size=n)
    beta = rng.normal(1.0, cfg.beta_std, size=n)
    market = rng.normal(0.0, cfg.market_vol, size=t_total) if cfg.market_vol > 0 else np.zeros(t_total)
    t_scale = np.sqrt((cfg.tail_df - 2.0) / cfg.tail_df)  # unit-variance Student-t
    shocks = rng.standard_t(cfg.tail_df, size=(t_total, n)) * t_scale
    rho = cfg.signal_ic * 0.5 ** (np.arange(t_total) / cfg.signal_halflife)

    returns = np.zeros((t_total, n))
    signal = np.zeros((t_total, n))
    returns[0] = beta * market[0] + vol * shocks[0]
    for t in range(t_total - 1):
        lo = max(0, t - k + 1)
        signal[t] = _cs_zscore(returns[lo : t + 1].mean(axis=0))
        idio = rho[t] * signal[t] + np.sqrt(1.0 - rho[t] ** 2) * shocks[t + 1]
        returns[t + 1] = beta * market[t + 1] + vol * idio
    signal[-1] = _cs_zscore(returns[max(0, t_total - k) :].mean(axis=0))

    period_ns = int(cfg.period.total_seconds() * 1e9)
    ts = to_ns(cfg.start) + period_ns * np.arange(t_total, dtype=np.int64)
    symbols = tuple(f"{cfg.symbol_prefix}{i:02d}{cfg.symbol_suffix}" for i in range(n))
    return SyntheticCrossSection(ReturnPanel(ts, symbols, returns), signal, rho)
