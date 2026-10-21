"""Cutoff-aware rolling-origin evaluation.

At every origin ``t`` (a bar index) the model sees only data known at ``ts[t]``: the context
``returns[t-L+1 : t+1]`` and, for history-fitted models, the history up to ``t``. It forecasts the
cumulative return over ``returns[t+1 : t+1+h]``. An origin counts for a model only when
``ts[t] > pretraining_cutoff + embargo``; earlier origins are optionally scored as "contaminated" and
reported apart from the ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from tsfm_arena.data.base import DataSource
from tsfm_arena.data.panel import FloatArray, IntArray, ReturnPanel
from tsfm_arena.evaluation import metrics
from tsfm_arena.evaluation.bootstrap import Interval, bootstrap_ci
from tsfm_arena.logging import get_logger
from tsfm_arena.models.base import Forecaster
from tsfm_arena.models.spec import ModelSpec
from tsfm_arena.settings import EvaluationSettings
from tsfm_arena.timeutil import from_ns, iso, to_ns

log = get_logger(__name__)
BoolArray = npt.NDArray[np.bool_]


def is_eligible(origin_ns: int, cutoff: datetime | None, embargo: timedelta) -> bool:
    """True when the origin lies strictly after ``cutoff + embargo`` (always True without a cutoff)."""
    return cutoff is None or origin_ns > to_ns(cutoff + embargo)


def rolling_origins(n_periods: int, min_history: int, horizon: int, step: int) -> IntArray:
    """Origin indices with ``min_history`` bars up to and including the origin and a full target."""
    return np.arange(min_history - 1, n_periods - horizon, step, dtype=np.int64)


@dataclass(frozen=True)
class ModelRun:
    """Raw per-origin output of one model."""

    name: str
    origin_ts: IntArray  # (O,)
    eligible: BoolArray  # (O,)
    means: FloatArray  # (O, N)
    quantiles: FloatArray  # (O, N, Q)
    realised: FloatArray  # (O, N)

    def subset(self, mask: BoolArray) -> ModelRun:
        return ModelRun(
            self.name,
            self.origin_ts[mask],
            self.eligible[mask],
            self.means[mask],
            self.quantiles[mask],
            self.realised[mask],
        )


@dataclass(frozen=True)
class Scores:
    """Metrics of one model over a set of origins; intervals are stationary-bootstrap CIs."""

    n_origins: int
    first_origin: str | None
    last_origin: str | None
    mae: Interval
    crps: float
    ic: metrics.ICSummary
    ic_mean: Interval
    sharpe_net: Interval
    sharpe_gross: float
    turnover: float
    psr: float
    dispersion: metrics.Dispersion
    net_returns: FloatArray
    origin_ts: IntArray


class Arena:
    """Runs forecasters over rolling origins of one panel."""

    def __init__(
        self,
        cfg: EvaluationSettings,
        source: DataSource,
        as_of: datetime,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
    ) -> None:
        self.cfg = cfg
        self.source = source
        loaded = source.load(start, as_of, as_of, symbols or None)
        self.panel: ReturnPanel = loaded.panel
        self.fingerprint = loaded.fingerprint
        self.realised_all = self.panel.forward_returns(cfg.horizon)
        self.origins = rolling_origins(self.panel.n_periods, cfg.min_history, cfg.horizon, cfg.step)
        if self.origins.size == 0:
            raise ValueError(
                f"panel of {self.panel.n_periods} bars has no origins for min_history "
                f"{cfg.min_history} and horizon {cfg.horizon}"
            )

    @property
    def periods_per_origin(self) -> float:
        """Origins per year: Sharpe ratios annualise per-origin returns (overlapping when horizon > step)."""
        return self.cfg.periods_per_year / self.cfg.step

    def eligibility(self, spec: ModelSpec) -> BoolArray:
        return np.array(
            [
                is_eligible(int(self.panel.ts[t]), spec.pretraining_cutoff, self.cfg.embargo)
                for t in self.origins
            ],
            dtype=np.bool_,
        )

    def _history(self, t: int) -> ReturnPanel:
        if not self.source.revisable:
            return self.panel.rows(0, t + 1)
        known = from_ns(int(self.panel.ts[t]))
        loaded = self.source.load(from_ns(int(self.panel.ts[0])), known, known, self.panel.symbols)
        return loaded.panel.select(self.panel.symbols)

    def _context(self, history: ReturnPanel, length: int) -> FloatArray:
        window = history.returns[-length:].T
        if window.shape[1] < length:
            pad = np.full((window.shape[0], length - window.shape[1]), np.nan)
            window = np.concatenate([pad, window], axis=1)
        return window

    def run(self, spec: ModelSpec, forecaster: Forecaster) -> ModelRun:
        """Forecast at every eligible origin (and at contaminated ones when configured)."""
        cfg = self.cfg
        eligible = self.eligibility(spec)
        use = eligible | cfg.report_contaminated
        idx = self.origins[use]
        n, q = self.panel.n_assets, len(cfg.quantiles)
        means = np.full((idx.size, n), np.nan)
        quants = np.full((idx.size, n, q), np.nan)
        last_fit: int | None = None
        for k, t in enumerate(idx):
            history = self._history(int(t))
            if forecaster.needs_fit and (last_fit is None or k - last_fit >= cfg.refit_every):
                forecaster.fit(history, cfg.horizon)
                last_fit = k
            context = self._context(history, forecaster.context_length)
            forecast = forecaster.predict(context, cfg.horizon, cfg.quantiles)
            means[k], quants[k] = forecast.mean, forecast.quantiles
        log.info(
            "model_evaluated",
            model=spec.name,
            origins=int(idx.size),
            eligible=int(eligible.sum()),
            contaminated=int((~eligible[use]).sum()),
        )
        return ModelRun(spec.name, self.panel.ts[idx], eligible[use], means, quants, self.realised_all[idx])

    def score(self, run: ModelRun) -> Scores:
        """Metrics over all origins of ``run`` (subset it first to score a window)."""
        cfg, boot = self.cfg, self.cfg.bootstrap
        ls = cfg.long_short
        n = run.origin_ts.size
        ics = metrics.rank_ic_series(run.means, run.realised) if n else np.empty(0)
        weights = np.array([metrics.long_short_weights(m, ls.fraction, ls.gross) for m in run.means])
        weights = weights.reshape(n, self.panel.n_assets)
        book = metrics.long_short_returns(weights, run.realised, ls.cost_bps)
        abs_err = np.abs(run.means - run.realised)
        per_origin_mae = np.array([np.nanmean(row) if (~np.isnan(row)).any() else np.nan for row in abs_err])

        def ci(values: FloatArray, stat: Callable[[FloatArray], float]) -> Interval:
            return bootstrap_ci(values, stat, boot.mean_block, boot.n_resamples, boot.confidence, boot.seed)

        return Scores(
            n_origins=int(n),
            first_origin=iso(int(run.origin_ts[0])) if n else None,
            last_origin=iso(int(run.origin_ts[-1])) if n else None,
            mae=ci(per_origin_mae, lambda v: float(np.mean(v))),
            crps=metrics.crps_from_quantiles(run.quantiles, run.realised, cfg.quantiles),
            ic=metrics.ic_summary(ics),
            ic_mean=ci(ics, lambda v: float(np.mean(v))),
            sharpe_net=ci(book.net_returns, lambda v: metrics.sharpe(v, self.periods_per_origin)),
            sharpe_gross=metrics.sharpe(book.gross_returns, self.periods_per_origin),
            turnover=float(book.turnover.mean()) if n else float("nan"),
            psr=metrics.probabilistic_sharpe(book.net_returns, 0.0),
            dispersion=metrics.forecast_dispersion(run.means, run.realised, cfg.collapse_threshold),
            net_returns=book.net_returns,
            origin_ts=run.origin_ts,
        )
