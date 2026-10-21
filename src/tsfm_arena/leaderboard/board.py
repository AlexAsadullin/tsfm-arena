"""Leaderboard: per-model rows on each model's eligible window, on the common window, and the
contaminated (pre-cutoff) scores that are shown but never ranked."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel

from tsfm_arena.evaluation import metrics
from tsfm_arena.evaluation.arena import Arena, Scores
from tsfm_arena.evaluation.bootstrap import Interval
from tsfm_arena.ledger.client import Verdict
from tsfm_arena.pipeline.evaluate import Entry


class CI(BaseModel):
    estimate: float | None
    low: float | None
    high: float | None

    @classmethod
    def of(cls, interval: Interval) -> CI:
        return cls(estimate=_f(interval.estimate), low=_f(interval.low), high=_f(interval.high))


def _f(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


class ScoreRow(BaseModel):
    n_origins: int
    first_origin: str | None
    last_origin: str | None
    rank_ic: CI
    icir: float | None
    ic_t_stat: float | None
    ic_hit_rate: float | None
    sharpe_net: CI
    sharpe_gross: float | None
    turnover: float | None
    psr: float | None
    mae: CI
    crps: float | None
    dispersion_ratio: float | None
    collapse_rate: float | None

    @classmethod
    def of(cls, s: Scores) -> ScoreRow:
        return cls(
            n_origins=s.n_origins,
            first_origin=s.first_origin,
            last_origin=s.last_origin,
            rank_ic=CI.of(s.ic_mean),
            icir=_f(s.ic.icir),
            ic_t_stat=_f(s.ic.t_stat),
            ic_hit_rate=_f(s.ic.hit_rate),
            sharpe_net=CI.of(s.sharpe_net),
            sharpe_gross=_f(s.sharpe_gross),
            turnover=_f(s.turnover),
            psr=_f(s.psr),
            mae=CI.of(s.mae),
            crps=_f(s.crps),
            dispersion_ratio=_f(s.dispersion.ratio),
            collapse_rate=_f(s.dispersion.collapse_rate),
        )


class Row(BaseModel):
    model: str
    adapter: str
    strategy_id: str
    pretraining_cutoff: str | None
    status: Literal["ok", "skipped"]
    reason: str | None
    eligible: ScoreRow | None
    common: ScoreRow | None
    contaminated: ScoreRow | None
    dsr: float | None
    dsr_source: Literal["S6", "local"] | None
    gate_passed: bool | None


class Leaderboard(BaseModel):
    schema_version: Literal["1"] = "1"
    generated_at: datetime
    family: str
    dataset: str
    dataset_fingerprint: str | None
    horizon: int
    embargo_days: float
    cost_bps: float
    n_trials: int
    confidence: float
    common_window: tuple[str, str] | None
    trials_sink: str | None
    rows: list[Row]


def local_dsr(entries: list[Entry]) -> dict[str, float]:
    """DSR of each scored entry, deflated by the number of scored entries in this run and the variance
    of their per-period Sharpe ratios (Bailey & Lopez de Prado 2014)."""
    scored = [e for e in entries if e.eligible is not None and e.eligible.n_origins > 2]
    srs = np.array([metrics.per_period_sharpe(e.eligible.net_returns) for e in scored if e.eligible])
    finite = srs[np.isfinite(srs)]
    variance = float(finite.var(ddof=1)) if finite.size > 1 else 0.0
    return {
        e.spec.name: metrics.deflated_sharpe(e.eligible.net_returns, len(scored), variance)
        for e in scored
        if e.eligible is not None
    }


def common_window_scores(
    arena: Arena, entries: list[Entry]
) -> tuple[dict[str, Scores], tuple[str, str] | None]:
    """Scores restricted to origins that are eligible for every scored model."""
    runs = [e.run for e in entries if e.run is not None]
    if not runs:
        return {}, None
    common = set(runs[0].origin_ts[runs[0].eligible].tolist())
    for run in runs[1:]:
        common &= set(run.origin_ts[run.eligible].tolist())
    if len(common) < 3:
        return {}, None
    out = {}
    for run in runs:
        mask = np.isin(run.origin_ts, np.fromiter(common, dtype=np.int64)) & run.eligible
        out[run.name] = arena.score(run.subset(mask))
    any_scores = next(iter(out.values()))
    window = (any_scores.first_origin or "", any_scores.last_origin or "")
    return out, window


def build_leaderboard(
    arena: Arena,
    entries: list[Entry],
    family: str,
    verdicts: dict[str, Verdict | None],
    generated_at: datetime,
    trials_sink: str | None,
) -> Leaderboard:
    cfg = arena.cfg
    dsr = local_dsr(entries)
    common, window = common_window_scores(arena, entries)
    rows = []
    for e in entries:
        verdict = verdicts.get(e.strategy_id)
        row_dsr: float | None
        source: Literal["S6", "local"] | None
        passed: bool | None
        if verdict is not None and verdict.dsr is not None:
            row_dsr, source, passed = verdict.dsr, "S6", verdict.passed
        elif e.spec.name in dsr:
            row_dsr, source, passed = _f(dsr[e.spec.name]), "local", None
        else:
            row_dsr, source, passed = None, None, None
        cutoff = e.spec.pretraining_cutoff
        rows.append(
            Row(
                model=e.spec.name,
                adapter=e.spec.adapter,
                strategy_id=e.strategy_id,
                pretraining_cutoff=cutoff.isoformat() if cutoff else None,
                status=e.status,
                reason=e.reason,
                eligible=ScoreRow.of(e.eligible) if e.eligible and e.eligible.n_origins else None,
                common=ScoreRow.of(common[e.spec.name]) if e.spec.name in common else None,
                contaminated=ScoreRow.of(e.contaminated) if e.contaminated else None,
                dsr=row_dsr,
                dsr_source=source,
                gate_passed=passed,
            )
        )
    rows.sort(key=lambda r: -(r.eligible.rank_ic.estimate or -np.inf) if r.eligible else np.inf)
    return Leaderboard(
        generated_at=generated_at,
        family=family,
        dataset=arena.source.name,
        dataset_fingerprint=arena.fingerprint,
        horizon=cfg.horizon,
        embargo_days=cfg.embargo.total_seconds() / 86_400,
        cost_bps=cfg.long_short.cost_bps,
        n_trials=len(dsr),
        confidence=cfg.bootstrap.confidence,
        common_window=window,
        trials_sink=trials_sink,
        rows=rows,
    )
