"""Evaluate registry models in the arena and turn the results into S6 trial records."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from tsfm_arena.evaluation.arena import Arena, ModelRun, Scores
from tsfm_arena.ledger.client import Actor, ReturnsSeries, TrialRecord
from tsfm_arena.logging import get_logger
from tsfm_arena.models.base import CutoffViolationError, ModelUnavailableError
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.models.spec import ModelSpec
from tsfm_arena.settings import Settings
from tsfm_arena.timeutil import iso, utc_now

log = get_logger(__name__)


@dataclass
class Entry:
    """One evaluated (model, config): a trial."""

    spec: ModelSpec
    status: Literal["ok", "skipped"]
    reason: str | None
    config_hash: str
    started_at: datetime
    finished_at: datetime
    run: ModelRun | None = None
    eligible: Scores | None = None
    contaminated: Scores | None = None

    @property
    def strategy_id(self) -> str:
        return f"tsfm-arena:{self.spec.name}:{self.config_hash}"


def eval_config(settings: Settings) -> dict[str, object]:
    """The evaluation settings that define a trial together with the model spec."""
    return settings.evaluation.model_dump(mode="json", exclude={"bootstrap", "report_contaminated"})


def evaluate_specs(arena: Arena, specs: list[ModelSpec], factory: ModelFactory) -> list[Entry]:
    """Build and score every spec; unavailable models become ``skipped`` entries with the reason."""
    entries: list[Entry] = []
    extra = eval_config(factory.settings)
    for spec in specs:
        started = utc_now()
        config_hash = spec.config_hash(extra)
        try:
            forecaster = factory.build(spec)
        except (ModelUnavailableError, CutoffViolationError) as exc:
            log.warning("model_skipped", model=spec.name, reason=str(exc))
            entries.append(Entry(spec, "skipped", str(exc), config_hash, started, utc_now()))
            continue
        run = arena.run(spec, forecaster)
        eligible = run.subset(run.eligible)
        contaminated = run.subset(~run.eligible)
        entries.append(
            Entry(
                spec,
                "ok",
                None,
                config_hash,
                started,
                utc_now(),
                run,
                arena.score(eligible),
                arena.score(contaminated) if contaminated.origin_ts.size else None,
            )
        )
    return entries


def git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def family_name(settings: Settings, source_name: str) -> str:
    return f"{settings.ledger.family_prefix}/rank-ic/{source_name}-h{settings.evaluation.horizon}"


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def trial_records(
    entries: list[Entry], settings: Settings, arena: Arena, code_version: str | None, tags: list[str]
) -> list[TrialRecord]:
    """One ``TrialRecord`` per scored entry, with the eligible-window net long-short returns."""
    records = []
    family = family_name(settings, arena.source.name)
    actor = Actor(kind=settings.ledger.actor_kind, id=settings.ledger.actor_id)
    for e in entries:
        if e.status != "ok" or e.eligible is None or e.eligible.n_origins == 0:
            continue
        s = e.eligible
        records.append(
            TrialRecord(
                family=family,
                strategy_id=e.strategy_id,
                source=settings.ledger.source,
                actor=actor,
                params={
                    "model": e.spec.model_dump(mode="json"),
                    "evaluation": eval_config(settings),
                },
                dataset_fingerprint=arena.fingerprint,
                code_version=code_version,
                seed=settings.seed,
                started_at=e.started_at,
                finished_at=e.finished_at,
                periods_per_year=settings.evaluation.periods_per_year / settings.evaluation.step,
                returns=ReturnsSeries(
                    ts=[iso(int(t)) for t in s.origin_ts], values=[float(v) for v in s.net_returns]
                ),
                metrics={
                    "rank_ic": _finite(s.ic.mean),
                    "icir": _finite(s.ic.icir),
                    "sharpe_net": _finite(s.sharpe_net.estimate),
                    "mae": _finite(s.mae.estimate),
                    "crps": _finite(s.crps),
                    "dispersion_ratio": _finite(s.dispersion.ratio),
                    "n_origins": s.n_origins,
                },
                tags=[*tags, e.spec.adapter],
            )
        )
    return records
