"""Experiments behind ``docs/report.md`` (``make report``).

1. Objective comparison: an MSE-trained PatchTST, the same network rank-tuned with each rank loss, a
   rank-loss model trained from scratch, and the baselines, every PatchTST variant over several seeds.
2. Paired tests on per-origin rank IC between variants.
3. Look-ahead inflation: rank IC on contaminated (pre-cutoff) origins vs eligible origins.
4. Forecast collapse: dispersion ratio per model.
5. Chronos zero-shot and LoRA rank-tuned, when weights are available (downloads allowed or cached).

Every evaluated (model, config, seed) is one trial in the same S6 family, so the DSR of each row is
deflated by all of them. Tables go to ``<project data>/reports/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from tsfm_arena.data.panel import FloatArray, IntArray
from tsfm_arena.evaluation import metrics
from tsfm_arena.evaluation.arena import ModelRun
from tsfm_arena.evaluation.bootstrap import Interval, bootstrap_ci
from tsfm_arena.leaderboard.render import render_markdown
from tsfm_arena.ledger.client import LedgerClient
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.models.spec import ModelSpec, Registry, load_registry
from tsfm_arena.pipeline.evaluate import Entry
from tsfm_arena.pipeline.run import build_ledger, run_arena, synthetic_source, train_models
from tsfm_arena.settings import Settings


def variant_specs(settings: Settings, registry: Registry) -> dict[str, list[ModelSpec]]:
    """PatchTST variants per objective, one spec per seed (names ``<objective>-s<seed>``)."""
    rep = settings.report
    base, rank = registry.get(rep.base_model), registry.get(rep.rank_model)
    if base.training is None or rank.training is None:
        raise ValueError("report base and rank models need training sections")
    out: dict[str, list[ModelSpec]] = {"mse": [], "listnet-scratch": []}
    out.update({loss: [] for loss in rep.rank_losses})
    for seed in rep.seeds:
        mse_name = f"mse-s{seed}"
        out["mse"].append(
            base.model_copy(update={"name": mse_name, "path": Path(f"checkpoints/report/{mse_name}.pt")})
        )
        for loss in rep.rank_losses:
            name = f"{loss}-s{seed}"
            training = rank.training.model_copy(update={"init_from": mse_name, "rank_loss": loss})
            out[loss].append(
                rank.model_copy(
                    update={"name": name, "path": Path(f"checkpoints/report/{name}.pt"), "training": training}
                )
            )
        name = f"listnet-scratch-s{seed}"
        scratch = rank.training.model_copy(
            update={"init_from": None, "rank_loss": "listnet", "epochs": settings.training.epochs}
        )
        out["listnet-scratch"].append(
            rank.model_copy(
                update={"name": name, "path": Path(f"checkpoints/report/{name}.pt"), "training": scratch}
            )
        )
    return out


@dataclass(frozen=True)
class VariantSummary:
    name: str
    n_seeds: int
    n_origins: int
    ic_seed_mean: float
    ic_seed_std: float
    ic: Interval  # seed-averaged per-origin IC, bootstrap CI
    sharpe: Interval  # seed-averaged net returns, bootstrap CI
    dsr_mean: float
    dispersion: float
    contaminated_ic: float | None
    ic_series: FloatArray
    origin_ts: IntArray
    contaminated_ts: IntArray


def _nanmean(values: FloatArray, axis: int) -> FloatArray:
    """``nanmean`` that returns NaN for all-NaN slices without a warning (e.g. constant forecasts)."""
    count = (~np.isnan(values)).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _ic_series(run: ModelRun) -> FloatArray:
    return metrics.rank_ic_series(run.means, run.realised)


def summarise(
    name: str, entries: list[Entry], dsr: dict[str, float | None], settings: Settings, ppy: float
) -> VariantSummary:
    boot = settings.evaluation.bootstrap
    runs = [e.run for e in entries if e.run is not None]
    eligible = [r.subset(r.eligible) for r in runs]
    contaminated = [r.subset(~r.eligible) for r in runs]
    ic_by_seed = np.array([_ic_series(r) for r in eligible])
    ic_avg = _nanmean(ic_by_seed, axis=0)
    net = np.array([e.eligible.net_returns for e in entries if e.eligible is not None]).mean(axis=0)
    seed_ic = _nanmean(ic_by_seed, axis=1)
    cont = [float(_nanmean(_ic_series(r)[None, :], axis=1)[0]) for r in contaminated if r.origin_ts.size]
    dsrs = [d for e in entries if (d := dsr.get(e.spec.name)) is not None]
    disp = [e.eligible.dispersion.ratio for e in entries if e.eligible is not None]

    def ci(values: FloatArray, stat: Callable[[FloatArray], float]) -> Interval:
        return bootstrap_ci(values, stat, boot.mean_block, boot.n_resamples, boot.confidence, boot.seed)

    return VariantSummary(
        name=name,
        n_seeds=len(runs),
        n_origins=int(ic_avg.size),
        ic_seed_mean=float(seed_ic.mean()),
        ic_seed_std=float(seed_ic.std(ddof=1)) if seed_ic.size > 1 else float("nan"),
        ic=ci(ic_avg, lambda v: float(np.mean(v))),
        sharpe=ci(net, lambda v: metrics.sharpe(v, ppy)),
        dsr_mean=float(np.mean(dsrs)) if dsrs else float("nan"),
        dispersion=float(np.mean(disp)) if disp else float("nan"),
        contaminated_ic=float(np.mean(cont)) if cont else None,
        ic_series=ic_avg,
        origin_ts=eligible[0].origin_ts,
        contaminated_ts=contaminated[0].origin_ts,
    )


@dataclass(frozen=True)
class PairedTest:
    a: str
    b: str
    n: int
    mean_diff: Interval
    t_stat: float
    p_value: float


def paired_ic_test(a: VariantSummary, b: VariantSummary, settings: Settings) -> PairedTest:
    """Student's paired t-test on per-origin IC differences over the origins both series share
    (H0: mean difference is zero), plus a stationary-bootstrap CI of the mean difference that allows
    for serial dependence."""
    boot = settings.evaluation.bootstrap
    _, ia, ib = np.intersect1d(a.origin_ts, b.origin_ts, return_indices=True)
    x, y = a.ic_series[ia], b.ic_series[ib]
    ok = ~(np.isnan(x) | np.isnan(y))
    test = ttest_rel(x[ok], y[ok])
    diff = x[ok] - y[ok]
    interval = bootstrap_ci(
        diff, lambda v: float(np.mean(v)), boot.mean_block, boot.n_resamples, boot.confidence, boot.seed
    )
    return PairedTest(a.name, b.name, int(ok.sum()), interval, float(test.statistic), float(test.pvalue))


@dataclass(frozen=True)
class Inflation:
    """Rank IC before vs after a model's cutoff, next to the reference rule on the same two windows.

    On synthetic data no model can have memorised outcomes, so the reference ratio measures how much
    of the gap comes from the decay of the planted signal; the rest is overfitting to training dates.
    """

    name: str
    contaminated_ic: float
    eligible_ic: float
    reference_contaminated_ic: float
    reference_eligible_ic: float


def _mean_on(ref: VariantSummary, ts: IntArray) -> float:
    values = ref.ic_series[np.isin(ref.origin_ts, ts)]
    values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else float("nan")


def inflation_rows(summaries: list[VariantSummary], reference: VariantSummary) -> list[Inflation]:
    return [
        Inflation(
            s.name,
            s.contaminated_ic,
            s.ic_seed_mean,
            _mean_on(reference, s.contaminated_ts),
            _mean_on(reference, s.origin_ts),
        )
        for s in summaries
        if s.contaminated_ic is not None
    ]


def _ratio(a: float, b: float) -> str:
    return _fmt(a / b, 2) if b and np.isfinite(a) and np.isfinite(b) else "n/a"


def _fmt(v: float | None, d: int = 3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def _ci(i: Interval, d: int = 3) -> str:
    return f"{_fmt(i.estimate, d)} [{_fmt(i.low, d)}, {_fmt(i.high, d)}]"


def render_tables(
    summaries: list[VariantSummary],
    tests: list[PairedTest],
    inflation: list[Inflation],
    reference: str,
    leaderboard_md: str,
    settings: Settings,
) -> str:
    pct = round(settings.evaluation.bootstrap.confidence * 100)
    lines = [
        "# Report tables (generated by make report)",
        "",
        f"Intervals are {pct}% stationary-bootstrap intervals over origins "
        f"({settings.evaluation.bootstrap.n_resamples} resamples, mean block "
        f"{settings.evaluation.bootstrap.mean_block:g}). Seed-level rows average the per-origin series "
        "over seeds before bootstrapping.",
        "",
        "## Table 1. Objectives and baselines on the eligible window",
        "",
        "| Model | Seeds | Origins | Rank IC mean over seeds (sd) | Rank IC [CI] | Net Sharpe [CI] "
        "| Mean DSR (local) | Dispersion | Contaminated IC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        sd = f"{_fmt(s.ic_seed_mean)} ({_fmt(s.ic_seed_std)})" if s.n_seeds > 1 else _fmt(s.ic_seed_mean)
        lines.append(
            f"| {s.name} | {s.n_seeds} | {s.n_origins} | {sd} | {_ci(s.ic)} | {_ci(s.sharpe, 2)} | "
            f"{_fmt(s.dsr_mean)} | {_fmt(s.dispersion, 2)} | {_fmt(s.contaminated_ic)} |"
        )
    lines += [
        "",
        "## Table 2. Paired tests on per-origin rank IC",
        "",
        "| A | B | Origins | Mean IC difference A - B [CI] | Paired t | p (two-sided) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {t.a} | {t.b} | {t.n} | {_ci(t.mean_diff, 4)} | {_fmt(t.t_stat, 2)} | {_fmt(t.p_value, 4)} |"
        for t in tests
    ]
    lines += [
        "",
        "## Table 3. Look-ahead inflation: rank IC before and after each model's cutoff",
        "",
        f"The reference columns score {reference} on the same origins; its ratio is the part of the gap "
        "that the decay of the planted signal explains.",
        "",
        "| Model | IC before cutoff | IC after cutoff | Ratio | Reference IC before | Reference IC after "
        "| Reference ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.name} | {_fmt(r.contaminated_ic)} | {_fmt(r.eligible_ic)} | "
        f"{_ratio(r.contaminated_ic, r.eligible_ic)} | {_fmt(r.reference_contaminated_ic)} | "
        f"{_fmt(r.reference_eligible_ic)} | {_ratio(r.reference_contaminated_ic, r.reference_eligible_ic)} |"
        for r in inflation
    ]
    lines += ["", "## Table 4. Full arena leaderboard of this run", ""]
    lines += [line.replace("# TSFM Arena leaderboard", "").strip("\n") for line in [leaderboard_md]]
    return "\n".join(lines) + "\n"


def run_report(
    settings: Settings, factory: ModelFactory | None = None, ledger: LedgerClient | None = None
) -> dict[str, Path]:
    factory = factory or ModelFactory(settings)
    ledger = ledger or build_ledger(settings)
    rep = settings.report
    source = synthetic_source(settings)
    registry = load_registry(settings.models_path)
    variants = variant_specs(settings, registry)
    for seed_index, seed in enumerate(rep.seeds):
        seeded = settings.model_copy(update={"seed": seed})
        seeded_factory = factory.with_settings(seeded)
        specs = [group[seed_index] for group in variants.values()]
        train_models(specs, Registry(models=specs), seeded_factory, source)
    tsfm = [registry.get(n) for n in rep.tsfm_models]
    train_models(tsfm, registry, factory, source)
    baselines = [registry.get(n) for n in rep.baselines]
    all_specs = baselines + tsfm + [s for group in variants.values() for s in group]
    out_dir = settings.project_path("reports")
    result = run_arena(settings, all_specs, source, factory, ledger, out_dir / "arena", ["report"])

    dsr = {r.model: r.dsr for r in result.leaderboard.rows}
    by_name = {e.spec.name: e for e in result.entries}
    ppy = settings.evaluation.periods_per_year / settings.evaluation.step
    summaries: dict[str, VariantSummary] = {}
    for spec in baselines + tsfm:
        entry = by_name[spec.name]
        if entry.status == "ok":
            summaries[spec.name] = summarise(spec.name, [entry], dsr, settings, ppy)
    for objective, specs in variants.items():
        group = [by_name[s.name] for s in specs if by_name[s.name].status == "ok"]
        if group:
            summaries[f"patchtst-{objective}"] = summarise(f"patchtst-{objective}", group, dsr, settings, ppy)

    tests = []
    pairs = [(f"patchtst-{loss}", "patchtst-mse") for loss in rep.rank_losses]
    pairs += [("patchtst-listnet", "patchtst-listnet-scratch"), ("patchtst-mse", "ar-ridge")]
    pairs += [(f"patchtst-{loss}", "ar-ridge") for loss in rep.rank_losses]
    pairs += [(f"{m}-rank", m) for m in rep.tsfm_models]
    pairs += [(m, "patchtst-mse") for m in rep.tsfm_models] + [(m, "ar-ridge") for m in rep.tsfm_models]
    for a, b in pairs:
        if a in summaries and b in summaries:
            tests.append(paired_ic_test(summaries[a], summaries[b], settings))

    reference = summaries[rep.reference_model]
    inflation = inflation_rows(list(summaries.values()), reference)
    tables = render_tables(
        list(summaries.values()),
        tests,
        inflation,
        rep.reference_model,
        render_markdown(result.leaderboard),
        settings,
    )
    # The arena paths come first, so its leaderboard "json" key cannot replace the report payload.
    paths = {**result.paths, "tables": out_dir / "report_tables.md", "report_json": out_dir / "report.json"}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths["tables"].write_text(tables, encoding="utf-8")
    payload = {
        "summaries": [
            {k: v for k, v in s.__dict__.items() if k not in {"ic_series", "origin_ts", "contaminated_ts"}}
            | {"ic": s.ic.__dict__, "sharpe": s.sharpe.__dict__}
            for s in summaries.values()
        ],
        "tests": [t.__dict__ | {"mean_diff": t.mean_diff.__dict__} for t in tests],
        "inflation": [r.__dict__ for r in inflation],
        "trials_logged": result.n_trials_logged,
        "trials_sink": result.trials_sink,
    }
    paths["report_json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return paths
