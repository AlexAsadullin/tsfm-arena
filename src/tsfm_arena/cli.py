"""Command-line interface: ``tsfm-arena reproduce | report | train | evaluate | models | serve``."""

from __future__ import annotations

from typing import Annotated

import typer

from tsfm_arena.data.factory import build_source
from tsfm_arena.logging import configure_logging
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.models.spec import load_registry
from tsfm_arena.pipeline.run import ArenaResult, build_ledger, reproduce, run_arena, train_models
from tsfm_arena.settings import Settings, load_settings

app = typer.Typer(add_completion=False, help="Cutoff-aware TSFM benchmark on financial returns.")


def _settings() -> Settings:
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_json)
    return settings


def _summary(result: ArenaResult) -> None:
    board = result.leaderboard
    typer.echo(f"family {board.family}: {result.n_trials_logged} trials logged via {result.trials_sink}")
    typer.echo(f"{'model':<26}{'origins':>8}{'rank IC':>9}{'90% CI':>18}{'net SR':>8}{'DSR':>7}")
    for row in board.rows:
        s = row.eligible
        if s is None:
            typer.echo(f"{row.model:<26}  skipped: {row.reason}")
            continue
        ic = s.rank_ic
        ci = f"[{ic.low:.3f}, {ic.high:.3f}]" if ic.low is not None and ic.high is not None else "n/a"
        sr = s.sharpe_net.estimate
        typer.echo(
            f"{row.model:<26}{s.n_origins:>8}{(ic.estimate or float('nan')):>9.3f}{ci:>18}"
            f"{(sr if sr is not None else float('nan')):>8.2f}"
            f"{(row.dsr if row.dsr is not None else float('nan')):>7.2f}"
        )
    for name, path in result.paths.items():
        typer.echo(f"{name}: {path}")


@app.command("reproduce")
def reproduce_cmd() -> None:
    """Offline end-to-end demo on the synthetic cross-section."""
    _summary(reproduce(_settings()))


@app.command("report")
def report_cmd() -> None:
    """Run the research-report experiments and write tables under the data root."""
    from tsfm_arena.pipeline.report import run_report

    paths = run_report(_settings())
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@app.command("models")
def models_cmd() -> None:
    """List registry models with their declared pretraining cutoffs."""
    settings = _settings()
    for spec in load_registry(settings.models_path).models:
        cutoff = spec.pretraining_cutoff.isoformat() if spec.pretraining_cutoff else "none"
        state = "enabled" if spec.enabled else "disabled"
        typer.echo(f"{spec.name:<26}{spec.adapter:<17}{cutoff:<27}{state}")


@app.command("train")
def train_cmd(
    names: Annotated[list[str], typer.Argument(help="Registry models to train (parents train first).")],
) -> None:
    """Train in-house models (PatchTST, Chronos LoRA) on the configured source up to their cutoffs."""
    settings = _settings()
    registry = load_registry(settings.models_path)
    trained = train_models(
        [registry.get(n) for n in names], registry, ModelFactory(settings), build_source(settings)
    )
    typer.echo(f"trained: {', '.join(trained) or 'nothing'}")


@app.command("evaluate")
def evaluate_cmd(
    names: Annotated[list[str] | None, typer.Argument(help="Models to evaluate (default: enabled).")] = None,
) -> None:
    """Evaluate models on the configured source and write the leaderboard."""
    settings = _settings()
    registry = load_registry(settings.models_path)
    specs = [registry.get(n) for n in names] if names else registry.enabled()
    result = run_arena(
        settings,
        specs,
        build_source(settings),
        ModelFactory(settings),
        build_ledger(settings),
        settings.project_path("leaderboard"),
        ["evaluate"],
    )
    _summary(result)


@app.command("serve")
def serve_cmd() -> None:
    """Run the forecast service (contract 6.5)."""
    import uvicorn

    from tsfm_arena.service.app import create_app

    settings = _settings()
    uvicorn.run(create_app(settings), host=settings.service.host, port=settings.service.port)
