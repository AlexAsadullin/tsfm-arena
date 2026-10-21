"""The arena pipeline: train in-house models, evaluate, log trials to S6, write the leaderboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tsfm_arena.data.base import DataSource
from tsfm_arena.data.factory import build_source
from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.data.sources import InMemorySource, write_long
from tsfm_arena.data.synthetic import generate_cross_section
from tsfm_arena.evaluation.arena import Arena
from tsfm_arena.leaderboard.board import Leaderboard, build_leaderboard
from tsfm_arena.leaderboard.render import write_leaderboard
from tsfm_arena.ledger.client import LedgerClient, Verdict
from tsfm_arena.logging import get_logger
from tsfm_arena.models.base import ModelUnavailableError
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.models.spec import ModelSpec, Registry, load_registry
from tsfm_arena.pipeline.evaluate import Entry, evaluate_specs, family_name, git_sha, trial_records
from tsfm_arena.pipeline.training import train_spec
from tsfm_arena.settings import Settings
from tsfm_arena.timeutil import from_ns, utc_now

log = get_logger(__name__)
_REPO = Path(__file__).resolve().parents[3]


def build_ledger(settings: Settings) -> LedgerClient:
    cfg = settings.ledger
    return LedgerClient(cfg.base_url, cfg.mode, settings.ledger_inbox_dir, cfg.source, cfg.timeout_s)


def synthetic_source(settings: Settings) -> InMemorySource:
    """Generate the synthetic cross-section and keep a Parquet copy in the project data subtree, so the
    same data can be re-read through the local-file source."""
    cross_section = generate_cross_section(settings.data.synthetic)
    write_long(cross_section.panel, settings.project_path(settings.data.local.path))
    return InMemorySource(cross_section.panel, name="synthetic")


def train_models(
    specs: list[ModelSpec], registry: Registry, factory: ModelFactory, source: DataSource
) -> list[str]:
    """Train every spec with a training section (parents first); return the names that were trained.

    Chronos LoRA adapters train only when downloads are allowed and the setting asks for it, because
    a LoRA adapter on randomly initialised weights would be meaningless.
    """
    settings = factory.settings
    panel: ReturnPanel | None = None
    trained = []
    for spec in registry.training_order([s.name for s in specs]):
        if spec.adapter == "chronos" and not (
            settings.allow_downloads and settings.reproduce.run_lora_when_downloads_allowed
        ):
            log.info("lora_training_skipped", model=spec.name, reason="downloads disabled")
            continue
        if panel is None:
            panel = source.load(None, utc_now(), utc_now(), settings.data.symbols or None).panel
        try:
            train_spec(spec, registry, factory, panel)
        except ModelUnavailableError as exc:
            log.warning("training_skipped", model=spec.name, reason=str(exc))
            continue
        trained.append(spec.name)
    return trained


@dataclass
class ArenaResult:
    leaderboard: Leaderboard
    entries: list[Entry]
    paths: dict[str, Path]
    trials_sink: str
    n_trials_logged: int


def run_arena(
    settings: Settings,
    specs: list[ModelSpec],
    source: DataSource,
    factory: ModelFactory,
    ledger: LedgerClient,
    out_dir: Path,
    tags: list[str],
) -> ArenaResult:
    """Evaluate ``specs`` on ``source``, log one trial per scored spec and write the leaderboard."""
    probe = source.load(None, utc_now(), utc_now(), settings.data.symbols or None)
    as_of = from_ns(int(probe.panel.ts[-1]))
    arena = Arena(settings.evaluation, source, as_of, settings.data.symbols or None)
    entries = evaluate_specs(arena, specs, factory)
    records = trial_records(entries, settings, arena, git_sha(_REPO), tags)
    sink = ledger.log_trials(records)
    verdicts: dict[str, Verdict | None] = (
        {r.strategy_id: ledger.verdict(r.strategy_id) for r in records} if sink == "http" else {}
    )
    family = family_name(settings, source.name)
    board = build_leaderboard(arena, entries, family, verdicts, utc_now(), sink)
    paths = write_leaderboard(board, out_dir)
    return ArenaResult(board, entries, paths, sink, len(records))


def reproduce(
    settings: Settings, factory: ModelFactory | None = None, ledger: LedgerClient | None = None
) -> ArenaResult:
    """The offline end-to-end demo: synthetic cross-section, training, arena, trials, leaderboard."""
    factory = factory or ModelFactory(settings)
    ledger = ledger or build_ledger(settings)
    kind = settings.reproduce.source
    source = synthetic_source(settings) if kind == "synthetic" else build_source(settings, kind)
    registry = load_registry(settings.models_path)
    specs = registry.enabled()
    train_models(specs, registry, factory, source)
    return run_arena(
        settings, specs, source, factory, ledger, settings.project_path("leaderboard"), ["reproduce"]
    )
