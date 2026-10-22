"""End-to-end: tiny reproduce and report runs fully offline under a temporary data root."""

from __future__ import annotations

import json

import pytest

from tsfm_arena.ledger.client import TrialRecord
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.models.spec import ModelSpec
from tsfm_arena.pipeline.report import run_report
from tsfm_arena.pipeline.training import train_spec
from tsfm_arena.pipeline.run import build_ledger, reproduce
from tsfm_arena.settings import Settings


def test_reproduce_end_to_end(settings: Settings, factory: ModelFactory) -> None:
    result = reproduce(settings, factory)
    board = result.leaderboard
    rows = {r.model: r for r in board.rows}
    # Chronos LoRA is not trained offline, so it is reported as skipped with the reason.
    assert rows["chronos-tiny-rank"].status == "skipped"
    assert "not trained" in (rows["chronos-tiny-rank"].reason or "")
    assert "kronos-small" not in rows  # disabled in the registry
    for name in ("random-walk", "ar-ridge", "patchtst-finance", "patchtst-rank", "chronos-tiny"):
        assert rows[name].status == "ok"
    rank = rows["patchtst-rank"]
    assert rank.eligible is not None and rank.contaminated is not None
    assert rank.eligible.first_origin is not None
    assert rank.eligible.first_origin > "2024-04-22"  # cutoff 2024-04-15 + 7-day embargo
    assert rows["random-walk"].contaminated is None
    assert rank.dsr_source == "local" and rank.gate_passed is None
    assert board.common_window is not None
    assert result.trials_sink == "inbox"
    assert result.n_trials_logged == board.n_trials == 7
    lines = build_ledger(settings).inbox_path().read_text().splitlines()
    record = TrialRecord.model_validate_json(lines[0])
    assert record.source == "s1" and record.family == board.family
    assert record.returns is not None and len(record.returns.ts) == len(record.returns.values) > 0
    assert record.dataset_fingerprint is not None and record.dataset_fingerprint.startswith("sha256:")
    md = result.paths["markdown"].read_text()
    assert "local, S6 unavailable" in md and "Contaminated origins" in md and "Not evaluated" in md
    html = result.paths["html"].read_text()
    assert html.startswith("<!doctype html>") and "<svg" in html
    assert json.loads(result.paths["json"].read_text())["schema_version"] == "1"
    assert settings.project_path(settings.data.local.path).exists()


def test_reproduce_from_local_source(settings: Settings, factory: ModelFactory) -> None:
    reproduce(settings, factory)  # writes the synthetic Parquet copy
    local = settings.model_copy(
        update={"reproduce": settings.reproduce.model_copy(update={"source": "local"})}
    )
    result = reproduce(local, factory.with_settings(local))
    assert result.leaderboard.dataset.startswith("local:")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_report_tables(settings: Settings, factory: ModelFactory) -> None:
    paths = run_report(settings, factory)
    tables = paths["tables"].read_text()
    for name in ("patchtst-mse", "patchtst-listnet", "patchtst-listnet-scratch", "ar-ridge", "chronos-tiny"):
        assert f"| {name} |" in tables
    assert "Paired t" in tables
    payload = json.loads(paths["json"].read_text())
    mse = next(s for s in payload["summaries"] if s["name"] == "patchtst-mse")
    assert mse["n_seeds"] == 2 and mse["contaminated_ic"] is not None
    assert {t["a"] for t in payload["tests"]} >= {"patchtst-listnet", "patchtst-mse"}


def test_lora_training_path_when_downloads_allowed(settings: Settings, factory: ModelFactory) -> None:
    from tsfm_arena.data.sources import InMemorySource
    from tsfm_arena.data.synthetic import generate_cross_section
    from tsfm_arena.models.spec import load_registry
    from tsfm_arena.pipeline.run import train_models

    allowed = settings.model_copy(update={"allow_downloads": True})
    fac = factory.with_settings(allowed)
    registry = load_registry(settings.models_path)
    source = InMemorySource(generate_cross_section(settings.data.synthetic).panel)
    trained = train_models([registry.get("chronos-tiny-rank")], registry, fac, source)
    assert trained == ["chronos-tiny-rank"]
    forecaster = fac.build(registry.get("chronos-tiny-rank"))
    assert forecaster.name == "chronos-tiny-rank"
    untrainable = ModelSpec(
        name="rw", adapter="random_walk", pretraining_cutoff="2024-01-01T00:00:00Z", training={}
    )
    panel = source.load(None, None, None).panel
    with pytest.raises(ValueError, match="no training procedure"):
        train_spec(untrainable, registry, fac, panel)
    with pytest.raises(ValueError, match="no training section"):
        train_spec(registry.get("random-walk"), registry, fac, panel)
