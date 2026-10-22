from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tsfm_arena import cli
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.pipeline import run as run_module
from tsfm_arena.settings import Settings


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, settings: Settings, factory: ModelFactory) -> Settings:
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "ModelFactory", lambda s: factory)
    monkeypatch.setattr(run_module, "ModelFactory", lambda s: factory)
    return settings


def test_cli_models_train_evaluate_reproduce(patched: Settings) -> None:
    runner = CliRunner()
    out = runner.invoke(cli.app, ["models"])
    assert out.exit_code == 0 and "patchtst-rank" in out.output
    out = runner.invoke(cli.app, ["train", "patchtst-rank"])
    assert out.exit_code == 0, out.output
    assert "patchtst-finance, patchtst-rank" in out.output
    out = runner.invoke(cli.app, ["evaluate", "random-walk", "patchtst-rank"])
    assert out.exit_code == 0, out.output
    assert "patchtst-rank" in out.output and "markdown:" in out.output
    out = runner.invoke(cli.app, ["reproduce"])
    assert out.exit_code == 0, out.output
    assert "trials logged via inbox" in out.output and "skipped" in out.output
    assert Path(out.output.strip().splitlines()[-1].split(": ")[1]).exists()


def test_cli_settings_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRADEQ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TSFM_LOG_LEVEL", "ERROR")
    assert cli._settings().data_root == tmp_path
