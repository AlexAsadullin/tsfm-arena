from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from tsfm_arena.settings import Settings, default_settings_file, load_settings
from tsfm_arena.timeutil import ensure_utc, from_ns, iso, to_ns
from tests.conftest import tiny_overrides


def test_repo_settings_load_with_data_root_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRADEQ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TSFM_EVALUATION__LONG_SHORT__COST_BPS", "3.5")
    monkeypatch.delenv("TSFM_SETTINGS_FILE", raising=False)
    s = load_settings(_env_file=None)
    assert s.data_root == tmp_path
    assert s.evaluation.long_short.cost_bps == 3.5
    assert s.service.port == 8604
    assert s.evaluation.embargo == timedelta(days=30)
    assert s.project_dir == tmp_path / "tsfm-arena"
    assert s.ledger_inbox_dir == tmp_path / "research-ledger" / "inbox"
    assert s.models_path == default_settings_file().parent / "models.yaml"
    assert s.project_path("/abs/x") == Path("/abs/x")


def test_settings_file_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TSFM_SETTINGS_FILE", str(tmp_path / "s.yaml"))
    assert default_settings_file() == tmp_path / "s.yaml"


def test_quantiles_must_increase(tmp_path: Path) -> None:
    overrides = tiny_overrides(tmp_path)
    overrides["evaluation"]["quantiles"] = [0.5, 0.1]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_time_roundtrip_and_naive_rejected() -> None:
    dt = datetime(2024, 5, 6, 7, 8, 9, 123456, tzinfo=UTC)
    assert from_ns(to_ns(dt)) == dt
    assert iso(to_ns(dt)) == "2024-05-06T07:08:09.123456Z"
    assert ensure_utc(datetime(2024, 1, 1, 3, tzinfo=timezone(timedelta(hours=3)))).hour == 0
    with pytest.raises(ValueError, match="naive"):
        to_ns(datetime(2024, 1, 1))
