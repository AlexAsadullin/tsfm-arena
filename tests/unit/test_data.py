from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm_arena.data.factory import build_source
from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.data.sources import InMemorySource, LocalFileSource, write_long
from tsfm_arena.data.synthetic import generate_cross_section
from tsfm_arena.evaluation.metrics import rank_ic_series
from tsfm_arena.settings import Settings
from tsfm_arena.timeutil import to_ns


def _panel() -> ReturnPanel:
    ts = np.array([10, 20, 30, 40], dtype=np.int64)
    returns = np.array([[0.1, 0.0], [0.2, -0.1], [0.3, np.nan], [0.4, 0.1]])
    return ReturnPanel(ts, ("A", "B"), returns)


def test_panel_validation() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ReturnPanel(np.array([2, 1], dtype=np.int64), ("A",), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="shape"):
        ReturnPanel(np.array([1, 2], dtype=np.int64), ("A", "B"), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="unique"):
        ReturnPanel(np.array([1], dtype=np.int64), ("A", "A"), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="1-D"):
        ReturnPanel(np.zeros((1, 1), dtype=np.int64), ("A",), np.zeros((1, 1)))


def test_slicing_and_forward_returns() -> None:
    p = _panel()
    assert p.upto(25).n_periods == 2
    assert p.between(20, 30).ts.tolist() == [20, 30]
    assert p.select(["B"]).symbols == ("B",)
    with pytest.raises(KeyError):
        p.select(["C"])
    fwd = p.forward_returns(2)
    assert fwd[0, 0] == pytest.approx(0.5)
    assert np.isnan(fwd[0, 1]) and np.isnan(fwd[2:]).all()
    assert np.isnan(p.rows(0, 1).forward_returns(3)).all()


def test_fingerprint_is_content_hash() -> None:
    p = _panel()
    assert p.fingerprint() == _panel().fingerprint()
    changed = ReturnPanel(p.ts, p.symbols, p.returns + 1e-9)
    assert changed.fingerprint() != p.fingerprint()


def test_from_long_prices_and_returns() -> None:
    frame = pd.DataFrame(
        {
            "t": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
            "s": ["A"] * 3 + ["B"] * 3,
            "p": [100.0, 110.0, 99.0, 10.0, 10.0, 20.0],
        }
    )
    p = ReturnPanel.from_long(frame, "t", "s", "p", "price")
    assert p.n_periods == 2
    assert p.returns[0, 0] == pytest.approx(np.log(1.1))
    assert p.returns[1, 1] == pytest.approx(np.log(2.0))
    assert p.ts[0] == to_ns(datetime(2024, 1, 2, tzinfo=UTC))
    back = ReturnPanel.from_long(p.to_long(), "ts", "symbol", "ret", "return")
    np.testing.assert_allclose(back.returns, p.returns)
    with pytest.raises(ValueError, match="positive"):
        ReturnPanel.from_long(frame.assign(p=-1.0), "t", "s", "p", "price")
    with pytest.raises(KeyError):
        ReturnPanel.from_long(frame, "t", "s", "missing", "price")
    with pytest.raises(ValueError, match="empty"):
        ReturnPanel.from_long(frame.iloc[:0], "t", "s", "p", "price")


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_local_file_source_roundtrip(tmp_path: Path, panel: ReturnPanel, suffix: str) -> None:
    path = write_long(panel, tmp_path / f"d{suffix}")
    src = LocalFileSource(path, "ts", "symbol", "ret", "return")
    known = datetime(2024, 3, 1, tzinfo=UTC)
    out = src.load(None, None, known, [panel.symbols[1]])
    assert out.panel.ts[-1] <= to_ns(known)
    assert out.panel.symbols == (panel.symbols[1],)
    assert out.fingerprint is not None and out.fingerprint.startswith("sha256:")
    np.testing.assert_allclose(out.panel.returns[:, 0], panel.upto(to_ns(known)).returns[:, 1])


def test_local_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalFileSource(tmp_path / "nope.parquet", "ts", "symbol", "ret", "return")
    bad = tmp_path / "x.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        LocalFileSource(bad, "ts", "symbol", "ret", "return")


def test_in_memory_source_respects_known_at(panel: ReturnPanel) -> None:
    src = InMemorySource(panel)
    end = datetime(2024, 6, 1, tzinfo=UTC)
    known = datetime(2024, 2, 1, tzinfo=UTC)
    out = src.load(None, end, known)
    assert out.panel.ts[-1] == to_ns(known)
    assert not src.revisable


def test_build_source_kinds(settings: Settings, panel: ReturnPanel) -> None:
    assert build_source(settings, "synthetic").name == "synthetic"
    write_long(panel, settings.project_path(settings.data.local.path))
    assert build_source(settings, "local").name.startswith("local:")
    assert build_source(settings, "pit").revisable


def test_synthetic_is_deterministic_and_plants_a_decaying_signal(settings: Settings) -> None:
    cfg = settings.data.synthetic.model_copy(update={"n_assets": 40, "n_periods": 600, "signal_halflife": 300})
    a, b = generate_cross_section(cfg), generate_cross_section(cfg)
    np.testing.assert_array_equal(a.panel.returns, b.panel.returns)
    assert a.planted_ic[0] == pytest.approx(cfg.signal_ic)
    assert a.planted_ic[300] == pytest.approx(cfg.signal_ic / 2)
    ic = rank_ic_series(a.signal[:-1], a.panel.forward_returns(1)[:-1])
    early, late = np.nanmean(ic[:200]), np.nanmean(ic[-200:])
    assert early > 0.1
    assert late < early
    assert len(a.panel.symbols[0]) > 0 and a.panel.symbols[0].startswith(cfg.symbol_prefix)
