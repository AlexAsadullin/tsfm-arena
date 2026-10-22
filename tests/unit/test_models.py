from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from tests.conftest import FIXTURES, tiny_chronos_model
from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.models import chronos, kronos
from tsfm_arena.models.base import CutoffViolationError, Forecast, ModelUnavailableError, nan_std
from tsfm_arena.models.baselines import ARRidge, CrossSectionalMomentum, HistoricalMean, RandomWalk, ridge_fit
from tsfm_arena.models.factory import ModelFactory, check_training_cutoff
from tsfm_arena.models.patchtst import (
    PatchTSTConfig,
    PatchTSTForecaster,
    PatchTSTNet,
    load_checkpoint,
    save_checkpoint,
)
from tsfm_arena.models.spec import ModelSpec, Registry, load_registry
from tsfm_arena.settings import Settings

LEVELS = [0.1, 0.5, 0.9]


def test_forecast_sorts_crossing_quantiles() -> None:
    f = Forecast(np.zeros(1), np.array([[0.3, 0.1, 0.2]]), (0.1, 0.5, 0.9))
    assert f.quantiles.tolist() == [[0.1, 0.2, 0.3]]
    with pytest.raises(ValueError, match="shapes"):
        Forecast(np.zeros(2), np.zeros((1, 3)), (0.1, 0.5, 0.9))


def test_nan_std_floor() -> None:
    ctx = np.array([[1.0, 3.0, np.nan], [np.nan, np.nan, np.nan], [2.0, 2.0, 2.0]])
    np.testing.assert_allclose(nan_std(ctx, 1e-3), [np.sqrt(2.0), 1e-3, 1e-3])


def test_random_walk_and_historical_mean() -> None:
    ctx = np.array([[0.01, -0.01, 0.01, -0.01], [0.02, 0.02, 0.02, 0.02]])
    rw = RandomWalk("rw", 4)
    rw.fit(ReturnPanel(np.array([1], dtype=np.int64), ("a", "b"), np.zeros((1, 2))), 1)
    f = rw.predict(ctx, 4, LEVELS)
    assert f.mean.tolist() == [0.0, 0.0]
    assert f.quantiles[0, 2] == pytest.approx(ctx[0].std(ddof=1) * 2 * 1.2815515, rel=1e-6)
    hm = HistoricalMean("hm", 4).predict(ctx, 2, LEVELS)
    assert hm.mean.tolist() == pytest.approx([0.0, 0.04])


def test_ridge_recovers_ar_coefficient(rng: np.random.Generator) -> None:
    t, n, phi = 2000, 5, 0.3
    r = np.zeros((t, n))
    for i in range(1, t):
        r[i] = phi * r[i - 1] + rng.standard_normal(n) * 0.01
    panel = ReturnPanel(np.arange(t, dtype=np.int64), tuple("abcde"), r)
    model = ARRidge("ar", lags=2, alpha=1e-6, train_window=t)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(r[-5:].T, 1, LEVELS)
    model.fit(panel, 1)
    assert model._coef is not None
    assert model._coef[0] == pytest.approx(phi, abs=0.05)
    assert abs(model._coef[1]) < 0.05
    f = model.predict(r[-5:].T, 1, LEVELS)
    assert np.corrcoef(f.mean, r[-1])[0, 1] > 0.9
    with pytest.raises(ValueError, match="too few"):
        ARRidge("ar", 3, 1.0, 3).fit(panel, 1)
    intercept, coef = ridge_fit(np.arange(10.0)[:, None], 2 * np.arange(10.0) + 1, 0.0)
    assert intercept == pytest.approx(1.0) and coef[0] == pytest.approx(2.0)


def test_xs_momentum_scores_relative_trend() -> None:
    ctx = np.array([[0.0, 0.03, 0.03], [0.0, -0.03, -0.03], [0.0, 0.0, 0.0]])
    f = CrossSectionalMomentum("m", 2, 3).predict(ctx, 1, LEVELS)
    assert f.mean[0] > f.mean[2] > f.mean[1]
    assert f.mean.sum() == pytest.approx(0.0)


def _net(horizon: int = 1) -> PatchTSTNet:
    torch.manual_seed(0)
    cfg = PatchTSTConfig(
        context_length=16, patch_len=4, stride=4, d_model=8, n_heads=2, n_layers=1, d_ff=16,
        dropout=0.0, horizon=horizon, levels=LEVELS,
    )  # fmt: skip
    return PatchTSTNet(cfg)


def test_patchtst_shapes_monotone_quantiles_and_nan(tmp_path: Path, rng: np.random.Generator) -> None:
    net = _net()
    ctx = rng.normal(0, 0.02, (5, 10))
    ctx[0, :4] = np.nan
    fc = PatchTSTForecaster("p", net).predict(ctx, 1, LEVELS)  # left-padded to 16
    assert fc.mean.shape == (5,) and np.isfinite(fc.quantiles).all()
    _, quant = net(torch.randn(3, 16))
    assert bool((quant.diff(dim=1) >= 0).all())
    path = save_checkpoint(tmp_path / "c.pt", net, {"train_end": "2024-01-01T00:00:00Z"})
    loaded, meta = load_checkpoint(path)
    assert meta["train_end"].startswith("2024")
    again = PatchTSTForecaster("p", loaded).predict(ctx, 1, LEVELS)
    np.testing.assert_allclose(again.mean, fc.mean)
    with pytest.raises(ValueError, match="horizon"):
        PatchTSTForecaster("p", net).predict(ctx, 2, LEVELS)
    with pytest.raises(ValueError, match="quantile"):
        PatchTSTForecaster("p", net).predict(ctx, 1, [0.5])
    with pytest.raises(ValueError, match="patch_len"):
        PatchTSTNet(net.cfg.model_copy(update={"patch_len": 32}))


def test_registry_order_cycles_and_hash() -> None:
    reg = load_registry(FIXTURES / "models.yaml")
    order = [s.name for s in reg.training_order(["patchtst-rank", "random-walk"])]
    assert order == ["patchtst-finance", "patchtst-rank"]
    assert reg.get("kronos-small") not in reg.enabled()
    with pytest.raises(KeyError):
        reg.get("nope")
    spec = reg.get("random-walk")
    assert spec.config_hash({"a": 1}) == spec.config_hash({"a": 1}) != spec.config_hash({"a": 2})
    a = ModelSpec(name="a", adapter="patchtst", pretraining_cutoff=None, training={"init_from": "b"})
    b = ModelSpec(name="b", adapter="patchtst", pretraining_cutoff=None, training={"init_from": "a"})
    with pytest.raises(ValueError, match="cycle"):
        Registry(models=[a, b]).training_order(["a"])
    with pytest.raises(ValueError, match="unique"):
        Registry(models=[a, a])


def test_cutoff_check() -> None:
    spec = ModelSpec(
        name="m", adapter="patchtst", pretraining_cutoff=datetime(2024, 1, 1, tzinfo=UTC), path=Path("x")
    )
    check_training_cutoff(spec, {"train_end": "2023-12-31T00:00:00Z"})
    with pytest.raises(CutoffViolationError, match="after the declared"):
        check_training_cutoff(spec, {"train_end": "2024-01-02T00:00:00Z"})
    with pytest.raises(CutoffViolationError, match="declare"):
        check_training_cutoff(spec.model_copy(update={"pretraining_cutoff": None}), {"train_end": "x"})


def test_factory_builds_and_refuses(settings: Settings, factory: ModelFactory) -> None:
    reg = load_registry(settings.models_path)
    for name in ("random-walk", "hist-mean", "ar-ridge", "xs-momentum", "chronos-tiny"):
        assert factory.build(reg.get(name)).name == name
    for name in ("patchtst-finance", "chronos-tiny-rank"):
        with pytest.raises(ModelUnavailableError, match="not trained"):
            factory.build(reg.get(name))
    with pytest.raises(ModelUnavailableError, match="Kronos"):
        factory.build(reg.get("kronos-small"))
    bad = reg.get("chronos-tiny").model_copy(update={"params": {"context_length": 8, "input_mode": "x"}})
    with pytest.raises(ValueError, match="input_mode"):
        factory.build(bad)
    with pytest.raises(ModelUnavailableError, match="hf_id"):
        factory.build(bad.model_copy(update={"hf_id": None}))
    with pytest.raises(ModelUnavailableError, match="checkpoint path"):
        factory.checkpoint_path(reg.get("random-walk"))
    path = factory.checkpoint_path(reg.get("patchtst-finance"))
    save_checkpoint(path, _net(), {"train_end": "2025-01-01T00:00:00Z"})
    with pytest.raises(CutoffViolationError):
        factory.build(reg.get("patchtst-finance"))


def test_chronos_adapter_with_tiny_bolt(
    factory: ModelFactory, settings: Settings, rng: np.random.Generator
) -> None:
    spec = load_registry(settings.models_path).get("chronos-tiny")
    fc = factory.build(spec)
    ctx = rng.normal(0, 0.02, (4, 40))
    ctx[1, :30] = np.nan
    out = fc.predict(ctx, 1, LEVELS)
    assert out.mean.shape == (4,) and out.quantiles.shape == (4, 3)
    with pytest.raises(ValueError, match="horizon 1"):
        fc.predict(ctx, 2, LEVELS)
    level = chronos.ChronosForecaster("c", fc.pipeline, 32, "log_price")  # type: ignore[attr-defined]
    assert level.predict(ctx, 3, LEVELS).quantiles.shape == (4, 3)
    empty = np.full((1, 40), np.nan)
    assert np.isfinite(fc.predict(empty, 1, LEVELS).mean).all()


class _ListPipeline:
    """Mimics Chronos-2, which returns one (1, H, Q) tensor per univariate input."""

    def predict_quantiles(self, inputs: Any, prediction_length: int, quantile_levels: list[float]) -> Any:
        q = [torch.full((1, prediction_length, len(quantile_levels)), float(x[-1])) for x in inputs]
        m = [torch.full((1, prediction_length), float(x[-1])) for x in inputs]
        return q, m


def test_chronos_adapter_stacks_list_outputs() -> None:
    fc = chronos.ChronosForecaster("c2", _ListPipeline(), 8, "returns")
    out = fc.predict(np.array([[0.0, 0.1], [0.0, -0.2]]), 1, LEVELS)
    assert out.mean.tolist() == pytest.approx([0.1, -0.2])


def test_lora_injection_save_and_load(tmp_path: Path, settings: Settings) -> None:
    model = chronos.add_lora(tiny_chronos_model(), settings.training.lora)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.fill_(0.5)
    path = chronos.save_lora(tmp_path / "l.pt", model, {"train_end": "2024-01-01T00:00:00Z"})
    fresh = chronos.add_lora(tiny_chronos_model(), settings.training.lora)
    assert chronos.load_lora(path, fresh)["train_end"].startswith("2024")
    for k, v in chronos.lora_state_dict(fresh).items():
        torch.testing.assert_close(v, chronos.lora_state_dict(model)[k])
    torch.save({"lora": {}, "meta": {}}, tmp_path / "empty.pt")
    with pytest.raises(ValueError, match="lacks"):
        chronos.load_lora(tmp_path / "empty.pt", fresh)


def test_load_pipeline_offline_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="TSFM_ALLOW_DOWNLOADS"):
        chronos.load_pipeline("test/does-not-exist", tmp_path, "cpu", allow_downloads=False)


class _FakeKronos:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, df: pd.DataFrame, x_timestamp: pd.Series, y_timestamp: pd.Series, pred_len: int,
                T: float, top_p: float, sample_count: int, verbose: bool) -> pd.DataFrame:  # fmt: skip
        assert list(df.columns) == list(kronos.CANDLE_COLUMNS) and len(y_timestamp) == pred_len
        self.calls += 1
        last = float(df["close"].iloc[-1])
        growth = 1.0 + 0.01 * (self.calls % 3)
        return pd.DataFrame({"close": [last * growth**k for k in range(1, pred_len + 1)]})


def test_kronos_adapter_with_fake_predictor(settings: Settings) -> None:
    candles = kronos.candles_from_returns(np.array([0.1, np.nan, -0.1]))
    assert candles["close"].iloc[-1] == pytest.approx(1.0)
    assert (candles["high"] >= candles[["open", "close"]].max(axis=1)).all()
    fake = _FakeKronos()
    fc = kronos.KronosForecaster("k", fake, 600, settings.kronos, seed=1)
    assert fc.context_length == settings.kronos.max_context
    out = fc.predict(np.zeros((2, 5)), 2, LEVELS)
    assert fake.calls == 2 * settings.kronos.n_paths
    assert out.quantiles[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert out.quantiles[0, 2] == pytest.approx(2 * np.log(1.02), rel=1e-6)


def test_kronos_loader_needs_checkout(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="checkout"):
        kronos.load_predictor("a", "b", settings.kronos, tmp_path, "cpu", False)
    (tmp_path / "model").mkdir()
    cfg = settings.kronos.model_copy(update={"repo_path": tmp_path})
    with pytest.raises(ModelUnavailableError, match="could not be loaded"):
        kronos.load_predictor("a", "b", cfg, tmp_path, "cpu", False)
