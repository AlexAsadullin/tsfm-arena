"""Shared fixtures: tiny CPU settings over a temporary data root; no network, no downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import T5Config

from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.data.synthetic import generate_cross_section
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def tiny_overrides(data_root: Path) -> dict[str, Any]:
    return {
        "data_root": data_root,
        "models_file": FIXTURES / "models.yaml",
        "allow_downloads": False,
        "log_level": "WARNING",
        "data": {
            "synthetic": {
                "n_assets": 8,
                "n_periods": 240,
                "start": "2024-01-01T00:00:00Z",
                "signal_ic": 0.3,
                "signal_halflife": 1000,
                "momentum_window": 3,
            }
        },
        "evaluation": {
            "context_length": 16,
            "min_history": 40,
            "refit_every": 20,
            "embargo": "P7D",
            "bootstrap": {"n_resamples": 50, "mean_block": 5},
        },
        "ledger": {"mode": "inbox"},
        "training": {
            "patchtst": {
                "context_length": 16,
                "patch_len": 4,
                "stride": 4,
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
                "d_ff": 16,
                "dropout": 0.0,
            },
            "epochs": 2,
            "dates_per_batch": 16,
            "lora": {"r": 2, "alpha": 4, "epochs": 1, "dates_per_batch": 16},
        },
        "report": {
            "seeds": [1, 2],
            "rank_losses": ["listnet"],
            "baselines": ["random-walk", "ar-ridge"],
            "tsfm_models": ["chronos-tiny"],
        },
    }


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for var in ("TSFM_ALLOW_DOWNLOADS", "TSFM_SETTINGS_FILE", "TRADEQ_DATA_DIR"):
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None, **tiny_overrides(tmp_path / "data"))  # type: ignore[arg-type]


@pytest.fixture
def panel(settings: Settings) -> ReturnPanel:
    return generate_cross_section(settings.data.synthetic).panel


def tiny_chronos_model(prediction_length: int = 4) -> Any:
    """A randomly initialised Chronos-Bolt model small enough for CPU tests."""
    from chronos.chronos_bolt import ChronosBoltModelForForecasting

    torch.manual_seed(0)
    config = T5Config(
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=8,
        dense_act_fn="relu",
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        chronos_config={
            "context_length": 32,
            "prediction_length": prediction_length,
            "input_patch_size": 8,
            "input_patch_stride": 8,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        architectures=["ChronosBoltModelForForecasting"],
    )
    return ChronosBoltModelForForecasting(config)


def tiny_pipeline_loader(hf_id: str, cache_dir: Path, device: str, allow_downloads: bool) -> Any:
    from chronos import ChronosBoltPipeline

    return ChronosBoltPipeline(model=tiny_chronos_model())


@pytest.fixture
def factory(settings: Settings) -> ModelFactory:
    return ModelFactory(settings, pipeline_loader=tiny_pipeline_loader)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
