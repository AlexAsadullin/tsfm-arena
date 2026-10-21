"""Build a :class:`Forecaster` from a registry entry."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from tsfm_arena.models import chronos, kronos
from tsfm_arena.models.base import CutoffViolationError, Forecaster, ModelUnavailableError
from tsfm_arena.models.baselines import ARRidge, HistoricalMean, RandomWalk
from tsfm_arena.models.patchtst import PatchTSTForecaster, load_checkpoint
from tsfm_arena.models.spec import ModelSpec
from tsfm_arena.settings import Settings
from tsfm_arena.timeutil import ensure_utc

PipelineLoader = Callable[[str, Path, str, bool], Any]
KronosLoader = Callable[..., Any]


def check_training_cutoff(spec: ModelSpec, meta: dict[str, Any]) -> None:
    """Refuse a checkpoint whose training data reaches past the declared ``pretraining_cutoff``."""
    if spec.pretraining_cutoff is None:
        raise CutoffViolationError(f"{spec.name}: a trained checkpoint must declare pretraining_cutoff")
    train_end = ensure_utc(datetime.fromisoformat(str(meta["train_end"]).replace("Z", "+00:00")))
    if train_end > spec.pretraining_cutoff:
        raise CutoffViolationError(
            f"{spec.name}: trained on data up to {train_end.isoformat()}, after the declared "
            f"cutoff {spec.pretraining_cutoff.isoformat()}"
        )


class ModelFactory:
    """Builds forecasters; the heavy loaders are injected so tests never download weights."""

    def __init__(
        self,
        settings: Settings,
        pipeline_loader: PipelineLoader = chronos.load_pipeline,
        kronos_loader: KronosLoader = kronos.load_predictor,
    ) -> None:
        self.settings = settings
        self._pipeline_loader = pipeline_loader
        self._kronos_loader = kronos_loader

    def build(self, spec: ModelSpec) -> Forecaster:
        p = spec.params
        if spec.adapter == "random_walk":
            return RandomWalk(spec.name, int(p["context_length"]))
        if spec.adapter == "historical_mean":
            return HistoricalMean(spec.name, int(p["context_length"]))
        if spec.adapter == "ar_ridge":
            return ARRidge(spec.name, int(p["lags"]), float(p["alpha"]), int(p["train_window"]))
        if spec.adapter == "patchtst":
            return self._patchtst(spec)
        if spec.adapter == "chronos":
            return self._chronos(spec)
        return self._kronos(spec)

    def checkpoint_path(self, spec: ModelSpec) -> Path:
        if spec.path is None:
            raise ModelUnavailableError(f"{spec.name}: no checkpoint path in the registry")
        return self.settings.project_path(spec.path)

    def _patchtst(self, spec: ModelSpec) -> Forecaster:
        path = self.checkpoint_path(spec)
        if not path.exists():
            raise ModelUnavailableError(f"{spec.name}: checkpoint {path} not trained yet")
        net, meta = load_checkpoint(path)
        check_training_cutoff(spec, meta)
        return PatchTSTForecaster(spec.name, net, meta)

    def load_chronos_pipeline(self, spec: ModelSpec) -> Any:
        if spec.hf_id is None:
            raise ModelUnavailableError(f"{spec.name}: hf_id is required for the chronos adapter")
        s = self.settings
        return self._pipeline_loader(spec.hf_id, s.hf_cache_dir, s.device, s.allow_downloads)

    def _chronos(self, spec: ModelSpec) -> Forecaster:
        pipeline = self.load_chronos_pipeline(spec)
        if spec.path is not None:
            path = self.checkpoint_path(spec)
            if not path.exists():
                raise ModelUnavailableError(f"{spec.name}: LoRA adapter {path} not trained yet")
            model = chronos.add_lora(pipeline.model, self.settings.training.lora)
            check_training_cutoff(spec, chronos.load_lora(path, model))
            model.eval()
        mode = spec.params.get("input_mode", "returns")
        if mode not in ("returns", "log_price"):
            raise ValueError(f"{spec.name}: unknown input_mode {mode!r}")
        return chronos.ChronosForecaster(spec.name, pipeline, int(spec.params["context_length"]), mode)

    def _kronos(self, spec: ModelSpec) -> Forecaster:
        if spec.hf_id is None or spec.tokenizer_id is None:
            raise ModelUnavailableError(f"{spec.name}: kronos needs hf_id and tokenizer_id")
        s = self.settings
        predictor = self._kronos_loader(
            spec.hf_id, spec.tokenizer_id, s.kronos, s.hf_cache_dir, s.device, s.allow_downloads
        )
        return kronos.KronosForecaster(
            spec.name, predictor, int(spec.params["context_length"]), s.kronos, s.seed
        )
