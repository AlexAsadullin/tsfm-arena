"""Chronos-2 / Chronos-Bolt adapter (``chronos-forecasting``), plus LoRA adapters for Chronos-Bolt."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import Tensor, nn

from tsfm_arena.data.panel import FloatArray, ReturnPanel
from tsfm_arena.models.base import Forecast, ModelUnavailableError
from tsfm_arena.settings import LoraSettings

InputMode = Literal["returns", "log_price"]


class QuantilePipeline(Protocol):
    """The part of a Chronos pipeline the adapter uses."""

    def predict_quantiles(
        self, inputs: Any, prediction_length: int, quantile_levels: list[float]
    ) -> tuple[Any, Any]: ...


def _stack(values: Any) -> Tensor:
    """Chronos-Bolt returns one tensor; Chronos-2 returns a list with one ``(1, H, ...)`` tensor per
    univariate input. Normalise both to ``(N, H, ...)``."""
    if isinstance(values, list):
        return torch.cat([v.reshape(-1, *v.shape[-(v.ndim - 1) :]) for v in values])
    return torch.as_tensor(values)


class ChronosForecaster:
    """Zero-shot (or LoRA-tuned) Chronos forecaster.

    In ``returns`` mode the model sees the return series and forecasts the next ``horizon`` returns; the
    cumulative quantiles are exact for ``horizon = 1`` only, so other horizons are refused. In
    ``log_price`` mode the model sees the cumulative log price (rebased to 0 at the origin) and the
    forecast of the level ``horizon`` bars ahead gives exact cumulative-return quantiles for any horizon.
    """

    needs_fit = False

    def __init__(
        self, name: str, pipeline: QuantilePipeline, context_length: int, input_mode: InputMode
    ) -> None:
        self.name = name
        self.pipeline = pipeline
        self.context_length = context_length
        self.input_mode = input_mode

    def fit(self, history: ReturnPanel, horizon: int) -> None:
        """Pretrained; nothing to fit at evaluation time."""

    def predict(self, context: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast:
        if self.input_mode == "returns" and horizon != 1:
            raise ValueError("returns input mode gives exact cumulative quantiles only for horizon 1")
        window = context[:, -self.context_length :]
        if self.input_mode == "log_price":
            level = np.nancumsum(window, axis=1)
            series = np.where(np.isnan(window), np.nan, level - level[:, -1:])
        else:
            series = window
        inputs = [torch.as_tensor(row[~np.isnan(row)], dtype=torch.float32) for row in series]
        inputs = [x if x.numel() > 0 else torch.zeros(1) for x in inputs]
        quant, mean = self.pipeline.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=list(levels)
        )
        q = _stack(quant).double().numpy()  # (N, H, Q)
        m = _stack(mean).double().numpy()  # (N, H)
        return Forecast(m[:, horizon - 1].copy(), q[:, horizon - 1, :].copy(), tuple(levels))


def load_pipeline(hf_id: str, cache_dir: Path, device: str, allow_downloads: bool) -> Any:
    """Load a Chronos pipeline from the HF cache under the data root; download only when allowed."""
    from chronos import BaseChronosPipeline

    try:
        return BaseChronosPipeline.from_pretrained(
            hf_id, cache_dir=str(cache_dir), device_map=device, local_files_only=not allow_downloads
        )
    except OSError as exc:
        hint = "" if allow_downloads else " (set TSFM_ALLOW_DOWNLOADS=true to download it)"
        raise ModelUnavailableError(f"Chronos weights {hf_id!r} not available{hint}: {exc}") from exc


def add_lora(model: Any, cfg: LoraSettings) -> Any:
    """Freeze ``model`` and inject LoRA matrices (Hu et al. 2021) into the T5 encoder and decoder.

    The adapters go into the two T5 stacks rather than the top-level Chronos-Bolt module, because peft's
    tied-embedding check calls ``get_input_embeddings``, which Chronos-Bolt does not implement.
    """
    model.requires_grad_(False)
    lora = LoraConfig(
        r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, target_modules=list(cfg.target_modules)
    )
    for stack in (model.encoder, model.decoder):
        inject_adapter_in_model(lora, stack)
    return model


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def save_lora(path: Path, model: nn.Module, meta: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora_state_dict(model), "meta": meta}, path)
    return path


def load_lora(path: Path, model: nn.Module) -> dict[str, Any]:
    """Load saved LoRA weights into a model that already has adapters injected; return the metadata."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    missing = set(lora_state_dict(model)) - set(blob["lora"])
    if missing:
        raise ValueError(f"LoRA checkpoint lacks {len(missing)} adapter tensors, e.g. {sorted(missing)[0]}")
    model.load_state_dict(blob["lora"], strict=False)
    meta: dict[str, Any] = dict(blob["meta"])
    return meta
