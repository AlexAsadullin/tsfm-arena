"""Model registry entries (``config/models.yaml``)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from tsfm_arena.timeutil import ensure_utc

AdapterName = Literal["random_walk", "historical_mean", "ar_ridge", "patchtst", "chronos", "kronos"]


class TrainingSpec(BaseModel):
    """How an in-house model (or a LoRA adapter) is trained; unset values fall back to settings."""

    init_from: str | None = None
    point_loss: Literal["none", "mse"] = "none"
    rank_loss: Literal["none", "pairwise", "listnet", "spearman"] = "none"
    quantile_weight: float = Field(default=1.0, ge=0)
    rank_weight: float = Field(default=0.0, ge=0)
    epochs: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)


class ModelSpec(BaseModel):
    name: str
    adapter: AdapterName
    hf_id: str | None = None
    tokenizer_id: str | None = None
    path: Path | None = None
    pretraining_cutoff: datetime | None
    requires_download: bool = False
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    training: TrainingSpec | None = None

    @field_validator("pretraining_cutoff")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    def config_hash(self, extra: dict[str, Any] | None = None) -> str:
        """Short stable hash of this spec plus evaluation settings: one (model, config) = one trial."""
        payload = {"spec": self.model_dump(mode="json"), "extra": extra or {}}
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


class Registry(BaseModel):
    models: list[ModelSpec]

    @field_validator("models")
    @classmethod
    def _unique(cls, value: list[ModelSpec]) -> list[ModelSpec]:
        names = [m.name for m in value]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        return value

    def get(self, name: str) -> ModelSpec:
        for spec in self.models:
            if spec.name == name:
                return spec
        raise KeyError(f"unknown model {name!r}")

    def enabled(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]

    def training_order(self, names: list[str]) -> list[ModelSpec]:
        """Specs with a training section among ``names``, parents (``init_from``) before children."""
        ordered: list[ModelSpec] = []
        seen: set[str] = set()

        def visit(name: str, stack: tuple[str, ...]) -> None:
            if name in stack:
                raise ValueError(f"init_from cycle: {' -> '.join((*stack, name))}")
            spec = self.get(name)
            if spec.training is None or name in seen:
                return
            if spec.training.init_from:
                visit(spec.training.init_from, (*stack, name))
            seen.add(name)
            ordered.append(spec)

        for name in names:
            visit(name, ())
        return ordered


def load_registry(path: Path) -> Registry:
    with path.open() as handle:
        return Registry.model_validate(yaml.safe_load(handle))
