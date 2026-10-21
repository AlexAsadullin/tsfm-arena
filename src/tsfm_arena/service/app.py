"""``forecast-service``: FastAPI app implementing contract 6.5 (``/v1/forecast``, ``/v1/rank``).

Every request reads data as known at ``known_at`` from the configured source. When cutoff enforcement is
on, a ``known_at`` inside a model's pretraining window (``<= cutoff + embargo``) is refused with 409,
because such a forecast may reflect memorised outcomes.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from tsfm_arena import __version__
from tsfm_arena.data.base import DataSource, LeakageRejectedError
from tsfm_arena.data.factory import build_source
from tsfm_arena.evaluation.arena import is_eligible
from tsfm_arena.models.base import CutoffViolationError, Forecast, Forecaster, ModelUnavailableError
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.models.spec import ModelSpec, Registry, load_registry
from tsfm_arena.settings import Settings
from tsfm_arena.timeutil import ensure_utc, to_ns


class ForecastRequest(BaseModel):
    schema_version: Literal["1"] = "1"
    model: str
    symbols: list[str] = Field(min_length=1)
    horizon: int = Field(ge=1)
    known_at: datetime
    quantiles: list[float] | None = None

    @field_validator("known_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class RankRequest(BaseModel):
    schema_version: Literal["1"] = "1"
    model: str
    symbols: list[str] = Field(min_length=1)
    horizon: int = Field(ge=1)
    known_at: datetime

    @field_validator("known_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class SymbolForecast(BaseModel):
    symbol: str
    horizon: int
    mean: float
    quantiles: dict[str, float]


class ForecastResponse(BaseModel):
    schema_version: Literal["1"] = "1"
    model: str
    known_at: datetime
    forecasts: list[SymbolForecast]
    model_cutoff: datetime | None


class RankEntry(BaseModel):
    symbol: str
    score: float


class RankResponse(BaseModel):
    schema_version: Literal["1"] = "1"
    model: str
    known_at: datetime
    ranks: list[RankEntry]
    model_cutoff: datetime | None


class ModelInfo(BaseModel):
    name: str
    adapter: str
    pretraining_cutoff: datetime | None
    enabled: bool


def quantile_key(level: float) -> str:
    """Wire key of a quantile level, e.g. ``0.1 -> "0.1"``."""
    return f"{level:g}"


class ForecastEngine:
    """Loads models lazily (LRU cache) and answers forecasts as known at a time."""

    def __init__(self, settings: Settings, registry: Registry, factory: ModelFactory, source: DataSource):
        self.settings = settings
        self.registry = registry
        self.factory = factory
        self.source = source
        self._cache: OrderedDict[str, Forecaster] = OrderedDict()

    def spec(self, name: str) -> ModelSpec:
        try:
            spec = self.registry.get(name)
        except KeyError as exc:
            raise HTTPException(404, f"unknown model {name!r}") from exc
        if not spec.enabled:
            raise HTTPException(404, f"model {name!r} is disabled")
        return spec

    def model(self, spec: ModelSpec) -> Forecaster:
        if spec.name in self._cache:
            self._cache.move_to_end(spec.name)
            return self._cache[spec.name]
        try:
            forecaster = self.factory.build(spec)
        except (ModelUnavailableError, CutoffViolationError) as exc:
            raise HTTPException(503, str(exc)) from exc
        self._cache[spec.name] = forecaster
        if len(self._cache) > self.settings.service.model_cache_size:
            self._cache.popitem(last=False)
        return forecaster

    def forecast(
        self, name: str, symbols: list[str], horizon: int, known_at: datetime, levels: list[float]
    ) -> tuple[ModelSpec, Forecast]:
        spec = self.spec(name)
        embargo = self.settings.evaluation.embargo
        if self.settings.service.enforce_cutoff and not is_eligible(
            to_ns(known_at), spec.pretraining_cutoff, embargo
        ):
            raise HTTPException(
                409,
                f"known_at {known_at.isoformat()} is inside the pretraining window of {name} "
                f"(cutoff {spec.pretraining_cutoff}, embargo {embargo})",
            )
        forecaster = self.model(spec)
        try:
            loaded = self.source.load(None, known_at, known_at, symbols)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except LeakageRejectedError as exc:
            raise HTTPException(409, str(exc)) from exc
        history = loaded.panel
        if history.n_periods < 2:
            raise HTTPException(422, f"no history known at {known_at.isoformat()}")
        try:
            if forecaster.needs_fit:
                forecaster.fit(history, horizon)
            context = history.returns[-forecaster.context_length :].T
            return spec, forecaster.predict(np.ascontiguousarray(context), horizon, levels)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


def create_app(
    settings: Settings,
    registry: Registry | None = None,
    factory: ModelFactory | None = None,
    source: DataSource | None = None,
) -> FastAPI:
    engine = ForecastEngine(
        settings,
        registry or load_registry(settings.models_path),
        factory or ModelFactory(settings),
        source or build_source(settings),
    )
    app = FastAPI(title="tsfm-arena forecast-service", version=__version__)
    default_levels = list(settings.evaluation.quantiles)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models")
    def models() -> list[ModelInfo]:
        return [
            ModelInfo(
                name=m.name, adapter=m.adapter, pretraining_cutoff=m.pretraining_cutoff, enabled=m.enabled
            )
            for m in engine.registry.models
        ]

    @app.post("/v1/forecast")
    def forecast(req: ForecastRequest) -> ForecastResponse:
        levels = req.quantiles or default_levels
        if any(not 0 < q < 1 for q in levels) or sorted(set(levels)) != levels:
            raise HTTPException(422, "quantiles must be strictly increasing levels in (0, 1)")
        spec, fc = engine.forecast(req.model, req.symbols, req.horizon, req.known_at, levels)
        items = [
            SymbolForecast(
                symbol=sym,
                horizon=req.horizon,
                mean=float(fc.mean[i]),
                quantiles={quantile_key(q): float(fc.quantiles[i, j]) for j, q in enumerate(levels)},
            )
            for i, sym in enumerate(req.symbols)
        ]
        return ForecastResponse(
            model=req.model, known_at=req.known_at, forecasts=items, model_cutoff=spec.pretraining_cutoff
        )

    @app.post("/v1/rank")
    def rank(req: RankRequest) -> RankResponse:
        spec, fc = engine.forecast(req.model, req.symbols, req.horizon, req.known_at, default_levels)
        order = np.argsort(-fc.mean, kind="stable")
        ranks = [RankEntry(symbol=req.symbols[i], score=float(fc.mean[i])) for i in order]
        return RankResponse(
            model=req.model, known_at=req.known_at, ranks=ranks, model_cutoff=spec.pretraining_cutoff
        )

    return app
