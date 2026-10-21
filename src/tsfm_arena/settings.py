"""Typed settings: defaults from ``config/settings.yaml``, overridden by ``TSFM_*`` env variables.

The data root is the platform-wide ``TRADEQ_DATA_DIR``. The settings file location can be changed with
``TSFM_SETTINGS_FILE``; by default it is ``config/settings.yaml`` at the repository root.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

SETTINGS_FILE_ENV = "TSFM_SETTINGS_FILE"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def default_settings_file() -> Path:
    """Return the settings file named by ``TSFM_SETTINGS_FILE`` or the repository default."""
    override = os.environ.get(SETTINGS_FILE_ENV)
    return Path(override) if override else _REPO_ROOT / "config" / "settings.yaml"


class SyntheticSettings(BaseModel):
    """Synthetic cross-section with a planted, weak, decaying momentum signal."""

    n_assets: int = Field(gt=1)
    n_periods: int = Field(gt=10)
    start: datetime
    period: timedelta
    seed: int
    vol_low: float = Field(gt=0)
    vol_high: float = Field(gt=0)
    market_vol: float = Field(ge=0)
    beta_std: float = Field(ge=0)
    tail_df: float = Field(gt=2)
    momentum_window: int = Field(ge=1)
    signal_ic: float = Field(ge=0, lt=1)
    signal_halflife: float = Field(gt=0)
    symbol_prefix: str
    symbol_suffix: str


class LocalSettings(BaseModel):
    path: Path
    time_col: str
    symbol_col: str
    value_col: str
    value_kind: Literal["return", "price"]


class PitSettings(BaseModel):
    base_url: str
    dataset: str
    timeout_s: float = Field(gt=0)
    format: Literal["arrow", "json"]
    mode: Literal["research", "live"]
    time_col: str
    symbol_col: str
    value_col: str
    value_kind: Literal["return", "price"]
    history_start: datetime


class DataSettings(BaseModel):
    source: Literal["synthetic", "local", "pit"]
    symbols: list[str]
    synthetic: SyntheticSettings
    local: LocalSettings
    pit: PitSettings


class LongShortSettings(BaseModel):
    fraction: float = Field(gt=0, le=0.5)
    gross: float = Field(gt=0)
    cost_bps: float = Field(ge=0)


class BootstrapSettings(BaseModel):
    n_resamples: int = Field(ge=10)
    mean_block: float = Field(ge=1)
    confidence: float = Field(gt=0, lt=1)
    seed: int


class EvaluationSettings(BaseModel):
    horizon: int = Field(ge=1)
    step: int = Field(ge=1)
    context_length: int = Field(ge=2)
    min_history: int = Field(ge=2)
    embargo: timedelta
    refit_every: int = Field(ge=1)
    quantiles: list[float]
    periods_per_year: float = Field(gt=0)
    report_contaminated: bool
    collapse_threshold: float = Field(ge=0)
    long_short: LongShortSettings
    bootstrap: BootstrapSettings

    @field_validator("quantiles")
    @classmethod
    def _check_quantiles(cls, value: list[float]) -> list[float]:
        if not value or any(not 0.0 < q < 1.0 for q in value) or sorted(set(value)) != value:
            raise ValueError("quantiles must be strictly increasing levels in (0, 1)")
        return value


class LedgerSettings(BaseModel):
    base_url: str
    mode: Literal["auto", "http", "inbox"]
    timeout_s: float = Field(gt=0)
    inbox_subdir: str
    source: str
    family_prefix: str
    actor_kind: Literal["human", "agent", "robot"]
    actor_id: str


class ServiceSettings(BaseModel):
    host: str
    port: int = Field(gt=0, lt=65536)
    enforce_cutoff: bool
    model_cache_size: int = Field(ge=1)


class PatchTSTSettings(BaseModel):
    context_length: int = Field(ge=2)
    patch_len: int = Field(ge=1)
    stride: int = Field(ge=1)
    d_model: int = Field(ge=2)
    n_heads: int = Field(ge=1)
    n_layers: int = Field(ge=1)
    d_ff: int = Field(ge=1)
    dropout: float = Field(ge=0, lt=1)


class LoraSettings(BaseModel):
    r: int = Field(ge=1)
    alpha: int = Field(ge=1)
    dropout: float = Field(ge=0, lt=1)
    target_modules: list[str]
    epochs: int = Field(ge=1)
    dates_per_batch: int = Field(ge=1)
    learning_rate: float = Field(gt=0)


class TrainingSettings(BaseModel):
    patchtst: PatchTSTSettings
    epochs: int = Field(ge=1)
    dates_per_batch: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    rank_temperature: float = Field(gt=0)
    lora: LoraSettings


class KronosSettings(BaseModel):
    repo_path: Path | None
    bar: timedelta
    max_context: int = Field(ge=2)
    n_paths: int = Field(ge=2)
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)


class ReproduceSettings(BaseModel):
    source: Literal["synthetic", "local", "pit"]
    run_lora_when_downloads_allowed: bool


class ReportSettings(BaseModel):
    seeds: list[int] = Field(min_length=1)
    rank_losses: list[Literal["pairwise", "listnet", "spearman"]]
    base_model: str
    rank_model: str
    baselines: list[str]
    tsfm_models: list[str]


class Settings(BaseSettings):
    """All tsfm-arena settings. Construct with :func:`load_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="TSFM_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    data_root: Path = Field(validation_alias=AliasChoices("data_root", "TRADEQ_DATA_DIR"))
    project_subdir: str
    models_file: Path
    allow_downloads: bool
    hf_cache_subdir: str
    device: str
    seed: int
    log_level: str
    log_json: bool
    data: DataSettings
    evaluation: EvaluationSettings
    ledger: LedgerSettings
    service: ServiceSettings
    training: TrainingSettings
    kronos: KronosSettings
    reproduce: ReproduceSettings
    report: ReportSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=default_settings_file())
        return init_settings, env_settings, dotenv_settings, yaml_source

    @property
    def project_dir(self) -> Path:
        """This project's subtree of the data root."""
        return self.data_root / self.project_subdir

    @property
    def hf_cache_dir(self) -> Path:
        return self.data_root / self.hf_cache_subdir

    @property
    def ledger_inbox_dir(self) -> Path:
        return self.data_root / self.ledger.inbox_subdir

    @property
    def models_path(self) -> Path:
        path = self.models_file
        return path if path.is_absolute() else default_settings_file().parent / path

    def project_path(self, relative: Path | str) -> Path:
        """Resolve a path relative to the project data subtree (absolute paths pass through)."""
        path = Path(relative)
        return path if path.is_absolute() else self.project_dir / path


def load_settings(**overrides: object) -> Settings:
    """Load settings from YAML + env (+ ``.env``); keyword overrides win over everything."""
    return Settings(**overrides)  # type: ignore[arg-type]
