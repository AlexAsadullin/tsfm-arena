"""Build the configured data source."""

from __future__ import annotations

from typing import Literal

from tsfm_arena.data.base import DataSource
from tsfm_arena.data.pit import PitSource
from tsfm_arena.data.sources import InMemorySource, LocalFileSource
from tsfm_arena.data.synthetic import generate_cross_section
from tsfm_arena.settings import Settings


def build_source(
    settings: Settings, kind: Literal["synthetic", "local", "pit"] | None = None
) -> DataSource:
    """Return the source named by ``kind`` (default: ``settings.data.source``)."""
    kind = kind or settings.data.source
    if kind == "synthetic":
        cross_section = generate_cross_section(settings.data.synthetic)
        return InMemorySource(cross_section.panel, name="synthetic")
    if kind == "local":
        cfg = settings.data.local
        return LocalFileSource(
            settings.project_path(cfg.path), cfg.time_col, cfg.symbol_col, cfg.value_col, cfg.value_kind
        )
    return PitSource(settings.data.pit)
