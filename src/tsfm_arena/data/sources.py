"""Revision-free sources: an in-memory panel (synthetic) and local Parquet/CSV files."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from tsfm_arena.data.base import SourceResult
from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.timeutil import to_ns


def slice_known(
    panel: ReturnPanel,
    start: datetime | None,
    end: datetime | None,
    known_at: datetime | None,
    symbols: Sequence[str] | None,
) -> ReturnPanel:
    """Slice a revision-free panel: a bar is known once it has closed, so ``ts <= known_at``."""
    end_ns = None if end is None else to_ns(end)
    if known_at is not None:
        known_ns = to_ns(known_at)
        end_ns = known_ns if end_ns is None else min(end_ns, known_ns)
    out = panel.between(None if start is None else to_ns(start), end_ns)
    return out.select(symbols) if symbols else out


class InMemorySource:
    """Wraps a fixed panel, e.g. the synthetic cross-section."""

    revisable = False

    def __init__(self, panel: ReturnPanel, name: str = "memory") -> None:
        self.name = name
        self._panel = panel
        self._fingerprint = panel.fingerprint()

    def load(
        self,
        start: datetime | None,
        end: datetime | None,
        known_at: datetime | None,
        symbols: Sequence[str] | None = None,
    ) -> SourceResult:
        return SourceResult(slice_known(self._panel, start, end, known_at, symbols), self._fingerprint)


class LocalFileSource:
    """Long-format Parquet or CSV file with a time, a symbol and a value column."""

    revisable = False

    def __init__(
        self,
        path: Path,
        time_col: str,
        symbol_col: str,
        value_col: str,
        value_kind: Literal["return", "price"],
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"local data file not found: {path}")
        self.name = f"local:{path.name}"
        self._path = path
        frame = self._read(path)
        self._panel = ReturnPanel.from_long(frame, time_col, symbol_col, value_col, value_kind)
        self._fingerprint = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"unsupported file type {suffix!r}; use .parquet or .csv")

    def load(
        self,
        start: datetime | None,
        end: datetime | None,
        known_at: datetime | None,
        symbols: Sequence[str] | None = None,
    ) -> SourceResult:
        return SourceResult(slice_known(self._panel, start, end, known_at, symbols), self._fingerprint)


def write_long(panel: ReturnPanel, path: Path) -> Path:
    """Write ``panel`` as a long Parquet/CSV file (columns ``ts``, ``symbol``, ``ret``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = panel.to_long()
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path
