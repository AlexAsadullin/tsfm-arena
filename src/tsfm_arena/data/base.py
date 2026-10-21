"""Data-source boundary: every source returns a :class:`ReturnPanel` as known at a given time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from tsfm_arena.data.panel import ReturnPanel


@dataclass(frozen=True)
class SourceResult:
    """A panel plus the fingerprint that identifies exactly which data produced it."""

    panel: ReturnPanel
    fingerprint: str | None


class LeakageRejectedError(RuntimeError):
    """A point-in-time source refused a query that would read data not yet known."""

    def __init__(self, message: str, event: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.event = event or {}


@runtime_checkable
class DataSource(Protocol):
    """A source of returns.

    ``revisable`` is True when the answer depends on ``known_at`` (a bitemporal store with revisions);
    the arena then re-queries the context at every origin with ``known_at = origin``. Revision-free
    sources (synthetic, frozen files) are loaded once and sliced.
    """

    name: str
    revisable: bool

    def load(
        self,
        start: datetime | None,
        end: datetime | None,
        known_at: datetime | None,
        symbols: Sequence[str] | None = None,
    ) -> SourceResult: ...
