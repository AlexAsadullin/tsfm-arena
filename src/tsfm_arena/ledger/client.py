"""Client for the Research Ledger (S6), contract 6.1: trial capture and verdicts.

Trials go to ``POST /v1/trials/batch``. When the ledger is unreachable (mode ``auto``) or disabled (mode
``inbox``), they are appended as JSON lines to
``<data root>/research-ledger/inbox/<source>-<yyyymmdd>.jsonl``, which the ledger ingests idempotently by
``trial_id``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from tsfm_arena.logging import get_logger
from tsfm_arena.timeutil import utc_now

log = get_logger(__name__)


class Actor(BaseModel):
    kind: Literal["human", "agent", "robot"]
    id: str


class ReturnsSeries(BaseModel):
    ts: list[str]
    values: list[float]


class TrialRecord(BaseModel):
    schema_version: Literal["1"] = "1"
    trial_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    family: str
    strategy_id: str
    source: str
    actor: Actor
    params: dict[str, Any]
    dataset_fingerprint: str | None
    code_version: str | None
    seed: int | None
    started_at: datetime
    finished_at: datetime
    periods_per_year: float
    returns: ReturnsSeries | None = None
    returns_path: str | None = None
    metrics: dict[str, Any]
    tags: list[str]


class Verdict(BaseModel):
    schema_version: Literal["1"] = "1"
    strategy_id: str
    n_trials: int
    effective_trials: float
    sharpe: float | None
    psr: float | None
    dsr: float | None
    pbo: float | None = None
    min_trl: float | None
    passed: bool
    reasons: list[str]
    thresholds: dict[str, Any]
    leakage_flags: list[Any]
    computed_at: datetime


class LedgerUnavailableError(RuntimeError):
    """The ledger could not be reached and the mode forbids the inbox fallback."""


class LedgerClient:
    def __init__(
        self,
        base_url: str,
        mode: Literal["auto", "http", "inbox"],
        inbox_dir: Path,
        source: str,
        timeout_s: float,
        client: httpx.Client | None = None,
    ) -> None:
        self.mode = mode
        self.inbox_dir = inbox_dir
        self.source = source
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def log_trials(self, records: list[TrialRecord]) -> Literal["http", "inbox"]:
        """Send ``records``; return where they went."""
        if not records:
            return "inbox" if self.mode == "inbox" else "http"
        if self.mode != "inbox":
            try:
                payload = [r.model_dump(mode="json") for r in records]
                response = self._client.post("/v1/trials/batch", json=payload)
                response.raise_for_status()
                return "http"
            except httpx.HTTPError as exc:
                if self.mode == "http":
                    raise LedgerUnavailableError(f"ledger rejected or unreachable: {exc}") from exc
                log.warning("ledger_unreachable_using_inbox", error=str(exc))
        self._append_inbox(records)
        return "inbox"

    def inbox_path(self) -> Path:
        return self.inbox_dir / f"{self.source}-{utc_now():%Y%m%d}.jsonl"

    def _append_inbox(self, records: list[TrialRecord]) -> None:
        path = self.inbox_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")

    def verdict(self, strategy_id: str) -> Verdict | None:
        """The ledger's verdict, or None when the ledger is unreachable, disabled or has none."""
        if self.mode == "inbox":
            return None
        try:
            response = self._client.get(f"/v1/strategies/{strategy_id}/verdict")
        except httpx.HTTPError:
            return None
        if response.status_code != httpx.codes.OK:
            return None
        return Verdict.model_validate(response.json())
