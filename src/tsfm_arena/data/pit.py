"""Client for the Point-in-time Store (I2), contract 6.3: ``POST /v1/query`` as known at T."""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

import httpx
import pandas as pd
import pyarrow.ipc as pa_ipc
from pydantic import BaseModel

from tsfm_arena.data.base import LeakageRejectedError, SourceResult
from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.settings import PitSettings

ARROW_STREAM = "application/vnd.apache.arrow.stream"
FINGERPRINT_HEADER = "X-Dataset-Fingerprint"


class PitQuery(BaseModel):
    """Request body of ``POST /v1/query`` (contract 6.3)."""

    schema_version: Literal["1"] = "1"
    dataset: str
    start: str
    end: str
    known_at: str
    columns: list[str] | None = None
    filters: dict[str, Any] | None = None
    mode: Literal["research", "live"] = "research"


class PitError(RuntimeError):
    """The store answered with an unexpected status or payload."""


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class PitSource:
    """A :class:`~tsfm_arena.data.base.DataSource` backed by I2.

    Symbols are sent as ``filters = {<symbol_col>: [..]}``. The store's 409 leakage refusal is raised as
    :class:`LeakageRejectedError` carrying the ``LeakageEvent`` body, so callers can never silently fall
    back to leaking data.
    """

    revisable = True

    def __init__(self, cfg: PitSettings, client: httpx.Client | None = None) -> None:
        self.name = f"pit:{cfg.dataset}"
        self._cfg = cfg
        self._client = client or httpx.Client(base_url=cfg.base_url, timeout=cfg.timeout_s)
        self.query_count = 0

    def load(
        self,
        start: datetime | None,
        end: datetime | None,
        known_at: datetime | None,
        symbols: Sequence[str] | None = None,
    ) -> SourceResult:
        if end is None or known_at is None:
            raise ValueError("a point-in-time query needs explicit end and known_at")
        cfg = self._cfg
        query = PitQuery(
            dataset=cfg.dataset,
            start=_iso(start or cfg.history_start),
            end=_iso(end),
            known_at=_iso(known_at),
            columns=[cfg.time_col, cfg.symbol_col, cfg.value_col],
            filters={cfg.symbol_col: list(symbols)} if symbols else None,
            mode=cfg.mode,
        )
        params = {"format": "json"} if cfg.format == "json" else None
        response = self._client.post("/v1/query", json=query.model_dump(), params=params)
        self.query_count += 1
        if response.status_code == httpx.codes.CONFLICT:
            raise LeakageRejectedError(f"I2 rejected query as leaking: {response.text}", _json_or_empty(response))
        if response.status_code != httpx.codes.OK:
            raise PitError(f"I2 query failed with {response.status_code}: {response.text[:200]}")
        frame = self._decode(response)
        panel = ReturnPanel.from_long(frame, cfg.time_col, cfg.symbol_col, cfg.value_col, cfg.value_kind)
        if symbols:
            panel = panel.select(list(symbols))
        return SourceResult(panel, response.headers.get(FINGERPRINT_HEADER))

    @staticmethod
    def _decode(response: httpx.Response) -> pd.DataFrame:
        if response.headers.get("content-type", "").startswith(ARROW_STREAM):
            reader = pa_ipc.open_stream(io.BytesIO(response.content))
            return reader.read_all().to_pandas()
        payload = response.json()
        rows = payload.get("rows", payload.get("data")) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise PitError("JSON response has no row list ('rows' or 'data')")
        return pd.DataFrame(rows)


def _json_or_empty(response: httpx.Response) -> dict[str, object]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
