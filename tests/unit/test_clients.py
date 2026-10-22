"""Contract tests for the I2 (6.3) and S6 (6.1) clients with mocked HTTP."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import respx

from tsfm_arena.data.base import LeakageRejectedError
from tsfm_arena.data.pit import ARROW_STREAM, FINGERPRINT_HEADER, PitError, PitSource
from tsfm_arena.ledger.client import Actor, LedgerClient, LedgerUnavailableError, TrialRecord, Verdict
from tsfm_arena.settings import Settings

PIT = "http://pit.test"
LEDGER = "http://ledger.test"
KNOWN = datetime(2024, 1, 4, tzinfo=UTC)
ROWS = {
    "valid_time": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"] * 2,
    "symbol": ["A"] * 3 + ["B"] * 3,
    "close": [10.0, 11.0, 12.0, 5.0, 5.0, 4.0],
}


def _pit(settings: Settings, fmt: str) -> PitSource:
    cfg = settings.data.pit.model_copy(update={"base_url": PIT, "format": fmt})
    return PitSource(cfg, httpx.Client(base_url=PIT))


def _arrow_bytes() -> bytes:
    table = pa.table(ROWS)
    sink = io.BytesIO()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


@respx.mock
def test_pit_arrow_query_sends_known_at_and_returns_fingerprint(settings: Settings) -> None:
    route = respx.post(f"{PIT}/v1/query").mock(
        return_value=httpx.Response(
            200,
            content=_arrow_bytes(),
            headers={"content-type": ARROW_STREAM, FINGERPRINT_HEADER: "sha256:ab"},
        )
    )
    out = _pit(settings, "arrow").load(None, KNOWN, KNOWN, ["B", "A"])
    body = json.loads(route.calls[0].request.content)
    assert body["known_at"] == "2024-01-04T00:00:00Z" and body["mode"] == "research"
    assert body["schema_version"] == "1" and body["filters"] == {"symbol": ["B", "A"]}
    assert out.fingerprint == "sha256:ab"
    assert out.panel.symbols == ("B", "A") and out.panel.n_periods == 2


@respx.mock
def test_pit_json_query(settings: Settings) -> None:
    rows = [dict(zip(ROWS, vals, strict=True)) for vals in zip(*ROWS.values(), strict=True)]
    route = respx.post(f"{PIT}/v1/query").mock(return_value=httpx.Response(200, json={"rows": rows}))
    out = _pit(settings, "json").load(None, KNOWN, KNOWN)
    assert route.calls[0].request.url.params["format"] == "json"
    assert out.panel.symbols == ("A", "B")


@respx.mock
def test_pit_leakage_and_errors(settings: Settings) -> None:
    event = {"source": "i2", "severity": "block", "kind": "end_after_known_at"}
    respx.post(f"{PIT}/v1/query").mock(return_value=httpx.Response(409, json=event))
    with pytest.raises(LeakageRejectedError) as info:
        _pit(settings, "json").load(None, KNOWN, KNOWN)
    assert info.value.event["severity"] == "block"
    respx.post(f"{PIT}/v1/query").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(PitError, match="500"):
        _pit(settings, "json").load(None, KNOWN, KNOWN)
    respx.post(f"{PIT}/v1/query").mock(return_value=httpx.Response(200, json={"nothing": 1}))
    with pytest.raises(PitError, match="row list"):
        _pit(settings, "json").load(None, KNOWN, KNOWN)
    respx.post(f"{PIT}/v1/query").mock(return_value=httpx.Response(409, text="not json"))
    with pytest.raises(LeakageRejectedError):
        _pit(settings, "json").load(None, KNOWN, KNOWN)
    with pytest.raises(ValueError, match="explicit"):
        _pit(settings, "json").load(None, None, KNOWN)


def _record() -> TrialRecord:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return TrialRecord(
        family="tsfm-arena/rank-ic/test",
        strategy_id="tsfm-arena:m:abc",
        source="s1",
        actor=Actor(kind="robot", id="tsfm-arena"),
        params={},
        dataset_fingerprint=None,
        code_version=None,
        seed=1,
        started_at=now,
        finished_at=now,
        periods_per_year=365,
        metrics={"rank_ic": 0.1},
        tags=[],
    )


def _ledger(tmp_path: Path, mode: str) -> LedgerClient:
    return LedgerClient(LEDGER, mode, tmp_path, "s1", 1.0, httpx.Client(base_url=LEDGER))  # type: ignore[arg-type]


@respx.mock
def test_ledger_http_then_inbox_fallback(tmp_path: Path) -> None:
    route = respx.post(f"{LEDGER}/v1/trials/batch").mock(return_value=httpx.Response(200, json=[]))
    assert _ledger(tmp_path, "auto").log_trials([_record()]) == "http"
    sent = json.loads(route.calls[0].request.content)[0]
    assert sent["schema_version"] == "1" and sent["actor"]["kind"] == "robot"
    respx.post(f"{LEDGER}/v1/trials/batch").mock(side_effect=httpx.ConnectError("down"))
    client = _ledger(tmp_path, "auto")
    assert client.log_trials([_record(), _record()]) == "inbox"
    lines = client.inbox_path().read_text().splitlines()
    assert client.inbox_path().name.startswith("s1-") and len(lines) == 2
    assert TrialRecord.model_validate_json(lines[0]).family == "tsfm-arena/rank-ic/test"
    with pytest.raises(LedgerUnavailableError):
        _ledger(tmp_path, "http").log_trials([_record()])
    assert _ledger(tmp_path, "inbox").log_trials([]) == "inbox"


@respx.mock
def test_ledger_verdict(tmp_path: Path) -> None:
    verdict = {
        "schema_version": "1", "strategy_id": "x", "n_trials": 3, "effective_trials": 2.5, "sharpe": 1.0,
        "psr": 0.9, "dsr": 0.7, "pbo": None, "min_trl": 100, "passed": False, "reasons": ["dsr below"],
        "thresholds": {"min_dsr": 0.95}, "leakage_flags": [], "computed_at": "2024-01-01T00:00:00Z",
    }  # fmt: skip
    respx.get(f"{LEDGER}/v1/strategies/x/verdict").mock(return_value=httpx.Response(200, json=verdict))
    respx.get(f"{LEDGER}/v1/strategies/y/verdict").mock(return_value=httpx.Response(404))
    respx.get(f"{LEDGER}/v1/strategies/z/verdict").mock(side_effect=httpx.ConnectError("down"))
    client = _ledger(tmp_path, "auto")
    got = client.verdict("x")
    assert isinstance(got, Verdict) and got.dsr == 0.7
    assert client.verdict("y") is None and client.verdict("z") is None
    assert _ledger(tmp_path, "inbox").verdict("x") is None
