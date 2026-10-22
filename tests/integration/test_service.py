"""Contract 6.5 tests for the forecast service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.data.sources import InMemorySource
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.models.patchtst import save_checkpoint
from tsfm_arena.models.spec import TrainingSpec, load_registry
from tsfm_arena.service.app import create_app, quantile_key
from tsfm_arena.settings import Settings
from tsfm_arena.timeutil import from_ns, iso
from tsfm_arena.training.trainer import train_patchtst


@pytest.fixture
def client(settings: Settings, factory: ModelFactory, panel: ReturnPanel) -> TestClient:
    return TestClient(create_app(settings, factory=factory, source=InMemorySource(panel)))


def test_health_and_models(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    models = {m["name"]: m for m in client.get("/v1/models").json()}
    assert models["patchtst-rank"]["pretraining_cutoff"].startswith("2024-04-15")


def test_forecast_and_rank(client: TestClient, panel: ReturnPanel) -> None:
    syms = list(panel.symbols[:4])
    known = iso(int(panel.ts[150]))
    body = {
        "model": "ar-ridge",
        "symbols": syms,
        "horizon": 1,
        "known_at": known,
        "quantiles": [0.1, 0.5, 0.9],
    }
    res = client.post("/v1/forecast", json=body)
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["schema_version"] == "1" and payload["model_cutoff"] is None
    first = payload["forecasts"][0]
    assert first["symbol"] == syms[0] and set(first["quantiles"]) == {"0.1", "0.5", "0.9"}
    assert first["quantiles"]["0.1"] <= first["quantiles"]["0.9"]
    ranks = client.post(
        "/v1/rank", json={"model": "xs-momentum", "symbols": syms, "horizon": 1, "known_at": known}
    )
    scores = [r["score"] for r in ranks.json()["ranks"]]
    assert scores == sorted(scores, reverse=True) and len(scores) == 4
    default = client.post("/v1/forecast", json=body | {"quantiles": None}).json()
    assert len(default["forecasts"][0]["quantiles"]) == 9
    assert quantile_key(0.25) == "0.25"


def test_forecast_errors(
    client: TestClient, settings: Settings, factory: ModelFactory, panel: ReturnPanel
) -> None:
    syms = list(panel.symbols[:2])
    late, early = iso(int(panel.ts[200])), iso(int(panel.ts[60]))
    base = {"symbols": syms, "horizon": 1, "known_at": late}
    assert client.post("/v1/forecast", json=base | {"model": "nope"}).status_code == 404
    assert client.post("/v1/forecast", json=base | {"model": "kronos-small"}).status_code == 404
    assert client.post("/v1/forecast", json=base | {"model": "patchtst-rank"}).status_code == 503
    leak = client.post("/v1/forecast", json=base | {"model": "chronos-tiny", "known_at": early})
    assert leak.status_code == 409 and "pretraining window" in leak.json()["detail"]
    assert (
        client.post("/v1/forecast", json=base | {"model": "ar-ridge", "symbols": ["ZZZ"]}).status_code == 404
    )
    bad_q = client.post("/v1/forecast", json=base | {"model": "ar-ridge", "quantiles": [0.9, 0.1]})
    assert bad_q.status_code == 422
    too_early = base | {"model": "ar-ridge", "known_at": iso(int(panel.ts[0]))}
    assert client.post("/v1/forecast", json=too_early).status_code == 422
    spec = load_registry(settings.models_path).get("patchtst-finance")
    cutoff = spec.pretraining_cutoff
    assert cutoff is not None
    trained = train_patchtst(panel, cutoff, TrainingSpec(point_loss="mse", epochs=1), settings.training, 1,
                             settings.evaluation.quantiles, 1)  # fmt: skip
    save_checkpoint(factory.checkpoint_path(spec), trained.net, trained.meta)
    ok = client.post("/v1/forecast", json=base | {"model": "patchtst-finance"})
    assert ok.status_code == 200
    wrong_h = client.post("/v1/forecast", json=base | {"model": "patchtst-finance", "horizon": 3})
    assert wrong_h.status_code == 422
    assert from_ns(int(panel.ts[200])) > cutoff
