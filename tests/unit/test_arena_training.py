from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from tests.conftest import tiny_chronos_model
from tsfm_arena.data.base import SourceResult
from tsfm_arena.data.panel import FloatArray, ReturnPanel
from tsfm_arena.data.sources import InMemorySource, slice_known
from tsfm_arena.evaluation.arena import Arena, is_eligible, rolling_origins
from tsfm_arena.models.base import Forecast
from tsfm_arena.models.chronos import add_lora
from tsfm_arena.models.spec import ModelSpec, TrainingSpec
from tsfm_arena.settings import Settings
from tsfm_arena.timeutil import from_ns, to_ns
from tsfm_arena.training.lora_tune import lora_rank_tune
from tsfm_arena.training.trainer import build_samples, train_patchtst


def test_eligibility_is_strict_after_cutoff_plus_embargo() -> None:
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)
    boundary = to_ns(cutoff + timedelta(days=7))
    assert not is_eligible(boundary, cutoff, timedelta(days=7))
    assert is_eligible(boundary + 1, cutoff, timedelta(days=7))
    assert is_eligible(0, None, timedelta(days=7))
    assert rolling_origins(10, 3, 2, 2).tolist() == [2, 4, 6]


class _Recorder:
    """Forecaster that remembers what it was shown, to prove the arena never leaks the future."""

    needs_fit = True
    context_length = 5

    def __init__(self) -> None:
        self.name = "recorder"
        self.contexts: list[FloatArray] = []
        self.fit_ends: list[int] = []

    def fit(self, history: ReturnPanel, horizon: int) -> None:
        self.fit_ends.append(int(history.ts[-1]))

    def predict(self, context: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast:
        self.contexts.append(context.copy())
        mean = context[:, -1]
        return Forecast(mean, np.repeat(mean[:, None], len(levels), axis=1), tuple(levels))


def test_arena_uses_only_known_data_and_respects_cutoff(settings: Settings, panel: ReturnPanel) -> None:
    as_of = from_ns(int(panel.ts[-1]))
    arena = Arena(settings.evaluation, InMemorySource(panel), as_of)
    cutoff = from_ns(int(panel.ts[100]))
    spec = ModelSpec(name="recorder", adapter="random_walk", pretraining_cutoff=cutoff)
    rec = _Recorder()
    run = arena.run(spec, rec)
    assert run.origin_ts.size == arena.origins.size  # contaminated origins are scored too
    first_ok = run.origin_ts[run.eligible][0]
    assert first_ok > to_ns(cutoff + settings.evaluation.embargo)
    assert not run.eligible[run.origin_ts <= to_ns(cutoff + settings.evaluation.embargo)].any()
    for k, t in enumerate(arena.origins):
        np.testing.assert_array_equal(rec.contexts[k][:, -1], panel.returns[t])  # last bar = origin bar
        np.testing.assert_array_equal(run.realised[k], panel.returns[t + 1])
    assert all(
        end <= ts
        for end, ts in zip(rec.fit_ends, run.origin_ts[:: settings.evaluation.refit_every], strict=False)
    )
    assert len(rec.fit_ends) == int(np.ceil(arena.origins.size / settings.evaluation.refit_every))
    scores = arena.score(run.subset(run.eligible))
    assert scores.n_origins == int(run.eligible.sum())
    assert scores.ic_mean.low is not None and scores.ic_mean.low <= scores.ic_mean.estimate
    assert arena.score(run.subset(np.zeros(run.origin_ts.size, dtype=bool))).n_origins == 0


def test_arena_without_contaminated_scoring(settings: Settings, panel: ReturnPanel) -> None:
    cfg = settings.evaluation.model_copy(update={"report_contaminated": False})
    arena = Arena(cfg, InMemorySource(panel), from_ns(int(panel.ts[-1])))
    spec = ModelSpec(name="r", adapter="random_walk", pretraining_cutoff=from_ns(int(panel.ts[150])))
    run = arena.run(spec, _Recorder())
    assert run.eligible.all() and run.origin_ts[0] > panel.ts[150]
    with pytest.raises(ValueError, match="no origins"):
        Arena(
            cfg.model_copy(update={"min_history": 10_000}), InMemorySource(panel), from_ns(int(panel.ts[-1]))
        )


class _RevisableSource:
    """A bitemporal fake: every call must carry known_at; later vintages revise the past."""

    name = "fake-pit"
    revisable = True

    def __init__(self, panel: ReturnPanel) -> None:
        self.panel = panel
        self.calls: list[datetime | None] = []

    def load(
        self, start: datetime | None, end: datetime | None, known_at: datetime | None,
        symbols: Sequence[str] | None = None,
    ) -> SourceResult:  # fmt: skip
        self.calls.append(known_at)
        return SourceResult(slice_known(self.panel, start, end, known_at, symbols), "sha256:fake")


def test_arena_requeries_revisable_sources_at_each_origin(settings: Settings, panel: ReturnPanel) -> None:
    src = _RevisableSource(panel)
    arena = Arena(settings.evaluation, src, from_ns(int(panel.ts[-1])))
    rec = _Recorder()
    arena.run(ModelSpec(name="r", adapter="random_walk", pretraining_cutoff=None), rec)
    assert len(src.calls) == 1 + arena.origins.size
    assert src.calls[1] == from_ns(int(panel.ts[arena.origins[0]]))
    assert arena.fingerprint == "sha256:fake"


def test_build_samples_never_reads_past_cutoff(panel: ReturnPanel) -> None:
    known = panel.upto(int(panel.ts[99]))
    s = build_samples(known, 16, 2)
    assert s.contexts.shape == (100 - 16 - 2 + 1, panel.n_assets, 16)
    assert s.target_end_ts == known.ts[-1]
    np.testing.assert_array_equal(s.contexts[0][:, -1], panel.returns[15])
    np.testing.assert_allclose(s.targets[0], panel.returns[16] + panel.returns[17])
    with pytest.raises(ValueError, match="too short"):
        build_samples(known.rows(0, 10), 16, 1)


@pytest.mark.parametrize("rank", ["none", "pairwise", "spearman"])
def test_train_patchtst_learns_and_records_cutoff(settings: Settings, panel: ReturnPanel, rank: str) -> None:
    cutoff = from_ns(int(panel.ts[120]))
    spec = TrainingSpec(point_loss="mse", rank_loss=rank, rank_weight=1.0, epochs=4)  # type: ignore[arg-type]
    res = train_patchtst(panel, cutoff, spec, settings.training, 1, settings.evaluation.quantiles, 3)
    assert res.epoch_losses[-1] < res.epoch_losses[0]
    assert datetime.fromisoformat(res.meta["train_end"].replace("Z", "+00:00")) <= cutoff
    cont = train_patchtst(
        panel, cutoff, spec, settings.training, 1, settings.evaluation.quantiles, 3, res.net
    )
    assert cont.net is res.net
    with pytest.raises(ValueError, match="different horizon"):
        train_patchtst(panel, cutoff, spec, settings.training, 2, settings.evaluation.quantiles, 3, res.net)


def test_lora_rank_tune_updates_only_adapters(settings: Settings, panel: ReturnPanel) -> None:
    model = add_lora(tiny_chronos_model(), settings.training.lora)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    spec = TrainingSpec(rank_loss="listnet", rank_weight=1.0, quantile_weight=0.2, learning_rate=0.01)
    meta = lora_rank_tune(model, panel, from_ns(int(panel.ts[120])), spec, settings.training, 1, 32, 1)
    assert meta["n_dates"] > 0 and len(meta["epoch_losses"]) == settings.training.lora.epochs
    for n, p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n])
    assert any(not torch.equal(p, lora_before[n]) for n, p in model.named_parameters() if n in lora_before)
    with pytest.raises(ValueError, match="rank loss"):
        lora_rank_tune(model, panel, from_ns(int(panel.ts[120])), TrainingSpec(), settings.training, 1, 32, 1)
    with pytest.raises(ValueError, match="trainable"):
        lora_rank_tune(tiny_chronos_model().requires_grad_(False), panel, from_ns(int(panel.ts[120])),
                       spec, settings.training, 1, 32, 1)  # fmt: skip
