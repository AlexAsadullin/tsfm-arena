from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from tsfm_arena.evaluation import metrics
from tsfm_arena.evaluation.bootstrap import bootstrap_ci, stationary_indices


def test_crps_matches_gaussian_closed_form() -> None:
    # Closed form for N(mu, s^2): s * (z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)), Gneiting & Raftery 2007.
    levels = np.linspace(0.0025, 0.9975, 400)
    mu, s = 0.01, 0.02
    y = np.array([-0.03, 0.0, 0.01, 0.05])
    q = np.tile(mu + s * norm.ppf(levels), (y.size, 1))
    z = (y - mu) / s
    exact = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert metrics.crps_from_quantiles(q, y, levels) == pytest.approx(exact.mean(), rel=0.01)
    assert np.isnan(metrics.crps_from_quantiles(q[:1] * np.nan, y[:1], levels))


def test_pinball_and_mae() -> None:
    assert metrics.pinball(np.array([1.0]), np.array([3.0]), 0.9)[0] == pytest.approx(1.8)
    assert metrics.pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.1)
    assert metrics.mae(np.array([1.0, np.nan]), np.array([2.0, 5.0])) == 1.0
    assert np.isnan(metrics.mae(np.array([np.nan]), np.array([1.0])))


def test_spearman_ties_nan_and_degenerate() -> None:
    assert metrics.spearman(np.array([1.0, 2.0, 3.0, np.nan]), np.array([3.0, 2.0, 1.0, 9.0])) == pytest.approx(-1)
    assert metrics.spearman(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(0.8660254)
    assert np.isnan(metrics.spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    assert np.isnan(metrics.spearman(np.zeros(4), np.arange(4.0)))


def test_ic_summary() -> None:
    s = metrics.ic_summary(np.array([0.1, 0.3, np.nan, -0.1]))
    assert s.n == 3 and s.mean == pytest.approx(0.1)
    assert s.icir == pytest.approx(0.1 / 0.2)
    assert s.t_stat == pytest.approx(0.5 * np.sqrt(3))
    assert s.hit_rate == pytest.approx(2 / 3)
    assert np.isnan(metrics.ic_summary(np.array([0.1])).mean)


@given(st.lists(st.floats(-1, 1), min_size=2, max_size=40), st.floats(0.05, 0.5), st.floats(0.5, 3))
@settings(max_examples=50, deadline=None)
def test_long_short_weights_are_dollar_neutral(scores: list[float], fraction: float, gross: float) -> None:
    w = metrics.long_short_weights(np.array(scores), fraction, gross)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(w).sum() == pytest.approx(gross)


def test_long_short_weights_pick_extremes_and_skip_nan() -> None:
    w = metrics.long_short_weights(np.array([0.3, np.nan, -0.2, 0.1, 0.0]), 0.25, 2.0)
    assert w.tolist() == [1.0, 0.0, -1.0, 0.0, 0.0]
    assert not metrics.long_short_weights(np.array([np.nan, 1.0]), 0.5, 2.0).any()


def test_long_short_returns_with_costs() -> None:
    w = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    r = np.log1p(np.array([[0.01, -0.01], [0.02, 0.0], [0.0, 0.0]]))
    res = metrics.long_short_returns(w, r, cost_bps=10)
    np.testing.assert_allclose(res.turnover, [2.0, 0.0, 4.0])
    np.testing.assert_allclose(res.gross_returns, [0.02, 0.02, 0.0])
    np.testing.assert_allclose(res.net_returns, [0.018, 0.02, -0.004])


def test_sharpe_psr_closed_form() -> None:
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.004, -0.01, 0.015])
    sr = r.mean() / r.std(ddof=1)
    assert metrics.sharpe(r, 365) == pytest.approx(sr * np.sqrt(365))
    assert metrics.per_period_sharpe(r) == pytest.approx(sr)
    from scipy.stats import kurtosis, skew

    g3, g4 = skew(r), kurtosis(r, fisher=False)
    expected = norm.cdf(sr * np.sqrt(r.size - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    assert metrics.probabilistic_sharpe(r, 0.0) == pytest.approx(expected)
    assert np.isnan(metrics.sharpe(np.ones(5), 365))
    assert np.isnan(metrics.probabilistic_sharpe(np.ones(5), 0.0))
    assert np.isnan(metrics.per_period_sharpe(np.array([1.0])))


def test_expected_max_sharpe_matches_monte_carlo(rng: np.random.Generator) -> None:
    # E[max of N standard normals] for N = 100 is about 2.508; the Bailey-Lopez de Prado approximation is close.
    sims = rng.standard_normal((20_000, 100)).max(axis=1).mean()
    assert metrics.expected_max_sharpe(100, 1.0) == pytest.approx(sims, abs=0.05)
    assert metrics.expected_max_sharpe(1, 1.0) == 0.0
    assert metrics.expected_max_sharpe(10, 0.0) == 0.0


def test_dsr_is_psr_deflated(rng: np.random.Generator) -> None:
    r = rng.normal(0.001, 0.01, 500)
    psr = metrics.probabilistic_sharpe(r, 0.0)
    assert metrics.deflated_sharpe(r, 1, 0.01) == pytest.approx(psr)
    assert metrics.deflated_sharpe(r, 50, 0.01) < metrics.deflated_sharpe(r, 5, 0.01) < psr


def test_forecast_dispersion() -> None:
    realised = np.array([[1.0, -1.0, 0.0], [2.0, 0.0, -2.0]])
    d = metrics.forecast_dispersion(realised * 0.05, realised, threshold=0.1)
    assert d.ratio == pytest.approx(0.05) and d.collapse_rate == 1.0
    assert np.isnan(metrics.forecast_dispersion(realised, np.zeros_like(realised), 0.1).ratio)


def test_stationary_bootstrap(rng: np.random.Generator) -> None:
    idx = stationary_indices(50, 5.0, 20, rng)
    assert idx.shape == (20, 50) and idx.min() >= 0 and idx.max() < 50
    runs = (np.diff(idx, axis=1) % 50 == 1).mean()
    assert 0.7 < runs < 0.9  # continuation probability is 1 - 1/5
    x = rng.normal(0.5, 1.0, 400)
    ci = bootstrap_ci(x, lambda v: float(v.mean()), 5, 300, 0.9, 1)
    assert ci.low < ci.estimate < ci.high
    # i.i.d. data: the 90% interval width is close to 2 * 1.645 * sd / sqrt(n) = 0.082.
    assert ci.high - ci.low == pytest.approx(2 * 1.645 / np.sqrt(400), rel=0.25)
    short = bootstrap_ci(np.array([1.0, np.nan]), lambda v: float(v.mean()), 5, 10, 0.9, 1)
    assert short.estimate == 1.0 and np.isnan(short.low)
    nan_stat = bootstrap_ci(x, lambda v: float("nan"), 5, 10, 0.9, 1)
    assert np.isnan(nan_stat.high)
