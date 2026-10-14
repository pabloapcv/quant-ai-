from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from atlas.backtest import metrics as m
from atlas.backtest.crosssection import backtest_signal
from atlas.backtest.cv import CombinatorialPurgedCV, PurgedKFold
from atlas.backtest.trials import TrialLog


# ------------------------------------------------------------------ metrics
def test_sharpe_and_drawdown() -> None:
    r = np.array([0.01, -0.02, 0.03, 0.01])
    assert m.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert math.isnan(m.sharpe([0.01]))
    assert math.isnan(m.sharpe([0.01, 0.01]))
    assert m.max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    assert m.max_drawdown([]) == 0.0
    assert m.annualize_sharpe(0.1, 12) == pytest.approx(0.1 * math.sqrt(12))
    assert m.sortino([0.02, -0.01, 0.03]) > 0
    assert math.isnan(m.sortino([0.01, 0.02]))


def test_newey_west_matches_t_for_iid() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1.0, 5000)
    plain_t = x.mean() / (x.std(ddof=0) / math.sqrt(x.size))
    assert m.newey_west_tstat(x, lags=0) == pytest.approx(plain_t, rel=1e-6)
    assert abs(m.newey_west_tstat(x) - plain_t) < 0.3
    assert math.isnan(m.newey_west_tstat([1.0, 2.0]))


def test_psr_dsr_and_expected_max() -> None:
    rng = np.random.default_rng(1)
    good = rng.normal(0.01, 0.04, 600)
    psr = m.probabilistic_sharpe_ratio(good)
    assert 0.9 < psr <= 1.0
    assert m.probabilistic_sharpe_ratio(good, sr_benchmark=1.0) < 0.01
    assert m.expected_max_sharpe(1, 0.01) == 0.0
    assert m.expected_max_sharpe(100, 0.01) > m.expected_max_sharpe(10, 0.01) > 0
    assert m.deflated_sharpe_ratio(good, 1) == pytest.approx(psr)
    assert m.deflated_sharpe_ratio(good, 1000, 0.01) < psr
    assert math.isnan(m.probabilistic_sharpe_ratio([0.1, 0.2]))


def test_pbo_detects_noise_vs_skill() -> None:
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.02, (480, 10))
    assert 0.25 < m.pbo_cscv(noise, n_splits=10) < 0.75  # pure noise: coin flip
    skilled = noise.copy()
    skilled[:, 3] += 0.01
    assert m.pbo_cscv(skilled, n_splits=10) < 0.1
    assert math.isnan(m.pbo_cscv(noise[:, :1]))
    assert 0.0 <= m.pbo_cscv(noise, n_splits=16, max_combinations=50) <= 1.0


# ------------------------------------------------------------------ cv
def _assert_purged(train: np.ndarray, test: np.ndarray, horizon: int, embargo: int) -> None:
    test_lo, test_hi = test.min(), test.max() + horizon
    for i in train:
        overlaps = i + horizon >= test.min() and i <= test_hi
        in_embargo = test_hi < i <= test_hi + embargo
        assert not overlaps or i > test_hi + embargo
        assert not in_embargo
    _ = test_lo


def test_purged_kfold() -> None:
    cv = PurgedKFold(n_splits=5, horizon=3, embargo=2)
    splits = list(cv.split(100))
    assert len(splits) == 5
    assert np.array_equal(np.sort(np.concatenate([t for _, t in splits])), np.arange(100))
    for train, test in splits:
        assert not set(train) & set(test)
        _assert_purged(train, test, 3, 2)
    with pytest.raises(ValueError, match="n_splits"):
        list(PurgedKFold(n_splits=1).split(10))


def test_combinatorial_purged_cv() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=1, embargo=1)
    assert cv.n_splits == 15
    assert cv.n_paths == 5
    splits = list(cv.split(120))
    assert len(splits) == 15
    for train, test in splits:
        assert not set(train) & set(test)
        assert len(train) + len(test) < 120  # something was purged
    with pytest.raises(ValueError, match="n_test_groups"):
        list(CombinatorialPurgedCV(n_groups=4, n_test_groups=4).split(40))


# ------------------------------------------------------------------ cross-section
def _panel(skill: float, n_dates: int = 120, n_assets: int = 20, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_dates):
        d = date(2000 + t // 12, t % 12 + 1, 1)
        signal = rng.normal(size=n_assets)
        fwd = skill * signal * 0.01 + rng.normal(0, 0.05, n_assets)
        rows += [
            {"date": d, "asset": f"A{i}", "signal": signal[i], "fwd_ret": fwd[i]}
            for i in range(n_assets)
        ]
    return pl.DataFrame(rows)


def test_backtest_signal_with_skill() -> None:
    rep = backtest_signal(_panel(skill=2.0), "skill", n_quantiles=5, cost_bps=0)
    s = rep.summary()
    assert s["ic_mean"] > 0.1
    assert s["ic_tstat_nw"] > 3
    assert s["monotonicity"] > 0.8
    assert s["sharpe_net"] == pytest.approx(s["sharpe_gross"])
    assert rep.ic_decay[1] > 0.1
    assert abs(rep.ic_decay[2]) < 0.1  # i.i.d. signal: no predictive power at lag 2
    assert set(rep.quantile_mean_annual) == {1, 2, 3, 4, 5}
    # full rebalance each month with fresh random signals: turnover ~ 2 (in + out, both legs)
    assert 1.0 < s["turnover_mean"] <= 4.0


def test_costs_and_null_signal() -> None:
    gross = backtest_signal(_panel(skill=0.0, seed=3), "noise", cost_bps=0)
    net = backtest_signal(_panel(skill=0.0, seed=3), "noise", cost_bps=50)
    assert net.sharpe_net < gross.sharpe_net
    assert abs(gross.ic_mean) < 0.03
    assert gross.dsr < 0.99


def test_backtest_requires_breadth_and_supports_regimes() -> None:
    p = _panel(skill=1.0, n_assets=4)
    with pytest.raises(ValueError, match="assets"):
        backtest_signal(p, "thin")
    p = _panel(skill=1.0)
    regimes = (
        p.select("date")
        .unique()
        .with_columns(
            pl.when(pl.col("date").dt.year() < 2005)
            .then(pl.lit("early"))
            .otherwise(pl.lit("late"))
            .alias("regime")
        )
    )
    rep = backtest_signal(p, "r", regimes=regimes)
    assert set(rep.regime_ic) == {"early", "late"}


# ------------------------------------------------------------------ trials
def test_trial_log(tmp_path: Path) -> None:
    log = TrialLog(tmp_path)
    assert log.stats().n_trials == 1
    assert log.stats().sr_variance is None
    log.record(
        family="f", name="a", universe="u", params={"x": 1}, sharpe_per_period=0.1, n_periods=10
    )
    log.record(
        family="f", name="a", universe="u", params={"x": 1}, sharpe_per_period=0.1, n_periods=10
    )
    log.record(
        family="f", name="a", universe="u", params={"x": 2}, sharpe_per_period=-0.1, n_periods=10
    )
    log.record(
        family="f", name="b", universe="v", params={}, sharpe_per_period=float("nan"), n_periods=10
    )
    st = log.stats(universe="u")
    assert st.n_trials == 2  # duplicates of the same config are one trial
    assert st.sr_variance == pytest.approx(0.02)
    assert log.stats().n_trials == 3
    assert len(log.entries()) == 4
