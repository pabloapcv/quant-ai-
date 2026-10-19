from datetime import date

import numpy as np
import polars as pl
import pytest

from atlas.quality.drift import backtest_ic_stats, feature_psi, live_ic_drift, psi


def test_psi_null_and_shift() -> None:
    rng = np.random.default_rng(0)
    null = [psi(rng.normal(size=120), rng.normal(size=36)) for _ in range(300)]
    shifted = [psi(rng.normal(size=120), rng.normal(size=36) + 1.5) for _ in range(300)]
    assert np.median(null) < 0.2
    assert np.median(shifted) > 1.0
    assert np.isnan(psi(np.ones(3), np.ones(3)))  # too few observations
    assert psi(np.arange(100.0), np.arange(100.0)) == pytest.approx(0.0, abs=1e-12)


def test_feature_psi_windows() -> None:
    months = pl.date_range(date(2000, 1, 1), date(2014, 12, 1), "1mo", eager=True)
    n = len(months)
    stable = np.random.default_rng(1).normal(size=n)
    jump = stable.copy()
    jump[-36:] += 3.0
    m = pl.DataFrame({"event_time": months, "stable": stable, "jump": jump, "empty": [None] * n})
    out = feature_psi(m, date(2014, 12, 1))
    assert set(out) == {"stable", "jump"}
    assert out["jump"] > 1.0 > out["stable"]


def test_backtest_ic_and_live_drift() -> None:
    dates = [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)]
    fc = pl.DataFrame(
        {
            "date": [d for d in dates for _ in range(3)],
            "score": [1.0, 2.0, 3.0] * 3,
            "excess": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3],
        }
    )
    st = backtest_ic_stats(fc)
    assert st["n"] == 3
    assert st["mean"] == pytest.approx(1 / 3)
    bt = {"mean": 0.05, "sd": 0.2, "n": 600}
    assert live_ic_drift([0.0] * 5, bt) is None  # not enough live months yet
    assert live_ic_drift([0.0] * 6, bt) is None  # within 2 standard errors (floor -0.113)
    hit = live_ic_drift([-0.2] * 6, bt)
    assert hit is not None
    assert hit["floor"] == pytest.approx(0.05 - 2 * 0.2 / np.sqrt(6))
