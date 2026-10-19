from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from atlas.models import nowcast as nc
from atlas.pit.cube import VintageCube
from tests.conftest import make_obs_frame, utc


def test_mariano_murasawa_aggregation() -> None:
    idx = pd.period_range("2020-01", "2020-12", freq="M")
    cols = {i.name: np.full(len(idx), 1.0) for i in nc.INPUTS}
    q = nc._quarterly(pd.DataFrame(cols, index=idx))
    # constant 1% monthly growth -> 3% growth of the quarterly average, no acceleration
    assert q.loc[pd.Period("2020Q4", "Q"), "real_pce"] == pytest.approx(3.0)
    assert q.loc[pd.Period("2020Q4", "Q"), "inventory_accel"] == pytest.approx(0.0)
    assert np.isnan(q.loc[pd.Period("2020Q1", "Q"), "real_pce"])  # needs 5 months


def _gdp_cube() -> VintageCube:
    rows = [
        # 2018Q4 level known since January; 2019Q1 advance on 2019-04-26 12:30
        (nc.GDP_SERIES, date(2018, 10, 1), utc(2019, 1, 30), 100.0),
        (nc.GDP_SERIES, date(2019, 1, 1), utc(2019, 4, 26, 12, 30), 100.5),
        (nc.GDP_SERIES, date(2019, 1, 1), utc(2019, 5, 30, 12, 30), 100.6),  # 2nd estimate
        (nc.GDPNOW_SERIES, date(2019, 1, 1), utc(2019, 4, 18, 4), 2.4),
        (nc.GDPNOW_SERIES, date(2019, 1, 1), utc(2019, 4, 26, 4), 2.7),
        (nc.GDPNOW_SERIES, date(2019, 4, 1), utc(2019, 4, 30, 4), 1.3),
    ]
    return VintageCube(make_obs_frame(rows))


def test_advance_release_and_gdpnow_point_in_time() -> None:
    cube = _gdp_cube()
    q = pd.Period("2019Q1", "Q")
    rel = nc.advance_release(cube, q)
    assert rel is not None
    t, growth = rel
    assert t == utc(2019, 4, 26, 12, 30)
    assert growth == pytest.approx((1.005**4 - 1) * 100)  # advance, not the 2nd estimate
    assert nc.gdpnow_as_of(cube, q, t - timedelta(hours=1)) == 2.7
    assert nc.gdpnow_as_of(cube, q, utc(2019, 4, 20)) == 2.4
    assert nc.advance_release(cube, pd.Period("2019Q2", "Q")) is None
    now = nc.gdp_nowcast(cube, utc(2019, 5, 1))
    assert now is not None
    assert now["quarter"] == "2019Q2"
    assert now["gdp_saar_pct"] == 1.3
    assert nc.gdp_nowcast(cube, utc(2019, 1, 1)) is None


def _synthetic_cube(last_quarter_shock: float) -> VintageCube:
    """Every input and GDP load on one AR factor; the factor is shocked in 2019Q1."""
    rng = np.random.default_rng(3)
    months = pd.period_range("1993-01", "2019-03", freq="M")
    f = np.zeros(len(months))
    for t in range(1, len(f)):
        f[t] = 0.6 * f[t - 1] + rng.normal()
    f[-3:] += last_quarter_shock
    known = utc(2019, 4, 20)
    rows = []
    for k, inp in enumerate(nc.INPUTS):
        growth = 0.2 + 0.3 * f + 0.15 * rng.normal(size=len(f))
        level = (
            100 * np.exp(np.cumsum(growth) / 100)
            if inp.transform == "logdiff"
            else np.cumsum(growth)
        )
        n_obs = len(months) - (1 if k % 2 else 0)  # ragged edge: half lack the last month
        rows += [
            (inp.series_id, months[i].start_time.date(), known, float(level[i]))
            for i in range(n_obs)
        ]
    quarters = pd.period_range("1993Q1", "2018Q4", freq="Q")
    fq = pd.Series(f, index=months).groupby(months.asfreq("Q")).mean()
    gdp_growth = 0.5 + 0.6 * fq.reindex(quarters).to_numpy() + 0.1 * rng.normal(size=len(quarters))
    gdp_level = 100 * np.exp(np.cumsum(gdp_growth) / 100)
    rows += [
        (nc.GDP_SERIES, q.start_time.date(), known, float(v))
        for q, v in zip(quarters, gdp_level, strict=True)
    ]
    return VintageCube(make_obs_frame(rows))


def test_nowcast_follows_the_data() -> None:
    target = pd.Period("2019Q1", "Q")
    at = utc(2019, 4, 25)
    high, params = nc.nowcast(_synthetic_cube(+3.0), target, at)
    low, _ = nc.nowcast(_synthetic_cube(-3.0), target, at)
    assert high.months_observed["payrolls"] == 3
    assert high.months_observed["unemployment"] == 2  # ragged edge seen
    assert high.gdp_saar > low.gdp_saar + 2.0  # a boom quarter nowcasts well above a bust
    assert high.dfm_saar > low.dfm_saar
    assert params.size > 10
