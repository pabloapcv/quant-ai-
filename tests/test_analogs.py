from datetime import date

import numpy as np
import polars as pl
import pytest

from atlas.models.regime.analogs import find_analogs, forward_excess, summarize_forward


def _comps(n: int = 240, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    months = pl.date_range(
        date(2000, 1, 1), pl.date(2000, 1, 1).dt.offset_by(f"{n - 1}mo"), "1mo", eager=True
    )
    walk = np.cumsum(rng.normal(0, 0.3, size=(n, 3)), axis=0)
    return pl.DataFrame(
        {
            "event_time": months,
            "growth": walk[:, 0],
            "inflation": walk[:, 1],
            "financial_conditions": walk[:, 2],
        }
    )


def test_planted_analog_is_found_first() -> None:
    comps = _comps()
    # Make month 60 (and its 3m-earlier month) replicate the last four months exactly.
    cols = ["growth", "inflation", "financial_conditions"]
    tail = comps.tail(4).select(cols).to_numpy()
    arr = comps.select(cols).to_numpy().copy()
    arr[57:61] = tail
    comps = comps.with_columns(pl.Series(c, arr[:, i]) for i, c in enumerate(cols))
    res = find_analogs(comps, k=5)
    assert res is not None
    assert res.analogs[0].month == comps["event_time"][60]
    assert res.analogs[0].distance == pytest.approx(0.0, abs=1e-9)


def test_exclusion_and_declustering() -> None:
    comps = _comps()
    res = find_analogs(comps, k=8, min_gap=6, exclude_recent=24)
    assert res is not None
    cur = res.current_month
    idx = [a.month.year * 12 + a.month.month for a in res.analogs]
    assert all(cur.year * 12 + cur.month - i >= 24 for i in idx)
    assert all(abs(a - b) >= 6 for i, a in enumerate(idx) for b in idx[i + 1 :])
    assert [a.distance for a in res.analogs] == sorted(a.distance for a in res.analogs)


def test_current_month_is_point_in_time() -> None:
    comps = _comps()
    res = find_analogs(comps, current_month=date(2010, 6, 15))
    assert res is not None
    assert res.current_month == date(2010, 6, 1)
    assert all(a.month <= date(2008, 6, 1) for a in res.analogs)


def test_forward_excess_compounds_next_months() -> None:
    months = pl.date_range(date(2020, 1, 1), date(2020, 12, 1), "1mo", eager=True)
    panel = pl.concat(
        [
            pl.DataFrame({"date": months, "asset": "A", "ret": [0.10] * 12}),
            pl.DataFrame({"date": months, "asset": "B", "ret": [0.0] * 12}),
        ]
    )
    ex = forward_excess(panel, [date(2020, 1, 1)], 2)
    a = ex.filter(pl.col("asset") == "A")["excess"].item()
    assert a == pytest.approx((1.1**2 - 1) / 2)  # A minus the A/B average
    # Window running past the data is dropped, not truncated.
    assert forward_excess(panel, [date(2020, 11, 1)], 3).is_empty()
    summ = summarize_forward(ex)
    assert summ["asset"].to_list() == ["A", "B"]
    assert summ.filter(pl.col("asset") == "A")["hit_rate"].item() == 1.0
