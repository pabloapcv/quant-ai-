from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from atlas.signals.library import (
    lottery,
    low_beta,
    macro_tilt,
    near_high,
    neg_skew,
    seasonality,
)


def _panel(n_months: int = 72, n_assets: int = 6, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    months = [date(2010 + t // 12, t % 12 + 1, 1) for t in range(n_months)]
    rows = []
    for a in range(n_assets):
        for d in months:
            rows.append({"date": d, "asset": f"A{a}", "ret": float(rng.normal(0.01, 0.03))})
    return pl.DataFrame(rows)


def test_seasonality_looks_at_the_coming_calendar_month() -> None:
    p = _panel()
    # A0 earns +20% every January; nothing else is seasonal.
    p = p.with_columns(
        pl.when((pl.col("asset") == "A0") & (pl.col("date").dt.month() == 1))
        .then(0.20)
        .otherwise(pl.col("ret"))
        .alias("ret")
    )
    out = seasonality(5).compute(p)
    dec = out.filter(pl.col("date") == date(2015, 12, 1)).sort("signal", descending=True)
    assert dec["asset"][0] == "A0"  # December's signal predicts January
    assert dec["signal"][0] == pytest.approx(0.20)
    jun = out.filter((pl.col("date") == date(2015, 6, 1)) & (pl.col("asset") == "A0"))
    assert jun["signal"][0] < 0.1


def test_low_beta_and_lottery_and_skew_signs() -> None:
    p = _panel(n_months=96)
    mkt = p.group_by("date").agg(pl.col("ret").mean().alias("m"))
    # A0 is a levered version of the market: high beta, so the lowest low-beta score.
    p = (
        p.join(mkt, on="date")
        .with_columns(
            pl.when(pl.col("asset") == "A0")
            .then(3 * pl.col("m"))
            .otherwise(pl.col("ret"))
            .alias("ret")
        )
        .drop("m")
    )
    lb = low_beta(60).compute(p).filter(pl.col("date") == date(2017, 12, 1)).sort("signal")
    assert lb["asset"][0] == "A0"
    p2 = p.with_columns(
        pl.when((pl.col("asset") == "A1") & (pl.col("date") == date(2017, 6, 1)))
        .then(0.5)
        .otherwise(pl.col("ret"))
        .alias("ret")
    )
    lot = lottery(12).compute(p2).filter(pl.col("date") == date(2017, 12, 1)).sort("signal")
    assert lot["asset"][0] == "A1"  # had a lottery-like month: lowest score
    sk = neg_skew(60).compute(p2).filter(pl.col("date") == date(2017, 12, 1)).sort("signal")
    assert sk["asset"][0] == "A1"  # right-skewed: lowest negative-skew score


def test_near_high_is_zero_at_a_new_high() -> None:
    p = _panel().with_columns(
        pl.when(pl.col("asset") == "A0").then(0.02).otherwise(pl.col("ret")).alias("ret")
    )
    out = near_high(12).compute(p).filter(pl.col("asset") == "A0").drop_nulls("signal")
    assert out["signal"].abs().max() == pytest.approx(0.0)  # always at its high
    worst = near_high(12).compute(p).drop_nulls("signal")["signal"].to_numpy().max()
    assert worst <= 1e-12  # never above the high


def test_macro_tilt_uses_beta_times_recent_move() -> None:
    p = _panel(n_months=96)
    dates = p.select("date").unique().sort("date")
    rng = np.random.default_rng(5)
    f = dates.with_columns(pl.Series("x_doil", rng.normal(0, 0.05, dates.height)))
    p = p.join(f, on="date").with_columns(
        pl.when(pl.col("asset") == "A0")
        .then(pl.col("ret") + 0.5 * pl.col("x_doil"))  # A0 gains when oil rises
        .otherwise(pl.col("ret"))
        .alias("ret")
    )
    out = macro_tilt("x_doil", 60, 3).compute(p)
    last = out.filter(pl.col("date") == date(2017, 12, 1))
    move = float(f.filter(pl.col("date") <= date(2017, 12, 1)).tail(3)["x_doil"].to_numpy().sum())
    a0 = float(last.filter(pl.col("asset") == "A0")["signal"][0])
    assert np.sign(a0) == np.sign(move)  # positive exposure x recent oil move
    missing = macro_tilt("x_dy10").compute(p.drop("x_doil"))
    assert missing["signal"].null_count() == missing.height  # no factor column: no signal
