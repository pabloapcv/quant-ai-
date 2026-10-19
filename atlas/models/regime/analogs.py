"""Historical analogs (PRD §7.6): past months whose macro state looks most like today.

State vector per month: the growth / inflation / financial-conditions composites plus their
3-month changes (direction matters: a slowdown that is deepening is not one that is turning).
Distance is Mahalanobis on that vector, using the covariance of the whole history, so
correlated pillars are not double counted.

Selection rules:

* Months within ``exclude_recent`` of the current month are skipped; they are neighbours in
  time, not analogs.
* Analogs are de-clustered: a candidate within ``min_gap`` months of a better match is
  skipped, so one episode cannot fill all ``k`` slots.

Forward returns use months ``d+1 .. d+h`` after analog month ``d``, as excess over the
equal-weight average of the universe. Composites come from the vintage known at the as-of
date (later revisions of the analog months included). This is a descriptive lookup, not a
backtested signal.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl
from pydantic import BaseModel

STATE_COLUMNS = ("growth", "inflation", "financial_conditions")


class Analog(BaseModel):
    month: date
    distance: float
    state: dict[str, float]


class AnalogResult(BaseModel):
    current_month: date
    current_state: dict[str, float]
    analogs: list[Analog]
    n_candidates: int


def state_matrix(comps: pl.DataFrame, columns: Sequence[str] = STATE_COLUMNS) -> pl.DataFrame:
    """``event_time`` + levels + 3-month changes, rows with any gap dropped."""
    cols = list(columns)
    return (
        comps.sort("event_time")
        .select("event_time", *cols)
        .with_columns((pl.col(c) - pl.col(c).shift(3)).alias(f"{c}_chg3m") for c in cols)
        .drop_nulls()
    )


def find_analogs(
    comps: pl.DataFrame,
    *,
    current_month: date | None = None,
    k: int = 10,
    min_gap: int = 6,
    exclude_recent: int = 24,
    columns: Sequence[str] = STATE_COLUMNS,
) -> AnalogResult | None:
    """Nearest past months to the latest complete state at or before ``current_month``."""
    states = state_matrix(comps, columns)
    if current_month is not None:
        states = states.filter(pl.col("event_time") <= current_month)
    if states.height < 2:
        return None
    now = states.tail(1)
    current_month = now["event_time"].item()

    names = [c for c in states.columns if c != "event_time"]
    x = states.select(names).to_numpy().astype(np.float64)
    x0 = now.select(names).to_numpy().astype(np.float64)[0]
    inv = np.linalg.pinv(np.cov(x, rowvar=False))
    diff = x - x0
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff).clip(min=0.0))

    months = states["event_time"].to_list()
    idx = [_month_index(m) for m in months]
    cur = _month_index(current_month)
    candidates = [i for i in range(len(months)) if idx[i] <= cur - exclude_recent]
    chosen: list[int] = []
    for i in sorted(candidates, key=lambda j: dist[j]):
        if all(abs(idx[i] - idx[j]) >= min_gap for j in chosen):
            chosen.append(i)
            if len(chosen) == k:
                break

    level = list(columns)
    return AnalogResult(
        current_month=current_month,
        current_state={c: float(now[c].item()) for c in level},
        analogs=[
            Analog(
                month=months[i],
                distance=float(dist[i]),
                state={c: float(states[c][i]) for c in level},
            )
            for i in chosen
        ],
        n_candidates=len(candidates),
    )


def forward_excess(panel: pl.DataFrame, months: Sequence[date], horizon: int) -> pl.DataFrame:
    """Compounded return over ``d+1 .. d+horizon`` minus the equal-weight average.

    ``panel`` has ``date`` (month start), ``asset``, ``ret`` (that month's return). Returns
    ``month``, ``asset``, ``excess``; an asset needs every month of the window to count.
    """
    rets = panel.select(pl.col("date").dt.truncate("1mo"), "asset", "ret").drop_nulls()
    windows = (
        pl.DataFrame({"month": list(months)}, schema={"month": pl.Date})
        .with_columns(pl.int_ranges(1, horizon + 1).alias("step"))
        .explode("step")
        .with_columns(pl.col("month").dt.offset_by(pl.format("{}mo", pl.col("step"))).alias("date"))
    )
    cum = (
        windows.join(rets, on="date")
        .group_by("month", "asset")
        .agg(pl.len().alias("n"), ((pl.col("ret") + 1).product() - 1).alias("cum"))
        .filter(pl.col("n") == horizon)
    )
    return (
        cum.with_columns((pl.col("cum") - pl.col("cum").mean().over("month")).alias("excess"))
        .select("month", "asset", "excess")
        .sort("month", "asset")
    )


def summarize_forward(excess: pl.DataFrame) -> pl.DataFrame:
    """Per asset across analogs: mean / median excess, share positive, analogs counted."""
    return (
        excess.group_by("asset")
        .agg(
            pl.col("excess").mean().alias("mean"),
            pl.col("excess").median().alias("median"),
            (pl.col("excess") > 0).mean().alias("hit_rate"),
            pl.len().alias("n"),
        )
        .sort("mean", descending=True)
    )


def _month_index(d: date) -> int:
    return d.year * 12 + d.month - 1
