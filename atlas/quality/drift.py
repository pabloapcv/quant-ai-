"""Drift monitoring (PRD §12): feature distribution shift and live-vs-backtest skill.

* **Feature PSI.** Population stability index of each regime feature over the last
  ``current_months`` against the ``reference_months`` before them, on reference-quintile bins.
  A diagnostic, not an alert: the conventional 0.25 "significant shift" level was built for
  large samples of roughly independent observations. Here the current window is three years
  of monthly, highly persistent z-scores, so (measured 1990-2026) almost every feature is
  above 0.25 in almost every quarter, and even i.i.d. data has a 95th percentile near 0.34
  at n = 36. Use it to see which inputs sit furthest from their recent history.
* **Live IC.** Mean rank IC of the last ``window`` scored live months against the walk-forward
  backtest's mean IC. Under the backtest's distribution the live mean has standard error
  ``sd / sqrt(n)``; it is flagged when it falls more than ``sigmas`` of those below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl
from numpy.typing import NDArray

PSI_CONVENTIONAL = 0.25  # large-sample rule of thumb; see the module docstring


def psi(reference: NDArray[np.float64], current: NDArray[np.float64], bins: int = 5) -> float:
    """PSI of ``current`` vs ``reference`` on reference-quantile bins (NaNs dropped).

    Bin shares get add-half (Jeffreys) smoothing: with a few dozen monthly observations some
    bins are empty by chance, and clipping them to a tiny epsilon would make PSI explode.
    """
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if ref.size < 2 * bins or cur.size < bins:
        return math.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    k = edges.size + 1
    e = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=k) + 0.5
    a = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=k) + 0.5
    e_share, a_share = e / e.sum(), a / a.sum()
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def feature_psi(
    matrix: pl.DataFrame,
    end_month: date,
    *,
    current_months: int = 36,
    reference_months: int = 120,
) -> dict[str, float]:
    """PSI per feature column of ``matrix`` (``event_time`` + features), finite values only."""
    rows = matrix.filter(pl.col("event_time") <= end_month).sort("event_time")
    cur = rows.tail(current_months)
    ref = rows.head(max(rows.height - current_months, 0)).tail(reference_months)
    out = {}
    for col in rows.columns:
        if col == "event_time":
            continue
        value = psi(ref[col].cast(pl.Float64).to_numpy(), cur[col].cast(pl.Float64).to_numpy())
        if math.isfinite(value):
            out[col] = value
    return out


def backtest_ic_stats(forecasts: pl.DataFrame, start: date | None = None) -> dict[str, float]:
    """Mean / s.d. / count of monthly rank IC (score vs realized excess) in walk-forward output."""
    fc = forecasts.drop_nulls(["score", "excess"])
    if start is not None:
        fc = fc.filter(pl.col("date") >= start)
    ic = (
        fc.group_by("date")
        .agg(pl.corr("score", "excess", method="spearman").alias("ic"))["ic"]
        .drop_nulls()
        .drop_nans()
        .to_numpy()
    )
    if ic.size < 2:
        return {}
    return {"mean": float(ic.mean()), "sd": float(ic.std(ddof=1)), "n": int(ic.size)}


def live_ic_drift(
    live_ics: Sequence[float],
    backtest: dict[str, float],
    *,
    window: int = 6,
    sigmas: float = 2.0,
) -> dict[str, float] | None:
    """Stats for the last ``window`` live ICs if their mean is too far below the backtest."""
    recent = [x for x in live_ics if math.isfinite(x)][-window:]
    if len(recent) < window or not backtest.get("sd"):
        return None
    live = float(np.mean(recent))
    floor = backtest["mean"] - sigmas * backtest["sd"] / math.sqrt(len(recent))
    if live >= floor:
        return None
    return {"live_mean": live, "backtest_mean": backtest["mean"], "floor": floor, "n": len(recent)}
