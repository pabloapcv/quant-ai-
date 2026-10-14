"""12-month-ahead recession probability: probit on curve, labor, financial conditions.

Label discipline: NBER dates arrive 6-21 months late, so when fitting as of month ``T`` we
only train on months ``m <= T - horizon - label_lag``. The label for ``m`` is 1 if any of
months ``m+1 .. m+horizon`` is an NBER recession month, using the NBER series as known at
the as-of date (never a later vintage).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel


class RecessionResult(BaseModel):
    prob_12m: float
    n_obs: int
    n_positive: int
    train_end: date
    coefficients: dict[str, float]
    inputs: dict[str, float]


def forward_label(recession: pl.Expr, horizon: int) -> pl.Expr:
    """1 if any of the next ``horizon`` periods is a recession (null where unknown)."""
    ahead = recession.shift(-1)
    return ahead.reverse().rolling_max(window_size=horizon, min_samples=horizon).reverse()


def fit_recession_probit(
    features: pl.DataFrame,
    recession: pl.DataFrame,
    columns: Sequence[str],
    *,
    current_month: date,
    horizon: int = 12,
    label_lag: int = 12,
    min_positive: int = 6,
) -> RecessionResult | None:
    """Fit ``features`` (event_time + columns) against ``recession`` (event_time, value)."""
    cols = list(columns)
    rec = recession.select("event_time", pl.col("value").alias("rec"))
    df = (
        features.select("event_time", *cols)
        .join(rec, on="event_time", how="left")
        .sort("event_time")
        .with_columns(forward_label(pl.col("rec"), horizon).alias("y"))
    )
    train_end = pl.select(pl.lit(current_month).dt.offset_by(f"-{horizon + label_lag}mo")).item()
    train = df.filter(pl.col("event_time") <= train_end).drop_nulls([*cols, "y"])
    n_pos = int(train["y"].sum()) if train.height else 0
    if n_pos < min_positive or train.height - n_pos < min_positive:
        return None

    now = df.filter(pl.col("event_time") == current_month)
    if now.is_empty() or any(now[c].item() is None for c in cols):
        return None

    x = sm.add_constant(train.select(cols).to_numpy().astype(np.float64), has_constant="add")
    y = train["y"].to_numpy().astype(np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Probit(y, x).fit(disp=0, maxiter=200)
    except (np.linalg.LinAlgError, ValueError):
        return None
    params = np.asarray(fit.params, dtype=np.float64)
    if not np.all(np.isfinite(params)):
        return None

    x_now = np.concatenate([[1.0], now.select(cols).to_numpy().astype(np.float64)[0]])
    prob = float(fit.model.cdf(x_now @ params))
    return RecessionResult(
        prob_12m=prob,
        n_obs=train.height,
        n_positive=n_pos,
        train_end=train_end,
        coefficients=dict(zip(["const", *cols], map(float, params), strict=True)),
        inputs={c: float(now[c].item()) for c in cols},
    )
