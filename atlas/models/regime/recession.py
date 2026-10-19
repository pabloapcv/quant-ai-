"""12-month-ahead recession probability: probit on curve, labor, financial conditions.

Label discipline: NBER dates arrive 6-21 months late, so when fitting as of month ``T`` we
only train on months ``m <= T - horizon - label_lag``. The label for ``m`` is 1 if any of
months ``m+1 .. m+horizon`` is an NBER recession month, using the NBER series as known at
the as-of date (never a later vintage).

Why the Sahm weight is negative (and is left that way by default): the target is "any NBER
recession month in the next 12 months", and the real-time Sahm rule peaks *after* recessions
end, because unemployment keeps rising past the trough (late 2009, mid 2020, 1983). A high
Sahm reading therefore often precedes a recession-free year. Paired with the 12-month
unemployment change (+), it separates an early downturn (both rising -> higher odds) from a
post-trough economy (Sahm high, unemployment no longer rising -> lower odds). Walk-forward
1995-2026: unconstrained AUC 0.90 / Brier 0.109 vs sign-constrained 0.82 / 0.160 (worse than
the 0.146 base rate). Known limitation: early in a labour-market turn the two weights partly
offset, so the model reacts slowly to the first rise in unemployment.

Optional sign discipline: given ``expected_signs`` the probit is refit without any input whose
coefficient comes out against its economic sign, dropping the most significant offender
first (with one offender this is the sign-constrained MLE, its coefficient pinned at zero).
``RegimeEngine(recession_sign_constraint=True)`` turns it on.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel
from scipy import stats


class RecessionResult(BaseModel):
    prob_12m: float
    n_obs: int
    n_positive: int
    train_end: date
    coefficients: dict[str, float]
    inputs: dict[str, float]
    dropped: list[str] = []  # inputs excluded because their fitted sign was wrong


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
    expected_signs: Mapping[str, int] | None = None,
) -> RecessionResult | None:
    """Fit ``features`` (event_time + columns) against ``recession`` (event_time, value).

    ``expected_signs`` maps a column to +1/-1 (the direction that raises recession odds);
    columns not in it are unrestricted.
    """
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

    signs = dict(expected_signs or {})
    active = list(cols)
    dropped: list[str] = []
    while True:
        fitted = _probit(train, active)
        if fitted is None:
            return None
        params, z = fitted
        wrong = [
            (-signs[c] * z[i + 1], c)
            for i, c in enumerate(active)
            if signs.get(c, 0) * params[i + 1] < 0
        ]
        if not wrong:
            break
        worst = max(wrong)[1]
        active.remove(worst)
        dropped.append(worst)
        if not active:
            return None

    x_now = np.concatenate([[1.0], now.select(active).to_numpy().astype(np.float64)[0]])
    prob = float(stats.norm.cdf(x_now @ params))
    return RecessionResult(
        prob_12m=prob,
        n_obs=train.height,
        n_positive=n_pos,
        train_end=train_end,
        coefficients=dict(zip(["const", *active], map(float, params), strict=True)),
        inputs={c: float(now[c].item()) for c in cols},
        dropped=dropped,
    )


def _probit(train: pl.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """(params, z-stats) with the constant first, or None if the fit fails."""
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
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.nan_to_num(params / np.asarray(fit.bse, dtype=np.float64), nan=0.0)
    return params, z
