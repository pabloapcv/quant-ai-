"""Synthetic first-release rows for history that predates a source's vintage coverage.

Without them an observation from 1960 first archived in 1990 (or a dataset that is only
ever published as a current snapshot) would be "unknown" for decades, making long
backtests impossible. Instead we add a row at ``period_end + publication_lag`` carrying
the earliest known value, flagged ``is_approx_vintage=True``. Strict consumers exclude
these with ``include_approx=False``.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from atlas.pit.schema import Frequency, period_end

APPROX_GRACE = timedelta(days=30)


def approx_first_releases(
    df: pl.DataFrame,
    frequency: Frequency,
    publication_lag_days: int,
    *,
    grace: timedelta = APPROX_GRACE,
) -> pl.DataFrame:
    """``df``: ``event_time``, ``vintage_date``, ``value`` → approx rows (same columns + flag).

    A row is synthesized only when the first real vintage arrives more than ``grace`` after
    the period would normally have been published.
    """
    expected = period_end(pl.col("event_time"), frequency) + pl.duration(days=publication_lag_days)
    first = (
        df.sort("vintage_date")
        .group_by("event_time")
        .agg(pl.col("vintage_date").first(), pl.col("value").first())
        .with_columns(expected.alias("expected_release"))
    )
    return first.filter(
        pl.col("value").is_not_null()
        & (pl.col("vintage_date") > pl.col("expected_release") + grace)
    ).select(
        "event_time",
        pl.col("expected_release").alias("vintage_date"),
        "value",
        pl.lit(True).alias("is_approx_vintage"),
    )
