"""Composable, strictly causal transforms on a single series' history.

Every transform takes and returns a frame with columns ``event_time`` (Date) and ``value``
(Float64), sorted ascending on a regular monthly grid. Each output value at period ``m``
depends only on inputs at periods ``<= m`` (PRD §6.5): no centered windows, no full-sample
normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import polars as pl

_V = pl.col("value")


class Transform(Protocol):
    def apply(self, df: pl.DataFrame) -> pl.DataFrame: ...


@dataclass(frozen=True)
class Monthly:
    """Aggregate to calendar months on a gap-free grid (missing months become null).

    For daily/weekly inputs ``mean`` gives the monthly average; ``last`` the month-end value.
    Monthly inputs pass through unchanged apart from gap filling, so period shifts below
    always mean calendar months.
    """

    how: Literal["mean", "last"] = "mean"

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df.select("event_time", "value")
        agg = _V.mean() if self.how == "mean" else _V.sort_by("event_time").last()
        out = (
            df.group_by(pl.col("event_time").dt.truncate("1mo"))
            .agg(agg.alias("value"))
            .sort("event_time")
        )
        return out.upsample(time_column="event_time", every="1mo").select("event_time", "value")


@dataclass(frozen=True)
class PctChange:
    periods: int

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns((100.0 * (_V / _V.shift(self.periods) - 1.0)).alias("value"))


@dataclass(frozen=True)
class Annualized:
    """Compound annualized % change over ``periods`` months."""

    periods: int
    periods_per_year: int = 12

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        ratio = _V / _V.shift(self.periods)
        return df.with_columns(
            (100.0 * (ratio.pow(self.periods_per_year / self.periods) - 1.0)).alias("value")
        )


@dataclass(frozen=True)
class Diff:
    periods: int

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(_V.diff(self.periods).alias("value"))


@dataclass(frozen=True)
class RollingZ:
    """Trailing z-score (window includes the current period, never later ones)."""

    window: int = 120
    min_samples: int = 36

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        mean = _V.rolling_mean(window_size=self.window, min_samples=self.min_samples)
        std = _V.rolling_std(window_size=self.window, min_samples=self.min_samples)
        z = pl.when(std > 0).then((_V - mean) / std).otherwise(None)
        return df.with_columns(z.alias("value"))


def apply_all(df: pl.DataFrame, steps: tuple[Transform, ...]) -> pl.DataFrame:
    out = df.select("event_time", pl.col("value").cast(pl.Float64)).sort("event_time")
    for step in steps:
        out = step.apply(out)
    # inf/NaN can appear from ratios on zero or negative levels; they are missing data
    return out.with_columns(pl.when(_V.is_finite()).then(_V).otherwise(None).alias("value"))
