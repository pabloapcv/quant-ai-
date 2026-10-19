"""Feature specs and point-in-time feature computation (PRD §6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

import polars as pl

from atlas.features.transforms import Monthly, Transform, apply_all
from atlas.pit.cube import VintageCube


class Pillar(StrEnum):
    GROWTH = "growth"
    INFLATION = "inflation"
    FINANCIAL = "financial_conditions"  # higher = tighter / more stress
    RECESSION = "recession"  # raw inputs for the recession model, not composited


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    series_id: str
    pillar: Pillar
    steps: tuple[Transform, ...]
    sign: Literal[1, -1] = 1  # orientation in its composite; recession: expected probit sign
    aggregate: Literal["mean", "last"] = "mean"
    description: str = ""

    def compute(self, history: pl.DataFrame) -> pl.DataFrame:
        """Feature series from one as-of vintage of its input (``event_time``, ``value``)."""
        return apply_all(history, (Monthly(self.aggregate), *self.steps))


def feature_matrix(
    cube: VintageCube,
    specs: Sequence[FeatureSpec],
    at: date | datetime,
    *,
    end_month: date,
    ffill_limit: int = 3,
    include_approx: bool = True,
) -> pl.DataFrame:
    """Wide monthly matrix (``event_time`` + one column per feature) as known at ``at``.

    Rows run to ``end_month``. The ragged edge — features whose latest release lags the
    others — is forward-filled up to ``ffill_limit`` months, never further.
    """
    histories = cube.as_of_many(
        sorted({s.series_id for s in specs}), at, include_approx=include_approx
    )
    columns: list[pl.DataFrame] = []
    for spec in specs:
        series = spec.compute(histories[spec.series_id]).rename({"value": spec.name})
        columns.append(series.filter(pl.col("event_time") <= end_month))

    starts = [c["event_time"].min() for c in columns if c.height]
    if not starts:
        return pl.DataFrame(schema={"event_time": pl.Date, **{s.name: pl.Float64 for s in specs}})
    first = min(d for d in starts if isinstance(d, date))
    grid = pl.DataFrame({"event_time": pl.date_range(first, end_month, interval="1mo", eager=True)})
    out = grid
    for col in columns:
        out = out.join(col, on="event_time", how="left")
    return out.sort("event_time").with_columns(
        pl.col(s.name).forward_fill(limit=ffill_limit) for s in specs
    )
