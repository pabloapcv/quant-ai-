"""Canonical bitemporal observation schema (PRD §5.1).

Every fact carries two clocks:

* ``event_time``      the period the value describes. Follows the FRED convention: the
                      period *start* date for M/Q/SA/A series, the stated date for D/W/BW.
* ``knowledge_time``  when the value became public (UTC). Nothing may read a row before it.
"""

from __future__ import annotations

from enum import StrEnum

import polars as pl


class Frequency(StrEnum):
    DAILY = "D"
    WEEKLY = "W"
    BIWEEKLY = "BW"
    MONTHLY = "M"
    QUARTERLY = "Q"
    SEMIANNUAL = "SA"
    ANNUAL = "A"


UTC_DATETIME = pl.Datetime("us", "UTC")

OBSERVATION_SCHEMA: dict[str, pl.DataType] = {
    "series_id": pl.String(),
    "source": pl.String(),
    "source_series_id": pl.String(),
    "event_time": pl.Date(),
    "knowledge_time": UTC_DATETIME,
    "value": pl.Float64(),
    "unit": pl.String(),
    "frequency": pl.String(),
    "vintage_id": pl.String(),
    "sa_flag": pl.Boolean(),
    "is_approx_vintage": pl.Boolean(),
    "ingest_run_id": pl.String(),
    "ingested_at": UTC_DATETIME,
}

# A row is uniquely identified by what it describes and when it became known.
KEY: tuple[str, ...] = ("series_id", "event_time", "knowledge_time")

_PERIOD_LENGTH: dict[Frequency, str] = {
    Frequency.MONTHLY: "1mo",
    Frequency.QUARTERLY: "3mo",
    Frequency.SEMIANNUAL: "6mo",
    Frequency.ANNUAL: "1y",
}


def empty_observations() -> pl.DataFrame:
    return pl.DataFrame(schema=OBSERVATION_SCHEMA)


def conform(df: pl.DataFrame) -> pl.DataFrame:
    """Select and cast to the canonical schema, in canonical column order."""
    missing = set(OBSERVATION_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"observations frame missing columns: {sorted(missing)}")
    return df.select(pl.col(name).cast(dtype) for name, dtype in OBSERVATION_SCHEMA.items())


def period_end(event_time: pl.Expr, frequency: Frequency) -> pl.Expr:
    """Last calendar day of the period that starts at ``event_time``."""
    length = _PERIOD_LENGTH.get(frequency)
    if length is None:  # D / W / BW: FRED dates the observation at (or near) period end
        return event_time
    return event_time.dt.offset_by(length).dt.offset_by("-1d")
