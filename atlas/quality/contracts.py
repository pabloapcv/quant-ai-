"""Data contracts (PRD §5.6). A failing contract blocks the write — bad data never lands."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

from atlas.pit.schema import KEY, UTC_DATETIME, Frequency


class DataQualityError(Exception):
    """Raised when a frame violates its contract."""


OBSERVATIONS_CONTRACT = pa.DataFrameSchema(
    {
        "series_id": pa.Column(pl.String, nullable=False),
        "source": pa.Column(pl.String, nullable=False),
        "source_series_id": pa.Column(pl.String, nullable=False),
        "event_time": pa.Column(pl.Date, nullable=False),
        "knowledge_time": pa.Column(UTC_DATETIME, nullable=False),
        "value": pa.Column(pl.Float64, nullable=True),  # null = withdrawn in that vintage
        "unit": pa.Column(pl.String, nullable=True),
        "frequency": pa.Column(pl.String, pa.Check.isin([f.value for f in Frequency])),
        "vintage_id": pa.Column(pl.String, nullable=False),
        "sa_flag": pa.Column(pl.Boolean, nullable=True),
        "is_approx_vintage": pa.Column(pl.Boolean, nullable=False),
        "ingest_run_id": pa.Column(pl.String, nullable=False),
        "ingested_at": pa.Column(UTC_DATETIME, nullable=False),
    },
    unique=list(KEY),
    strict=True,
)


def check_observations(df: pl.DataFrame) -> None:
    try:
        OBSERVATIONS_CONTRACT.validate(df, lazy=True)
    except (SchemaErrors, SchemaError) as exc:
        raise DataQualityError(f"observations contract violated:\n{exc}") from None

    problems: list[str] = []
    non_finite = df.filter(pl.col("value").is_infinite() | pl.col("value").is_nan())
    if non_finite.height:
        problems.append(f"{non_finite.height} non-finite values")

    # A value cannot be public before the period it describes has started.
    premature = df.filter(
        pl.col("knowledge_time")
        < pl.col("event_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    if premature.height:
        sample = premature.select("series_id", "event_time", "knowledge_time").head(5)
        problems.append(f"{premature.height} rows known before their period starts:\n{sample}")

    if problems:
        raise DataQualityError("observations contract violated: " + "; ".join(problems))


# Maximum gap between successive releases before a series is considered stale (PRD §6.5).
MAX_STALENESS: dict[Frequency, timedelta] = {
    Frequency.DAILY: timedelta(days=5),
    Frequency.WEEKLY: timedelta(days=14),
    Frequency.BIWEEKLY: timedelta(days=21),
    Frequency.MONTHLY: timedelta(days=45),
    Frequency.QUARTERLY: timedelta(days=120),
    Frequency.SEMIANNUAL: timedelta(days=220),
    Frequency.ANNUAL: timedelta(days=400),
}


@dataclass(frozen=True)
class StaleSeries:
    series_id: str
    last_knowledge_time: datetime | None
    max_staleness: timedelta


def find_stale(
    last_known: dict[str, datetime | None],
    frequencies: Iterable[tuple[str, Frequency]],
    now: datetime,
) -> list[StaleSeries]:
    """Series whose most recent release is older than their frequency allows."""
    stale = []
    for series_id, freq in frequencies:
        last = last_known.get(series_id)
        limit = MAX_STALENESS[freq]
        if last is None or now - last > limit:
            stale.append(StaleSeries(series_id, last, limit))
    return stale
