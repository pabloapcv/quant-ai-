"""Time helpers: every cutoff the store sees is an explicit UTC instant."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

NEW_YORK = ZoneInfo("America/New_York")


def to_cutoff(at: date | datetime) -> datetime:
    """Normalize an as-of point to an aware UTC datetime.

    A bare ``date`` means "everything public by the end of that day in New York" — the
    natural cutoff for a signal computed after the US close. Naive datetimes are rejected:
    silently guessing a timezone is how look-ahead bugs start.
    """
    if isinstance(at, datetime):
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError(f"naive datetime {at!r}: pass an aware datetime or a date")
        return at.astimezone(UTC)
    return datetime.combine(at, time.max, tzinfo=NEW_YORK).astimezone(UTC)


def et_wall_time_to_utc(day: pl.Expr, wall_time: time) -> pl.Expr:
    """Combine a Date expression with a New York wall-clock time, returning UTC datetimes."""
    return (
        day.dt.combine(wall_time, time_unit="us")
        .dt.replace_time_zone("America/New_York", ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
    )
