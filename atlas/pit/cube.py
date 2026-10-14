"""In-memory vintage cube for walk-forward work.

Re-running a model on every month-end since 1990 means hundreds of as-of queries. Reading
Parquet each time is wasteful; the cube loads every vintage once and answers ``as_of`` from
memory using exactly the same rule as ``PITStore.as_of`` (``resolve_as_of``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import polars as pl

from atlas.pit.schema import OBSERVATION_SCHEMA
from atlas.pit.store import resolve_as_of


class VintageCube:
    def __init__(self, observations: pl.DataFrame) -> None:
        self._by_series: dict[str, pl.DataFrame] = {
            str(sid): part.sort("knowledge_time")
            for (sid,), part in observations.partition_by("series_id", as_dict=True).items()
        }

    @property
    def series_ids(self) -> list[str]:
        return sorted(self._by_series)

    @property
    def latest_knowledge_date(self) -> date | None:
        """Calendar date (UTC) of the most recent fact in the cube."""
        times = [f["knowledge_time"].max() for f in self._by_series.values() if f.height]
        latest = max((t for t in times if isinstance(t, datetime)), default=None)
        return None if latest is None else latest.date()

    def __contains__(self, series_id: object) -> bool:
        return series_id in self._by_series

    def as_of(
        self, series_id: str, at: date | datetime, *, include_approx: bool = True
    ) -> pl.DataFrame:
        """One series' full history as known at ``at`` (empty if unknown or not yet public)."""
        frame = self._by_series.get(series_id)
        if frame is None:
            return pl.DataFrame(schema=OBSERVATION_SCHEMA)
        return resolve_as_of(frame.lazy(), at, include_approx=include_approx).collect()

    def as_of_many(
        self, series_ids: Sequence[str], at: date | datetime, *, include_approx: bool = True
    ) -> dict[str, pl.DataFrame]:
        return {s: self.as_of(s, at, include_approx=include_approx) for s in series_ids}
