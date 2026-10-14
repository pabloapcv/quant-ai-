"""Append-only bitemporal observation store on Parquet (PRD §5.1, §5.4 silver layer).

Layout::

    {data_dir}/silver/observations/series_id={ID}/part-{ts}-{uuid}.parquet

Parts are immutable: a write only ever adds a new part containing rows the store has not
seen before. That makes snapshots trivially reproducible (a snapshot is a list of part
files) and makes re-ingesting the same payload a no-op.

All reads are point-in-time. ``as_of(t)`` returns, for each (series, period), the latest
vintage whose ``knowledge_time <= t`` — never anything published after ``t``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from atlas.pit.clock import to_cutoff
from atlas.pit.schema import KEY, OBSERVATION_SCHEMA, conform
from atlas.quality.contracts import check_observations

_EVENT_KEY = ["series_id", "event_time"]


@dataclass(frozen=True)
class WriteResult:
    series_id: str
    rows_in: int
    rows_new: int
    rows_conflicting: int  # same (series, period, knowledge_time) but a different value
    part: Path | None


class ReadOnlyStoreError(RuntimeError):
    pass


class PITStore:
    def __init__(self, data_dir: Path, *, pinned_parts: Iterable[Path] | None = None) -> None:
        self.data_dir = data_dir
        self.root = data_dir / "silver" / "observations"
        self._pinned = (
            None if pinned_parts is None else frozenset(p.resolve() for p in pinned_parts)
        )

    # ------------------------------------------------------------------ layout
    def series_dir(self, series_id: str) -> Path:
        return self.root / f"series_id={series_id}"

    def series_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        ids = sorted(p.name.removeprefix("series_id=") for p in self.root.glob("series_id=*"))
        return [s for s in ids if self.part_files([s])]

    def part_files(self, series_ids: Sequence[str] | None = None) -> list[Path]:
        if series_ids is None:
            parts = sorted(self.root.glob("series_id=*/part-*.parquet"))
        else:
            parts = sorted(p for s in series_ids for p in self.series_dir(s).glob("part-*.parquet"))
        if self._pinned is not None:
            parts = [p for p in parts if p.resolve() in self._pinned]
        return parts

    # ------------------------------------------------------------------ write
    def write(self, df: pl.DataFrame) -> list[WriteResult]:
        if self._pinned is not None:
            raise ReadOnlyStoreError("store is pinned to a snapshot and is read-only")
        df = conform(df)
        check_observations(df)

        results = []
        for (series_id,), incoming in df.partition_by("series_id", as_dict=True).items():
            results.append(self._write_series(str(series_id), incoming))
        return results

    def _write_series(self, series_id: str, incoming: pl.DataFrame) -> WriteResult:
        existing = (
            self.scan([series_id])
            .select(*KEY, pl.col("value").alias("_existing_value"), pl.lit(True).alias("_hit"))
            .collect()
        )
        joined = incoming.join(existing, on=list(KEY), how="left")
        is_new = pl.col("_hit").is_null()
        is_conflict = pl.col("_hit") & pl.col("_existing_value").ne_missing(pl.col("value"))
        to_write = joined.filter(is_new | is_conflict)
        n_conflict = to_write.filter(pl.col("_hit").is_not_null()).height

        part = None
        if to_write.height:
            part = self._atomic_write(series_id, to_write.select(OBSERVATION_SCHEMA.keys()))
        return WriteResult(
            series_id=series_id,
            rows_in=incoming.height,
            rows_new=to_write.height - n_conflict,
            rows_conflicting=n_conflict,
            part=part,
        )

    def _atomic_write(self, series_id: str, df: pl.DataFrame) -> Path:
        directory = self.series_dir(series_id)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        final = directory / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = final.with_suffix(".parquet.tmp")
        df.sort(list(KEY)).write_parquet(tmp, compression="zstd", statistics=True)
        os.replace(tmp, final)
        return final

    # ------------------------------------------------------------------ read
    def scan(self, series_ids: Sequence[str] | None = None) -> pl.LazyFrame:
        """All known rows (every vintage), with cross-part duplicates resolved.

        If the same key was written twice with different values (a source restated a
        vintage), the most recently ingested row wins.
        """
        parts = self.part_files(series_ids)
        if not parts:
            return pl.LazyFrame(schema=OBSERVATION_SCHEMA)
        lf = pl.scan_parquet(parts)
        if series_ids is not None:
            lf = lf.filter(pl.col("series_id").is_in(list(series_ids)))
        return (
            lf.group_by(KEY)
            .agg(pl.all().sort_by("ingested_at").last())
            .select(OBSERVATION_SCHEMA.keys())
        )

    def as_of(
        self,
        series_ids: Sequence[str],
        at: date | datetime,
        *,
        include_approx: bool = True,
        event_start: date | None = None,
        event_end: date | None = None,
    ) -> pl.DataFrame:
        """The full history of each series exactly as it was known at ``at``."""
        cutoff = to_cutoff(at)
        lf = self.scan(series_ids).filter(pl.col("knowledge_time") <= cutoff)
        if not include_approx:
            lf = lf.filter(~pl.col("is_approx_vintage"))
        if event_start is not None:
            lf = lf.filter(pl.col("event_time") >= event_start)
        if event_end is not None:
            lf = lf.filter(pl.col("event_time") <= event_end)
        return (
            _latest_vintage(lf, _EVENT_KEY)
            # a null in the latest vintage means the value was withdrawn: do not resurrect it
            .filter(pl.col("value").is_not_null())
            .sort(_EVENT_KEY)
            .collect()
        )

    def history(self, series_id: str, event_time: date) -> pl.DataFrame:
        """Every vintage ever recorded for one observation (its revision path)."""
        return (
            self.scan([series_id])
            .filter(pl.col("event_time") == event_time)
            .sort("knowledge_time")
            .collect()
        )

    def realtime_panel(
        self,
        series_ids: Sequence[str],
        dates: Sequence[date | datetime],
        *,
        include_approx: bool = True,
    ) -> pl.DataFrame:
        """For each as-of date: the most recent period's value, as known on that date.

        This is the series a model would actually have seen when re-run on each date —
        the right input for backtests. Returns one row per (as_of, series_id).
        """
        if not dates:
            raise ValueError("dates must not be empty")
        cutoffs = pl.DataFrame(
            {"as_of": [to_cutoff(d) for d in dates]},
            schema={"as_of": OBSERVATION_SCHEMA["knowledge_time"]},
        )
        obs = self.scan(series_ids)
        if not include_approx:
            obs = obs.filter(~pl.col("is_approx_vintage"))
        known = cutoffs.lazy().join_where(obs, pl.col("knowledge_time") <= pl.col("as_of"))
        vintages = _latest_vintage(known, ["as_of", *_EVENT_KEY]).filter(
            pl.col("value").is_not_null()
        )
        return (
            vintages.group_by(["as_of", "series_id"])
            .agg(pl.all().sort_by("event_time").last())
            .select(
                "as_of",
                "series_id",
                "event_time",
                "value",
                "knowledge_time",
                "vintage_id",
                "is_approx_vintage",
            )
            .sort(["as_of", "series_id"])
            .collect()
        )

    def last_knowledge_times(self) -> dict[str, datetime | None]:
        df = self.scan().group_by("series_id").agg(pl.col("knowledge_time").max()).collect()
        return dict(zip(df["series_id"].to_list(), df["knowledge_time"].to_list(), strict=True))


def _latest_vintage(lf: pl.LazyFrame, by: list[str]) -> pl.LazyFrame:
    return lf.group_by(by).agg(pl.all().sort_by("knowledge_time").last())
