from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from atlas.pit.schema import OBSERVATION_SCHEMA, conform
from atlas.pit.store import PITStore

# (series_id, event_time, knowledge_time, value) — value None means withdrawn in that vintage
Row = tuple[str, date, datetime, float | None]
MakeObs = Callable[..., pl.DataFrame]


def utc(y: int, m: int, d: int, hh: int = 12, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def make_obs_frame(
    rows: Sequence[Row],
    *,
    run_id: str = "run_test",
    approx: bool = False,
    ingested_at: datetime | None = None,
) -> pl.DataFrame:
    stamp = ingested_at or datetime(2026, 1, 1, tzinfo=UTC)
    data = [
        {
            "series_id": sid,
            "source": "test",
            "source_series_id": sid,
            "event_time": ev,
            "knowledge_time": kt,
            "value": v,
            "unit": "index",
            "frequency": "M",
            "vintage_id": ("approx:" if approx else "") + kt.date().isoformat(),
            "sa_flag": True,
            "is_approx_vintage": approx,
            "ingest_run_id": run_id,
            "ingested_at": stamp,
        }
        for sid, ev, kt, v in rows
    ]
    return conform(pl.DataFrame(data, schema=OBSERVATION_SCHEMA))


@pytest.fixture
def make_obs() -> MakeObs:
    return make_obs_frame


@pytest.fixture
def store(tmp_path: Path) -> PITStore:
    return PITStore(tmp_path / "data")
