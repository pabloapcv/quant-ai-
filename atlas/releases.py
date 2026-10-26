"""Release-calendar ingestion (PRD §9): fetch a series minutes after it is published.

The daily pipeline runs once in the morning; headline numbers (CPI, payrolls, retail sales)
come out at fixed Eastern-time slots during the day. For each catalog series with a stated
``release_time_et``, FRED's release calendar says whether its release is out today; the
series is then ingested at its release time and polled every ``poll_s`` seconds for up to
``max_wait`` until the new vintage appears (FRED usually posts within minutes).

Pure parts (``plan``) are separate from waiting and fetching (``execute``), so the schedule
can be tested without a clock or the network. The release map (series -> FRED release id)
is cached in ``data/reference/fred_release_map.json`` and refreshed after 30 days.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from atlas.catalog import CONSERVATIVE_RELEASE_TIME, SeriesSpec

ET = ZoneInfo("America/New_York")
MAP_MAX_AGE = timedelta(days=30)


@dataclass(frozen=True)
class Slot:
    at: datetime  # UTC
    series: tuple[str, ...]  # catalog ids


@dataclass
class SlotResult:
    at: datetime
    series: tuple[str, ...]
    attempts: int = 0
    new_rows: int = 0
    finished_at: datetime | None = None
    note: str = ""
    alerts: list[str] = field(default_factory=list)  # alert messages raised after this slot


@dataclass
class ReleaseDayReport:
    day: date
    slots: list[SlotResult] = field(default_factory=list)


def timed_fred_series(specs: Iterable[SeriesSpec]) -> list[SeriesSpec]:
    """FRED series with a stated release time (the rest wait for the morning batch)."""
    return [
        s for s in specs if s.source == "fred" and s.release_time_et != CONSERVATIVE_RELEASE_TIME
    ]


def plan(
    specs: Sequence[SeriesSpec],
    release_of: Mapping[str, int | None],
    releasing_today: set[int],
    day: date,
) -> list[Slot]:
    """Release slots for ``day``: series grouped by their Eastern-time release moment."""
    groups: dict[datetime, list[str]] = {}
    for s in timed_fred_series(specs):
        rid = release_of.get(s.id)
        if rid is None or rid not in releasing_today:
            continue
        at = datetime.combine(day, s.release_time_et, tzinfo=ET).astimezone(UTC)
        groups.setdefault(at, []).append(s.id)
    return [Slot(at, tuple(sorted(ids))) for at, ids in sorted(groups.items())]


async def execute(
    slots: Sequence[Slot],
    ingest: Callable[[Sequence[str]], Awaitable[int]],
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    delay: timedelta = timedelta(minutes=1),
    poll_s: float = 60.0,
    max_wait: timedelta = timedelta(minutes=30),
    on_new_data: Callable[[SlotResult], Awaitable[None]] | None = None,
) -> list[SlotResult]:
    """Wait for each slot, then ingest until new rows arrive or ``max_wait`` passes.

    ``on_new_data`` runs after a slot that brought new rows (the CLI evaluates the alert
    rules there, so a regime-probability cross is flagged minutes after the release).
    """
    results = []
    for slot in slots:
        res = SlotResult(slot.at, slot.series)
        start = slot.at + delay
        if now() > slot.at + max_wait + delay:
            res.note = "slot already past: catch-up ingest only"
        while now() < start:
            await sleep(min(60.0, (start - now()).total_seconds()))
        deadline = max(now(), start) + max_wait
        while True:
            res.attempts += 1
            res.new_rows += await ingest(slot.series)
            if res.new_rows > 0 or res.note or now() >= deadline:
                break
            await sleep(poll_s)
        if res.new_rows == 0 and not res.note:
            res.note = f"no new data within {int(max_wait.total_seconds() // 60)} minutes"
        if res.new_rows > 0 and on_new_data is not None:
            await on_new_data(res)
        res.finished_at = now()
        results.append(res)
    return results


# ---------------------------------------------------------------------------- FRED wiring
def map_path(data_dir: Path) -> Path:
    return data_dir / "reference" / "fred_release_map.json"


async def release_map(
    conn: Any, specs: Sequence[SeriesSpec], data_dir: Path
) -> dict[str, int | None]:
    """Catalog id -> FRED release id, cached for ``MAP_MAX_AGE``."""
    path = map_path(data_dir)
    if path.exists():
        cached = json.loads(path.read_text())
        fresh = datetime.now(UTC) - datetime.fromisoformat(cached["built_at"]) < MAP_MAX_AGE
        wanted = {s.id for s in timed_fred_series(specs)}
        if fresh and wanted <= set(cached["map"]):
            return {k: v for k, v in cached["map"].items()}
    out: dict[str, int | None] = {}
    for s in timed_fred_series(specs):
        out[s.id] = await conn.series_release(s.source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"built_at": datetime.now(UTC).isoformat(), "map": out}, indent=2))
    return out


def report_json(report: ReleaseDayReport) -> str:
    return json.dumps(
        {
            "day": report.day.isoformat(),
            "slots": [
                {
                    "at": r.at.isoformat(),
                    "series": list(r.series),
                    "attempts": r.attempts,
                    "new_rows": r.new_rows,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                    "note": r.note,
                }
                for r in report.slots
            ],
        },
        indent=2,
    )
