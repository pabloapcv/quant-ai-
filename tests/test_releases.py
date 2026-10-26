from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta

from atlas.catalog import SeriesSpec
from atlas.releases import Slot, execute, plan


def _spec(sid: str, at: time | None, source: str = "fred") -> SeriesSpec:
    extra = {} if at is None else {"release_time_et": at}
    return SeriesSpec(
        id=sid, source=source, source_id=sid.replace(".", ""), frequency="M",
        publication_lag_days=10, description="d", **extra,
    )  # fmt: skip


def test_plan_groups_todays_timed_series() -> None:
    specs = [
        _spec("US.CPI.CORE", time(8, 30)),
        _spec("US.CPI.HEADLINE", time(8, 30)),
        _spec("US.RETAIL", time(8, 30)),
        _spec("US.FED.ASSETS", time(16, 30)),
        _spec("US.UNTIMED", None),  # no stated time: left to the morning batch
        _spec("EA.SERIES", time(10, 0), source="ecb"),  # not FRED
    ]
    rel = {"US.CPI.CORE": 10, "US.CPI.HEADLINE": 10, "US.RETAIL": 9, "US.FED.ASSETS": 20}
    slots = plan(specs, rel, releasing_today={10, 20}, day=date(2026, 11, 12))
    assert [(s.at, s.series) for s in slots] == [
        (datetime(2026, 11, 12, 13, 30, tzinfo=UTC), ("US.CPI.CORE", "US.CPI.HEADLINE")),  # EST
        (datetime(2026, 11, 12, 21, 30, tzinfo=UTC), ("US.FED.ASSETS",)),
    ]
    summer = plan(specs, rel, {10}, date(2026, 7, 15))
    assert summer[0].at == datetime(2026, 7, 15, 12, 30, tzinfo=UTC)  # EDT


class Clock:
    def __init__(self, start: datetime) -> None:
        self.t = start

    def now(self) -> datetime:
        return self.t

    async def sleep(self, s: float) -> None:
        self.t += timedelta(seconds=s)


async def test_execute_waits_then_polls_until_new_data() -> None:
    clock = Clock(datetime(2026, 11, 12, 13, 0, tzinfo=UTC))
    slot = Slot(datetime(2026, 11, 12, 13, 30, tzinfo=UTC), ("US.CPI.CORE",))
    calls: list[datetime] = []

    async def ingest(ids: Sequence[str]) -> int:
        calls.append(clock.now())
        return 5 if len(calls) == 3 else 0  # data shows up on the third try

    (res,) = await execute([slot], ingest, now=clock.now, sleep=clock.sleep, poll_s=60)
    assert calls[0] == datetime(2026, 11, 12, 13, 31, tzinfo=UTC)  # release + 1 minute
    assert res.attempts == 3
    assert res.new_rows == 5
    assert res.note == ""


async def test_execute_gives_up_and_catches_up() -> None:
    clock = Clock(datetime(2026, 11, 12, 13, 0, tzinfo=UTC))
    slot = Slot(datetime(2026, 11, 12, 13, 30, tzinfo=UTC), ("US.CPI.CORE",))

    async def never(ids: Sequence[str]) -> int:
        return 0

    (res,) = await execute(
        [slot], never, now=clock.now, sleep=clock.sleep, poll_s=60, max_wait=timedelta(minutes=5)
    )
    assert res.new_rows == 0
    assert "no new data within 5 minutes" in res.note
    assert 5 <= res.attempts <= 7

    late = Clock(datetime(2026, 11, 12, 18, 0, tzinfo=UTC))  # started long after the release
    (res2,) = await execute([slot], never, now=late.now, sleep=late.sleep)
    assert res2.attempts == 1
    assert "catch-up" in res2.note


async def test_alert_hook_runs_minutes_after_new_data_only() -> None:
    clock = Clock(datetime(2026, 11, 12, 13, 0, tzinfo=UTC))
    slots = [
        Slot(datetime(2026, 11, 12, 13, 30, tzinfo=UTC), ("US.CPI.CORE",)),
        Slot(datetime(2026, 11, 12, 15, 0, tzinfo=UTC), ("US.UMICH",)),
    ]
    hooked: list[datetime] = []

    async def ingest(ids: Sequence[str]) -> int:
        return 3 if ids == ("US.CPI.CORE",) else 0

    async def hook(res: object) -> None:
        hooked.append(clock.now())

    await execute(
        slots,
        ingest,
        now=clock.now,
        sleep=clock.sleep,
        max_wait=timedelta(minutes=5),
        on_new_data=hook,
    )
    assert hooked == [datetime(2026, 11, 12, 13, 31, tzinfo=UTC)]  # within 15 minutes (story 4)
