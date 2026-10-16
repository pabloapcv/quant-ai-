from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
import respx

from atlas.catalog import SeriesSpec
from atlas.connectors.base import RawStore, RetryPolicy, TransientHTTPError
from atlas.connectors.fred import FRED_BASE_URL, FredConnector, FredError
from atlas.ingest import ingest_series
from atlas.pit.schema import Frequency
from atlas.pit.store import PITStore

KEY = "secret-test-key-123"
FAST = RetryPolicy(max_attempts=3, initial_wait_s=0.001, max_wait_s=0.002)

CPI = SeriesSpec(
    id="US.CPI.CORE",
    source="fred",
    source_id="CPILFESL",
    frequency=Frequency.MONTHLY,
    publication_lag_days=16,
    release_time_et=time(8, 30),
    description="Core CPI",
)

SERIES_META = {
    "seriess": [
        {
            "id": "CPILFESL",
            "title": "Core CPI",
            "frequency_short": "M",
            "units": "Index 1982-1984=100",
            "seasonal_adjustment_short": "SA",
            "last_updated": "2026-09-11 07:44:02-05",
        }
    ]
}


def obs(
    rows: list[tuple[str, str, str, str]], count: int | None = None, offset: int = 0
) -> dict[str, Any]:
    return {
        "count": len(rows) if count is None else count,
        "offset": offset,
        "observations": [
            {"realtime_start": rs, "realtime_end": re_, "date": d, "value": v}
            for rs, re_, d, v in rows
        ],
    }


# ALFRED history for two periods. 1980-01 predates archived vintages (first vintage 1990),
# 2020-03 has a first print and a revision. The 2020-03 middle row repeats the value
# (as chunked requests produce) and must be collapsed.
ALFRED_ROWS = [
    ("1990-05-01", "9999-12-31", "1980-01-01", "80.1"),
    ("2020-04-10", "2020-05-11", "2020-03-01", "265.0"),
    ("2020-05-12", "2021-04-12", "2020-03-01", "265.0"),
    ("2021-04-13", "9999-12-31", "2020-03-01", "265.3"),
    ("2020-05-12", "9999-12-31", "2020-04-01", "."),
]


def fred(transport_routes: respx.MockRouter, **kw: Any) -> FredConnector:
    return FredConnector(
        KEY, retry=FAST, transport=httpx.MockTransport(transport_routes.handler), **kw
    )


def route(
    router: respx.MockRouter,
    endpoint: str,
    handler: Callable[[httpx.Request], httpx.Response] | dict[str, Any],
) -> None:
    r = router.get(FRED_BASE_URL + endpoint)
    if callable(handler):
        r.mock(side_effect=handler)
    else:
        r.mock(return_value=httpx.Response(200, json=handler))


@pytest.fixture
def router() -> respx.MockRouter:
    return respx.MockRouter(assert_all_called=False)


async def test_fetch_builds_bitemporal_rows(router: respx.MockRouter) -> None:
    route(router, "series", SERIES_META)
    route(router, "series/observations", obs(ALFRED_ROWS))
    async with fred(router) as conn:
        df = await conn.fetch(CPI)

    mar = df.filter(pl.col("event_time") == date(2020, 3, 1))
    assert mar["value"].to_list() == [265.0, 265.3]  # unchanged middle vintage collapsed
    # 2020-04-10 08:30 EDT == 12:30 UTC
    assert mar["knowledge_time"][0] == datetime(2020, 4, 10, 12, 30, tzinfo=UTC)
    assert mar["vintage_id"].to_list() == ["2020-04-10", "2021-04-13"]
    assert df["unit"].unique().to_list() == ["Index 1982-1984=100"]
    assert df["sa_flag"].all()

    apr = df.filter(pl.col("event_time") == date(2020, 4, 1))
    assert apr["value"].to_list() == [None]  # "." -> null

    # Pre-vintage period: real 1990 vintage + synthetic first release at
    # period end (1980-01-31) + 16 days = 1980-02-16 08:30 EST (13:30 UTC).
    jan80 = df.filter(pl.col("event_time") == date(1980, 1, 1)).sort("knowledge_time")
    assert jan80["is_approx_vintage"].to_list() == [True, False]
    assert jan80["knowledge_time"][0] == datetime(1980, 2, 16, 13, 30, tzinfo=UTC)
    assert jan80["vintage_id"][0] == "approx:1980-02-16"


async def test_backfill_can_be_disabled(router: respx.MockRouter) -> None:
    route(router, "series", SERIES_META)
    route(router, "series/observations", obs(ALFRED_ROWS))
    async with fred(router) as conn:
        df = await conn.fetch(CPI, backfill_pre_vintage=False)
    assert not df["is_approx_vintage"].any()


async def test_current_mode_stamps_today(router: respx.MockRouter) -> None:
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(req.url.params)
        return httpx.Response(200, json=obs([("2026-10-14", "2026-10-14", "2026-08-01", "330.1")]))

    route(router, "series", SERIES_META)
    route(router, "series/observations", handler)
    async with fred(router, today=date(2026, 10, 14)) as conn:
        df = await conn.fetch(CPI, vintages=False, backfill_pre_vintage=False)
    assert seen["realtime_start"] == seen["realtime_end"] == "2026-10-14"
    assert df["knowledge_time"].item() == datetime(2026, 10, 14, 12, 30, tzinfo=UTC)


async def test_pagination(router: respx.MockRouter) -> None:
    pages = [ALFRED_ROWS[:2], ALFRED_ROWS[2:]]

    def handler(req: httpx.Request) -> httpx.Response:
        offset = int(req.url.params["offset"])
        page = pages[0] if offset == 0 else pages[1]
        return httpx.Response(200, json=obs(page, count=len(ALFRED_ROWS), offset=offset))

    route(router, "series", SERIES_META)
    route(router, "series/observations", handler)
    async with fred(router) as conn:
        df = await conn.fetch(CPI, backfill_pre_vintage=False)
    assert df.filter(pl.col("event_time") == date(2020, 3, 1)).height == 2


async def test_too_many_vintages_falls_back_to_chunks(
    router: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    import atlas.connectors.fred as fred_mod

    monkeypatch.setattr(fred_mod, "VINTAGES_PER_CHUNK", 2)
    windows: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        p = req.url.params
        if p["realtime_start"] == "1776-07-04":
            return httpx.Response(
                400,
                json={"error_code": 400, "error_message": "Bad Request. Too many vintage dates."},
            )
        windows.append((p["realtime_start"], p["realtime_end"]))
        rows = [r for r in ALFRED_ROWS if p["realtime_start"] <= r[0] <= p["realtime_end"]]
        return httpx.Response(200, json=obs(rows))

    route(router, "series", SERIES_META)
    route(
        router,
        "series/vintagedates",
        {"count": 3, "vintage_dates": ["2020-04-10", "2020-05-12", "2021-04-13"]},
    )
    route(router, "series/observations", handler)
    async with fred(router) as conn:
        df = await conn.fetch(CPI, backfill_pre_vintage=False)
    assert windows == [("2020-04-10", "2021-04-12"), ("2021-04-13", "9999-12-31")]
    assert df.filter(pl.col("event_time") == date(2020, 3, 1))["value"].to_list() == [265.0, 265.3]


async def test_retries_transient_errors(router: respx.MockRouter) -> None:
    calls = {"n": 0}

    def flaky(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json=SERIES_META)

    route(router, "series", flaky)
    async with fred(router) as conn:
        info = await conn.series_info("CPILFESL")
    assert info.units == "Index 1982-1984=100"
    assert calls["n"] == 3


async def test_gives_up_after_max_attempts(router: respx.MockRouter) -> None:
    route(router, "series", lambda req: httpx.Response(429))
    async with fred(router) as conn:
        with pytest.raises(TransientHTTPError):
            await conn.series_info("CPILFESL")


async def test_errors_never_leak_api_key(router: respx.MockRouter) -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed {req.url}", request=req)

    route(router, "series", boom)
    async with fred(router) as conn:
        with pytest.raises(TransientHTTPError) as exc:
            await conn.series_info("CPILFESL")
    assert KEY not in str(exc.value)
    assert exc.value.__cause__ is None


async def test_api_error_raises_fred_error(router: respx.MockRouter) -> None:
    route(
        router,
        "series",
        lambda req: httpx.Response(
            400,
            json={"error_code": 400, "error_message": "Bad Request. The series does not exist."},
        ),
    )
    async with fred(router) as conn:
        with pytest.raises(FredError, match="does not exist"):
            await conn.series_info("NOPE")


async def test_raw_payloads_archived_without_key(router: respx.MockRouter, tmp_path: Path) -> None:
    route(router, "series", SERIES_META)
    raw = RawStore(tmp_path)
    async with fred(router, raw_store=raw, run_id="run_x") as conn:
        await conn.series_info("CPILFESL")
    files = list((tmp_path / "raw" / "fred").rglob("*.json.zst"))
    assert len(files) == 1
    envelope = RawStore.read(files[0])
    assert envelope["request"]["endpoint"] == "series"
    assert "api_key" not in envelope["request"]["params"]
    assert KEY not in files[0].read_bytes().decode("latin-1")


async def test_release_calendar(router: respx.MockRouter) -> None:
    route(
        router,
        "releases/dates",
        {
            "count": 1,
            "release_dates": [
                {"release_id": 10, "release_name": "Consumer Price Index", "date": "2026-11-12"}
            ],
        },
    )
    async with fred(router) as conn:
        rel = await conn.release_calendar(date(2026, 11, 1), date(2026, 11, 30))
    assert rel[0].release_id == 10
    assert rel[0].date == date(2026, 11, 12)


async def test_rejects_wrong_source_and_missing_key(router: respx.MockRouter) -> None:
    with pytest.raises(ValueError, match="api_key"):
        FredConnector("")
    async with fred(router) as conn:
        with pytest.raises(ValueError, match="not fred"):
            await conn.fetch(CPI.model_copy(update={"source": "bls"}))


async def test_ingest_end_to_end_is_idempotent(router: respx.MockRouter, tmp_path: Path) -> None:
    route(router, "series", SERIES_META)
    route(router, "series/observations", obs(ALFRED_ROWS))
    store = PITStore(tmp_path)
    async with fred(router) as conn:
        first = await ingest_series([CPI], conn, store)
        second = await ingest_series([CPI], conn, store)
    assert first[0].ok
    assert first[0].rows_new == first[0].rows_fetched > 0
    assert second[0].rows_new == 0

    # As known at end of 2020: Mar-2020 first print, Apr-2020 withdrawn, 1980 approx value.
    known = store.as_of(["US.CPI.CORE"], date(2020, 12, 31))
    assert known.select("event_time", "value").rows() == [
        (date(1980, 1, 1), 80.1),
        (date(2020, 3, 1), 265.0),
    ]


async def test_ingest_reports_failures_per_series(router: respx.MockRouter, tmp_path: Path) -> None:
    route(
        router,
        "series",
        lambda req: httpx.Response(400, json={"error_code": 400, "error_message": "nope"}),
    )
    async with fred(router) as conn:
        reports = await ingest_series([CPI], conn, PITStore(tmp_path))
    assert not reports[0].ok
    assert "nope" in (reports[0].error or "")


async def test_holiday_rows_posted_early_are_clamped(router: respx.MockRouter) -> None:
    """FRED fills a holiday with the prior close, posted the trading day before the holiday."""
    vix = CPI.model_copy(
        update={
            "id": "US.VOL.VIX",
            "source_id": "VIXCLS",
            "frequency": Frequency.DAILY,
            "publication_lag_days": 1,
            "release_time_et": time(23, 59),
        }
    )
    meta = {"seriess": [{**SERIES_META["seriess"][0], "id": "VIXCLS", "frequency_short": "D"}]}
    rows = [
        ("2026-07-03", "9999-12-31", "2026-07-02", "16.1"),
        ("2026-07-03", "9999-12-31", "2026-07-04", "16.1"),  # July 4th, posted on the 3rd
    ]
    route(router, "series", meta)
    route(router, "series/observations", obs(rows))
    async with fred(router) as conn:
        df = await conn.fetch(vix, backfill_pre_vintage=False)
    july4 = df.filter(pl.col("event_time") == date(2026, 7, 4))
    assert july4["knowledge_time"].item().date() >= date(2026, 7, 4)
    assert july4["vintage_id"].item() == "2026-07-04"
