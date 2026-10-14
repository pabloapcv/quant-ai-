from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest

from atlas.catalog import SeriesSpec
from atlas.connectors.base import RawStore, RetryPolicy
from atlas.connectors.kenfrench import KenFrenchConnector, KenFrenchError, KFSource, parse_section
from atlas.ingest import ingest_series
from atlas.pit.backfill import approx_first_releases
from atlas.pit.schema import Frequency
from atlas.pit.store import PITStore

VW = "Average Value Weighted Returns -- Monthly"
INDUSTRY_CSV = f"""This file was created using the 202608 CRSP database.\r
Missing data are indicated by -99.99 or -999.\r
\r
\r
  {VW}\r
,NoDur,Durbl,Hlth \r
202601,   1.00,   2.00, -99.99\r
202602,  -1.50,   3.00,   0.50\r
202603,   2.25,  -999,   1.00\r
\r
\r
  Average Equal Weighted Returns -- Monthly\r
,NoDur,Durbl,Hlth \r
202601,   9.00,   9.00,   9.00\r
\r
  Average Value Weighted Returns -- Annual\r
,NoDur,Durbl,Hlth \r
2026,   5.00,   5.00,   5.00\r
"""
FACTORS_CSV = """Preamble line\r
\r
,Mkt-RF,SMB,RF\r
196307,   -0.39,   -0.48,    0.27\r
196308,    5.08,   -0.80,    0.25\r
\r
 Annual Factors: January-December\r
,Mkt-RF,SMB,RF\r
1964,  10.00,   1.00,   3.00\r
"""

FAST = RetryPolicy(max_attempts=2, initial_wait_s=0.001, max_wait_s=0.002)
FETCHED = datetime(2026, 10, 14, 15, 0, tzinfo=UTC)


def zipped(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text.encode("latin-1"))
    return buf.getvalue()


def spec(
    column: str = "NoDur", dataset: str = "12_Industry_Portfolios", section: str = VW
) -> SeriesSpec:
    return SeriesSpec(
        id=f"FF.IND12.{column.upper()}",
        source="kenfrench",
        source_id=f"{dataset}|{section}|{column}",
        frequency=Frequency.MONTHLY,
        publication_lag_days=45,
        description="test",
    )


def transport(calls: list[httpx.Request], *, status: int = 200) -> httpx.MockTransport:
    files = {
        "12_Industry_Portfolios_CSV.zip": zipped("12_Industry_Portfolios.csv", INDUSTRY_CSV),
        "F-F_Research_Data_5_Factors_2x3_CSV.zip": zipped("f.csv", FACTORS_CSV),
    }

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        name = req.url.path.rsplit("/", 1)[-1]
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, content=files[name]) if name in files else httpx.Response(404)

    return httpx.MockTransport(handler)


def test_parse_titled_section() -> None:
    df = parse_section(INDUSTRY_CSV, VW)
    assert df.columns == ["event_time", "NoDur", "Durbl", "Hlth"]  # padded names stripped
    assert df["event_time"].to_list() == [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)]
    assert df["Hlth"].to_list() == [None, 0.5, 1.0]  # -99.99 -> null
    assert df["Durbl"].to_list() == [2.0, 3.0, None]  # -999 -> null


def test_parse_first_table_skips_annual_and_errors() -> None:
    df = parse_section(FACTORS_CSV)
    assert df.height == 2
    assert df["Mkt-RF"].to_list() == [-0.39, 5.08]
    with pytest.raises(KenFrenchError, match="not found"):
        parse_section(INDUSTRY_CSV, "No Such Section")
    with pytest.raises(KenFrenchError, match="no table header"):
        parse_section("just text\nno table")
    with pytest.raises(KenFrenchError, match="no monthly rows"):
        parse_section("  T\n,A\n2026,  1.0\n", "T")


def test_source_id_parsing() -> None:
    assert KFSource.parse("a|b|c") == KFSource("a", "b", "c")
    with pytest.raises(ValueError, match="dataset"):
        KFSource.parse("a|b")


async def test_fetch_stamps_snapshot_at_fetch_time(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    async with KenFrenchConnector(
        transport=transport(calls), fetched_at=FETCHED, raw_store=RawStore(tmp_path), retry=FAST
    ) as kf:
        nodur = await kf.fetch(spec("NoDur"))
        durbl = await kf.fetch(spec("Durbl"))
    assert len(calls) == 1  # one download shared by every column of the file
    assert calls[0].headers["accept"] == "*/*"

    current = nodur.filter(~pl.col("is_approx_vintage"))
    assert current["knowledge_time"].unique().to_list() == [FETCHED]
    assert current["value"].to_list() == [1.0, -1.5, 2.25]
    approx = nodur.filter(pl.col("is_approx_vintage"))
    # Jan: period end Jan 31 + 45d = Mar 17, well before the fetch -> approx row exists
    assert approx["event_time"].to_list()[0] == date(2026, 1, 1)
    assert approx["knowledge_time"][0].date() == date(2026, 3, 17) + timedelta(days=1)  # 23:59 ET
    assert durbl.filter(~pl.col("is_approx_vintage")).height == 2  # -999 dropped
    assert list((tmp_path / "raw" / "kenfrench").rglob("*.json.zst"))


async def test_fetch_errors_are_cached_and_reported() -> None:
    calls: list[httpx.Request] = []
    async with KenFrenchConnector(transport=transport(calls, status=406), retry=FAST) as kf:
        for col in ("NoDur", "Durbl"):
            with pytest.raises(KenFrenchError, match="406"):
                await kf.fetch(spec(col))
    assert len(calls) == 1

    async with KenFrenchConnector(transport=transport([]), retry=FAST) as kf:
        with pytest.raises(KenFrenchError, match="column"):
            await kf.fetch(spec("Nope"))
        with pytest.raises(ValueError, match="not kenfrench"):
            await kf.fetch(spec().model_copy(update={"source": "fred"}))


async def test_daily_snapshot_reingest_adds_nothing(tmp_path: Path) -> None:
    store = PITStore(tmp_path)
    s = spec("NoDur")
    async with KenFrenchConnector(transport=transport([]), fetched_at=FETCHED) as kf:
        first = await ingest_series([s], kf, store)
    async with KenFrenchConnector(
        transport=transport([]), fetched_at=FETCHED + timedelta(days=1)
    ) as kf:
        second = await ingest_series([s], kf, store)
    assert first[0].rows_new > 0
    assert second[0].rows_new == 0  # unchanged values at a later fetch are redundant

    # Known now: all three months; strict mode only sees the real snapshot rows.
    now = store.as_of([s.id], FETCHED + timedelta(days=2))
    assert now["value"].to_list() == [1.0, -1.5, 2.25]
    assert not store.as_of([s.id], FETCHED + timedelta(days=2), include_approx=False).is_empty()
    # Before the fetch, only months whose approx release (period end + 45d) has passed:
    # Jan -> Mar 17, Feb -> Apr 14 visible on Apr 30; Mar -> May 15 not yet.
    visible = store.as_of([s.id], date(2026, 4, 30))["event_time"].to_list()
    assert visible == [date(2026, 1, 1), date(2026, 2, 1)]


def test_approx_first_releases() -> None:
    df = pl.DataFrame(
        {
            "event_time": [date(2020, 1, 1), date(2026, 9, 1), date(2020, 2, 1)],
            "vintage_date": [date(2026, 10, 1)] * 3,
            "value": [1.0, 2.0, None],
        }
    )
    out = approx_first_releases(df, Frequency.MONTHLY, 10)
    assert out["event_time"].to_list() == [date(2020, 1, 1)]  # recent & null rows skipped
    assert out["vintage_date"].to_list() == [date(2020, 2, 10)]
    assert out["is_approx_vintage"].all()
    _ = time
