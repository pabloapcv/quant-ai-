from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest
from typer.testing import CliRunner

from atlas.catalog import Catalog, SeriesSpec
from atlas.connectors.base import RawStore, RetryPolicy
from atlas.connectors.sdmx import SdmxConnector, SdmxError, SdmxSource, parse_csv, parse_period
from atlas.ingest import ingest_series
from atlas.pit.schema import Frequency
from atlas.pit.store import PITStore

REPO = Path(__file__).resolve().parents[1]
FAST = RetryPolicy(max_attempts=2, initial_wait_s=0.001, max_wait_s=0.002)
FETCHED = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)

OECD_CSV = """DATAFLOW,REF_AREA,FREQ,MEASURE,UNIT_MEASURE,ACTIVITY,ADJUSTMENT,TRANSFORMATION,TIME_HORIZ,METHODOLOGY,TIME_PERIOD,OBS_VALUE,OBS_STATUS
X,USA,M,SHARE,IX,_Z,_Z,_Z,_Z,_Z,2026-01,100.5,A
X,USA,M,SHARE,IX,_Z,_Z,_Z,_Z,_Z,2026-02,101.0,A
X,USA,M,IRLT,PA,_Z,_Z,_Z,_Z,_Z,2026-01,4.2,A
X,DEU,M,SHARE,IX,_Z,_Z,_Z,_Z,_Z,2026-01,90.0,A
X,DEU,M,SHARE,IX,_Z,_Z,_Z,_Z,_Z,2026-02,,A
"""
IMF_CSV = """DATAFLOW,COUNTRY,INDEX_TYPE,COICOP_1999,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.STA:CPI,USA,CPI,_T,IX,M,2026-M01,149.16
IMF.STA:CPI,USA,CPI,_T,IX,M,2026-M02,149.86
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-09-01", date(2026, 9, 1)),
        ("2026-09", date(2026, 9, 1)),
        ("2026-M09", date(2026, 9, 1)),
        ("2026-Q3", date(2026, 7, 1)),
        ("2026-S2", date(2026, 7, 1)),
        ("2026", date(2026, 1, 1)),
        ("2026-A1", date(2026, 1, 1)),
    ],
)
def test_parse_period(text: str, expected: date) -> None:
    assert parse_period(text) == expected


def test_parse_period_rejects_garbage() -> None:
    with pytest.raises(SdmxError, match="TIME_PERIOD"):
        parse_period("Sept 2026")


def test_source_parsing() -> None:
    src = SdmxSource.parse("FLOW|A+B.M|REF_AREA=USA,MEASURE=SHARE")
    assert src == SdmxSource("FLOW", "A+B.M", (("REF_AREA", "USA"), ("MEASURE", "SHARE")))
    assert SdmxSource.parse("EXR|M.USD.EUR.SP00.A|").selector == ()
    assert SdmxSource.parse("EXR|M.USD.EUR.SP00.A").selector == ()
    with pytest.raises(ValueError, match="flow"):
        SdmxSource.parse("onlyone")
    with pytest.raises(ValueError, match="selector"):
        SdmxSource.parse("F|K|REF_AREA")


def test_parse_csv_selects_and_guards_ambiguity() -> None:
    usa = parse_csv(OECD_CSV, (("REF_AREA", "USA"), ("MEASURE", "SHARE")))
    assert usa.rows() == [(date(2026, 1, 1), 100.5), (date(2026, 2, 1), 101.0)]
    deu = parse_csv(OECD_CSV, (("REF_AREA", "DEU"), ("MEASURE", "SHARE")))
    assert deu["value"].to_list() == [90.0, None]
    with pytest.raises(SdmxError, match="series"):
        parse_csv(OECD_CSV, (("REF_AREA", "USA"),))  # SHARE and IRLT both match
    with pytest.raises(SdmxError, match="no observations"):
        parse_csv(OECD_CSV, (("REF_AREA", "FRA"),))
    with pytest.raises(SdmxError, match="dimension"):
        parse_csv(OECD_CSV, (("NOPE", "X"),))
    with pytest.raises(SdmxError, match="not SDMX-CSV"):
        parse_csv("a,b\n1,2\n")
    assert parse_csv(IMF_CSV)["event_time"].to_list() == [date(2026, 1, 1), date(2026, 2, 1)]


def spec(sid: str, source_id: str, source: str = "oecd", lag: int = 20) -> SeriesSpec:
    return SeriesSpec(
        id=sid,
        source=source,
        source_id=source_id,
        frequency=Frequency.MONTHLY,
        publication_lag_days=lag,
        description="t",
    )


def transport(calls: list[httpx.Request], body: str, status: int = 200) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


KEY = "FLOW|USA+DEU.M.SHARE+IRLT......|"


async def test_connector_shares_one_download(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    async with SdmxConnector(
        "oecd",
        transport=transport(calls, OECD_CSV),
        fetched_at=FETCHED,
        retry=FAST,
        raw_store=RawStore(tmp_path),
        start_period="2026-01",
    ) as conn:
        usa = await conn.fetch(spec("USA.EQUITY.SHARE", KEY + "REF_AREA=USA,MEASURE=SHARE"))
        deu = await conn.fetch(spec("DEU.EQUITY.SHARE", KEY + "REF_AREA=DEU,MEASURE=SHARE"))
    assert len(calls) == 1
    assert calls[0].url.params["format"] == "csvfile"
    assert calls[0].url.params["startPeriod"] == "2026-01"
    assert calls[0].headers["accept"] == "text/csv"
    current = usa.filter(~pl.col("is_approx_vintage"))
    assert current["knowledge_time"].unique().to_list() == [FETCHED]
    assert current["value"].to_list() == [100.5, 101.0]
    approx = usa.filter(pl.col("is_approx_vintage")).sort("event_time")
    assert approx["event_time"].to_list() == [date(2026, 1, 1), date(2026, 2, 1)]
    assert approx["knowledge_time"][0].date() == date(2026, 2, 20) + timedelta(days=1)  # 23:59 ET
    assert deu.filter(~pl.col("is_approx_vintage")).height == 1  # empty value dropped
    assert list((tmp_path / "raw" / "oecd").rglob("*.json.zst"))


async def test_connector_errors_and_agencies() -> None:
    calls: list[httpx.Request] = []
    async with SdmxConnector(
        "oecd", transport=transport(calls, "Not enough key values", 403), retry=FAST
    ) as conn:
        for sid in ("A.B", "C.D"):
            with pytest.raises(SdmxError, match="403"):
                await conn.fetch(spec(sid, KEY + "REF_AREA=USA,MEASURE=SHARE"))
    assert len(calls) == 1  # failure cached
    with pytest.raises(ValueError, match="agency"):
        SdmxConnector("fed")
    async with SdmxConnector("imf", transport=transport([], IMF_CSV), fetched_at=FETCHED) as conn:
        with pytest.raises(ValueError, match="not imf"):
            await conn.fetch(spec("USA.CPI.IMF", "F|K|"))
        out = await conn.fetch(
            spec("USA.CPI.IMF", "IMF.STA,CPI|USA.CPI._T.IX.M|", source="imf"), vintages=False
        )
    assert not out["is_approx_vintage"].any()


@pytest.mark.parametrize("agency", ["ecb", "bis", "imf"])
async def test_agency_url_shapes(agency: str) -> None:
    calls: list[httpx.Request] = []
    async with SdmxConnector(agency, transport=transport(calls, IMF_CSV), retry=FAST) as conn:
        await conn.fetch(spec("USA.CPI.IMF", "FLOW|USA.CPI._T.IX.M|", source=agency))
    path = calls[0].url.path
    assert "FLOW" in path
    assert "USA.CPI._T.IX.M" in path
    if agency == "bis":
        assert "/dataflow/BIS/FLOW/1.0/" in path


async def test_snapshot_reingest_is_idempotent(tmp_path: Path) -> None:
    store = PITStore(tmp_path)
    s = spec("USA.EQUITY.SHARE", KEY + "REF_AREA=USA,MEASURE=SHARE")
    async with SdmxConnector("oecd", transport=transport([], OECD_CSV), fetched_at=FETCHED) as conn:
        first = await ingest_series([s], conn, store)
    async with SdmxConnector(
        "oecd", transport=transport([], OECD_CSV), fetched_at=FETCHED + timedelta(days=1)
    ) as conn:
        second = await ingest_series([s], conn, store)
    assert first[0].rows_new > 0
    assert second[0].rows_new == 0


def test_catalog_global_entries_parse() -> None:
    cat = Catalog.load(REPO / "config" / "series.yaml")
    for source in ("ecb", "oecd", "bis", "imf"):
        specs = cat.by_source(source)
        assert specs, source
        for s in specs:
            SdmxSource.parse(s.source_id)  # every entry is well-formed


def test_cli_rejects_unknown_agency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    monkeypatch.setenv("ATLAS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        res = CliRunner().invoke(app, ["ingest", "sdmx", "--agency", "fed"])
        assert res.exit_code != 0
    finally:
        get_settings.cache_clear()
