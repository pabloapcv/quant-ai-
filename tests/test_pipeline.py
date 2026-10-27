from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from atlas.pipeline import latest_run, run_daily
from atlas.settings import Settings
from atlas.views.track import score_all, score_file
from tests.test_country import _country_store

REPO = Path(__file__).resolve().parents[1]


def _view(
    asset: str, score: float, prob: float, stance: str, as_of: str = "2020-01-01"
) -> dict[str, object]:
    return {
        "asset_id": f"COUNTRY.{asset}",
        "score": score,
        "prob_outperform": prob,
        "stance": stance,
        "horizon": "3M",
        "as_of": as_of,
        "expected_excess_return": {"mean": 0.0, "p10": -0.05, "p90": 0.05, "benchmark": "x"},
    }


def _returns(assets: dict[str, float], months: int = 6) -> pl.DataFrame:
    rows = []
    for a, r in assets.items():
        for m in range(months):
            rows.append({"date": date(2020, m + 1, 1), "asset": a, "ret": r})
    return pl.DataFrame(rows)


def test_score_file_and_pending(tmp_path: Path) -> None:
    d = tmp_path / "gold" / "views" / "ctry_test" / "3M"
    d.mkdir(parents=True)
    f = d / "2020-01-01.json"
    f.write_text(
        json.dumps(
            [
                _view("AAA", 2.0, 0.6, "overweight"),
                _view("BBB", 0.0, 0.5, "neutral"),
                _view("CCC", -2.0, 0.4, "underweight"),
            ]
        )
    )
    rets = _returns({"AAA": 0.03, "BBB": 0.01, "CCC": -0.02})
    s = score_file(f, rets)
    assert s is not None
    assert s.rank_ic == pytest.approx(1.0)
    assert s.n_calls == 2
    assert s.hit_rate == 1.0
    assert 0 <= s.brier <= 0.25
    # 3 compounded months: excess vs the mean is +7.1%, +0.9%, -8.0% -> only BBB in [-5%, +5%]
    assert s.coverage == pytest.approx(1 / 3)

    short = _returns({"AAA": 0.03, "BBB": 0.01, "CCC": -0.02}, months=3)  # only Feb-Mar after Jan
    assert score_file(f, short) is None  # 3-month outcome not fully realized

    rec = score_all(tmp_path, {"ctry_test": short})
    assert rec.pending == [("ctry_test", "3M", date(2020, 1, 1))]
    assert rec.summary() == {"months_scored": 0, "months_pending": 1}
    rec = score_all(tmp_path, {"ctry_test": rets})
    assert rec.summary()["months_scored"] == 1
    assert rec.summary()["hit_rate"] == 1.0
    assert score_all(tmp_path / "nothing", {}).summary()["months_scored"] == 0


async def test_run_daily_without_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_SLACK_WEBHOOK_URL", raising=False)  # never post from tests
    data = tmp_path / "data"
    _country_store(data)
    settings = Settings(data_dir=data, config_dir=REPO / "config")
    now = datetime(2026, 10, 16, 22, 0, tzinfo=UTC)
    rep = await run_daily(
        settings, ingest=False, views=(("ctry_dev_lcl", 3), ("ind12", 3)), now=now
    )
    names = [s.name for s in rep.steps]
    assert names == [
        "quality:staleness",
        "snapshot",
        "views:ctry_dev_lcl:3M",
        "challenger:ens_v1_gbm:ctry_dev_lcl:3M",
        "views:ind12:3M",
        "challenger:ens_v1_gbm:ind12:3M",
        "views:sleeves:3M",
        "track:score",
        "quality:drift",
        "alerts",
    ]
    by = {s.name: s for s in rep.steps}
    assert by["alerts"].ok
    assert "skipped" in by["quality:drift"].detail  # no macro data in this store
    mon = rep.ic_monitor["ctry_dev_lcl:3M"]
    assert mon["backtest"]["n"] > 100
    assert mon["live_ics"] == []
    assert [a["rule"] for a in rep.alerts] == ["data_quality"]  # the failed ind12 step
    assert "views:ind12:3M" in rep.alerts[0]["message"]
    assert rep.alerts[0]["delivery"] == {}  # no channel configured: log only
    assert by["views:ctry_dev_lcl:3M"].ok
    assert "frozen" in by["views:ctry_dev_lcl:3M"].detail
    assert not by["views:ind12:3M"].ok  # no industry data: recorded, run continues
    assert by["track:score"].ok
    assert rep.snapshot_id
    assert rep.snapshot_id.startswith("snap_2026-10-16T2200Z")
    assert not rep.ok
    run_file = latest_run(data)
    assert run_file is not None
    saved = json.loads(run_file.read_text())
    assert saved["ok"] is False
    assert "NaN" not in run_file.read_text()
    frozen = list((data / "gold" / "views" / "ctry_dev_lcl" / "3M").glob("*.json"))
    assert len(frozen) == 1
    assert by["challenger:ens_v1_gbm:ctry_dev_lcl:3M"].ok
    ch = list((data / "gold" / "challengers" / "ens_v1_gbm" / "ctry_dev_lcl" / "3M").glob("*.json"))
    assert len(ch) == 1
    assert json.loads(ch[0].read_text())[0]["provenance"]["model_version"] == "ens_v1_gbm"
    assert saved["track_challengers"]["ens_v1_gbm"]["months_pending"] == 1
    first = frozen[0].read_text()

    # A second run the same month keeps the first publication byte-for-byte.
    rep2 = await run_daily(
        settings,
        ingest=False,
        views=(("ctry_dev_lcl", 3),),
        now=datetime(2026, 10, 17, 22, 0, tzinfo=UTC),
    )
    assert {s.name: s.ok for s in rep2.steps}["views:ctry_dev_lcl:3M"]
    assert frozen[0].read_text() == first


async def test_run_daily_skips_fred_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import atlas.pipeline as pl_mod
    from atlas.text.ecb import EcbTextConnector
    from atlas.text.fed import FedTextConnector

    async def no_network(*_a: object, **_k: object) -> list[object]:
        return []

    async def no_fed(*_a: object, **_k: object) -> dict[str, object]:
        return {"releases": 0, "new": 0, "skipped": 0, "failed": []}

    monkeypatch.setattr(pl_mod, "ingest_series", no_network)

    async def no_ecb(*_a: object, **_k: object) -> dict[str, object]:
        return {"new": 0, "pages": 0, "failed": []}

    monkeypatch.setattr(FedTextConnector, "sync", no_fed)  # never hit federalreserve.gov
    monkeypatch.setattr(EcbTextConnector, "sync", no_ecb)  # nor ecb.europa.eu
    data = tmp_path / "data"
    _country_store(data)
    rep = await run_daily(Settings(data_dir=data, config_dir=REPO / "config"), views=())
    fred = next(s for s in rep.steps if s.name == "ingest:fred")
    assert not fred.ok
    assert "no FRED key" in fred.detail
    assert {s.name for s in rep.steps} >= {
        "ingest:kenfrench",
        "ingest:ecb",
        "ingest:oecd",
        "ingest:bis",
        "ingest:imf",
        "ingest:fed_text",
        "ingest:ecb_text",
        "text:tone",
    }
    tone = next(s for s in rep.steps if s.name == "text:tone")
    assert tone.ok  # no statements stored: nothing to read, no Claude call


def test_cli_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    _country_store(data)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        runner = CliRunner()
        res = runner.invoke(app, ["pipeline", "daily", "--skip-ingest"])
        assert "views:ctry_dev_lcl:3M" in res.output
        assert res.exit_code == 1  # industry universes have no data in this store
        res = runner.invoke(app, ["pipeline", "track"])
        assert res.exit_code == 0, res.output
        assert "months_pending" in res.output
    finally:
        get_settings.cache_clear()
    _ = np


async def test_failed_series_are_retried_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import atlas.pipeline as pl_mod
    from atlas.ingest import IngestReport
    from atlas.text.ecb import EcbTextConnector
    from atlas.text.fed import FedTextConnector

    calls: dict[str, int] = {}

    async def flaky(specs: list[object], *_a: object, **_k: object) -> list[IngestReport]:
        out = []
        for spec in specs:
            sid = spec.id  # type: ignore[attr-defined]
            calls[sid] = calls.get(sid, 0) + 1
            fail = sid == "FX.EURCNY" and calls[sid] == 1  # a transient 500 on the first try
            out.append(IngestReport(sid, error="HTTP 500" if fail else None))
        return out

    async def no_docs(*_a: object, **_k: object) -> dict[str, object]:
        return {"releases": 0, "new": 0, "skipped": 0, "pages": 0, "failed": []}

    monkeypatch.setattr(pl_mod, "ingest_series", flaky)
    monkeypatch.setattr(pl_mod, "RETRY_DELAY_S", 0.0)
    monkeypatch.setattr(FedTextConnector, "sync", no_docs)
    monkeypatch.setattr(EcbTextConnector, "sync", no_docs)
    monkeypatch.delenv("ATLAS_SLACK_WEBHOOK_URL", raising=False)
    data = tmp_path / "data"
    _country_store(data)
    rep = await run_daily(Settings(data_dir=data, config_dir=REPO / "config"), views=())
    by = {s.name: s for s in rep.steps}
    assert calls["FX.EURCNY"] == 2
    assert by["ingest:ecb"].ok
    assert "recovered on retry: FX.EURCNY" in by["ingest:ecb"].detail
    assert by["ingest:retry"].ok
    assert by["ingest:retry"].detail.startswith("1 recovered")


def test_latest_run_ignores_release_day_logs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20261022T101838Z.json").write_text("{}")
    (runs / "releases-2026-10-22.json").write_text("{}")  # sorts after the pipeline runs
    assert latest_run(tmp_path) == runs / "20261022T101838Z.json"
