from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atlas.catalog import Catalog, SeriesSpec
from atlas.pit.schema import Frequency
from atlas.pit.snapshot import create_snapshot, load_snapshot, store_from_snapshot, verify_snapshot
from atlas.pit.store import PITStore, ReadOnlyStoreError
from atlas.quality.contracts import find_stale
from tests.conftest import make_obs_frame, utc

REPO = Path(__file__).resolve().parents[1]
S = "US.CPI.CORE"


def test_repo_catalog_loads_and_parses_times() -> None:
    cat = Catalog.load(REPO / "config" / "series.yaml")
    assert len(cat) >= 30
    cpi = cat.get("US.CPI.CORE")
    assert cpi.source_id == "CPILFESL"
    assert cpi.release_time_et == time(8, 30)  # quoted in YAML, not base-60
    assert cat.get("US.RATES.10Y").release_time_et == time(23, 59)
    assert {s.source for s in cat} == {"fred", "kenfrench", "ecb", "oecd", "bis", "imf"}


def test_catalog_rejects_duplicates_and_unknown() -> None:
    spec = SeriesSpec(
        id="US.X.Y",
        source="fred",
        source_id="X",
        frequency=Frequency.DAILY,
        publication_lag_days=1,
        description="x",
    )
    with pytest.raises(ValueError, match="duplicate"):
        Catalog([spec, spec])
    with pytest.raises(KeyError):
        Catalog([spec]).get("US.NOPE")


def test_snapshot_pins_data(tmp_path: Path) -> None:
    data, cfg = tmp_path / "data", tmp_path / "config"
    cfg.mkdir()
    (cfg / "series.yaml").write_text("series: []\n")
    store = PITStore(data)
    store.write(make_obs_frame([(S, date(2020, 1, 1), utc(2020, 2, 13), 100.0)]))

    snap = create_snapshot(data, cfg, now=datetime(2026, 10, 14, 6, 0, tzinfo=UTC))
    assert snap.snapshot_id.startswith("snap_2026-10-14T0600Z_")
    assert load_snapshot(data, snap.snapshot_id) == snap
    assert verify_snapshot(data, snap) == []

    # Later ingests are invisible to the pinned store.
    store.write(make_obs_frame([(S, date(2020, 1, 1), utc(2021, 2, 19), 100.5)], run_id="later"))
    pinned = store_from_snapshot(data, snap)
    assert pinned.as_of([S], date(2026, 1, 1))["value"].item() == 100.0
    assert store.as_of([S], date(2026, 1, 1))["value"].item() == 100.5
    with pytest.raises(ReadOnlyStoreError):
        pinned.write(make_obs_frame([(S, date(2020, 2, 1), utc(2020, 3, 11), 1.0)]))

    # Tampering is detected.
    part = data / snap.files[0].path
    part.write_bytes(part.read_bytes() + b"x")
    assert verify_snapshot(data, snap) == [f"modified: {snap.files[0].path}"]
    with pytest.raises(ValueError, match="not reproducible"):
        store_from_snapshot(data, snap)


def test_find_stale() -> None:
    now = datetime(2026, 10, 14, tzinfo=UTC)
    last = {"A": now - timedelta(days=2), "B": now - timedelta(days=60), "C": None}
    stale = find_stale(
        last, [("A", Frequency.DAILY), ("B", Frequency.MONTHLY), ("C", Frequency.WEEKLY)], now
    )
    assert [s.series_id for s in stale] == ["B", "C"]


def test_cli_catalog_and_pit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    PITStore(data).write(make_obs_frame([(S, date(2020, 1, 1), utc(2020, 2, 13), 100.0)]))
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    monkeypatch.delenv("ATLAS_FRED_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    get_settings.cache_clear()
    runner = CliRunner()
    try:
        res = runner.invoke(app, ["catalog", "list"])
        assert res.exit_code == 0, res.output
        assert "US.CPI.CORE" in res.output

        res = runner.invoke(app, ["pit", "as-of", S, "--at", "2020-03-01"])
        assert res.exit_code == 0, res.output
        assert "100.0" in res.output

        res = runner.invoke(app, ["pit", "history", S, "2020-01-01"])
        assert res.exit_code == 0, res.output

        res = runner.invoke(app, ["pit", "status"])
        assert res.exit_code == 0, res.output
        assert "STALE" in res.output

        res = runner.invoke(app, ["snapshot", "create"])
        assert res.exit_code == 0, res.output
        snap_id = res.output.split()[0]
        res = runner.invoke(app, ["snapshot", "verify", snap_id])
        assert res.exit_code == 0, res.output

        res = runner.invoke(
            app,
            ["ingest", "fred", "US.CPI.CORE"],
            env={"ATLAS_FRED_API_KEY": "", "FRED_API_KEY": ""},
        )
        assert res.exit_code != 0
    finally:
        get_settings.cache_clear()


def test_find_stale_allows_publication_lag() -> None:
    now = datetime(2026, 10, 19, tzinfo=UTC)
    last = {"USD": now - timedelta(days=9)}  # daily series, published weekly
    assert [s.series_id for s in find_stale(last, [("USD", Frequency.DAILY)], now)] == ["USD"]
    assert find_stale(last, [("USD", Frequency.DAILY, 7)], now) == []
