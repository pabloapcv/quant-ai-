from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from atlas.models.timeseries import (
    SPREAD_DURATION,
    freeze_sleeves,
    sleeve_panel,
    stance_for,
    walk_forward,
)
from atlas.pit.store import PITStore
from tests.conftest import make_obs_frame


def _store(path: Path, months: int = 120) -> PITStore:
    rows = []
    for t in range(months):
        d = date(2010 + t // 12, t % 12 + 1, 28)
        known = datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(days=1)
        rows.append(("US.CREDIT.BAA10Y", d, known, 2.0 + 0.5 * ((t // 6) % 2)))  # 6m regimes
        rows.append(("CRYPTO.BTC", d, known, 100.0 * 1.01**t))
        rows.append(("US.RATES.FF", d, known, 1.2))
    store = PITStore(path)
    store.write(make_obs_frame(rows))
    return store


def test_panel_returns(tmp_path: Path) -> None:
    p = sleeve_panel(_store(tmp_path), at=datetime(2026, 1, 15, tzinfo=UTC))
    baa = p.filter(p["asset"] == "CREDIT.BAA").sort("date")
    r = baa.row(6, named=True)  # the spread widens 0.5pt into month 6
    assert r["ret"] == pytest.approx(2.0 / 1200 - SPREAD_DURATION * 0.5 / 100)
    btc = p.filter(p["asset"] == "CRYPTO.BTC").sort("date")
    assert btc.row(5, named=True)["ret"] == pytest.approx(0.01 - 1.2 / 1200)
    assert set(p["asset"]) == {"CREDIT.BAA", "CRYPTO.BTC"}  # no data for the other two


def test_running_month_is_excluded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    p = sleeve_panel(store, at=datetime(2019, 6, 15, tzinfo=UTC))  # knowledge ends mid-June
    assert p["date"].max() < date(2019, 6, 1)


def test_walk_forward_is_out_of_sample(tmp_path: Path) -> None:
    fc = walk_forward(
        sleeve_panel(_store(tmp_path, months=200), at=datetime(2027, 1, 15, tzinfo=UTC))
    )
    assert fc
    assert all(0.0 <= f.prob <= 1.0 for f in fc)
    first = min(f.date for f in fc if f.asset == "CREDIT.BAA")
    assert first >= date(2015, 1, 1)  # 12 months of trend + 5 years of value, then 60 to train
    btc = [f for f in fc if f.asset == "CRYPTO.BTC"]
    assert btc  # every past outcome positive: a smoothed base rate, not a skip
    assert all(0.5 < f.prob < 1.0 for f in btc)


def test_stances() -> None:
    assert stance_for(0.70) == "strong_overweight"
    assert stance_for(0.56) == "overweight"
    assert stance_for(0.50) == "neutral"
    assert stance_for(0.44) == "underweight"
    assert stance_for(0.30) == "strong_underweight"


def test_freeze_is_write_once(tmp_path: Path) -> None:
    store = _store(tmp_path / "s", months=200)
    ok, msg = freeze_sleeves(tmp_path, store)
    assert ok
    assert "frozen ->" in msg
    path = next((tmp_path / "gold" / "sleeves" / "3M").glob("*.json"))
    first = json.loads(path.read_text())
    assert {v["asset_id"] for v in first} == {"CREDIT.BAA", "CRYPTO.BTC"}
    assert "frozen earlier" in freeze_sleeves(tmp_path, store)[1]
