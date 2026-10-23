from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from atlas.pit.store import PITStore
from atlas.signals.commodity import COMMODITY_COMPONENTS, COMMODITY_NAMES, commodity_panel
from atlas.signals.universes import get_universe
from tests.conftest import make_obs_frame, utc

KNOWN = utc(2026, 1, 1)
MONTHS = [date(2020, m, 1) for m in range(1, 5)]


def _store(path: Path) -> PITStore:
    rows = []
    for t, d in enumerate(MONTHS):
        for i, a in enumerate(COMMODITY_NAMES):
            rows.append((f"CMDTY.{a}", d, KNOWN, 100.0 * (1 + 0.01 * i) ** t))
    store = PITStore(path)
    store.write(make_obs_frame(rows))
    return store


def test_price_changes(tmp_path: Path) -> None:
    p = commodity_panel(_store(tmp_path))
    assert p["asset"].n_unique() == len(COMMODITY_NAMES)
    assert p["date"].min() == MONTHS[1]  # the first month has no change
    copper = p.filter(pl.col("asset") == "COPPER")["ret"].to_list()
    i = list(COMMODITY_NAMES).index("COPPER")
    assert copper == pytest.approx([0.01 * i] * 3)


def test_commodity_universe() -> None:
    uni = get_universe("cmdty")
    assert uni.asset_name("OIL.BRENT") == "Crude oil (Brent)"
    assert set(COMMODITY_COMPONENTS) <= set(uni.signals)
    # monthly-average prices: no momentum that uses the latest month
    assert all(not n.startswith("mom_") or not n.endswith("_0") for n in uni.signals)
    for members in uni.families.values():
        assert all(s.params.get("skip", 1) >= 1 for s in members)


def test_missing_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ingest sdmx"):
        commodity_panel(PITStore(tmp_path))
