from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from atlas.pit.store import PITStore
from atlas.signals.rates import AREAS, DURATION, rates_panel
from atlas.signals.universes import get_universe
from tests.conftest import make_obs_frame, utc

KNOWN = utc(2026, 1, 1)
MONTHS = [date(2020 + t // 12, t % 12 + 1, 1) for t in range(6)]


def _store(path: Path) -> PITStore:
    rows = []
    for t, d in enumerate(MONTHS):
        for a in AREAS:
            y10 = 3.0 + (0.1 * t if a == "USA" else 0.0)  # US yields rise 10bp a month
            rows += [(f"{a}.RATES.10Y", d, KNOWN, y10), (f"{a}.RATES.3M", d, KNOWN, 1.0)]
    store = PITStore(path)
    store.write(make_obs_frame(rows))
    return store


def test_bond_return_from_yields(tmp_path: Path) -> None:
    p = rates_panel(_store(tmp_path))
    usa = p.filter((pl.col("asset") == "USA") & (pl.col("date") == MONTHS[2])).row(0, named=True)
    # carry 3.1%/12 minus duration x 10bp minus cash 1%/12
    assert usa["ret"] == pytest.approx(0.031 / 12 - DURATION * 0.001 - 0.01 / 12)
    deu = p.filter((pl.col("asset") == "DEU") & (pl.col("date") == MONTHS[2])).row(0, named=True)
    assert deu["ret"] == pytest.approx((0.03 - 0.01) / 12)  # unchanged yields: pure carry
    assert usa["y10"] == pytest.approx(3.1)  # signal inputs lag a month (publication)


def test_rates_universe(tmp_path: Path) -> None:
    uni = get_universe("rates_dev")
    assert uni.asset_class == "rates"
    sig = uni.signals["term_spread"].compute(rates_panel(_store(tmp_path)))
    last = sig.filter(pl.col("date") == MONTHS[-1]).sort("signal")
    assert last["asset"][-1] == "USA"  # steepest curve ranks first
