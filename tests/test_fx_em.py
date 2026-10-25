from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from atlas.pit.store import PITStore
from atlas.signals.fx import EM_CURRENCIES, fx_em_panel
from atlas.signals.universes import get_universe
from tests.conftest import make_obs_frame, utc

KNOWN = utc(2026, 1, 1)
MONTHS = [date(2020, m, 1) for m in range(1, 7)]


def _store(path: Path) -> PITStore:
    rows = []
    for t, d in enumerate(MONTHS):
        rows.append(("FX.EURUSD.EOP", d, KNOWN, 1.10))
        rows.append(("USA.RATES.3M", d, KNOWN, 2.0))
        for cur, (_, area) in EM_CURRENCIES.items():
            per_eur = 20.0 * (1.02**t) if cur == "MXN" else 5.0  # the peso loses ~2%/month
            rows.append((f"FX.EUR{cur}.EOP", d, KNOWN, per_eur))
            rows.append((f"{area}.RATES.3M", d, KNOWN, 8.0 if cur == "MXN" else 2.0))
    store = PITStore(path)
    store.write(make_obs_frame(rows))
    return store


def test_em_panel_has_no_euro_and_earns_carry(tmp_path: Path) -> None:
    p = fx_em_panel(_store(tmp_path))
    assert set(p["asset"]) == set(EM_CURRENCIES)
    mxn = p.filter((pl.col("asset") == "MXN") & (pl.col("date") == MONTHS[3])).row(0, named=True)
    assert mxn["carry"] == pytest.approx(6.0)
    assert mxn["ret"] == pytest.approx(1 / 1.02 - 1 + 6.0 / 1200)


def test_em_universe() -> None:
    uni = get_universe("fx_em")
    assert uni.asset_name("ZAR") == "South African rand"
    assert "slope_diff" not in uni.signals
    assert "carry" in uni.signals
    assert uni.ensemble_components is not None
    assert set(uni.ensemble_components) <= set(uni.signals)
