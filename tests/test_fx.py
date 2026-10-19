from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from atlas.pit.store import PITStore
from atlas.signals.fx import CURRENCIES, fx_panel
from atlas.signals.universes import get_universe
from tests.conftest import make_obs_frame, utc

KNOWN = utc(2026, 1, 1)
MONTHS = [date(2020 + t // 12, t % 12 + 1, 1) for t in range(8)]


def _store(path: Path, *, drop_gbp_rates_after: int | None = None) -> PITStore:
    rows = []
    for t, d in enumerate(MONTHS):
        rows.append(("FX.EURUSD.EOP", d, KNOWN, 1.10 * (1.01**t)))  # euro +1%/month vs USD
        for cur in CURRENCIES:
            if cur != "EUR":
                rows.append((f"FX.EUR{cur}.EOP", d, KNOWN, 2.0))  # constant per euro
        rows.append(("USA.RATES.3M", d, KNOWN, 2.0))
        for _, area in CURRENCIES.values():
            if area == "GBR" and drop_gbp_rates_after is not None and t > drop_gbp_rates_after:
                continue
            rows.append((f"{area}.RATES.3M", d, KNOWN, 2.0 + (6.0 if area == "AUS" else 0.0)))
    store = PITStore(path)
    store.write(make_obs_frame(rows))
    return store


def test_panel_returns_spot_plus_lagged_carry(tmp_path: Path) -> None:
    p = fx_panel(_store(tmp_path))
    assert set(p["asset"]) == set(CURRENCIES)

    def row(asset: str, d: date) -> dict[str, Any]:
        return p.filter((pl.col("asset") == asset) & (pl.col("date") == d)).row(0, named=True)

    # Every currency is constant per euro and the euro gains 1%/month: all spots +1%.
    eur = row("EUR", MONTHS[3])
    assert eur["ret"] == pytest.approx(0.01)
    assert eur["carry"] == pytest.approx(0.0)
    aud = row("AUD", MONTHS[3])
    assert aud["carry"] == pytest.approx(6.0)  # +6pp vs USD, known with a month's lag
    assert aud["ret"] == pytest.approx(0.01 + 6.0 / 1200)
    first = row("AUD", MONTHS[1])  # month 0's rate is the signal; no carry earned yet
    assert first["carry"] == pytest.approx(6.0)
    assert first["ret"] == pytest.approx(0.01)


def test_ragged_rates_forward_filled_three_months(tmp_path: Path) -> None:
    p = fx_panel(_store(tmp_path, drop_gbp_rates_after=2))
    gbp = p.filter(pl.col("asset") == "GBP").sort("date")["carry"].to_list()
    # rates exist for months 0-2; signal lags a month; three months carried forward.
    assert gbp[1:6] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert gbp[6] is None


def test_universe_registered(tmp_path: Path) -> None:
    uni = get_universe("fx_g10")
    assert uni.asset_class == "fx"
    assert {"carry", "mom_12_1"} <= set(uni.signals)
    assert uni.asset_name("JPY") == "Japanese yen"
    sig = uni.signals["carry"].compute(fx_panel(_store(tmp_path)))
    last = sig.filter(pl.col("date") == MONTHS[-2]).sort("signal")
    assert last["asset"][-1] == "AUD"  # highest carry ranks first
