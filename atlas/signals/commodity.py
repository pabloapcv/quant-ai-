"""Commodity price views on IMF primary commodity prices (PRD §3.1 commodities, Phase 5).

Panel: ``date`` (month start), ``asset`` (commodity id), ``ret`` (that month's change in the
USD benchmark price).

Data caveats, handled explicitly:

* IMF PCPS prices are **monthly averages** of benchmark spot prices, like the OECD share
  prices in ``atlas.signals.country``. Changes of averages are smoothed and positively
  autocorrelated at lag 1, so every direction signal here skips the latest month and there
  is no 1-month reversal or skip-0 momentum.
* They are **spot benchmark prices**, not futures returns: no roll yield, no collateral
  return. The views therefore rank *price changes* (what a buyer of the commodity pays),
  which is the right target for input-cost questions, not the return of an investable
  futures position. Futures-curve signals (basis, carry) need data we do not have.
* Market prices are not revised; the store stamps each month's average as known
  ``publication_lag_days`` (10) after the month ends.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from atlas.pit.store import PITStore
from atlas.signals.industry import SignalDef, long_term_reversal, low_volatility, momentum

COMMODITY_NAMES: dict[str, str] = {
    "OIL.BRENT": "Crude oil (Brent)",
    "GAS.US": "Natural gas (US)",
    "GAS.EU": "Natural gas (Europe)",
    "COAL.AU": "Coal",
    "COPPER": "Copper",
    "ALUMINUM": "Aluminum",
    "NICKEL": "Nickel",
    "ZINC": "Zinc",
    "IRONORE": "Iron ore",
    "GOLD": "Gold",
    "SILVER": "Silver",
    "WHEAT": "Wheat",
    "MAIZE": "Corn",
    "SOYBEANS": "Soybeans",
    "SUGAR": "Sugar",
    "COFFEE": "Coffee",
    "COTTON": "Cotton",
}
PREFIX = "CMDTY."


def commodity_panel(store: PITStore) -> pl.DataFrame:
    """(date x commodity) panel of monthly price changes, from the latest vintage."""
    ids = [PREFIX + a for a in COMMODITY_NAMES]
    px = store.as_of(ids, datetime.now(UTC)).select(
        pl.col("event_time").alias("date"),
        pl.col("series_id").str.strip_prefix(PREFIX).alias("asset"),
        pl.col("value").alias("px"),
    )
    if px.is_empty():
        raise ValueError("no commodity prices in the store — run `atlas ingest sdmx --agency imf`")
    return (
        px.sort(["asset", "date"])
        .with_columns((pl.col("px") / pl.col("px").shift(1) - 1).over("asset").alias("ret"))
        .drop("px")
        .filter(pl.col("ret").is_not_null())
    )


# Averaged prices: every momentum configuration skips at least one month.
COMMODITY_SIGNALS: dict[str, SignalDef] = {
    s.name: s
    for s in (
        momentum(12, 1),
        momentum(6, 1),
        momentum(3, 1),
        long_term_reversal(60, 12),
        low_volatility(36),
    )
}
COMMODITY_FAMILIES: dict[str, list[SignalDef]] = {
    "momentum": [momentum(lb, 1) for lb in (3, 6, 9, 12)],
    "long_term_reversal": [long_term_reversal(lb, 12) for lb in (36, 60)],
    "low_volatility": [low_volatility(w) for w in (24, 36, 60)],
}
# Ensemble components, chosen on the pre-2021 research window only (`atlas research gates
# --universe cmdty --exclude-holdout`, 2026-10-23), then tested once on the holdout:
# 5-year reversal, the commodity value proxy (Asness, Moskowitz & Pedersen 2013; t 2.89,
# positive IC in all four regimes) and low volatility (t 1.60, positive in all four).
# Momentum is left out: on monthly-average spot prices its research IC was negative (12-1
# t -0.22, 6-1 t -1.92), unlike the futures-return evidence (Miffre & Rallis 2007).
COMMODITY_COMPONENTS = ("ltr_60_12", "low_vol_36")
