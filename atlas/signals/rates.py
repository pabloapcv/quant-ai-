"""Government bond (duration) allocation across 9 developed markets (PRD §3.1 rates, Phase 5).

Panel: ``date`` (month start), ``asset`` (ISO3 area), ``ret`` (that month's 10-year bond
return in excess of local cash, local currency, i.e. currency-hedged), and ``y10``, ``y3m``,
``cli`` *as known at the end of the month* (lagged a month: OECD publishes about three weeks
after month end).

Returns are approximated from yields, the standard constant-maturity approach:
``ret_t = y10_{t-1}/12 - D * (y10_t - y10_{t-1}) - y3m_{t-1}/12`` (yields in decimals),
with modified duration ``D = 8`` for a 10-year par bond. OECD yields are monthly averages,
so returns are smoothed like the country equity series: momentum skips the latest month.
Germany stands in for the euro area.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from atlas.pit.store import PITStore
from atlas.signals.industry import SignalDef, momentum

AREAS: dict[str, str] = {
    "USA": "United States",
    "DEU": "Germany",
    "JPN": "Japan",
    "GBR": "United Kingdom",
    "CHE": "Switzerland",
    "AUS": "Australia",
    "CAN": "Canada",
    "SWE": "Sweden",
    "NOR": "Norway",
}
RATES_NAMES = {k: f"{v} 10Y government bond" for k, v in AREAS.items()}
DURATION = 8.0
PUBLICATION_LAG_MONTHS = 1


def _levels(store: PITStore, ids: list[str], col: str) -> pl.DataFrame:
    return (
        store.as_of(ids, datetime.now(UTC))
        .select(
            pl.col("event_time").alias("date"),
            pl.col("series_id").str.split(".").list.first().alias("asset"),
            pl.col("value").alias(col),
        )
        .unique(["date", "asset"], keep="last")
    )


def rates_panel(store: PITStore) -> pl.DataFrame:
    ids = list(AREAS)
    y10 = _levels(store, [f"{a}.RATES.10Y" for a in ids], "y10")
    if y10.is_empty():
        raise ValueError("no 10Y yields — run `atlas ingest sdmx --agency oecd`")
    y3m = _levels(store, [f"{a}.RATES.3M" for a in ids], "y3m")
    cli = _levels(store, [f"{a}.CLI" for a in ids], "cli")
    grid = y10.select("date").unique().join(y10.select("asset").unique(), how="cross")
    df = (
        grid.join(y10, on=["date", "asset"], how="left")
        .join(y3m, on=["date", "asset"], how="left")
        .join(cli, on=["date", "asset"], how="left")
        .sort(["asset", "date"])
    )
    prev10, prev3 = pl.col("y10").shift(1), pl.col("y3m").shift(1)
    ret = (prev10 / 1200 - DURATION * (pl.col("y10") - prev10) / 100 - prev3 / 1200).over("asset")
    df = df.with_columns(ret.alias("ret")).with_columns(
        pl.col(c).shift(PUBLICATION_LAG_MONTHS).over("asset") for c in ("y10", "y3m", "cli")
    )
    return df.filter(pl.col("ret").is_not_null() & pl.col("ret").is_finite())


# ---------------------------------------------------------------------------- signals
CARRY_REF = "Koijen, Moskowitz, Pedersen & Vrugt (2018); Ilmanen (1995) on bond risk premia"


def term_spread(smooth: int = 1) -> SignalDef:
    spread = pl.col("y10") - pl.col("y3m")
    expr = spread if smooth == 1 else spread.rolling_mean(smooth, min_samples=smooth).over("asset")
    return SignalDef(
        name="term_spread" if smooth == 1 else f"term_spread_avg_{smooth}",
        family="carry",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"smooth": smooth},
        description="10-year yield minus the 3-month rate (bond carry)"
        + ("" if smooth == 1 else f", {smooth}-month average"),
        reference=CARRY_REF,
    )


def yield_value(window: int = 60) -> SignalDef:
    expr = (pl.col("y10") - pl.col("y10").rolling_mean(window, min_samples=window * 2 // 3)).over(
        "asset"
    )
    return SignalDef(
        name=f"yield_vs_{window}m",
        family="value",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"window": window},
        description=f"10-year yield above its own {window}-month average (cheap bonds)",
        reference="Asness, Moskowitz & Pedersen (2013), value in government bonds",
    )


def growth_slowdown(months: int = 6) -> SignalDef:
    expr = -(pl.col("cli") - pl.col("cli").shift(months)).over("asset")
    return SignalDef(
        name=f"cli_fall_{months}",
        family="growth",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"months": months},
        description=f"Fall in the OECD leading indicator over {months} months (bonds rally)",
        reference="Ilmanen (2003); bond returns and the business cycle",
    )


RATES_SIGNALS: dict[str, SignalDef] = {
    s.name: s for s in (term_spread(), momentum(12, 1), yield_value(60), growth_slowdown(6))
}
RATES_FAMILIES: dict[str, list[SignalDef]] = {
    "carry": [term_spread(1), term_spread(3), term_spread(12)],
    "momentum": [momentum(lb, 1) for lb in (3, 6, 12)],
    "value": [yield_value(w) for w in (36, 60)],
    "growth": [growth_slowdown(m) for m in (3, 6, 12)],
}
