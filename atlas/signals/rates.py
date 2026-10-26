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
# Emerging markets with both an OECD 10-year and 3-month rate (India's 3-month rate starts in
# 2011). The duration-8 approximation overstates price sensitivity at high yields (about 6.5
# at 9%), so EM returns are approximate in level; the cross-sectional ranking is the target.
EM_AREAS: dict[str, str] = {
    "MEX": "Mexico",
    "KOR": "South Korea",
    "IND": "India",
    "CHN": "China",
    "POL": "Poland",
    "HUN": "Hungary",
    "CZE": "Czechia",
    "ZAF": "South Africa",
    "ISR": "Israel",
}
EM_RATES_NAMES = {k: f"{v} 10Y government bond" for k, v in EM_AREAS.items()}
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


def rates_panel(store: PITStore, areas: dict[str, str] | None = None) -> pl.DataFrame:
    ids = list(areas or AREAS)
    y10 = _levels(store, [f"{a}.RATES.10Y" for a in ids], "y10")
    if y10.is_empty():
        raise ValueError("no 10Y yields — run `atlas ingest sdmx --agency oecd`")
    y3m = _levels(store, [f"{a}.RATES.3M" for a in ids], "y3m")
    cli = _levels(store, [f"{a}.CLI" for a in ids], "cli")
    eq = _levels(store, [f"{a}.EQUITY.SHARE" for a in ids], "eq")
    grid = y10.select("date").unique().join(y10.select("asset").unique(), how="cross")
    df = (
        grid.join(y10, on=["date", "asset"], how="left")
        .join(y3m, on=["date", "asset"], how="left")
        .join(cli, on=["date", "asset"], how="left")
        .join(eq, on=["date", "asset"], how="left")
        .sort(["asset", "date"])
        .with_columns((pl.col("eq") / pl.col("eq").shift(12) - 1).over("asset").alias("eq_ret12"))
    )
    prev10, prev3 = pl.col("y10").shift(1), pl.col("y3m").shift(1)
    ret = (prev10 / 1200 - DURATION * (pl.col("y10") - prev10) / 100 - prev3 / 1200).over("asset")
    df = (
        df.with_columns(ret.alias("ret"))
        .with_columns(
            pl.col(c).shift(PUBLICATION_LAG_MONTHS).over("asset")
            for c in ("y10", "y3m", "cli", "eq_ret12")
        )
        .drop("eq")
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


def curve_flattening(months: int = 12) -> SignalDef:
    spread = pl.col("y10") - pl.col("y3m")
    expr = -(spread - spread.shift(months)).over("asset")
    return SignalDef(
        name=f"curve_flattening_{months}",
        family="curve_change",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"months": months},
        description=f"Flattening of the 10Y-3M curve over {months} months",
        reference="Cochrane & Piazzesi (2005) on the term structure of bond risk premia",
    )


def short_rate_fall(months: int = 12) -> SignalDef:
    expr = -(pl.col("y3m") - pl.col("y3m").shift(months)).over("asset")
    return SignalDef(
        name=f"short_rate_fall_{months}",
        family="policy_easing",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"months": months},
        description=f"Fall in the 3-month rate over {months} months (easing cycle)",
        reference="Ilmanen (2011), expected returns; bonds in easing cycles",
    )


def equity_weakness() -> SignalDef:
    return SignalDef(
        name="equity_weakness_12",
        family="flight_to_quality",
        build=lambda p: p.with_columns((-pl.col("eq_ret12")).alias("signal")),
        params={"months": 12},
        description="Weak local equity market over 12 months (flight to quality)",
        reference="Baele, Bekaert & Inghelbrecht (2010), stock-bond return comovement",
    )


RATES_SIGNALS: dict[str, SignalDef] = {
    s.name: s
    for s in (
        term_spread(),
        momentum(12, 1),
        yield_value(60),
        growth_slowdown(6),
        curve_flattening(12),
        short_rate_fall(12),
        equity_weakness(),
    )
}
RATES_FAMILIES: dict[str, list[SignalDef]] = {
    "carry": [term_spread(1), term_spread(3), term_spread(12)],
    "momentum": [momentum(lb, 1) for lb in (3, 6, 12)],
    "value": [yield_value(w) for w in (36, 60)],
    "growth": [growth_slowdown(m) for m in (3, 6, 12)],
    "curve_change": [curve_flattening(m) for m in (6, 12)],
    "policy_easing": [short_rate_fall(m) for m in (6, 12)],
    "flight_to_quality": [equity_weakness()],
}


def rates_em_panel(store: PITStore) -> pl.DataFrame:
    return rates_panel(store, EM_AREAS)


# Only 5 of the 9 EM areas have an OECD leading indicator (below the 6-asset minimum), so
# the growth-slowdown signal is left out of the EM set.
RATES_EM_SIGNALS: dict[str, SignalDef] = {
    n: sg for n, sg in RATES_SIGNALS.items() if sg.family != "growth"
}
RATES_EM_FAMILIES: dict[str, list[SignalDef]] = {
    k: v for k, v in RATES_FAMILIES.items() if k != "growth"
}
