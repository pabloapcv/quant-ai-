"""G10 currency allocation vs the US dollar (PRD §3.1 FX, Phase 5).

Panel: ``date`` (month start), ``asset`` (currency code), ``ret`` (that month's excess
return of holding the currency, in USD) and ``carry`` (annualized short-rate differential vs
the US dollar, percentage points, as known at the end of the month).

* Spot: ECB end-of-month euro reference rates, crossed into USD per unit of currency, from
  1999 (the euro's launch). End-of-month closes, so 1-month signals are allowed, unlike the
  monthly-average OECD equity prices.
* Return of a long position held through month ``t``: spot change over ``t`` plus the
  interest differential earned, ``(r_local - r_usd) / 12``, at the 3-month rates of ``t-1``
  (the rate locked in when the position is opened).
* Short rates are OECD 3-month interbank monthly averages, published about three weeks
  after month end, so the carry *signal* at the end of ``t`` uses ``t-1`` rates. German
  rates stand in for the euro area (Euribor and German bills track closely since 1999).
  A rate missing for up to three months is carried forward; longer gaps stay missing (the
  currency then has no carry signal, and its return omits the differential).
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from atlas.pit.store import PITStore
from atlas.signals.industry import SignalDef, long_term_reversal, momentum

CURRENCIES: dict[str, tuple[str, str]] = {
    "EUR": ("Euro", "DEU"),
    "JPY": ("Japanese yen", "JPN"),
    "GBP": ("British pound", "GBR"),
    "CHF": ("Swiss franc", "CHE"),
    "AUD": ("Australian dollar", "AUS"),
    "CAD": ("Canadian dollar", "CAN"),
    "SEK": ("Swedish krona", "SWE"),
    "NOK": ("Norwegian krone", "NOR"),
}
FX_NAMES = {k: v[0] for k, v in CURRENCIES.items()}
# Emerging-market currencies with both an ECB euro reference rate and an OECD 3-month rate.
# India's rate starts in 2011 (no carry before); Indonesia's stops at 2025-12, so it is left out.
EM_CURRENCIES: dict[str, tuple[str, str]] = {
    "MXN": ("Mexican peso", "MEX"),
    "KRW": ("South Korean won", "KOR"),
    "PLN": ("Polish zloty", "POL"),
    "HUF": ("Hungarian forint", "HUN"),
    "CZK": ("Czech koruna", "CZE"),
    "ZAR": ("South African rand", "ZAF"),
    "CNY": ("Chinese yuan", "CHN"),
    "ILS": ("Israeli shekel", "ISR"),
    "INR": ("Indian rupee", "IND"),
}
EM_NAMES = {k: v[0] for k, v in EM_CURRENCIES.items()}
RATE_LAG_MONTHS = 1
RATE_FFILL_MONTHS = 3  # ragged edge: a rate series may lag (the UK's stops early 2026)


def _levels(store: PITStore, ids: list[str]) -> pl.DataFrame:
    return store.as_of(ids, datetime.now(UTC)).select("series_id", "event_time", "value")


def _usd_per_unit(
    store: PITStore, currencies: dict[str, tuple[str, str]] = CURRENCIES
) -> pl.DataFrame:
    """date, asset, usd_per_unit at month end, via ECB euro crosses."""
    others = [c for c in currencies if c != "EUR"]
    fx = _levels(store, [f"FX.EUR{c}.EOP" for c in ["USD", *others]])
    wide = fx.pivot(index="event_time", on="series_id", values="value").sort("event_time")
    if "FX.EURUSD.EOP" not in wide.columns:
        raise ValueError("FX.EURUSD.EOP missing — run `atlas ingest sdmx --agency ecb`")
    usd_per_eur = pl.col("FX.EURUSD.EOP")
    frames = []
    if "EUR" in currencies:
        frames.append(
            wide.select(
                pl.col("event_time").alias("date"),
                pl.lit("EUR").alias("asset"),
                usd_per_eur.alias("px"),
            )
        )
    for c in others:
        col = f"FX.EUR{c}.EOP"
        if col in wide.columns:
            frames.append(
                wide.select(
                    pl.col("event_time").alias("date"),
                    pl.lit(c).alias("asset"),
                    (usd_per_eur / pl.col(col)).alias("px"),
                )
            )
    return pl.concat(frames).drop_nulls("px")


def _short_rates(
    store: PITStore, currencies: dict[str, tuple[str, str]] = CURRENCIES
) -> pl.DataFrame:
    """date, asset, rate (local 3-month rate minus the US one, pp) for month ``date``."""
    areas = {area: cur for cur, (_, area) in currencies.items()}
    raw = _levels(store, [f"{a}.RATES.3M" for a in [*areas, "USA"]]).select(
        pl.col("event_time").alias("date"),
        pl.col("series_id").str.split(".").list.first().alias("area"),
        pl.col("value").alias("rate"),
    )
    us = raw.filter(pl.col("area") == "USA").select("date", pl.col("rate").alias("us"))
    return (
        raw.filter(pl.col("area") != "USA")
        .join(us, on="date", how="inner")
        .with_columns(
            pl.col("area").replace_strict(areas).alias("asset"),
            (pl.col("rate") - pl.col("us")).alias("diff"),
        )
        .select("date", "asset", "diff")
    )


def fx_panel(store: PITStore, currencies: dict[str, tuple[str, str]] = CURRENCIES) -> pl.DataFrame:
    """Gap-free (date x currency) panel: excess return in USD, carry known at month end."""
    px = _usd_per_unit(store, currencies)
    if px.is_empty():
        raise ValueError("no ECB month-end FX rates — run `atlas ingest sdmx --agency ecb`")
    rates = _short_rates(store, currencies)
    grid = px.select("date").unique().join(px.select("asset").unique(), how="cross")
    df = (
        grid.join(px, on=["date", "asset"], how="left")
        .join(rates, on=["date", "asset"], how="left")
        .sort(["asset", "date"])
        .with_columns(
            pl.col("diff")
            .forward_fill(limit=RATE_FFILL_MONTHS)
            .shift(RATE_LAG_MONTHS)
            .over("asset")
            .alias("carry")
        )
        .with_columns(
            (
                pl.col("px") / pl.col("px").shift(1)
                - 1
                + pl.col("carry").shift(1).fill_null(0.0) / 1200
            )
            .over("asset")
            .alias("ret")
        )
    )
    df = df.join(_slope_and_equity(store, currencies), on=["date", "asset"], how="left")
    return df.drop("px", "diff").filter(pl.col("ret").is_not_null())


def _slope_and_equity(
    store: PITStore, currencies: dict[str, tuple[str, str]] = CURRENCIES
) -> pl.DataFrame:
    """date, asset, slope_diff, eq_mom: relative curve slope and local equity momentum.

    Both from OECD monthly data, lagged a month for publication like the rates. Equity
    momentum is 12-1 on monthly-average share prices (skipping the latest month); markets
    without an OECD share index (Switzerland, Sweden, Norway) have none.
    """
    areas = {area: cur for cur, (_, area) in currencies.items()}
    ids = [*areas, "USA"]
    empty = pl.DataFrame(
        schema={"date": pl.Date, "asset": pl.String, "slope_diff": pl.Float64, "eq_mom": pl.Float64}
    )
    raw = _levels(store, [f"{a}.RATES.{t}" for a in ids for t in ("10Y", "3M")])
    if raw.is_empty():
        return empty
    raw = raw.with_columns(
        pl.col("series_id").str.split(".").list.first().alias("area"),
        pl.col("series_id").str.split(".").list.last().alias("tenor"),
    )
    wide = raw.pivot(
        index=["event_time", "area"], on="tenor", values="value", aggregate_function="last"
    )
    if not {"10Y", "3M"} <= set(wide.columns):
        return empty
    wide = wide.with_columns((pl.col("10Y") - pl.col("3M")).alias("slope"))
    us = wide.filter(pl.col("area") == "USA").select("event_time", pl.col("slope").alias("us"))
    slope = (
        wide.filter(pl.col("area") != "USA")
        .join(us, on="event_time", how="inner")
        .select(
            pl.col("event_time").alias("date"),
            pl.col("area").replace_strict(areas).alias("asset"),
            (pl.col("slope") - pl.col("us")).alias("slope_diff"),
        )
    )
    eq = (
        _levels(store, [f"{a}.EQUITY.SHARE" for a in areas])
        .select(
            pl.col("event_time").alias("date"),
            pl.col("series_id")
            .str.split(".")
            .list.first()
            .replace_strict(areas, default=None)
            .alias("asset"),
            pl.col("value").alias("eq"),
        )
        .drop_nulls("asset")
    )
    out = slope.join(eq, on=["date", "asset"], how="full", coalesce=True).sort(["asset", "date"])
    lag = RATE_LAG_MONTHS
    return out.with_columns(
        pl.col("slope_diff").shift(lag).over("asset"),
        (pl.col("eq").shift(lag + 1) / pl.col("eq").shift(lag + 12) - 1)
        .over("asset")
        .alias("eq_mom"),
    ).select("date", "asset", "slope_diff", "eq_mom")


# ---------------------------------------------------------------------------- signals
CARRY_REF = "Lustig, Roussanov & Verdelhan (2011); Koijen, Moskowitz, Pedersen & Vrugt (2018)"


def carry(smooth: int = 1) -> SignalDef:
    expr = (
        pl.col("carry")
        if smooth == 1
        else pl.col("carry").rolling_mean(smooth, min_samples=smooth).over("asset")
    )
    return SignalDef(
        name="carry" if smooth == 1 else f"carry_avg_{smooth}",
        family="carry",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"smooth": smooth},
        description="Short-rate differential vs the US dollar"
        + ("" if smooth == 1 else f", {smooth}-month average"),
        reference=CARRY_REF,
    )


def carry_change(months: int = 12) -> SignalDef:
    expr = (pl.col("carry") - pl.col("carry").shift(months)).over("asset")
    return SignalDef(
        name=f"carry_chg_{months}",
        family="carry_change",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"months": months},
        description=f"Change in the rate differential vs the US over {months} months",
        reference="Bakshi & Panayotov (2013); rising relative rates attract capital",
    )


def slope_difference() -> SignalDef:
    return SignalDef(
        name="slope_diff",
        family="curve_slope",
        build=lambda p: p.with_columns(pl.col("slope_diff").alias("signal")),
        params={"slope": 1},
        description="Local 10Y-3M slope minus the US one (steeper relative curve)",
        reference="Ang & Chen (2010), yield curve predictors of foreign exchange returns",
    )


def equity_momentum() -> SignalDef:
    """Not in the FX signal set: only 5 of the 8 currencies have an OECD share index, below
    the backtester's 6-asset minimum for a cross-sectional ranking."""
    return SignalDef(
        name="equity_mom_12_1",
        family="cross_asset_momentum",
        build=lambda p: p.with_columns(pl.col("eq_mom").alias("signal")),
        params={"lookback": 12, "skip": 1},
        description="12-1 momentum of the local equity market (capital flows into winners)",
        reference="Hau & Rey (2006), exchange rates, equity prices and capital flows",
    )


FX_SIGNALS: dict[str, SignalDef] = {
    s.name: s
    for s in (
        carry(),
        momentum(12, 1),
        momentum(3, 0),
        long_term_reversal(60, 12),
        carry_change(12),
        slope_difference(),
    )
}
FX_FAMILIES: dict[str, list[SignalDef]] = {
    "carry": [carry(1), carry(3), carry(12)],
    "momentum": [momentum(lb, skip) for lb, skip in ((1, 0), (3, 0), (6, 0), (12, 1), (12, 0))],
    "long_term_reversal": [long_term_reversal(lb, 12) for lb in (36, 60)],
    "carry_change": [carry_change(m) for m in (6, 12)],
    "curve_slope": [slope_difference()],
}


def fx_em_panel(store: PITStore) -> pl.DataFrame:
    return fx_panel(store, EM_CURRENCIES)


# Emerging markets: the G10 definitions, minus the curve slope (few EM 10-year series). The
# ensemble components are chosen on the pre-2021 research window (see universes.py).
FX_EM_SIGNALS: dict[str, SignalDef] = {n: s for n, s in FX_SIGNALS.items() if n != "slope_diff"}
FX_EM_FAMILIES: dict[str, list[SignalDef]] = {
    k: v for k, v in FX_FAMILIES.items() if k != "curve_slope"
}
