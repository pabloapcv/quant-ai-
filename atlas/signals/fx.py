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
RATE_LAG_MONTHS = 1
RATE_FFILL_MONTHS = 3  # ragged edge: a rate series may lag (the UK's stops early 2026)


def _levels(store: PITStore, ids: list[str]) -> pl.DataFrame:
    return store.as_of(ids, datetime.now(UTC)).select("series_id", "event_time", "value")


def _usd_per_unit(store: PITStore) -> pl.DataFrame:
    """date, asset, usd_per_unit at month end, via ECB euro crosses."""
    others = [c for c in CURRENCIES if c != "EUR"]
    fx = _levels(store, [f"FX.EUR{c}.EOP" for c in ["USD", *others]])
    wide = fx.pivot(index="event_time", on="series_id", values="value").sort("event_time")
    if "FX.EURUSD.EOP" not in wide.columns:
        raise ValueError("FX.EURUSD.EOP missing — run `atlas ingest sdmx --agency ecb`")
    usd_per_eur = pl.col("FX.EURUSD.EOP")
    frames = [
        wide.select(
            pl.col("event_time").alias("date"),
            pl.lit("EUR").alias("asset"),
            usd_per_eur.alias("px"),
        )
    ]
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


def _short_rates(store: PITStore) -> pl.DataFrame:
    """date, asset, rate (local 3-month rate minus the US one, pp) for month ``date``."""
    areas = {area: cur for cur, (_, area) in CURRENCIES.items()}
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


def fx_panel(store: PITStore) -> pl.DataFrame:
    """Gap-free (date x currency) panel: excess return in USD, carry known at month end."""
    px = _usd_per_unit(store)
    if px.is_empty():
        raise ValueError("no ECB month-end FX rates — run `atlas ingest sdmx --agency ecb`")
    rates = _short_rates(store)
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
    return df.drop("px", "diff").filter(pl.col("ret").is_not_null())


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


FX_SIGNALS: dict[str, SignalDef] = {
    s.name: s for s in (carry(), momentum(12, 1), momentum(3, 0), long_term_reversal(60, 12))
}
FX_FAMILIES: dict[str, list[SignalDef]] = {
    "carry": [carry(1), carry(3), carry(12)],
    "momentum": [momentum(lb, skip) for lb, skip in ((1, 0), (3, 0), (6, 0), (12, 1), (12, 0))],
    "long_term_reversal": [long_term_reversal(lb, 12) for lb in (36, 60)],
}
