"""Country equity allocation signals on OECD/ECB data (PRD §6.1-6.3, Phase 5).

Panel: ``date`` (month start), ``asset`` (ISO3 country), ``ret`` (that month's return),
plus ``cli`` and ``y10`` *as known at the end of the month*.

Data caveats, handled explicitly:

* OECD share prices are **monthly averages**, not month-end closes. Returns of averages are
  smoothed and positively autocorrelated at lag 1, which manufactures fake 1-month momentum.
  Every return-based signal here therefore skips the latest month, and there is no 1-month
  reversal signal.
* Returns are **price only** (no dividends) — fine for cross-country *ranking*, which is
  what the signals do, but understated in level.
* ``ctry_usd`` converts with ECB end-of-month rates (from 1999, the euro's launch), all 14
  markets; ``ctry_dev_lcl`` is local currency from the 1950s for 9 developed markets only
  (see ``DEVELOPED``).
* CLI and 10Y yields for month ``m`` are published about two weeks after ``m`` ends, so at
  the end of month ``t`` the latest usable values are for ``t-1``: both are lagged a month.
  CLIs are heavily revised; their pre-vintage history is approx (see ``atlas.connectors.sdmx``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from atlas.pit.store import PITStore
from atlas.signals.industry import SignalDef, long_term_reversal, low_volatility, momentum
from atlas.signals.library import GENERIC_FAMILIES, GENERIC_SIGNALS

COUNTRIES: dict[str, tuple[str, str]] = {
    "USA": ("United States", "USD"),
    "DEU": ("Germany", "EUR"),
    "JPN": ("Japan", "JPY"),
    "GBR": ("United Kingdom", "GBP"),
    "FRA": ("France", "EUR"),
    "ITA": ("Italy", "EUR"),
    "CAN": ("Canada", "CAD"),
    "CHN": ("China", "CNY"),
    "IND": ("India", "INR"),
    "BRA": ("Brazil", "BRL"),
    "KOR": ("Korea", "KRW"),
    "AUS": ("Australia", "AUD"),
    "MEX": ("Mexico", "MXN"),
    "ESP": ("Spain", "EUR"),
}
# The wide universe (ctry_world): OECD share prices since 1999 converted with ECB month-end
# rates. Chile and Colombia have OECD prices but no ECB rate, so they are left out. Most of
# these have no OECD leading indicator any more (only IDN, TUR, ZAF): their CLI signals are
# simply missing and the ensemble uses the components that exist.
EXTRA_COUNTRIES: dict[str, tuple[str, str]] = {
    "NLD": ("Netherlands", "EUR"),
    "BEL": ("Belgium", "EUR"),
    "AUT": ("Austria", "EUR"),
    "FIN": ("Finland", "EUR"),
    "IRL": ("Ireland", "EUR"),
    "PRT": ("Portugal", "EUR"),
    "GRC": ("Greece", "EUR"),
    "CHE": ("Switzerland", "CHF"),
    "SWE": ("Sweden", "SEK"),
    "NOR": ("Norway", "NOK"),
    "DNK": ("Denmark", "DKK"),
    "NZL": ("New Zealand", "NZD"),
    "POL": ("Poland", "PLN"),
    "HUN": ("Hungary", "HUF"),
    "CZE": ("Czechia", "CZK"),
    "ZAF": ("South Africa", "ZAR"),
    "ISR": ("Israel", "ILS"),
    "TUR": ("Turkey", "TRY"),
    "IDN": ("Indonesia", "IDR"),
}
ALL_COUNTRIES = {**COUNTRIES, **EXTRA_COUNTRIES}
WORLD = tuple(ALL_COUNTRIES)
COUNTRY_NAMES = {k: v[0] for k, v in ALL_COUNTRIES.items()}
# Local-currency nominal returns are only comparable across low-inflation economies: with
# Brazil or Mexico in the set, decades of high inflation dominate the cross-section (they
# look like winners, and inflation persistence fakes momentum). The local-currency universe
# is therefore limited to developed markets; the USD universe keeps all 14.
DEVELOPED = ("USA", "DEU", "JPN", "GBR", "FRA", "ITA", "CAN", "AUS", "ESP")
PUBLICATION_LAG_MONTHS = 1


def _levels(store: PITStore, ids: list[str]) -> pl.DataFrame:
    if not ids:
        return pl.DataFrame(
            schema={"series_id": pl.String, "event_time": pl.Date, "value": pl.Float64}
        )
    return store.as_of(ids, datetime.now(UTC)).select("series_id", "event_time", "value")


def _usd_per_local(store: PITStore, countries: tuple[str, ...] | None = None) -> pl.DataFrame:
    """date, currency, usd_per_local (end of month) via EUR crosses."""
    ids = countries or tuple(COUNTRIES)
    curs = sorted({ALL_COUNTRIES[c][1] for c in ids} - {"EUR", "USD"})
    fx = _levels(store, [f"FX.EUR{c}.EOP" for c in ["USD", *curs]])
    wide = fx.pivot(index="event_time", on="series_id", values="value").sort("event_time")
    if "FX.EURUSD.EOP" not in wide.columns:
        raise ValueError("FX.EURUSD.EOP missing — run `atlas ingest sdmx --agency ecb`")
    usd_per_eur = pl.col("FX.EURUSD.EOP")
    frames = [
        wide.select(
            pl.col("event_time").alias("date"),
            pl.lit("EUR").alias("currency"),
            usd_per_eur.alias("usd_per_local"),
        ),
        wide.select(
            pl.col("event_time").alias("date"),
            pl.lit("USD").alias("currency"),
            pl.lit(1.0).alias("usd_per_local"),
        ),
    ]
    for c in curs:
        col = f"FX.EUR{c}.EOP"
        if col in wide.columns:
            frames.append(
                wide.select(
                    pl.col("event_time").alias("date"),
                    pl.lit(c).alias("currency"),
                    (usd_per_eur / pl.col(col)).alias("usd_per_local"),
                )
            )
    return pl.concat(frames).drop_nulls("usd_per_local")


def country_panel(
    store: PITStore, *, usd: bool = True, countries: tuple[str, ...] | None = None
) -> pl.DataFrame:
    """Gap-free (date x country) panel: ret (USD or local), cli and y10 as known at month end."""
    ids = list(countries or COUNTRIES)
    share = _levels(store, [f"{c}.EQUITY.SHARE" for c in ids])
    if share.is_empty():
        raise ValueError("no country share prices — run `atlas ingest sdmx --agency oecd`")
    df = share.select(
        pl.col("event_time").alias("date"),
        pl.col("series_id").str.split(".").list.first().alias("asset"),
        pl.col("value").alias("level"),
    )
    grid = df.select("date").unique().join(df.select("asset").unique(), how="cross")
    df = (
        grid.join(df, on=["date", "asset"], how="left")
        .sort(["asset", "date"])
        .with_columns(
            (pl.col("level") / pl.col("level").shift(1) - 1).over("asset").alias("ret_lcl")
        )
    )
    if usd:
        cur = pl.DataFrame({"asset": ids, "currency": [ALL_COUNTRIES[c][1] for c in ids]})
        fx = _usd_per_local(store, tuple(ids))
        df = (
            df.join(cur, on="asset", how="left")
            .join(fx, on=["date", "currency"], how="left")
            .sort(["asset", "date"])
            .with_columns(
                (
                    (1 + pl.col("ret_lcl"))
                    * (pl.col("usd_per_local") / pl.col("usd_per_local").shift(1))
                    - 1
                )
                .over("asset")
                .alias("ret"),
                (pl.col("usd_per_local") / pl.col("usd_per_local").shift(1) - 1)
                .over("asset")
                .alias("fx_ret"),
            )
            .drop("currency", "usd_per_local")
        )
    else:
        df = df.with_columns(pl.col("ret_lcl").alias("ret"))

    for col, suffix in (("cli", "CLI"), ("y10", "RATES.10Y"), ("y3m", "RATES.3M")):
        raw = _levels(store, [f"{c}.{suffix}" for c in ids]).select(
            pl.col("event_time").alias("date"),
            pl.col("series_id").str.split(".").list.first().alias("asset"),
            pl.col("value").alias(col),
        )
        df = (
            df.join(raw, on=["date", "asset"], how="left")
            .sort(["asset", "date"])
            .with_columns(
                pl.col(col).shift(PUBLICATION_LAG_MONTHS).over("asset")  # month t-1, known at t
            )
        )
    # index levels rounded to 0 in early hyperinflation years give inf/NaN returns
    df = df.with_columns(
        pl.when(pl.col("ret").is_finite()).then(pl.col("ret")).otherwise(None).alias("ret")
    )
    return df.drop("level", "ret_lcl").filter(
        pl.col("ret").is_not_null() | pl.col("cli").is_not_null()
    )


# ---------------------------------------------------------------------------- signals
CLI_REF = "OECD CLI methodology; Stock & Watson (2003) on leading indicators and asset returns"


def cli_change(months: int = 6) -> SignalDef:
    expr = (pl.col("cli") - pl.col("cli").shift(months)).over("asset")
    return SignalDef(
        name=f"cli_chg_{months}",
        family="leading_indicator",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"months": months},
        description=f"{months}-month change in the OECD leading indicator (improving growth)",
        reference=CLI_REF,
    )


def cli_level() -> SignalDef:
    return SignalDef(
        name="cli_level",
        family="leading_indicator",
        build=lambda p: p.with_columns((pl.col("cli") - 100.0).alias("signal")),
        params={"level": 1},
        description="OECD leading indicator relative to trend (100)",
        reference=CLI_REF,
    )


def yield_change(months: int = 12) -> SignalDef:
    expr = -(pl.col("y10") - pl.col("y10").shift(months)).over("asset")
    return SignalDef(
        name=f"yield_fall_{months}",
        family="rates",
        build=lambda p: p.with_columns(expr.alias("signal")),
        params={"months": months},
        description=f"Fall in the 10-year government yield over {months} months",
        reference="Campbell & Ammer (1993); discount-rate channel of equity returns",
    )


def term_spread_equity() -> SignalDef:
    return SignalDef(
        name="term_spread_eq",
        family="country_curve",
        build=lambda p: p.with_columns((pl.col("y10") - pl.col("y3m")).alias("signal")),
        params={"spread": 1},
        description="Local 10-year yield minus the 3-month rate (a steep curve signals growth)",
        reference="Estrella & Mishkin (1998); term spread and future growth",
    )


def fx_momentum(lookback: int = 12, skip: int = 1) -> SignalDef:
    window = lookback - skip
    expr = (
        pl.col("fx_ret")
        .log1p()
        .rolling_sum(window_size=window, min_samples=window)
        .shift(skip)
        .over("asset")
    )
    return SignalDef(
        name=f"fx_mom_{lookback}_{skip}",
        family="currency_momentum",
        build=lambda p: p.with_columns(
            (expr if "fx_ret" in p.columns else pl.lit(None, dtype=pl.Float64)).alias("signal")
        ),
        params={"lookback": lookback, "skip": skip},
        description=f"{lookback}-month momentum of the local currency vs the US dollar",
        reference="Menkhoff, Sarno, Schmeling & Schrimpf (2012), currency momentum",
    )


COUNTRY_SIGNALS: dict[str, SignalDef] = {
    s.name: s
    for s in (
        momentum(12, 1),
        momentum(6, 1),
        long_term_reversal(60, 12),
        low_volatility(36),
        cli_change(6),
        cli_level(),
        yield_change(12),
        term_spread_equity(),
        *GENERIC_SIGNALS,
    )
}
# The USD universe also sees each market's currency.
COUNTRY_USD_SIGNALS: dict[str, SignalDef] = {**COUNTRY_SIGNALS, "fx_mom_12_1": fx_momentum(12, 1)}

# Momentum must skip at least one month here (averaged prices): no skip-0 configurations.
COUNTRY_FAMILIES: dict[str, list[SignalDef]] = {
    "momentum": [momentum(lb, 1) for lb in (3, 6, 9, 12, 18)],
    "long_term_reversal": [long_term_reversal(lb, 12) for lb in (36, 60)],
    "low_volatility": [low_volatility(w) for w in (24, 36, 60)],
    "leading_indicator": [cli_change(m) for m in (3, 6, 12)] + [cli_level()],
    "rates": [yield_change(m) for m in (6, 12)],
    "country_curve": [term_spread_equity()],
    **{k: v for k, v in GENERIC_FAMILIES.items() if k != "intermediate_momentum"},
    # averaged prices: intermediate momentum already skips 7 months, fine as is
    "intermediate_momentum": GENERIC_FAMILIES["intermediate_momentum"],
}
COUNTRY_USD_FAMILIES: dict[str, list[SignalDef]] = {
    **COUNTRY_FAMILIES,
    "currency_momentum": [fx_momentum(lb, 1) for lb in (6, 12)],
}
