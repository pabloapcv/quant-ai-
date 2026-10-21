"""Signal library, part 2 (PRD §6.2-6.4): published cross-sectional anomalies that need only
returns, plus macro-sensitivity tilts. Each works on any research panel with ``date``,
``asset`` and ``ret`` (the month's return); the "market" is the panel's own equal-weight
average, so the same definitions serve industries and countries.

All signals at month ``t`` use returns through ``t`` (and macro changes known by ``t``);
the backtester scores them against month ``t+1``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from atlas.pit.store import PITStore
from atlas.signals.industry import SignalDef, momentum

MARKET = pl.col("ret").mean().over("date")  # equal-weight universe return


def _with_market(p: pl.DataFrame) -> pl.DataFrame:
    return p.with_columns(MARKET.alias("_mkt")).sort(["asset", "date"])


def seasonality(years: int = 10) -> SignalDef:
    """Average return in the same calendar month over the past ``years`` years."""
    lags = [12 * k for k in range(1, years + 1)]
    min_years = max(3, years // 2)

    def build(p: pl.DataFrame) -> pl.DataFrame:
        # month t's signal predicts t+1: use the same calendar month as t+1, i.e. lags 11, 23...
        cols = [pl.col("ret").shift(lag - 1).over("asset") for lag in lags]
        stacked = pl.concat_list(cols)
        n = stacked.list.eval(pl.element().is_not_null().sum()).list.first()
        return p.sort(["asset", "date"]).with_columns(
            pl.when(n >= min_years).then(stacked.list.mean()).otherwise(None).alias("signal")
        )

    return SignalDef(
        name=f"seasonality_{years}y",
        family="seasonality",
        build=build,
        params={"years": years},
        description=f"Average return in the coming calendar month over the past {years} years",
        reference="Heston & Sadka (2008), seasonality in the cross-section of returns",
    )


def _rolling_beta(y: pl.Expr, x: pl.Expr, window: int) -> pl.Expr:
    mp = max(24, window * 2 // 3)
    cov = (y * x).rolling_mean(window, min_samples=mp) - y.rolling_mean(
        window, min_samples=mp
    ) * x.rolling_mean(window, min_samples=mp)
    var = (x * x).rolling_mean(window, min_samples=mp) - x.rolling_mean(window, min_samples=mp) ** 2
    return cov / var


def low_beta(window: int = 60) -> SignalDef:
    def build(p: pl.DataFrame) -> pl.DataFrame:
        q = _with_market(p)
        beta = _rolling_beta(pl.col("ret"), pl.col("_mkt"), window).over("asset")
        return q.with_columns((-beta).alias("signal")).drop("_mkt")

    return SignalDef(
        name=f"low_beta_{window}",
        family="low_beta",
        build=build,
        params={"window": window},
        description=f"Negative {window}-month beta to the universe average (betting against beta)",
        reference="Frazzini & Pedersen (2014), betting against beta",
    )


def idiosyncratic_vol(window: int = 36) -> SignalDef:
    def build(p: pl.DataFrame) -> pl.DataFrame:
        q = _with_market(p)
        beta = _rolling_beta(pl.col("ret"), pl.col("_mkt"), window).over("asset")
        q = q.with_columns(beta.alias("_beta")).with_columns(
            (pl.col("ret") - pl.col("_beta") * pl.col("_mkt")).alias("_res")
        )
        sd = pl.col("_res").rolling_std(window, min_samples=max(24, window * 2 // 3)).over("asset")
        return q.with_columns((-sd).alias("signal")).drop("_mkt", "_beta", "_res")

    return SignalDef(
        name=f"idio_vol_{window}",
        family="idiosyncratic_vol",
        build=build,
        params={"window": window},
        description=f"Negative {window}-month residual volatility vs the universe average",
        reference="Ang, Hodrick, Xing & Zhang (2006), the cross-section of volatility",
    )


def near_high(months: int = 12) -> SignalDef:
    def build(p: pl.DataFrame) -> pl.DataFrame:
        q = p.sort(["asset", "date"])
        level = pl.col("ret").fill_null(0.0).log1p().cum_sum().over("asset")
        q = q.with_columns(level.alias("_lvl"))
        hi = pl.col("_lvl").rolling_max(months, min_samples=months).over("asset")
        return q.with_columns((pl.col("_lvl") - hi).alias("signal")).drop("_lvl")

    return SignalDef(
        name=f"high_{months}m",
        family="near_high",
        build=build,
        params={"months": months},
        description=f"Distance (log) below the {months}-month high of the return index",
        reference="George & Hwang (2004), the 52-week high and momentum investing",
    )


def lottery(months: int = 12) -> SignalDef:
    expr = -pl.col("ret").rolling_max(months, min_samples=months).over("asset")
    return SignalDef(
        name=f"max_ret_{months}m",
        family="lottery",
        build=lambda p: p.sort(["asset", "date"]).with_columns(expr.alias("signal")),
        params={"months": months},
        description=f"Negative maximum monthly return over {months} months (avoid lotteries)",
        reference="Bali, Cakici & Whitelaw (2011), maxing out",
    )


def neg_skew(window: int = 60) -> SignalDef:
    def build(p: pl.DataFrame) -> pl.DataFrame:
        q = p.sort(["asset", "date"])
        r = pl.col("ret")
        mp = max(36, window * 2 // 3)
        m = r.rolling_mean(window, min_samples=mp)
        sd = r.rolling_std(window, min_samples=mp)
        m3 = ((r - m) ** 3).rolling_mean(window, min_samples=mp)
        return q.with_columns((-(m3 / sd**3)).over("asset").alias("signal"))

    return SignalDef(
        name=f"neg_skew_{window}",
        family="skewness",
        build=build,
        params={"window": window},
        description=f"Negative {window}-month return skewness",
        reference="Harvey & Siddique (2000); Amaya et al. (2015) on realized skewness",
    )


# ---------------------------------------------------------------------------- macro tilts
MACRO_FACTORS = {
    "x_dy10": ("US.RATES.10Y", "diff"),  # month-end change in the 10-year yield, pp
    "x_doil": ("CMDTY.OIL.WTI", "pct"),  # month-end WTI return
}


def with_macro_factors(panel: pl.DataFrame, store: PITStore) -> pl.DataFrame:
    """Join monthly macro factor changes (``x_*`` columns) by date; absent data -> no column."""
    out = panel
    for col, (sid, kind) in MACRO_FACTORS.items():
        obs = store.as_of([sid], datetime.now(UTC)).drop_nulls("value")
        if obs.is_empty():
            continue
        lvl = (
            obs.sort("event_time")
            .group_by(pl.col("event_time").dt.truncate("1mo").alias("date"))
            .agg(pl.col("value").last())
            .sort("date")
        )
        chg = pl.col("value").diff() if kind == "diff" else pl.col("value").pct_change()
        out = out.join(lvl.select("date", chg.alias(col)), on="date", how="left")
    return out


def macro_tilt(factor: str, window: int = 60, trend: int = 3) -> SignalDef:
    """Exposure to a macro factor times the factor's recent move (PRD §6.4).

    Beta of the asset's return in excess of the universe average on the factor's monthly
    change (rolling ``window``), times the factor's change over the last ``trend`` months:
    assets that gain when the factor moves the way it has been moving score high.
    """
    label = {"x_dy10": "rates", "x_doil": "oil"}[factor]

    def build(p: pl.DataFrame) -> pl.DataFrame:
        if factor not in p.columns:
            return p.with_columns(pl.lit(None, dtype=pl.Float64).alias("signal"))
        q = _with_market(p).with_columns((pl.col("ret") - pl.col("_mkt")).alias("_xr"))
        beta = _rolling_beta(pl.col("_xr"), pl.col(factor), window).over("asset")
        move = pl.col(factor).rolling_sum(trend, min_samples=trend).over("asset")
        return q.with_columns((beta * move).alias("signal")).drop("_mkt", "_xr")

    return SignalDef(
        name=f"{label}_tilt" if (window, trend) == (60, 3) else f"{label}_tilt_{window}_{trend}",
        family=f"{label}_tilt",
        build=build,
        params={"window": window, "trend": trend},
        description=f"Sensitivity to {'the 10-year yield' if label == 'rates' else 'oil'} "
        f"({window}-month beta) times its {trend}-month move",
        reference="PRD §6.4 macro-to-sector sensitivity; Moskowitz, Ooi & Pedersen (2012) "
        "on trend persistence",
    )


GENERIC_SIGNALS: tuple[SignalDef, ...] = (
    momentum(12, 7),
    seasonality(10),
    low_beta(60),
    idiosyncratic_vol(36),
    near_high(12),
    lottery(12),
    neg_skew(60),
    macro_tilt("x_dy10"),
    macro_tilt("x_doil"),
)
GENERIC_FAMILIES: dict[str, list[SignalDef]] = {
    "intermediate_momentum": [momentum(12, 7), momentum(18, 7)],
    "seasonality": [seasonality(y) for y in (5, 10, 20)],
    "low_beta": [low_beta(w) for w in (36, 60)],
    "idiosyncratic_vol": [idiosyncratic_vol(w) for w in (24, 36, 60)],
    "near_high": [near_high(m) for m in (12, 24)],
    "lottery": [lottery(m) for m in (6, 12)],
    "skewness": [neg_skew(w) for w in (36, 60)],
    "rates_tilt": [macro_tilt("x_dy10", w, t) for w in (36, 60) for t in (1, 3)],
    "oil_tilt": [macro_tilt("x_doil", w, t) for w in (36, 60) for t in (1, 3)],
}
