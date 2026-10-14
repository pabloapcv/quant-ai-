"""Industry / sector cross-sectional signals on Ken French industry portfolios (PRD §6.2-6.3).

Panel convention: ``date`` = month start, ``ret`` = that month's simple return (fraction).
A signal at ``date`` uses information available at the end of that month; ``fwd_ret`` is
the next month's return.

Timing of each input:

* Returns are market data, knowable at month-end from prices; the Ken French publication
  lag is a distribution delay, so returns come from the latest vintage.
* Book-to-market (annual) follows the Fama-French convention: fiscal year ``y`` data is
  used from the end of June ``y+1`` (``value_panel``), never earlier.

Every signal carries a family (for the PBO test across its configurations) and a
literature reference (the PRD's economic-rationale gate).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import polars as pl

from atlas.pit.store import PITStore

UNIVERSES = {"ind12": "FF.IND12.", "ind49": "FF.IND49."}
BM_PREFIX = {"ind12": "FF.IND12BM.", "ind49": "FF.IND49BM."}
FORMATION_MONTH = 6  # June: when Fama-French form portfolios on prior fiscal-year data

Builder = Callable[[pl.DataFrame], pl.DataFrame]


# ---------------------------------------------------------------------------- panels
def latest_by_prefix(store: PITStore, prefix: str, at: date | datetime | None) -> pl.DataFrame:
    ids = [s for s in store.series_ids() if s.startswith(prefix)]
    if not ids:
        return pl.DataFrame(schema={"event_time": pl.Date, "asset": pl.String, "value": pl.Float64})
    obs = store.as_of(ids, at or datetime.now(UTC))  # default: latest vintage
    return obs.select(
        "event_time", pl.col("series_id").str.strip_prefix(prefix).alias("asset"), "value"
    )


def returns_panel(
    store: PITStore, universe: str, *, at: date | datetime | None = None
) -> pl.DataFrame:
    """Gap-free (date x asset) panel of monthly returns; missing months stay null."""
    raw = latest_by_prefix(store, UNIVERSES[universe], at)
    if raw.is_empty():
        raise ValueError(f"no {universe} series in the store — run `atlas ingest kenfrench`")
    df = raw.select(
        pl.col("event_time").alias("date"), "asset", (pl.col("value") / 100.0).alias("ret")
    )
    grid = df.select("date").unique().join(df.select("asset").unique(), how="cross")
    return grid.join(df, on=["date", "asset"], how="left").sort(["asset", "date"])


def fiscal_year_known(month: pl.Expr) -> pl.Expr:
    """Latest fiscal year whose BE/ME is usable at the end of ``month``."""
    shift = pl.when(month.dt.month() >= FORMATION_MONTH).then(1).otherwise(2)
    return (month.dt.year() - shift).cast(pl.Int32)


def value_panel(panel: pl.DataFrame, store: PITStore, universe: str) -> pl.DataFrame:
    """Adds ``bm`` (aggregate industry BE/ME) to a returns panel with Fama-French timing."""
    raw = latest_by_prefix(store, BM_PREFIX[universe], None)
    bm = raw.filter(pl.col("value") > 0).select(
        "asset",
        pl.col("event_time").dt.year().cast(pl.Int32).alias("fy"),
        pl.col("value").alias("bm"),
    )
    return (
        panel.with_columns(fiscal_year_known(pl.col("date")).alias("fy"))
        .join(bm, on=["asset", "fy"], how="left")
        .drop("fy")
        .sort(["asset", "date"])
    )


def research_panel(store: PITStore, universe: str) -> pl.DataFrame:
    """Returns + every characteristic the signal library needs."""
    return value_panel(returns_panel(store, universe), store, universe)


def with_forward_return(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.sort(["asset", "date"]).with_columns(
        pl.col("ret").shift(-1).over("asset").alias("fwd_ret")
    )


# ---------------------------------------------------------------------------- definitions
@dataclass(frozen=True)
class SignalDef:
    name: str
    family: str
    build: Builder  # panel -> panel with a "signal" column
    params: dict[str, float] = field(default_factory=dict)
    description: str = ""
    reference: str = ""
    vol_target: float | None = None  # annualized vol target for the long-short overlay
    ranking_of: str | None = None  # overlays: the signal whose cross-sectional ranking this reuses

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Adds ``signal`` and ``fwd_ret`` to a research panel."""
        return self.build(with_forward_return(panel))


def _xs_z(col: str) -> pl.Expr:
    """Cross-sectional z-score within each date."""
    c = pl.col(col)
    return (c - c.mean().over("date")) / c.std().over("date")


def _momentum_expr(lookback: int, skip: int) -> pl.Expr:
    window = lookback - skip
    return (
        pl.col("ret")
        .log1p()
        .rolling_sum(window_size=window, min_samples=window)
        .shift(skip)
        .over("asset")
    )


MOM_REF = "Jegadeesh & Titman (1993); Moskowitz & Grinblatt (1999), industry momentum"


def momentum(lookback: int, skip: int = 1) -> SignalDef:
    """Cumulative log return over months t-lookback+1 .. t-skip."""
    return SignalDef(
        name=f"mom_{lookback}_{skip}",
        family="momentum",
        build=lambda p: p.with_columns(_momentum_expr(lookback, skip).alias("signal")),
        params={"lookback": lookback, "skip": skip},
        description=f"{lookback}-month momentum skipping the last {skip} month(s)",
        reference=MOM_REF,
    )


def vol_scaled_momentum(target: float = 0.12) -> SignalDef:
    base = momentum(12, 1)
    return SignalDef(
        name="mom_12_1_vt",
        family="momentum",
        build=base.build,
        params={"lookback": 12, "skip": 1, "vol_target": target},
        description=f"12-1 momentum, long-short scaled to {target:.0%} trailing volatility",
        reference="Barroso & Santa-Clara (2015), momentum has its moments",
        vol_target=target,
        ranking_of=base.name,
    )


def reversal(months: int = 1) -> SignalDef:
    return SignalDef(
        name=f"reversal_{months}m",
        family="reversal",
        build=lambda p: p.with_columns((-_momentum_expr(months, 0)).alias("signal")),
        params={"months": months},
        description=f"Short-term reversal: negative last {months}-month return",
        reference="Jegadeesh (1990); Lehmann (1990)",
    )


def low_volatility(window: int = 36) -> SignalDef:
    expr = -pl.col("ret").rolling_std(window_size=window, min_samples=max(12, window * 2 // 3))
    return SignalDef(
        name=f"low_vol_{window}",
        family="low_volatility",
        build=lambda p: p.with_columns(expr.over("asset").alias("signal")),
        params={"window": window},
        description=f"Negative trailing {window}-month volatility",
        reference="Ang, Hodrick, Xing & Zhang (2006); Frazzini & Pedersen (2014)",
    )


def long_term_reversal(lookback: int = 60, skip: int = 12) -> SignalDef:
    return SignalDef(
        name=f"ltr_{lookback}_{skip}",
        family="long_term_reversal",
        build=lambda p: p.with_columns((-_momentum_expr(lookback, skip)).alias("signal")),
        params={"lookback": lookback, "skip": skip},
        description=f"Negative {lookback}-month return skipping the last {skip} months",
        reference="De Bondt & Thaler (1985)",
    )


VALUE_REF = "Fama & French (1992); Asness, Moskowitz & Pedersen (2013)"


def value(kind: str = "log_bm") -> SignalDef:
    """Industry book-to-market. ``kind``: log level, raw level, or vs own 5-year average."""
    exprs = {
        "log_bm": pl.col("bm").log(),
        "bm": pl.col("bm"),
        # cheap relative to its own history (60 months of the annual series, forward-filled)
        "bm_vs_5y": (pl.col("bm").log() - pl.col("bm").log().rolling_mean(60, min_samples=36)).over(
            "asset"
        ),
    }
    return SignalDef(
        name=f"value_{kind}",
        family="value",
        build=lambda p: p.with_columns(exprs[kind].alias("signal")),
        params={"kind_" + kind: 1},
        description={
            "log_bm": "Log book-to-market (cheap industries score high)",
            "bm": "Book-to-market level",
            "bm_vs_5y": "Log book-to-market relative to its own 5-year average",
        }[kind],
        reference=VALUE_REF,
    )


def value_momentum(weight_value: float = 0.5) -> SignalDef:
    """Blend of cross-sectional z-scores of log BE/ME and 12-1 momentum."""

    def build(p: pl.DataFrame) -> pl.DataFrame:
        return (
            p.with_columns(_momentum_expr(12, 1).alias("_mom"), pl.col("bm").log().alias("_val"))
            .with_columns(
                (weight_value * _xs_z("_val") + (1 - weight_value) * _xs_z("_mom")).alias("signal")
            )
            .drop("_mom", "_val")
        )

    return SignalDef(
        name=f"value_mom_{int(weight_value * 100)}",
        family="value_momentum",
        build=build,
        params={"weight_value": weight_value},
        description=f"{weight_value:.0%} value / {1 - weight_value:.0%} momentum z-score blend",
        reference="Asness, Moskowitz & Pedersen (2013), value and momentum everywhere",
    )


# Representative configuration of each idea (what tearsheets and gates report on).
SIGNALS: dict[str, SignalDef] = {
    s.name: s
    for s in (
        momentum(12, 1),
        vol_scaled_momentum(0.12),
        momentum(6, 1),
        reversal(1),
        low_volatility(36),
        long_term_reversal(60, 12),
        value("log_bm"),
        value_momentum(0.5),
    )
}

# Every configuration tried per idea (what PBO is computed across).
FAMILIES: dict[str, list[SignalDef]] = {
    "momentum": [momentum(lb, sk) for lb in (3, 6, 9, 12, 18, 24) for sk in (0, 1)],
    "reversal": [reversal(m) for m in (1, 2, 3)],
    "low_volatility": [low_volatility(w) for w in (12, 24, 36, 60)],
    "long_term_reversal": [long_term_reversal(lb, sk) for lb, sk in ((36, 12), (60, 12), (60, 0))],
    "value": [value(k) for k in ("log_bm", "bm", "bm_vs_5y")],
    "value_momentum": [value_momentum(w) for w in (0.25, 0.5, 0.75)],
}
