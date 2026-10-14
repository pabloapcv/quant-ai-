"""Industry / sector cross-sectional signals on Ken French industry portfolios (PRD §6.2-6.3).

Panel convention: ``date`` = month start, ``ret`` = that month's simple return (fraction).
A signal at ``date`` uses returns through ``date`` (month-end market data); ``fwd_ret`` is
the next month's return. Monthly returns are knowable at month-end from prices — the Ken
French publication lag is a distribution delay, so the panel is built from the latest
vintage (documented, deliberate; see ``atlas.connectors.kenfrench``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import polars as pl

from atlas.pit.store import PITStore

UNIVERSES = {"ind12": "FF.IND12.", "ind49": "FF.IND49."}


def returns_panel(
    store: PITStore, universe: str, *, at: date | datetime | None = None
) -> pl.DataFrame:
    """Gap-free (date x asset) panel of monthly returns; missing months stay null."""
    prefix = UNIVERSES[universe]
    ids = [s for s in store.series_ids() if s.startswith(prefix)]
    if not ids:
        raise ValueError(f"no {universe} series in the store — run `atlas ingest kenfrench`")
    obs = store.as_of(ids, at or datetime.now(UTC))  # default: latest vintage
    df = obs.select(
        pl.col("event_time").alias("date"),
        pl.col("series_id").str.strip_prefix(prefix).alias("asset"),
        (pl.col("value") / 100.0).alias("ret"),
    )
    grid = df.select("date").unique().join(df.select("asset").unique(), how="cross")
    return grid.join(df, on=["date", "asset"], how="left").sort(["asset", "date"])


def with_forward_return(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.sort(["asset", "date"]).with_columns(
        pl.col("ret").shift(-1).over("asset").alias("fwd_ret")
    )


SignalFn = Callable[[pl.DataFrame], pl.Expr]


@dataclass(frozen=True)
class SignalDef:
    name: str
    family: str
    fn: SignalFn
    params: dict[str, int]
    description: str

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Adds ``signal`` and ``fwd_ret`` to a ``returns_panel``."""
        return with_forward_return(panel).with_columns(self.fn(panel).over("asset").alias("signal"))


def momentum(lookback: int, skip: int = 1) -> SignalDef:
    """Cumulative log return over months t-lookback+1 .. t-skip (Jegadeesh-Titman style)."""
    window = lookback - skip

    def fn(_: pl.DataFrame) -> pl.Expr:
        return pl.col("ret").log1p().rolling_sum(window_size=window, min_samples=window).shift(skip)

    return SignalDef(
        name=f"mom_{lookback}_{skip}",
        family="momentum",
        fn=fn,
        params={"lookback": lookback, "skip": skip},
        description=f"{lookback}-month momentum skipping the last {skip} month(s)",
    )


def reversal() -> SignalDef:
    return SignalDef(
        name="reversal_1m",
        family="reversal",
        fn=lambda _: -pl.col("ret"),
        params={},
        description="Short-term (1-month) reversal: negative last-month return",
    )


def low_volatility(window: int = 36) -> SignalDef:
    def fn(_: pl.DataFrame) -> pl.Expr:
        return -pl.col("ret").rolling_std(window_size=window, min_samples=max(12, window * 2 // 3))

    return SignalDef(
        name=f"low_vol_{window}",
        family="low_volatility",
        fn=fn,
        params={"window": window},
        description=f"Negative trailing {window}-month volatility",
    )


def long_term_reversal(lookback: int = 60, skip: int = 12) -> SignalDef:
    """De Bondt-Thaler: losers over years t-5..t-1 outperform."""
    base = momentum(lookback, skip)
    return SignalDef(
        name=f"ltr_{lookback}_{skip}",
        family="long_term_reversal",
        fn=lambda p: -base.fn(p),
        params={"lookback": lookback, "skip": skip},
        description=f"Negative {lookback}-month return skipping the last {skip} months",
    )


SIGNALS: dict[str, SignalDef] = {
    s.name: s
    for s in (
        momentum(12, 1),
        momentum(6, 1),
        momentum(3, 0),
        reversal(),
        low_volatility(36),
        long_term_reversal(60, 12),
    )
}

FAMILIES: dict[str, list[SignalDef]] = {
    "momentum": [momentum(lb, sk) for lb in (3, 6, 9, 12, 18, 24) for sk in (0, 1)],
    "low_volatility": [low_volatility(w) for w in (12, 24, 36, 60)],
}
