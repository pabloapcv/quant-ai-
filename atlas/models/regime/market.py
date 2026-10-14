"""Market-based regime proxy: trend x volatility of the US equity market (PRD §8.4 gate).

Used to check that a signal works across regimes ("positive IC in >= 3 of 4 regimes") with
data available since 1926, before the FRED macro regimes are loaded.

* trend: trailing 12-month market excess return > 0 -> "up", else "down"
* volatility: trailing 12-month realized vol above its own expanding median -> "volatile"

Both use returns through the month in question only, so the label at month ``m`` is known
at the end of ``m`` — the same moment a signal dated ``m`` is formed.
"""

from __future__ import annotations

import polars as pl

from atlas.pit.store import PITStore
from atlas.signals.industry import latest_by_prefix

MARKET_SERIES_PREFIX = "FF.FACTOR.MKTRF"
REGIMES = ("up_calm", "up_volatile", "down_calm", "down_volatile")


def market_regimes(store: PITStore, *, window: int = 12, min_history: int = 36) -> pl.DataFrame:
    """``date``, ``regime`` (one of ``REGIMES``), plus the trend and vol readings."""
    mkt = latest_by_prefix(store, MARKET_SERIES_PREFIX, None).filter(pl.col("asset") == "")
    if mkt.is_empty():
        raise ValueError("FF.FACTOR.MKTRF missing — run `atlas ingest kenfrench FF.FACTOR.MKTRF`")
    r = (pl.col("value") / 100.0).log1p()
    df = mkt.sort("event_time").select(
        pl.col("event_time").alias("date"),
        r.rolling_sum(window, min_samples=window).alias("trend_12m"),
        r.rolling_std(window, min_samples=window).mul(12**0.5).alias("vol_12m"),
    )
    expanding_median = pl.col("vol_12m").rolling_median(
        window_size=df.height, min_samples=min_history
    )
    out = df.with_columns(expanding_median.alias("vol_median")).with_columns(
        pl.when(pl.col("trend_12m").is_null() | pl.col("vol_median").is_null())
        .then(None)
        .otherwise(
            pl.concat_str(
                pl.when(pl.col("trend_12m") > 0).then(pl.lit("up")).otherwise(pl.lit("down")),
                pl.when(pl.col("vol_12m") > pl.col("vol_median"))
                .then(pl.lit("volatile"))
                .otherwise(pl.lit("calm")),
                separator="_",
            )
        )
        .alias("regime")
    )
    return out.drop_nulls("regime").select("date", "regime", "trend_12m", "vol_12m")
