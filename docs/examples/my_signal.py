"""Example research signal: run it with

    ramon research signal --file docs/examples/my_signal.py --universe ind12

A signal adds a ``signal`` column to the research panel (one row per date and asset, with
that month's return in ``ret``). It must use only data up to each row's date: the
backtester scores it against the *next* month's return. Higher signal = expected to do
better than the other assets.
"""

import polars as pl

from atlas.signals.industry import SignalDef


def _build(lookback: int):
    def build(panel: pl.DataFrame) -> pl.DataFrame:
        # log return over months t-lookback+1 .. t-1 (skipping the latest month)
        mom = pl.col("ret").log1p().rolling_sum(lookback - 1, min_samples=lookback - 1)
        return (
            panel.sort(["asset", "date"])
            .with_columns(mom.shift(1).over("asset").alias("_mom"))
            # minus that month's cross-sectional mean
            .with_columns((pl.col("_mom") - pl.col("_mom").mean().over("date")).alias("signal"))
            .drop("_mom")
        )

    return build


def make(lookback: int) -> SignalDef:
    return SignalDef(
        name=f"my_mom_{lookback}",
        family="my_momentum",
        build=_build(lookback),
        params={"lookback": lookback},
        description=f"Demeaned {lookback}-1 momentum (an example)",
        reference="Jegadeesh & Titman (1993)",
    )


SIGNAL = make(9)
FAMILY = [make(lb) for lb in (6, 9, 12)]  # optional: variants for the PBO estimate
