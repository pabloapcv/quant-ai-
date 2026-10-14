"""Pillar composites: equal-weight mean of clipped, sign-oriented feature z-scores."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from atlas.features.registry import FeatureSpec

CLIP = 3.0


def _oriented(spec: FeatureSpec) -> pl.Expr:
    return pl.col(spec.name).clip(-CLIP, CLIP) * spec.sign


def composite(
    matrix: pl.DataFrame, specs: Sequence[FeatureSpec], name: str, *, min_features: int = 3
) -> pl.DataFrame:
    """``event_time``, ``{name}``, ``{name}_n``. Null where fewer than ``min_features`` exist."""
    present = [s for s in specs if s.name in matrix.columns]
    if not present:
        return matrix.select(
            "event_time",
            pl.lit(None, dtype=pl.Float64).alias(name),
            pl.lit(0, dtype=pl.UInt32).alias(f"{name}_n"),
        )
    n = pl.sum_horizontal(pl.col(s.name).is_not_null() for s in present).alias(f"{name}_n")
    mean = pl.mean_horizontal(_oriented(s) for s in present)
    return matrix.select(
        "event_time",
        pl.when(n >= min_features).then(mean).otherwise(None).alias(name),
        n,
    )


def contributions(matrix_row: dict[str, object], specs: Sequence[FeatureSpec]) -> dict[str, float]:
    """Each feature's share of the composite at one row (sums to the composite score)."""
    vals: dict[str, float] = {}
    for s in specs:
        v = matrix_row.get(s.name)
        if isinstance(v, int | float):
            vals[s.name] = max(-CLIP, min(CLIP, float(v))) * s.sign
    n = len(vals)
    return {k: v / n for k, v in vals.items()} if n else {}
