"""Rules-based growth x inflation quadrant with soft probabilities (PRD §7.1)."""

from __future__ import annotations

import math
from enum import StrEnum

import polars as pl
from scipy.stats import norm


class Quadrant(StrEnum):
    GOLDILOCKS = "goldilocks"  # growth above trend, inflation below
    OVERHEATING = "overheating"  # both above
    STAGFLATION = "stagflation"  # growth below, inflation above
    DEFLATIONARY_SLOWDOWN = "deflationary_slowdown"  # both below


def classify(growth: float, inflation: float) -> Quadrant:
    if growth > 0:
        return Quadrant.OVERHEATING if inflation > 0 else Quadrant.GOLDILOCKS
    return Quadrant.STAGFLATION if inflation > 0 else Quadrant.DEFLATIONARY_SLOWDOWN


def classify_expr(growth: pl.Expr, inflation: pl.Expr) -> pl.Expr:
    return (
        pl.when(growth.is_null() | inflation.is_null())
        .then(None)
        .when(growth > 0)
        .then(
            pl.when(inflation > 0)
            .then(pl.lit(Quadrant.OVERHEATING.value))
            .otherwise(pl.lit(Quadrant.GOLDILOCKS.value))
        )
        .otherwise(
            pl.when(inflation > 0)
            .then(pl.lit(Quadrant.STAGFLATION.value))
            .otherwise(pl.lit(Quadrant.DEFLATIONARY_SLOWDOWN.value))
        )
    )


def noise_scale(scores: pl.Series, *, horizon: int = 3, floor: float = 0.1) -> float:
    """Typical ``horizon``-month move of a composite: how far a reading can drift by noise.

    Used as the standard deviation when turning a composite level into P(level > 0).
    """
    moves = scores.drop_nulls().diff(horizon).drop_nulls()
    if moves.len() < 12:
        return 1.0
    sd = moves.std()
    return max(float(sd) if isinstance(sd, int | float) and math.isfinite(sd) else 1.0, floor)


def probabilities(
    growth: float, inflation: float, growth_sigma: float, inflation_sigma: float
) -> dict[Quadrant, float]:
    """P(each quadrant), treating growth and inflation readings as independent Gaussians."""
    pg = float(norm.cdf(growth / growth_sigma))
    pi = float(norm.cdf(inflation / inflation_sigma))
    return {
        Quadrant.GOLDILOCKS: pg * (1 - pi),
        Quadrant.OVERHEATING: pg * pi,
        Quadrant.STAGFLATION: (1 - pg) * pi,
        Quadrant.DEFLATIONARY_SLOWDOWN: (1 - pg) * (1 - pi),
    }
