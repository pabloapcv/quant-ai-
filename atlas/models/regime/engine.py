"""Macro regime engine (PRD §7.1, Layer 1).

Given an as-of date, everything is computed from data public by then:

1. Feature matrix from the as-of vintage of every input (``atlas.features``).
2. Growth / inflation / financial-conditions composites per month.
3. Quadrant + soft probabilities for the latest complete month.
4. Gaussian HMM on the composite history (optional, slower).
5. 12-month recession probit (optional).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
import polars as pl
from pydantic import BaseModel

from atlas.features.macro import FINANCIAL, GROWTH, INFLATION, RECESSION
from atlas.features.registry import FeatureSpec, feature_matrix
from atlas.models.regime.composites import composite, contributions
from atlas.models.regime.hmm import HMMResult, fit_regime_hmm
from atlas.models.regime.quadrant import Quadrant, classify, noise_scale, probabilities
from atlas.models.regime.recession import RecessionResult, fit_recession_probit
from atlas.pit.cube import VintageCube

RECESSION_SERIES = "US.NBER.RECESSION"
PILLARS: dict[str, tuple[FeatureSpec, ...]] = {
    "growth": GROWTH,
    "inflation": INFLATION,
    "financial_conditions": FINANCIAL,
}
HMM_COLUMNS = ("growth", "inflation", "financial_conditions")


class PillarSummary(BaseModel):
    score: float | None
    prob_above_trend: float | None
    change_3m: float | None
    n_features: int
    contributions: dict[str, float]


class RegimeReport(BaseModel):
    as_of: date
    data_month: date  # latest complete month the readings describe
    regime: Quadrant | None
    quadrant_probs: dict[Quadrant, float]
    growth: PillarSummary
    inflation: PillarSummary
    financial_conditions: PillarSummary
    hmm: HMMResult | None
    recession: RecessionResult | None
    missing_features: list[str]


def data_month_for(at: date | datetime) -> date:
    """The last complete calendar month before ``at``."""
    d = at.date() if isinstance(at, datetime) else at
    first = d.replace(day=1)
    return (
        first.replace(year=first.year - 1, month=12)
        if first.month == 1
        else first.replace(month=first.month - 1)
    )


class RegimeEngine:
    def __init__(
        self,
        cube: VintageCube,
        *,
        include_approx: bool = True,
        min_features: int = 3,
        ffill_limit: int = 3,
        recession_sign_constraint: bool = False,
    ) -> None:
        self.cube = cube
        self.include_approx = include_approx
        # Off by default: see atlas.models.regime.recession ("Why the Sahm weight is negative").
        self.recession_sign_constraint = recession_sign_constraint
        self.min_features = min_features
        self.ffill_limit = ffill_limit

    def _matrix(self, specs: Sequence[FeatureSpec], at: date | datetime, end: date) -> pl.DataFrame:
        usable = [s for s in specs if s.series_id in self.cube]
        return feature_matrix(
            self.cube,
            usable,
            at,
            end_month=end,
            ffill_limit=self.ffill_limit,
            include_approx=self.include_approx,
        )

    def composites(self, at: date | datetime) -> tuple[pl.DataFrame, pl.DataFrame]:
        """(composite history, pillar feature matrix), both as known at ``at``."""
        end = data_month_for(at)
        specs = [s for group in PILLARS.values() for s in group]
        matrix = self._matrix(specs, at, end)
        out = matrix.select("event_time")
        for name, group in PILLARS.items():
            comp = composite(matrix, group, name, min_features=self._min_for(name))
            out = out.join(comp, on="event_time", how="left")
        return out.sort("event_time"), matrix

    def _min_for(self, pillar: str) -> int:
        # Financial conditions has few long-history inputs; one standardized index suffices.
        return 1 if pillar == "financial_conditions" else self.min_features

    def run(
        self, at: date | datetime, *, with_hmm: bool = True, with_recession: bool = True
    ) -> RegimeReport:
        end = data_month_for(at)
        comps, matrix = self.composites(at)
        last = comps.filter(pl.col("event_time") == end)
        last_row = last.row(0, named=True) if last.height else {}
        mrow = matrix.filter(pl.col("event_time") == end)
        matrix_row = mrow.row(0, named=True) if mrow.height else {}

        summaries: dict[str, PillarSummary] = {}
        sigmas: dict[str, float] = {}
        for name, group in PILLARS.items():
            score = last_row.get(name)
            contrib = contributions(matrix_row, group) if score is not None else {}
            sigma = _uncertainty(comps[name] if name in comps.columns else None, contrib)
            sigmas[name] = sigma
            prev = comps.filter(pl.col("event_time") < end).tail(3)
            prev_score = prev[name][0] if prev.height == 3 else None
            summaries[name] = PillarSummary(
                score=score,
                prob_above_trend=None if score is None else _phi(score / sigma),
                change_3m=None if score is None or prev_score is None else score - prev_score,
                n_features=int(last_row.get(f"{name}_n") or 0),
                contributions=contrib,
            )

        g, i = summaries["growth"].score, summaries["inflation"].score
        regime = None if g is None or i is None else classify(g, i)
        probs = (
            {}
            if g is None or i is None
            else probabilities(g, i, sigmas["growth"], sigmas["inflation"])
        )

        hmm = self._hmm(comps) if with_hmm else None
        recession = self._recession(at, end) if with_recession else None

        all_specs = [s for group in PILLARS.values() for s in group]
        missing = sorted(s.name for s in all_specs if matrix_row.get(s.name) is None)
        return RegimeReport(
            as_of=at.date() if isinstance(at, datetime) else at,
            data_month=end,
            regime=regime,
            quadrant_probs=probs,
            growth=summaries["growth"],
            inflation=summaries["inflation"],
            financial_conditions=summaries["financial_conditions"],
            hmm=hmm,
            recession=recession,
            missing_features=missing,
        )

    def _hmm(self, comps: pl.DataFrame) -> HMMResult | None:
        hist = comps.select(HMM_COLUMNS).drop_nulls()
        if hist.is_empty():
            return None
        return fit_regime_hmm(hist.to_numpy().astype(np.float64), HMM_COLUMNS)

    def _recession(self, at: date | datetime, end: date) -> RecessionResult | None:
        if RECESSION_SERIES not in self.cube:
            return None
        features = self._matrix(RECESSION, at, end)
        rec = self.cube.as_of(RECESSION_SERIES, at, include_approx=self.include_approx)
        rec_monthly = (
            rec.select(pl.col("event_time").dt.truncate("1mo"), "value")
            .group_by("event_time")
            .agg(pl.col("value").max())
        )
        specs = [s for s in RECESSION if s.name in features.columns]
        if not specs:
            return None
        return fit_recession_probit(
            features,
            rec_monthly,
            [s.name for s in specs],
            current_month=end,
            expected_signs={s.name: s.sign for s in specs}
            if self.recession_sign_constraint
            else None,
        )


def _uncertainty(history: pl.Series | None, contrib: dict[str, float]) -> float:
    """Std. dev. of a composite reading: time-series noise ⊕ disagreement among inputs.

    * drift: typical 3-month move of the composite (``noise_scale``)
    * disagreement: cross-sectional s.d. of the oriented inputs / sqrt(n) — a standard error
    """
    drift = noise_scale(history) if history is not None else 1.0
    n = len(contrib)
    if n < 2:
        return drift
    values = np.array([c * n for c in contrib.values()])  # undo the 1/n share
    stderr = float(values.std(ddof=1)) / np.sqrt(n)
    return float(np.hypot(drift, stderr))


def _phi(z: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(z))
