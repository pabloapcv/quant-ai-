"""Scenario engine (PRD §7.6): asset returns under shocks to market factors.

Model: each asset's monthly return (as excess over the equal-weight universe average, plus the
average itself) is regressed on the same month's factor changes::

    r_a,t = alpha_a + sum_f beta_a,f * dx_f,t + e_a,t

A scenario fixes some factor moves ``s``. Factors not shocked move by their conditional
expectation given the shocked ones under a Gaussian model of the changes,
``x_u = Sigma_us Sigma_ss^-1 s`` (a +100bp 10Y shock usually comes with a 2Y move too), and
the response is ``beta . x``, alpha excluded. Bands: moving-block bootstrap of the months
(12-month blocks), refitting betas and covariance each draw; p10 / p90 of the response.

The response is the move in the month the shock happens, not a forecast over a longer
horizon. Factor levels are month-end values as known at the as-of date. Growth and inflation
surprise factors are not included: there is no point-in-time consensus series to build
surprises from. Credit uses Moody's Baa-10Y (daily since 1986), because FRED distributes
only a trailing three-year window of ICE BofA OAS history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel

from atlas.pit.store import PITStore

AVERAGE = "__average__"  # pseudo-asset: the equal-weight universe average return


@dataclass(frozen=True)
class Factor:
    name: str
    series_id: str
    change: Literal["diff", "pct"]
    unit: str
    description: str


FACTORS: tuple[Factor, ...] = (
    Factor("us10y", "US.RATES.10Y", "diff", "pp", "10Y Treasury yield"),
    Factor("us2y", "US.RATES.2Y", "diff", "pp", "2Y Treasury yield"),
    Factor("breakeven10y", "US.RATES.BE10", "diff", "pp", "10Y breakeven inflation"),
    Factor("credit_spread", "US.CREDIT.BAA10Y", "diff", "pp", "Moody's Baa minus 10Y"),
    Factor("usd", "US.USD.BROAD", "pct", "fraction", "Broad trade-weighted USD"),
    Factor("oil", "CMDTY.OIL.WTI", "pct", "fraction", "WTI crude"),
    Factor("vix", "US.VOL.VIX", "diff", "points", "VIX"),
)
FACTOR_NAMES = tuple(f.name for f in FACTORS)


def factor_changes(
    store: PITStore, at: date | datetime, factors: Sequence[Factor] = FACTORS
) -> pl.DataFrame:
    """``date`` (month start) + one column of monthly changes per factor, as known at ``at``."""
    obs = store.as_of([f.series_id for f in factors], at)
    month_end = (
        obs.drop_nulls("value")
        .sort("event_time")
        .group_by("series_id", pl.col("event_time").dt.truncate("1mo").alias("date"))
        .agg(pl.col("value").last())
    )
    out: pl.DataFrame | None = None
    for f in factors:
        lvl = month_end.filter(pl.col("series_id") == f.series_id).sort("date")
        prev = pl.col("value").shift(1)
        chg = pl.col("value") - prev if f.change == "diff" else pl.col("value") / prev - 1
        col = lvl.select("date", chg.alias(f.name))
        out = col if out is None else out.join(col, on="date", how="full", coalesce=True)
    assert out is not None
    return out.sort("date")


class ScenarioAsset(BaseModel):
    asset: str
    expected: float
    p10: float
    p90: float


class ScenarioResult(BaseModel):
    shocks: dict[str, float]
    implied_factor_moves: dict[str, float]  # unshocked factors' conditional moves
    assets: list[ScenarioAsset]  # excess vs the equal-weight average
    average: ScenarioAsset  # the equal-weight average itself
    n_months: int
    window: tuple[date, date]


@dataclass(frozen=True)
class ScenarioModel:
    factors: tuple[str, ...]
    assets: tuple[str, ...]  # last one is AVERAGE
    x: NDArray[np.float64]  # months x factors
    y: NDArray[np.float64]  # months x assets
    months: tuple[date, ...]

    def run(
        self, shocks: Mapping[str, float], *, n_boot: int = 500, block: int = 12, seed: int = 7
    ) -> ScenarioResult:
        unknown = sorted(set(shocks) - set(self.factors))
        if unknown:
            raise ValueError(f"unknown factor(s) {unknown}; choose from {list(self.factors)}")
        if not shocks:
            raise ValueError("give at least one factor shock")
        s_idx = [self.factors.index(f) for f in shocks]
        s = np.array([float(shocks[f]) for f in shocks])

        move = _full_move(self.x, s_idx, s)
        point = _betas(self.x, self.y).T @ move
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot, len(self.assets)))
        for b in range(n_boot):
            rows = _block_rows(len(self.months), block, rng)
            xb, yb = self.x[rows], self.y[rows]
            draws[b] = _betas(xb, yb).T @ _full_move(xb, s_idx, s)
        lo, hi = np.percentile(draws, [10, 90], axis=0)

        rows_out = [
            ScenarioAsset(asset=a, expected=float(point[i]), p10=float(lo[i]), p90=float(hi[i]))
            for i, a in enumerate(self.assets)
        ]
        avg = rows_out.pop()
        return ScenarioResult(
            shocks={f: float(v) for f, v in shocks.items()},
            implied_factor_moves={
                f: float(move[i]) for i, f in enumerate(self.factors) if f not in shocks
            },
            assets=sorted(rows_out, key=lambda r: -r.expected),
            average=avg,
            n_months=len(self.months),
            window=(self.months[0], self.months[-1]),
        )


def fit_scenario_model(
    changes: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    months: Sequence[date] | None = None,
    min_obs: int = 60,
) -> ScenarioModel:
    """``changes`` from ``factor_changes``; ``panel`` has ``date``, ``asset``, ``ret``.

    ``months`` restricts the fit (e.g. to months in the current regime).
    """
    factors = [c for c in changes.columns if c != "date"]
    rets = panel.select(pl.col("date").dt.truncate("1mo"), "asset", "ret").drop_nulls()
    wide = rets.pivot(on="asset", index="date", values="ret").sort("date")
    assets = sorted(c for c in wide.columns if c != "date")
    wide = wide.drop_nulls(assets)  # balanced panel: excess vs a stable average
    avg = pl.mean_horizontal(*assets)
    wide = wide.select("date", *[(pl.col(a) - avg).alias(a) for a in assets], avg.alias(AVERAGE))
    df = changes.drop_nulls(factors).join(wide, on="date").sort("date")
    if months is not None:
        df = df.filter(pl.col("date").is_in(list(months)))
    if df.height < min_obs:
        raise ValueError(f"only {df.height} months with every factor and asset; need {min_obs}")
    return ScenarioModel(
        factors=tuple(factors),
        assets=(*assets, AVERAGE),
        x=df.select(factors).to_numpy().astype(np.float64),
        y=df.select(*assets, AVERAGE).to_numpy().astype(np.float64),
        months=tuple(df["date"].to_list()),
    )


def _betas(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Slopes (factors x assets) of OLS with an intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(coef[1:], dtype=np.float64)


def _full_move(
    x: NDArray[np.float64], s_idx: list[int], s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Shocked factors at ``s``; the rest at their Gaussian conditional mean given ``s``."""
    cov = np.cov(x, rowvar=False)
    u_idx = [i for i in range(x.shape[1]) if i not in s_idx]
    move = np.zeros(x.shape[1])
    move[s_idx] = s
    if u_idx:
        css = cov[np.ix_(s_idx, s_idx)]
        cus = cov[np.ix_(u_idx, s_idx)]
        move[u_idx] = cus @ np.linalg.pinv(css) @ s
    return move


def _block_rows(n: int, block: int, rng: np.random.Generator) -> NDArray[np.int64]:
    starts = rng.integers(0, max(1, n - block + 1), size=-(-n // block))
    rows = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
    return np.asarray(np.minimum(rows, n - 1), dtype=np.int64)
