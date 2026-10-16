"""Live paper track record (PRD §8.5, Phase 8).

Scores frozen views — written once by ``atlas.views.publish`` before their outcome existed —
against what happened next. Unlike a backtest, nothing here can be tuned after the fact:
the forecast files are immutable, and outcomes come from the latest return data.

A view file for month ``m`` at horizon ``h`` is scored once returns for ``m+1 .. m+h`` exist
for its assets; until then it is reported as pending.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from atlas.models.expected_returns.ensemble import forward_return


@dataclass(frozen=True)
class ScoredMonth:
    universe: str
    horizon: str
    as_of: date
    n_assets: int
    rank_ic: float
    hit_rate: float | None  # non-neutral calls right vs the median asset
    n_calls: int
    brier: float
    coverage: float  # realized excess inside [p10, p90]


@dataclass(frozen=True)
class TrackRecord:
    scored: list[ScoredMonth] = field(default_factory=list)
    pending: list[tuple[str, str, date]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        if not self.scored:
            return {"months_scored": 0, "months_pending": len(self.pending)}
        ics = np.array([s.rank_ic for s in self.scored if math.isfinite(s.rank_ic)])
        calls = sum(s.n_calls for s in self.scored)
        hits = sum((s.hit_rate or 0) * s.n_calls for s in self.scored)
        return {
            "months_scored": len(self.scored),
            "months_pending": len(self.pending),
            "mean_rank_ic": float(ics.mean()) if ics.size else None,
            "directional_calls": calls,
            "hit_rate": hits / calls if calls else None,
            "brier": float(np.mean([s.brier for s in self.scored])),
            "coverage": float(np.mean([s.coverage for s in self.scored])),
        }


def _months(h: str) -> int:
    if not h.endswith("M") or not h[:-1].isdigit():
        raise ValueError(f"unsupported horizon {h!r}")
    return int(h[:-1])


def score_file(path: Path, returns: pl.DataFrame) -> ScoredMonth | None:
    """Score one frozen views file; None if its horizon has not fully elapsed yet."""
    views = json.loads(path.read_text())
    if not views:
        return None
    universe = path.parent.parent.name
    horizon = views[0]["horizon"]
    h = _months(horizon)
    as_of = date.fromisoformat(views[0]["as_of"])
    outcome = (
        returns.sort(["asset", "date"])
        .with_columns(forward_return(h).alias("fwd"))
        .filter(pl.col("date") == as_of)
        .select("asset", "fwd")
    )
    v = pl.DataFrame(
        {
            "asset": [x["asset_id"].rsplit(".", 1)[-1] for x in views],
            "score": [x["score"] for x in views],
            "prob": [x["prob_outperform"] for x in views],
            "stance": [x["stance"] for x in views],
            "p10": [x["expected_excess_return"]["p10"] for x in views],
            "p90": [x["expected_excess_return"]["p90"] for x in views],
        },
        schema={
            "asset": pl.String,
            "score": pl.Float64,
            "prob": pl.Float64,
            "stance": pl.String,
            "p10": pl.Float64,
            "p90": pl.Float64,
        },
    )
    df = v.join(outcome, on="asset", how="left")
    if df["fwd"].null_count() > 0:
        return None  # some outcome not realized (or not yet published) -> pending
    df = df.with_columns(
        (pl.col("fwd") - pl.col("fwd").mean()).alias("excess"),
        (pl.col("fwd") - pl.col("fwd").median()).alias("excess_med"),
    )
    ic = float(spearmanr(df["score"], df["fwd"]).statistic)
    untied = df.filter(pl.col("excess_med") != 0)
    calls = untied.filter(~pl.col("stance").eq("neutral"))
    hit = None
    if calls.height:
        right = (calls["excess_med"] > 0) == calls["stance"].str.contains("overweight")
        hit = float(np.mean(right.to_numpy()))
    probs = untied.drop_nulls("prob")
    y = (probs["excess_med"] > 0).cast(pl.Float64).to_numpy()
    brier = float(np.mean((probs["prob"].to_numpy() - y) ** 2)) if y.size else math.nan
    inside = df.drop_nulls(["p10", "p90"])
    within = ((inside["excess"] >= inside["p10"]) & (inside["excess"] <= inside["p90"])).to_numpy()
    coverage = float(np.mean(within)) if within.size else math.nan
    return ScoredMonth(
        universe=universe,
        horizon=horizon,
        as_of=as_of,
        n_assets=df.height,
        rank_ic=ic,
        hit_rate=hit,
        n_calls=calls.height,
        brier=brier,
        coverage=coverage,
    )


def score_all(data_dir: Path, returns_by_universe: dict[str, pl.DataFrame]) -> TrackRecord:
    root = data_dir / "gold" / "views"
    scored: list[ScoredMonth] = []
    pending: list[tuple[str, str, date]] = []
    if not root.exists():
        return TrackRecord()
    for path in sorted(root.glob("*/*/*.json")):
        universe, horizon = path.parent.parent.name, path.parent.name
        returns = returns_by_universe.get(universe)
        if returns is None:
            continue
        result = score_file(path, returns)
        if result is None:
            pending.append((universe, horizon, date.fromisoformat(path.stem)))
        else:
            scored.append(result)
    return TrackRecord(scored=scored, pending=pending)
