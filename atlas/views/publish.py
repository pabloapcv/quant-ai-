"""Turn ensemble forecasts into immutable ``View`` objects and score them (PRD §3, §8.5).

Views are written once per (universe, as-of month) to
``{data_dir}/gold/views/{universe}/{as_of}.json`` and never overwritten: the live track
record is computed from these frozen files, so it cannot be revised after the fact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from atlas import jsonsafe
from atlas.models.expected_returns.ensemble import EnsembleRun
from atlas.signals.universes import get_universe
from atlas.views.schema import (
    Driver,
    ExpectedReturn,
    Provenance,
    Stance,
    View,
    confidence_for,
    stance_with_hysteresis,
)


class ViewExistsError(RuntimeError):
    pass


def with_stances(forecasts: pl.DataFrame) -> pl.DataFrame:
    """Add ``stance`` per (asset, date), applying hysteresis through time per asset."""
    out = []
    for (_asset,), part in (
        forecasts.sort(["asset", "date"]).partition_by("asset", as_dict=True).items()
    ):
        prev: Stance | None = None
        stances = []
        for prob, agree in zip(part["prob"].to_list(), part["agreement"].to_list(), strict=True):
            prev = stance_with_hysteresis(prob, agree, prev) if prob is not None else prev
            stances.append(None if prev is None else prev.value)
        out.append(part.with_columns(pl.Series("stance", stances, dtype=pl.String)))
    return pl.concat(out).sort(["date", "asset"])


def build_views(
    run: EnsembleRun,
    universe: str,
    as_of: date | None = None,
    *,
    code_sha: str | None = None,
    n_trials: int | None = None,
    snapshot_id: str | None = None,
    created_at: datetime | None = None,
) -> list[View]:
    """Views for one month (default: the latest month with a score)."""
    uni = get_universe(universe)
    fc = with_stances(run.forecasts)
    scored = fc.drop_nulls("score")
    latest = scored["date"].max()
    month: date = as_of or (latest if isinstance(latest, date) else date.min)
    rows = scored.filter(pl.col("date") == month)
    if rows.is_empty():
        raise ValueError(f"no forecasts for {month}")
    weights = run.weights.filter(pl.col("date") == month).row(0, named=True)
    created = created_at or datetime.now(UTC)
    views = []
    for r in rows.iter_rows(named=True):
        drivers = [
            Driver(
                name=n,
                contribution=float(weights[n]) * float(r[f"z_{n}"]),
                value=float(r[f"z_{n}"]),
            )
            for n in run.component_list
            if n in weights and weights[n] > 0 and r.get(f"z_{n}") is not None
        ]
        drivers.sort(key=lambda d: -abs(d.contribution))
        stance = Stance(r["stance"]) if r["stance"] else Stance.NEUTRAL
        uid = hashlib.sha256(
            f"{universe}|{month}|{r['asset']}|{run.config.version}".encode()
        ).hexdigest()
        views.append(
            View(
                view_id=f"{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{uid[20:32]}",
                created_at=created,
                as_of=month,
                asset_id=f"{uni.asset_id_prefix}.{r['asset'].upper()}",
                asset_name=uni.asset_name(r["asset"]),
                asset_class=uni.asset_class,
                horizon=f"{run.config.horizon}M",
                stance=stance,
                score=max(-3.0, min(3.0, float(r["score"]))),
                expected_excess_return=ExpectedReturn(
                    mean=float(r["exp_excess"]) if r["exp_excess"] is not None else 0.0,
                    p10=r["p_lo"],
                    p90=r["p_hi"],
                    benchmark=f"equal-weight {universe}",
                ),
                prob_outperform=r["prob"],
                agreement=r["agreement"],
                confidence=confidence_for(r["prob"], r["agreement"]),
                drivers=drivers,
                invalidation=(
                    "Stance changes when the calibrated probability crosses its PRD threshold "
                    "by 0.02 at a monthly rebalance; reviewed if model agreement drops below 50%."
                ),
                provenance=Provenance(
                    model_version=run.config.version,
                    code_sha=code_sha,
                    data_through=month,
                    n_trials=n_trials,
                    snapshot_id=snapshot_id,
                ),
            )
        )
    return sorted(views, key=lambda v: -v.score)


def _comparable(view: dict[str, Any]) -> dict[str, Any]:
    """A view's forecast content, without run metadata (when, from which commit/snapshot)."""
    out = {k: v for k, v in view.items() if k != "created_at"}
    volatile = {"snapshot_id", "code_sha"}
    out["provenance"] = {k: v for k, v in view["provenance"].items() if k not in volatile}
    return out


def views_dir(data_dir: Path, universe: str, horizon: str) -> Path:
    """Frozen views for one universe and horizon: ``gold/views/<universe>/<horizon>/``."""
    return data_dir / "gold" / "views" / universe / horizon


def publish(views: list[View], data_dir: Path, universe: str) -> Path:
    """Write views once. Re-publishing identical content is a no-op; different content fails."""
    if not views:
        raise ValueError("no views to publish")
    month = views[0].as_of
    path = views_dir(data_dir, universe, views[0].horizon) / f"{month.isoformat()}.json"
    body = [_comparable(v.model_dump(mode="json")) for v in views]
    if path.exists():
        existing = [_comparable(e) for e in json.loads(path.read_text())]
        if existing == body:
            return path
        raise ViewExistsError(
            f"views for {universe} {month} already published with different content"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(jsonsafe.dumps([v.model_dump(mode="json") for v in views], indent=2))
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------- evaluation
@dataclass(frozen=True)
class ViewEvaluation:
    months: int
    ic_mean: float
    ic_tstat: float
    icir_annual: float
    brier: float
    brier_baseline: float  # always 0.5
    brier_improvement: float
    coverage: float  # share of realized excess returns inside [p10, p90]
    hit_rate: float  # directional calls (non-neutral) that were right
    ow_minus_uw_annual: float  # mean excess of overweights minus underweights, annualized
    reliability: pl.DataFrame  # prob bucket -> mean prob, realized frequency, n

    def summary(self) -> dict[str, float | int | None]:
        """Scalar metrics; undefined ones (e.g. no month with both OW and UW calls) are None."""
        out: dict[str, float | int | None] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, pl.DataFrame):
                continue
            out[k] = None if isinstance(v, float) and not math.isfinite(v) else v
        return out


def evaluate(run: EnsembleRun, start: date | None = None) -> ViewEvaluation:
    """Walk-forward scoring of every month's forecasts against what happened next."""
    from atlas.backtest.metrics import newey_west_tstat

    fc = with_stances(run.forecasts).drop_nulls(["score", "excess"])
    if start is not None:
        fc = fc.filter(pl.col("date") >= start)
    ic = (
        fc.group_by("date")
        .agg(pl.corr("score", "excess", method="spearman").alias("ic"))
        .sort("date")
    )
    ic_arr = ic["ic"].drop_nulls().drop_nans().to_numpy()
    sd = float(np.std(ic_arr, ddof=1)) if ic_arr.size > 1 else math.nan

    cal = (
        fc.drop_nulls("prob")
        .filter(pl.col("excess_med") != 0)
        .with_columns((pl.col("excess_med") > 0).cast(pl.Float64).alias("y"))
    )
    y = cal["y"].to_numpy()
    p = cal["prob"].to_numpy()
    brier = float(np.mean((p - y) ** 2)) if y.size else math.nan
    base = float(np.mean((0.5 - y) ** 2)) if y.size else math.nan
    rel = (
        cal.with_columns((pl.col("prob") * 10).floor().clip(0, 9).alias("bucket"))
        .group_by("bucket")
        .agg(
            pl.col("prob").mean().alias("mean_prob"),
            pl.col("y").mean().alias("realized"),
            pl.len().alias("n"),
        )
        .sort("bucket")
    )
    cov = fc.drop_nulls(["p_lo", "p_hi"])
    inside = ((cov["excess"] >= cov["p_lo"]) & (cov["excess"] <= cov["p_hi"])).to_numpy()
    coverage = float(inside.mean()) if inside.size else math.nan
    ow = pl.col("stance").is_in([Stance.OVERWEIGHT.value, Stance.STRONG_OVERWEIGHT.value])
    uw = pl.col("stance").is_in([Stance.UNDERWEIGHT.value, Stance.STRONG_UNDERWEIGHT.value])
    calls = fc.filter((ow | uw) & (pl.col("excess_med") != 0))
    hits = calls.select(
        ((pl.col("excess_med") > 0) == pl.col("stance").str.contains("overweight")).mean()
    ).item()
    spread = (
        fc.group_by("date")
        .agg(
            pl.col("excess").filter(ow).mean().alias("ow"),
            pl.col("excess").filter(uw).mean().alias("uw"),
        )
        .drop_nulls()
        .select((pl.col("ow") - pl.col("uw")).mean())
        .item()
    )
    return ViewEvaluation(
        months=ic.height,
        ic_mean=float(np.mean(ic_arr)) if ic_arr.size else math.nan,
        ic_tstat=newey_west_tstat(ic_arr),
        icir_annual=float(np.mean(ic_arr) / sd * math.sqrt(12)) if sd and sd > 0 else math.nan,
        brier=brier,
        brier_baseline=base,
        brier_improvement=(base - brier) / base if base else math.nan,
        coverage=coverage,
        hit_rate=float(hits) if hits is not None else math.nan,
        ow_minus_uw_annual=float(spread) * 12 / run.config.horizon
        if spread is not None
        else math.nan,
        reliability=rel,
    )
