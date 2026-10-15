"""Research runner shared by the CLI, gates and reports.

``run_signal`` is the one way a signal gets backtested: it applies any overlay, records the
trial (so multiple testing is charged for) and attaches the trial statistics the deflated
Sharpe ratio needs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import polars as pl

from atlas.backtest.crosssection import SignalReport, backtest_signal, vol_target
from atlas.backtest.gates import GateResult, evaluate_gates
from atlas.backtest.metrics import pbo_cscv, sharpe
from atlas.backtest.trials import TrialLog
from atlas.signals.industry import FAMILIES, SIGNALS, SignalDef


@dataclass(frozen=True)
class ResearchContext:
    panel: pl.DataFrame  # research_panel(...)
    universe: str
    log: TrialLog
    regimes: pl.DataFrame | None = None
    n_quantiles: int = 5
    cost_bps: float = 10.0
    start: date | None = None
    signals: dict[str, SignalDef] = field(default_factory=lambda: dict(SIGNALS))
    families: dict[str, list[SignalDef]] = field(default_factory=lambda: dict(FAMILIES))


def signal_panel(sig: SignalDef, ctx: ResearchContext) -> pl.DataFrame:
    panel = sig.compute(ctx.panel)
    if ctx.start is not None:
        panel = panel.filter(pl.col("date") >= ctx.start)
    return panel


def run_signal(sig: SignalDef, ctx: ResearchContext) -> SignalReport:
    rep = backtest_signal(
        signal_panel(sig, ctx),
        sig.name,
        n_quantiles=ctx.n_quantiles,
        cost_bps=ctx.cost_bps,
        regimes=ctx.regimes,
    )
    if sig.vol_target is not None:
        rep = vol_target(rep, sig.vol_target, cost_bps=ctx.cost_bps)
    ctx.log.record(
        family=sig.family,
        name=sig.name,
        universe=ctx.universe,
        params=sig.params,
        sharpe_per_period=sharpe(rep.long_short["net"].to_numpy()),
        n_periods=rep.long_short.height,
    )
    stats = ctx.log.stats(universe=ctx.universe)
    return replace(rep, n_trials=stats.n_trials, trial_sr_variance=stats.sr_variance)


def family_pbo(reports: Sequence[SignalReport], n_splits: int = 16) -> tuple[float, int]:
    """PBO across a family's configurations, on their common months. (pbo, months)."""
    net = reports[0].long_short.select("date", pl.col("net").alias(reports[0].name))
    for rep in reports[1:]:
        net = net.join(rep.long_short.select("date", pl.col("net").alias(rep.name)), on="date")
    return pbo_cscv(net.drop("date").to_numpy(), n_splits=n_splits), net.height


def _icir(panel: pl.DataFrame, col: str, periods_per_year: int = 12) -> float:
    ic = (
        panel.drop_nulls([col, "fwd_ret"])
        .group_by("date")
        .agg(pl.corr(col, "fwd_ret", method="spearman").alias("ic"))["ic"]
        .drop_nans()
        .drop_nulls()
        .to_numpy()
    )
    if ic.size < 12 or float(np.std(ic, ddof=1)) == 0:
        return math.nan
    return float(np.mean(ic) / np.std(ic, ddof=1) * math.sqrt(periods_per_year))


def ensemble_marginal_icir(panels: dict[str, pl.DataFrame]) -> tuple[float, dict[str, float]]:
    """ICIR of an equal-weight z-score ensemble, and each signal's marginal contribution.

    marginal(s) = ICIR(all) - ICIR(all but s). Positive means the ensemble is better with s.
    """
    names = list(panels)
    wide: pl.DataFrame | None = None
    for name, p in panels.items():
        z = p.select(
            "date",
            "asset",
            "fwd_ret",
            (
                (pl.col("signal") - pl.col("signal").mean().over("date"))
                / pl.col("signal").std().over("date")
            ).alias(name),
        )
        wide = z if wide is None else wide.join(z.drop("fwd_ret"), on=["date", "asset"], how="left")
    assert wide is not None

    def icir_of(cols: list[str]) -> float:
        return _icir(wide.with_columns(pl.mean_horizontal(cols).alias("_ens")), "_ens")

    full = icir_of(names)
    return full, {n: full - icir_of([c for c in names if c != n]) for n in names}


@dataclass(frozen=True)
class GateSuite:
    """Everything one full research pass produces for a universe."""

    universe: str
    reports: dict[str, SignalReport]  # representative signal -> report
    family_reports: dict[str, list[SignalReport]]
    pbo: dict[str, float]
    pbo_months: dict[str, int]
    ensemble_icir: float
    gates: dict[str, GateResult]
    n_trials: int
    signals: dict[str, SignalDef] = field(default_factory=dict)


def _try_run(sig: SignalDef, ctx: ResearchContext) -> SignalReport | None:
    """``run_signal``, or None when the signal's inputs are not in the store."""
    try:
        return run_signal(sig, ctx)
    except ValueError:
        return None


def run_gate_suite(ctx: ResearchContext, *, n_splits: int = 16) -> GateSuite:
    """Run every family (so the trial count is complete), then gate each signal."""
    family_reports: dict[str, list[SignalReport]] = {}
    for f, members in ctx.families.items():
        reps = [r for r in (_try_run(s, ctx) for s in members) if r is not None]
        if reps:
            family_reports[f] = reps
    pbo, months = {}, {}
    for f, reps in family_reports.items():
        if len(reps) >= 2:
            pbo[f], months[f] = family_pbo(reps, n_splits=n_splits)
    reports = {n: r for n, s in ctx.signals.items() if (r := _try_run(s, ctx)) is not None}
    # Overlays share their base signal's ranking, so only distinct rankings enter the ensemble.
    rankings = {n: s for n, s in ctx.signals.items() if s.ranking_of is None and n in reports}
    ens, marginal = ensemble_marginal_icir({n: signal_panel(s, ctx) for n, s in rankings.items()})
    gates = {
        n: evaluate_gates(
            s, reports[n], pbo=pbo.get(s.family), marginal_icir=marginal[s.ranking_of or n]
        )
        for n, s in ctx.signals.items()
        if n in reports and (s.ranking_of or n) in marginal
    }
    return GateSuite(
        universe=ctx.universe,
        reports=reports,
        family_reports=family_reports,
        pbo=pbo,
        pbo_months=months,
        ensemble_icir=ens,
        gates=gates,
        n_trials=ctx.log.stats(universe=ctx.universe).n_trials,
        signals=ctx.signals,
    )
