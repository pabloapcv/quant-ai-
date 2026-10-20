"""Locked holdout (PRD §8.2): 2021-2025 is opened once per model family, and every opening
is logged.

Research runs should stop before the holdout (``atlas research gates --exclude-holdout``,
``ResearchContext.end``): the last research month is the one whose forward return still
ends before ``HOLDOUT_START``. Opening it (``atlas research holdout open``) scores the
ensemble and each signal on holdout months only and appends the result to
``data/research/holdout_log.jsonl``; a second opening for the same universe and family is
refused unless explicitly forced, and is then logged as a reopening.

An opening is ``contaminated`` unless the caller states the family was researched with the
holdout excluded (``--clean``). Every universe built before the registry existed (October
2026) was developed and reported on the full sample, so its logged opening is contaminated:
its holdout numbers are out of sample for the walk-forward fits, but not a clean test of the
research choices (signals, components, settings) made while looking at those years.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

HOLDOUT_START = date(2021, 1, 1)
HOLDOUT_END = date(2025, 12, 31)


class HoldoutAlreadyOpened(RuntimeError):
    pass


class HoldoutLog:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "research" / "holdout_log.jsonl"

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def openings(self, universe: str, family: str) -> list[dict[str, Any]]:
        return [e for e in self.entries() if e["universe"] == universe and e["family"] == family]

    def record(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def in_holdout(col: str = "date") -> pl.Expr:
    return (pl.col(col) >= HOLDOUT_START) & (pl.col(col) <= HOLDOUT_END)


def holdout_scores(ctx: Any, cfg: Any) -> dict[str, Any]:
    """Ensemble and per-signal results on holdout months (dates whose outcome is inside it)."""
    from atlas.backtest.crosssection import backtest_signal
    from atlas.models.expected_returns.ensemble import EnsembleRun, walk_forward
    from atlas.signals.research import signal_panel
    from atlas.views.publish import evaluate

    full = replace(ctx, start=None, end=None)  # fits may use pre-holdout history
    run = walk_forward(full, cfg)
    h = cfg.horizon
    window = run.forecasts.filter(
        (pl.col("date") >= HOLDOUT_START) & (pl.col("date").dt.offset_by(f"{h}mo") <= HOLDOUT_END)
    )
    ens = evaluate(EnsembleRun(config=run.config, forecasts=window, weights=run.weights))
    signals: dict[str, dict[str, float]] = {}
    for name, sig in full.signals.items():
        if sig.ranking_of is not None:
            continue
        p = signal_panel(sig, full).filter(
            (pl.col("date") >= HOLDOUT_START) & (pl.col("date").dt.offset_by("1mo") <= HOLDOUT_END)
        )
        if p.drop_nulls(["signal", "fwd_ret"]).is_empty():
            continue
        rep = backtest_signal(p, name, n_quantiles=ctx.n_quantiles, cost_bps=ctx.cost_bps)
        signals[name] = {"ic_mean": rep.ic_mean, "ic_tstat": rep.ic_tstat, "months": rep.ic.height}
    summary = {
        k: v for k, v in ens.summary().items() if isinstance(v, int | float) and np.isfinite(v)
    }
    return {"ensemble": summary, "signals": signals}


def open_holdout(
    data_dir: Path,
    ctx: Any,
    cfg: Any,
    *,
    family: str,
    reason: str,
    code_sha: str,
    allow_reopen: bool = False,
    clean: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    log = HoldoutLog(data_dir)
    previous = log.openings(ctx.universe, family)
    if previous and not allow_reopen:
        raise HoldoutAlreadyOpened(
            f"holdout already opened for {ctx.universe}/{family} on {previous[0]['opened_at']}"
        )
    now = now or datetime.now(UTC)
    result = holdout_scores(ctx, cfg)
    entry = {
        "opened_at": now.isoformat(),
        "universe": ctx.universe,
        "family": family,
        "window": [HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()],
        "reopening": bool(previous),
        "contaminated": not clean,
        "reason": reason,
        "code_sha": code_sha,
        "config_version": getattr(cfg, "version", ""),
        **result,
    }
    log.record(entry)
    return entry
