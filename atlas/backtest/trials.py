"""Trial log: every backtest run is recorded so multiple testing can be charged for (DSR).

PRD §8.3 asks for the number of trials to be counted honestly. Each research run appends
one JSON line; ``stats()`` returns the trial count and the cross-trial variance of
per-period Sharpe ratios that ``deflated_sharpe_ratio`` needs. (MLflow replaces this file
in a later phase; the interface stays.)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrialStats:
    n_trials: int
    sr_variance: float | None


class TrialLog:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "research" / "trials.jsonl"

    def record(
        self,
        *,
        family: str,
        name: str,
        universe: str,
        params: dict[str, Any],
        sharpe_per_period: float,
        n_periods: int,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "family": family,
            "name": name,
            "universe": universe,
            "params": params,
            "sharpe_per_period": sharpe_per_period if math.isfinite(sharpe_per_period) else None,
            "n_periods": n_periods,
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def stats(self, *, universe: str | None = None) -> TrialStats:
        """Distinct (name, params) configurations tried on ``universe`` (all if None)."""
        seen: dict[str, float | None] = {}
        for e in self.entries():
            if universe is None or e["universe"] == universe:
                key = f"{e['name']}|{json.dumps(e['params'], sort_keys=True)}"
                seen[key] = e["sharpe_per_period"]
        srs = np.array([v for v in seen.values() if v is not None], dtype=float)
        var = float(srs.var(ddof=1)) if srs.size >= 2 else None
        return TrialStats(n_trials=max(len(seen), 1), sr_variance=var)
