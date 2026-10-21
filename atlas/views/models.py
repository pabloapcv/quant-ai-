"""Champion and challenger view models (PRD §9 champion/challenger, §8.5 live track).

The champion's views are the published ones (``gold/views``): what the assistant, alerts
and reports use. Each challenger's views are frozen alongside, with the same snapshot, into
``gold/challengers/<version>/`` and scored by the same live track, so a challenger can only
replace the champion on evidence it could not have been tuned to.

Challenger now: ``ens_v1_gbm``, the gradient-boosted combiner, which beat the IC-weighted
champion on 3 of 6 universes in the walk-forward and lost on the other 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CHAMPION = "ens_v1"
CHALLENGERS: dict[str, dict[str, Any]] = {"ens_v1_gbm": {"combiner": "gbm"}}


def views_root(data_dir: Path, version: str = CHAMPION) -> Path:
    if version == CHAMPION:
        return data_dir / "gold" / "views"
    if version not in CHALLENGERS:
        raise ValueError(f"unknown model version {version!r}")
    return data_dir / "gold" / "challengers" / version


def model_options(version: str) -> dict[str, Any]:
    """EnsembleConfig overrides for a model version (champion: none)."""
    if version == CHAMPION:
        return {}
    return dict(CHALLENGERS[version])


def all_versions() -> list[str]:
    return [CHAMPION, *CHALLENGERS]


MIN_SHARED_MONTHS = 12
PROMOTE_TSTAT = 2.0


def compare(champion: Any, challenger: Any) -> dict[str, Any]:
    """Paired test of live rank IC on the view-months both models have scored.

    ``champion`` / ``challenger`` are ``TrackRecord``s. The verdict follows the runbook:
    promote only with at least ``MIN_SHARED_MONTHS`` shared months and a paired t-stat of
    ``PROMOTE_TSTAT`` or more in the challenger's favour.
    """
    import math

    import numpy as np

    a = {(m.universe, m.horizon, m.as_of): m.rank_ic for m in champion.scored}
    b = {(m.universe, m.horizon, m.as_of): m.rank_ic for m in challenger.scored}
    keys = sorted(k for k in a.keys() & b.keys() if math.isfinite(a[k]) and math.isfinite(b[k]))
    diff = np.array([b[k] - a[k] for k in keys])
    n = len(keys)
    out: dict[str, Any] = {"shared_months": n}
    if n:
        out["champion_ic"] = float(np.mean([a[k] for k in keys]))
        out["challenger_ic"] = float(np.mean([b[k] for k in keys]))
        out["mean_difference"] = float(diff.mean())
    t = (
        float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n)))
        if n > 1 and diff.std(ddof=1) > 0
        else None
    )
    out["paired_tstat"] = t
    if n < MIN_SHARED_MONTHS:
        out["verdict"] = f"keep champion: {n} of {MIN_SHARED_MONTHS} shared scored months so far"
    elif t is not None and t >= PROMOTE_TSTAT:
        out["verdict"] = "challenger qualifies for promotion (as a new model version)"
    else:
        out["verdict"] = "keep champion: the challenger is not significantly better"
    return out
