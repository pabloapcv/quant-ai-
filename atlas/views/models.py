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
