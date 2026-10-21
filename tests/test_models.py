from __future__ import annotations

from datetime import date

import numpy as np

from atlas.views.models import compare
from atlas.views.track import ScoredMonth, TrackRecord


def _rec(ics: list[float], universe: str = "ind12") -> TrackRecord:
    months = [date(2027 + i // 12, i % 12 + 1, 1) for i in range(len(ics))]
    return TrackRecord(
        scored=[
            ScoredMonth(universe, "3M", m, 12, ic, None, 0, 0.25, 0.8)
            for m, ic in zip(months, ics, strict=True)
        ]
    )


def test_compare_needs_enough_shared_months() -> None:
    out = compare(_rec([0.0] * 6), _rec([0.3] * 6))
    assert out["shared_months"] == 6
    assert out["verdict"].startswith("keep champion: 6 of 12")


def test_compare_promotes_only_on_significant_paired_gain() -> None:
    rng = np.random.default_rng(1)
    base = list(rng.normal(0.02, 0.1, 24))
    better = [x + 0.08 + float(rng.normal(0, 0.02)) for x in base]
    same = [x + float(rng.normal(0, 0.05)) for x in base]
    win = compare(_rec(base), _rec(better))
    assert win["paired_tstat"] > 2
    assert "qualifies for promotion" in win["verdict"]
    tie = compare(_rec(base), _rec(same))
    assert "not significantly better" in tie["verdict"]
    other = compare(_rec(base), _rec(better, universe="fx_g10"))
    assert other["shared_months"] == 0  # months only pair within the same universe
