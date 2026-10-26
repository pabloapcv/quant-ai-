"""Context attached to each published view (PRD §3.2): regime, historical analogs, risks.

Everything is computed as of the view's month from the same store as the view itself, so
the daily pipeline and ``atlas reproduce`` give identical context:

* ``regime_context``: the growth x inflation quadrant with the highest probability, read
  from the macro data known at the end of the view's month.
* ``historical_analogs``: the 3 past months whose macro state is closest (the analog search
  skips the 24 most recent months, so every 3-month outcome was known by then) and the
  asset's 3-month return vs the equal-weight average after each.
* ``risks``: the two one-click what-if events in which the asset historically did worst
  against its peers in the same month, from a factor model fitted on months up to the view's
  month only. History, not a forecast.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from atlas.pit.store import PITStore
from atlas.signals.universes import get_universe

ANALOGS = 3
RISKS = 2
MIN_RISK = 0.005  # an event counts as a risk only below -0.5% vs the average


def _month_end(month: date) -> date:
    nxt = date(month.year + month.month // 12, month.month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def view_context(store: PITStore, universe: str, month: date) -> dict[str, dict[str, Any]]:
    """asset id (panel's) -> {regime_context, historical_analogs, risks}; {} without macro data."""
    from atlas.lens import PRESETS
    from atlas.models.regime.analogs import find_analogs, forward_excess
    from atlas.models.regime.engine import RegimeEngine, data_month_for
    from atlas.models.scenario import FACTORS, factor_changes, fit_scenario_model

    cube = store.load_cube()
    if "US.IP" not in cube or any(f.series_id not in cube for f in FACTORS):
        return {}
    at = _month_end(month)
    engine = RegimeEngine(cube)
    rep = engine.run(at, with_hmm=False)
    probs = {str(k): float(v) for k, v in rep.quadrant_probs.items()}
    top = max(probs, key=lambda k: probs[k]) if probs else str(rep.regime)
    regime = {
        "regime": top,
        "prob": round(probs.get(top, 0.0), 3),
        "data_month": str(rep.data_month),
    }

    panel = get_universe(universe).build_panel(store)
    assets = sorted(panel["asset"].unique().to_list())
    out: dict[str, dict[str, Any]] = {
        a: {"regime_context": regime, "historical_analogs": [], "risks": []} for a in assets
    }

    comps, _ = engine.composites(at)
    found = find_analogs(comps, current_month=data_month_for(at), k=ANALOGS)
    if found is not None:
        months = [an.month for an in found.analogs]
        ex = forward_excess(panel, months, 3)
        for an in found.analogs:
            rows = ex.filter(pl.col("month") == an.month)
            for r in rows.iter_rows(named=True):
                if r["asset"] in out:
                    out[r["asset"]]["historical_analogs"].append(
                        {
                            "period": an.month.isoformat()[:7],
                            "distance": round(an.distance, 2),
                            "fwd_3m_excess": round(r["excess"], 4),
                        }
                    )

    hist = panel.filter(pl.col("date") <= month)
    fit_months = hist["date"].unique().to_list()
    model = fit_scenario_model(factor_changes(store, at), hist, months=fit_months)
    effects: dict[str, list[tuple[float, str]]] = {a: [] for a in assets}
    for p in PRESETS:
        # point estimates only: the bootstrap bands are not part of the view
        res = model.run({f: float(x) for f, x in p["shocks"].items()}, n_boot=2)
        for row in res.assets:
            if row.asset in effects:
                effects[row.asset].append((row.expected, p["when"]))
    for asset, evs in effects.items():
        worst = sorted(e for e in evs if e[0] < -MIN_RISK)[:RISKS]
        out[asset]["risks"] = [
            f"Historically weak when {when}: {x * 100:+.1f}% vs the average in the same month"
            for x, when in worst
        ]
    return out
