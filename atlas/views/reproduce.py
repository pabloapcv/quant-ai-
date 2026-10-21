"""Rebuild frozen views from their snapshot (PRD §9: ``atlas reproduce <view_id>``).

A frozen view records the data snapshot and code commit it came from. Reproducing it:

1. find the view file holding ``view_id``;
2. verify the snapshot (every pinned part file present, hashes unchanged);
3. open the store with *only* the snapshot's parts (``PITStore(pinned_parts=...)``), so data
   ingested later is invisible;
4. rerun the view model and compare each rebuilt view's forecast content with the frozen
   one (run metadata such as ``created_at`` excluded, as when publishing).

If the current commit differs from the view's ``code_sha`` the rebuild still runs, and the
report says so: an exact reproduction then needs ``git checkout <code_sha>`` first. A view
frozen from a working tree with uncommitted changes (``code_dirty`` in its snapshot) cannot
be tied to any commit exactly; the report flags it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from atlas.backtest.trials import TrialLog
from atlas.models.expected_returns.ensemble import EnsembleConfig, EnsembleRun, walk_forward
from atlas.pit.store import PITStore
from atlas.signals.research import ResearchContext
from atlas.signals.universes import get_universe
from atlas.views.models import CHAMPION, all_versions, model_options, views_root
from atlas.views.publish import _comparable, build_views
from atlas.views.schema import View


def compute_views(
    store: PITStore,
    data_dir: Path,
    universe: str,
    horizon: int,
    *,
    as_of: date | None = None,
    code_sha: str | None = None,
    snapshot_id: str | None = None,
    version: str = CHAMPION,
) -> tuple[EnsembleRun, list[View], pl.DataFrame]:
    """The view model as the daily pipeline runs it (one definition for both).

    Returns the run, the views and the research panel (date, asset, ret, ...).
    """
    uni = get_universe(universe)
    ctx = ResearchContext(
        panel=uni.build_panel(store),
        universe=universe,
        log=TrialLog(data_dir),
        n_quantiles=3,
        signals=uni.signals,
        families=uni.families,
    )
    cfg = EnsembleConfig(
        horizon=horizon,
        components=uni.ensemble_components,
        version=version,
        **model_options(version),
    )
    run = walk_forward(ctx, cfg)
    views = build_views(run, universe, as_of, code_sha=code_sha, snapshot_id=snapshot_id)
    return run, views, ctx.panel


def find_view(data_dir: Path, view_id: str) -> tuple[Path, list[dict[str, Any]]] | None:
    """The frozen file holding ``view_id``, champion or challenger."""
    paths = [p for v in all_versions() for p in sorted(views_root(data_dir, v).glob("*/*/*.json"))]
    for path in paths:
        views = json.loads(path.read_text())
        if any(v.get("view_id") == view_id for v in views):
            return path, views
    return None


@dataclass
class Reproduction:
    view_id: str
    universe: str
    month: str
    snapshot_id: str | None
    snapshot_problems: list[str] = field(default_factory=list)
    code_sha: str | None = None
    code_dirty: bool | None = None  # uncommitted changes when the snapshot was taken
    current_sha: str | None = None
    matched: int = 0
    total: int = 0
    max_score_diff: float = 0.0
    differences: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return not self.snapshot_problems and self.total > 0 and self.matched == self.total

    @property
    def same_code(self) -> bool:
        return self.code_sha is not None and self.code_sha == self.current_sha


def reproduce(data_dir: Path, view_id: str, current_sha: str | None) -> Reproduction:
    from atlas.pit.snapshot import load_snapshot, verify_snapshot

    found = find_view(data_dir, view_id)
    if found is None:
        raise ValueError(f"no frozen view {view_id!r}")
    path, frozen = found
    universe = path.parent.parent.name
    head = next(v for v in frozen if v["view_id"] == view_id)
    prov = head["provenance"]
    rep = Reproduction(
        view_id=view_id,
        universe=universe,
        month=head["as_of"],
        snapshot_id=prov.get("snapshot_id"),
        code_sha=prov.get("code_sha"),
        current_sha=current_sha,
    )
    if not rep.snapshot_id:
        rep.snapshot_problems.append(
            "view has no snapshot id (frozen before snapshots were pinned)"
        )
        return rep
    try:
        snap = load_snapshot(data_dir, rep.snapshot_id)
    except FileNotFoundError:
        rep.snapshot_problems.append(f"snapshot {rep.snapshot_id} not found")
        return rep
    rep.code_dirty = snap.code_dirty
    rep.snapshot_problems = verify_snapshot(data_dir, snap)
    if rep.snapshot_problems:
        return rep
    store = PITStore(data_dir, pinned_parts=[data_dir / f.path for f in snap.files])
    horizon = int(str(head["horizon"]).rstrip("M"))
    _, rebuilt, _ = compute_views(
        store,
        data_dir,
        universe,
        horizon,
        as_of=date.fromisoformat(head["as_of"]),
        version=prov.get("model_version", CHAMPION),
    )
    old = {v["asset_id"]: v for v in frozen}
    rep.total = len(old)
    for v in rebuilt:
        new = v.model_dump(mode="json")
        prev = old.get(new["asset_id"])
        if prev is None:
            rep.differences.append(f"{new['asset_id']}: not in the frozen file")
            continue
        rep.max_score_diff = max(rep.max_score_diff, abs(new["score"] - prev["score"]))
        a, b = _comparable(prev), _comparable(new)
        if a == b:
            rep.matched += 1
        else:
            keys = sorted(k for k in a if a.get(k) != b.get(k))
            rep.differences.append(f"{new['asset_id']}: {', '.join(keys)} differ")
    missing = set(old) - {v.asset_id for v in rebuilt}
    rep.differences += [f"{a}: frozen but not rebuilt" for a in sorted(missing)]
    return rep
