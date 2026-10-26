from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from atlas.pit.snapshot import create_snapshot
from atlas.pit.store import PITStore
from atlas.views.publish import publish
from atlas.views.reproduce import compute_views, reproduce
from tests.conftest import make_obs_frame
from tests.test_country import _country_store

REPO = Path(__file__).resolve().parents[1]


def _freeze(data: Path) -> str:
    store = _country_store(data)
    snap = create_snapshot(data, REPO / "config")
    _, views, _ = compute_views(
        store, data, "ctry_dev_lcl", 3, code_sha="abc123", snapshot_id=snap.snapshot_id
    )
    publish(views, data, "ctry_dev_lcl")
    return views[0].view_id


def test_reproduce_ignores_later_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    view_id = _freeze(data)
    # New data arrives after the freeze: Japan's whole share-price history is revised into
    # a strong uptrend, visible to any as-of-now read.
    later = datetime.now(UTC)
    rows = [
        ("JPN.EQUITY.SHARE", date(1995 + t // 12, t % 12 + 1, 1), later, 100 * 1.02**t)
        for t in range(240)
    ]
    PITStore(data).write(make_obs_frame(rows))
    frozen_file = next((data / "gold" / "views" / "ctry_dev_lcl" / "3M").glob("*.json"))
    frozen = {v["asset_id"]: v["score"] for v in json.loads(frozen_file.read_text())}
    _, unpinned, _ = compute_views(PITStore(data), data, "ctry_dev_lcl", 3)
    assert frozen != {v.asset_id: v.score for v in unpinned}  # the new data does matter...
    rep = reproduce(data, view_id, current_sha="abc123")
    assert rep.snapshot_problems == []
    assert rep.total == 9
    assert rep.matched == 9, rep.differences  # ...but the pinned snapshot never sees it
    assert rep.exact
    assert rep.same_code
    assert reproduce(data, view_id, current_sha="other").same_code is False


def test_reproduce_reports_broken_snapshot(tmp_path: Path) -> None:
    data = tmp_path / "data"
    view_id = _freeze(data)
    part = next((data / "silver" / "observations").rglob("*.parquet"))
    part.write_bytes(part.read_bytes() + b"tampered")
    rep = reproduce(data, view_id, current_sha="abc123")
    assert rep.snapshot_problems
    assert not rep.exact


def test_views_frozen_before_context_fields_still_reproduce(tmp_path: Path) -> None:
    data = tmp_path / "data"
    view_id = _freeze(data)
    path = next((data / "gold" / "views" / "ctry_dev_lcl" / "3M").glob("*.json"))
    old = json.loads(path.read_text())
    for v in old:  # as written before regime_context / historical_analogs / risks existed
        for k in ("regime_context", "historical_analogs", "risks"):
            v.pop(k, None)
    path.write_text(json.dumps(old))
    rep = reproduce(data, view_id, current_sha="abc123")
    assert rep.matched == rep.total
    assert rep.total > 0


def test_context_absent_without_macro_data(tmp_path: Path) -> None:
    from atlas.views.reproduce import with_context

    data = tmp_path / "data"
    store = _country_store(data)  # country prices only: no US macro series
    _, views, _ = compute_views(store, data, "ctry_dev_lcl", 3)
    assert all(v.regime_context is None and v.risks == [] for v in views)
    assert with_context([], store, "ctry_dev_lcl") == []
