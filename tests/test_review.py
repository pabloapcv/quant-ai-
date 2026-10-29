from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from atlas.review import _drift, monthly_review, review_due, review_markdown


def _run(data: Path, name: str, steps: list[tuple[str, bool]], ok: bool) -> None:
    (data / "runs").mkdir(parents=True, exist_ok=True)
    body = {"ok": ok, "stale": [], "steps": [{"name": n, "ok": o} for n, o in steps]}
    (data / "runs" / name).write_text(json.dumps(body))


def test_review_decisions(tmp_path: Path) -> None:
    for i in range(4):
        _run(tmp_path, f"202610{10 + i:02d}T063000Z.json", [("ingest:ecb", False)], False)
    _run(tmp_path, "20261020T063000Z.json", [("ingest:ecb", True)], True)
    (tmp_path / "usage").mkdir()
    (tmp_path / "usage" / "2026-10-20.jsonl").write_text(
        json.dumps({"purpose": "chat", "weighted": 1000.5}) + "\n"
    )
    verdicts = {
        "ens_v1_gbm": {"verdict": "challenger qualifies for promotion (as a new model version)"}
    }
    r = monthly_review(tmp_path, "2026-10", {"months_scored": 0}, {"ens_v1_gbm": {}}, verdicts, [])
    assert r["runs"] == 5
    assert r["runs_ok"] == 1
    assert r["failed_steps"] == {"ingest:ecb": 4}
    assert r["usage"] == {"calls": 1, "weighted_tokens": 1000, "by_purpose": {"chat": 1000.5}}
    assert r["decisions"][0].startswith("Challenger ens_v1_gbm qualifies")
    assert "Fix 'ingest:ecb': it failed on 4 runs this month." in r["decisions"]
    md = review_markdown(r, date(2026, 11, 2))
    assert md.startswith("# Ramon monthly model review, 2026-10")


def test_drift_flag() -> None:
    mon = {
        "ind12:3M": {"backtest": {"mean": 0.05, "sd": 0.1}, "live_ics": [-0.2, -0.3, -0.25]},
        "fx_g10:3M": {"backtest": {"mean": 0.05, "sd": 0.1}, "live_ics": [0.05, 0.1, 0.0]},
        "cmdty:3M": {"backtest": {"mean": 0.05, "sd": 0.1}, "live_ics": [-0.9]},  # too few
    }
    assert _drift(mon) == ["ind12:3M"]


def test_review_due(tmp_path: Path) -> None:
    assert review_due(tmp_path, date(2026, 11, 2)) == "2026-10"
    assert review_due(tmp_path, date(2027, 1, 4)) == "2026-12"
    assert review_due(tmp_path, date(2026, 11, 9)) is None  # after the first week
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "review-2026-10.md").write_text("x")
    assert review_due(tmp_path, date(2026, 11, 3)) is None  # already written


def test_error_runs_do_not_count_as_latest_eval(tmp_path: Path) -> None:
    from atlas.review import _eval_scores

    ev = tmp_path / "evals"
    ev.mkdir()
    good = [{"score": 90.0, "error": ""}] * 120
    bad = [{"score": 0.0, "error": "credit balance is too low"}] * 100 + [
        {"score": 95.0, "error": ""}
    ] * 20
    (ev / "20261020T000000Z.json").write_text(json.dumps({"results": good}))
    (ev / "20261029T000000Z.json").write_text(json.dumps({"results": bad}))
    out = _eval_scores(tmp_path)
    assert out["latest_at"] == "20261020T000000Z"
    assert out["latest"] == 90.0
