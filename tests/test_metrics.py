from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from atlas.api import create_app
from atlas.metrics import Metric, collect, render
from atlas.settings import Settings
from tests.test_country import _country_store

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)


def test_render_prometheus_text() -> None:
    m = Metric("atlas_x", "An example", "gauge")
    m.add(1792479489.36927, step='ingest:"fred"')
    m.add(float("nan"), step="skipped")  # non-finite samples are dropped
    m.add(None)
    empty = Metric("atlas_empty", "No samples", "gauge")
    text = render([m, empty])
    assert text == (
        "# HELP atlas_x An example\n# TYPE atlas_x gauge\n"
        'atlas_x{step="ingest:\\"fred\\""} 1792479489.36927\n'
    )


def _run_file(data: Path) -> None:
    run = {
        "started_at": "2026-10-20T06:58:09+00:00",
        "ok": False,
        "steps": [
            {"name": "ingest:fred", "ok": True, "detail": "", "seconds": 1.5},
            {"name": "views:ind12:3M", "ok": False, "detail": "x", "seconds": 0.2},
        ],
        "stale": ["US.SENTIMENT.UMICH"],
        "track": {"months_scored": 0, "months_pending": 6},
        "feature_psi": {"payrolls_6m_ann": 3.2},
    }
    (data / "runs").mkdir(parents=True)
    (data / "runs" / "20261020T065809Z.json").write_text(json.dumps(run))
    (data / "alerts").mkdir()
    (data / "alerts" / "2026-10-20.jsonl").write_text(
        json.dumps({"status": "sent"}) + "\n" + json.dumps({"status": "cooldown"}) + "\n"
    )


def test_collect_reads_recorded_state(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _country_store(data)
    _run_file(data)
    text = render(collect(data, REPO / "config", daily_budget=3_000_000, now=NOW))
    assert "atlas_pipeline_last_run_ok 0" in text
    assert 'atlas_pipeline_step_ok{step="views:ind12:3M"} 0' in text
    assert "atlas_stale_series 1" in text
    assert 'atlas_live_track_months{model="ens_v1",status="pending"} 6' in text
    assert 'atlas_alerts_today{status="sent"} 1' in text
    assert "atlas_llm_daily_budget_tokens 3000000" in text
    assert 'atlas_source_data_age_seconds{source="oecd"}' in text
    assert 'atlas_documents{kind="statement"} 0' in text


def test_metrics_endpoint(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _country_store(data)
    _run_file(data)
    client = TestClient(create_app(Settings(data_dir=data, config_dir=REPO / "config")))
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "# TYPE atlas_pipeline_step_seconds gauge" in r.text
