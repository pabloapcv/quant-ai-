from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from atlas.agent.report import (
    HEADLINE_UNIVERSES,
    Report,
    ReportWriter,
    _changes,
    gather_facts,
    previous_facts,
    save,
)
from atlas.agent.tools import ToolKit
from tests.test_agent import FakeClient, response, text_block
from tests.test_country import _country_store

REPO = Path(__file__).resolve().parents[1]
FACTS: dict[str, Any] = {
    "report_date": "2026-10-16",
    "views": {
        "ind12": {
            "as_of_month": "2026-08-01",
            "views": [{"asset": "Energy", "score": 1.1, "stance": "neutral"}],
        }
    },
    "track_records": {"ind12": {"ic_mean": 0.0545, "hit_rate": 0.5247}},
}


def test_writer_and_red_team_grounded() -> None:
    client = FakeClient(
        [
            response(
                "end_turn",
                text_block(
                    "## Summary\nEnergy leads with a score of 1.1 (IC 0.055, as of 2026-08)."
                ),
            ),
            response(
                "end_turn",
                text_block("## Risks and counterpoints\n- A 52% hit rate is a slim edge."),
            ),
        ]
    )
    rep = ReportWriter(client).write(FACTS, date(2026, 10, 16))
    assert rep.unsupported == []
    md = rep.markdown()
    assert md.startswith("# Ramon weekly research note — 2026-10-16")
    assert "## Summary" in md
    assert "## Risks and counterpoints" in md
    assert "not investment advice" in md
    first, second = client.requests
    assert first["output_config"] == {"effort": "high"}
    assert first["fallbacks"] == "default"
    assert "tools" not in first  # the writer only sees facts
    (block,) = second["messages"][0]["content"]
    assert "Draft note" in block["text"]
    assert block["cache_control"] == {"type": "ephemeral"}  # facts prefix is cached


def test_writer_repairs_then_reports_unsupported() -> None:
    client = FakeClient(
        [
            response("end_turn", text_block("## Summary\nIC is 0.09.")),  # unsupported
            response("end_turn", text_block("## Summary\nIC is 0.055.")),  # repaired
            response(
                "end_turn", text_block("## Risks and counterpoints\n- Hit rate 61%.")
            ),  # unsupported
            response("end_turn", text_block("## Risks and counterpoints\n- Hit rate still 61%.")),
        ]
    )
    rep = ReportWriter(client).write(FACTS, date(2026, 10, 16))
    assert "0.055" in rep.body
    assert rep.unsupported == ["61%"]
    assert "Grounding: not traceable" in rep.markdown()
    assert "0.09" in client.requests[1]["messages"][-1]["content"]


def test_refusal_leaves_empty_section() -> None:
    client = FakeClient(
        [
            response("refusal"),
            response("end_turn", text_block("## Risks and counterpoints\n- None.")),
        ]
    )
    rep = ReportWriter(client, fallbacks=False).write(FACTS, date(2026, 10, 16))
    assert rep.body == ""
    assert "No narrative generated" in rep.markdown()
    assert "fallbacks" not in client.requests[0]


def test_changes_since_last_report() -> None:
    def facts(stance: str, score: float) -> dict[str, Any]:
        return {
            "views": {
                u: {"views": [{"asset": "Canada", "stance": stance, "score": score}]}
                for u in HEADLINE_UNIVERSES
            }
        }

    assert _changes(facts("neutral", 1.0), None) == []
    assert _changes(facts("neutral", 1.0), facts("neutral", 0.9)) == []
    moved = _changes(facts("overweight", 2.0), facts("neutral", 1.0))
    assert len(moved) == len(HEADLINE_UNIVERSES)
    assert moved[0]["stance_before"] == "neutral"
    assert moved[0]["stance_now"] == "overweight"


def test_gather_save_and_previous(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _country_store(data)
    kit = ToolKit(data, REPO / "config")
    reports = data / "reports"
    facts = gather_facts(kit, date(2026, 10, 9), reports)
    assert set(facts) == {
        "report_date",
        "macro_regime",
        "views",
        "views_1m",
        "track_records",
        "portfolios",
        "signal_gates",
        "historical_analogs",
        "central_banks",
        "changes_since_last_report",
    }
    assert set(facts["central_banks"]) == {"fomc", "ecb"}  # errors (no documents) reported
    assert "error" in facts["macro_regime"]  # no FRED data in this store: reported, not raised
    assert "error" in facts["historical_analogs"]
    assert "largest_tilts" in facts["portfolios"]["ctry_dev_lcl"]
    path = save(Report(as_of=date(2026, 10, 9), facts=facts), reports)
    assert path.exists()
    assert (path.parent / "facts.json").exists()
    assert "NaN" not in (path.parent / "facts.json").read_text()
    assert previous_facts(reports, date(2026, 10, 16)) == json.loads(
        (path.parent / "facts.json").read_text()
    )
    assert previous_facts(reports, date(2026, 10, 9)) is None


def test_cli_report_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    _country_store(data)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        res = CliRunner().invoke(app, ["report", "weekly", "--dry-run"])
        assert res.exit_code == 0, res.output
        assert list((data / "reports").glob("*/facts.json"))
    finally:
        get_settings.cache_clear()
    _ = SimpleNamespace


def test_cli_report_without_credits_still_writes_briefing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anthropic
    import httpx

    from atlas.agent import report as report_mod
    from atlas.agent import usage
    from atlas.cli import app
    from atlas.settings import get_settings

    def no_credits(*_: Any, **__: Any) -> Any:
        response = httpx.Response(400, request=httpx.Request("POST", "http://test"))
        raise anthropic.BadRequestError("credit balance is too low", response=response, body=None)

    data = tmp_path / "data"
    _country_store(data)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    monkeypatch.setattr(usage, "make_client", lambda *_: object())
    monkeypatch.setattr(report_mod.ReportWriter, "write", no_credits)
    get_settings.cache_clear()
    try:
        res = CliRunner().invoke(app, ["report", "weekly", "--no-send"])
        assert res.exit_code == 1
        assert "research note not written" in res.output
        assert list((data / "reports").glob("*/briefing.md"))
        assert not list((data / "reports").glob("*/report.md"))
    finally:
        get_settings.cache_clear()


def test_views_heatmap() -> None:
    from atlas.agent.report import heatmap

    facts = {
        "views": {
            "ind12": {
                "as_of_month": "2026-08-01",
                "views": [
                    {
                        "asset": "Energy",
                        "stance": "overweight",
                        "prob_beat_median": 0.58,
                        "score": 1.0,
                    },
                    {
                        "asset": "Telecom",
                        "stance": "underweight",
                        "prob_beat_median": 0.41,
                        "score": -1.0,
                    },
                ],
            }
        },
        "views_1m": {
            "ind12": {"views": [{"asset": "Energy", "stance": "neutral", "prob_beat_median": 0.52}]}
        },
    }
    lines = heatmap(facts)
    assert "| Energy | Neutral (52%) | OW (58%) |" in lines
    assert "| Telecom | n/a | UW (41%) |" in lines
    assert heatmap({}) == []
