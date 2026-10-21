from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from atlas.agent.assistant import FALLBACK_BETA, Assistant
from atlas.agent.grounding import check, extract_claims
from atlas.agent.tools import TOOL_SPECS, ToolKit
from tests.test_country import _country_store

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ grounding
TOOL_OUT = [
    json.dumps(
        {
            "ic_mean": 0.0647,
            "hit_rate": 0.5406,
            "icir_annual": 0.9128,
            "spread": -0.0123,
            "as_of_month": "2026-08-01",
            "months": 677,
        }
    )
]


@pytest.mark.parametrize(
    "text",
    [
        "IC is 0.065 with an ICIR of 0.91.",
        "Calls are right 54% of the time (54.1%).",
        "IC of 6.5% in percent terms.",
        "Data through 2026-08-01, over 677 months.",
        "Underweights lagged by 1.23%.",  # sign carried by the words
        "Positive in 3 of 4 regimes using 12-1 momentum (mom_12_1) over 3 months.",
        "A move of 91bp in percentage-point terms.",  # 0.9128 pp -> 91 bp
        "No numbers here at all.",
    ],
)
def test_grounded_answers_pass(text: str) -> None:
    rep = check(text, TOOL_OUT)
    assert rep.grounded, rep.unsupported
    assert rep.grounded_share == 1.0


@pytest.mark.parametrize(
    ("text", "bad"),
    [
        ("Calls are right 56% of the time.", "56%"),
        ("The spread is 9.6% a year.", "9.6%"),
        ("IC is 0.07 exactly 0.0700.", "0.0700"),
        ("We expect 450 bp of excess return.", "450 bp"),
    ],
)
def test_ungrounded_numbers_are_caught(text: str, bad: str) -> None:
    rep = check(text, TOOL_OUT)
    assert bad in [c.text for c in rep.unsupported]
    assert rep.grounded_share < 1.0


def test_claim_extraction() -> None:
    claims = extract_claims("Fell -0.3%, rose 1,250.5 and 25 bp; skip FX.EURUSD and Q5.")
    texts = [c.text for c in claims]
    assert "-0.3%" in texts
    assert "1,250.5" in texts
    assert "25 bp" in texts
    assert not any("EURUSD" in t or t == "5" for t in texts)
    assert claims[1].value == 1250.5
    assert claims[1].decimals == 1


# ------------------------------------------------------------------ toolkit
@pytest.fixture(scope="module")
def toolkit(tmp_path_factory: pytest.TempPathFactory) -> ToolKit:
    data = tmp_path_factory.mktemp("agentdata")
    _country_store(data)
    return ToolKit(data, REPO / "config")


def test_tool_specs_are_strict_ready() -> None:
    names = {t["name"] for t in TOOL_SPECS}
    assert {
        "get_views",
        "get_track_record",
        "get_signal_gates",
        "get_portfolio",
        "get_series",
        "search_catalog",
        "get_regime",
        "list_universes",
        "find_analogs",
        "run_scenario",
        "search_documents",
        "get_fomc_statement",
        "get_ecb_statement",
    } == names
    for t in TOOL_SPECS:
        schema = t["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_toolkit_calls(toolkit: ToolKit) -> None:
    out, err = toolkit.call("get_views", {"universe": "ctry_dev_lcl", "horizon": 3})
    views = json.loads(out)
    assert not err
    assert len(views["views"]) == 9
    assert {"stance", "prob_beat_median", "expected_excess_return", "top_drivers"} <= set(
        views["views"][0]
    )
    track, err = toolkit.call("get_track_record", {"universe": "ctry_dev_lcl", "horizon": 3})
    assert not err
    assert "ic_mean" in json.loads(track)
    pf, err = toolkit.call("get_portfolio", {"universe": "ctry_dev_lcl"})
    assert not err
    weights = json.loads(pf)["weights"]
    assert abs(sum(w["weight"] for w in weights) - 1) < 1e-3
    hits, _ = toolkit.call("search_catalog", {"query": "japan yield"})
    assert json.loads(hits)["matches"][0]["id"] == "JPN.RATES.10Y"
    series, err = toolkit.call("get_series", {"series_id": "USA.CLI", "last_n": 3, "as_of": ""})
    assert not err
    assert len(json.loads(series)["observations"]) == 3
    assert "NaN" not in out + track + pf


def test_toolkit_errors_are_returned_not_raised(toolkit: ToolKit) -> None:
    for name, args, needle in [
        ("nope", {}, "unknown tool"),
        ("get_views", {"universe": "mars", "horizon": 3}, "universe"),
        ("get_series", {"series_id": "NOT.A.SERIES", "last_n": 3, "as_of": ""}, "unknown series"),
        ("get_series", {"series_id": "US.CPI.CORE", "last_n": 3, "as_of": ""}, "no data"),
        ("get_regime", {"as_of": ""}, "FRED"),
    ]:
        out, err = toolkit.call(str(name), dict(args))  # type: ignore[call-overload]
        assert err, name
        assert needle in json.loads(out)["error"]


def test_toolkit_caches_results(toolkit: ToolKit, monkeypatch: pytest.MonkeyPatch) -> None:
    toolkit.call("list_universes", {})
    monkeypatch.setattr(toolkit, "_handlers", {**toolkit._handlers, "list_universes": None})
    out, err = toolkit.call("list_universes", {})  # served from cache
    assert not err
    assert "ind12" in out


# ------------------------------------------------------------------ assistant loop
def text_block(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def tool_block(i: str, name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=i, name=name, input=args)


def response(stop: str, *blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop, content=list(blocks), model="claude-opus-5-5")


class FakeClient:
    """Plays back scripted responses and records every request."""

    def __init__(self, script: list[SimpleNamespace]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.__dict__)))
        return self.script.pop(0)


class StubKit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        self.calls.append((name, arguments))
        if name == "get_track_record":
            return json.dumps({"ic_mean": 0.0647, "hit_rate": 0.5406}), False
        return json.dumps({"error": "no such data"}), True


def test_assistant_runs_parallel_tools_and_grounds() -> None:
    client = FakeClient(
        [
            response(
                "tool_use",
                text_block("Checking."),
                tool_block("t1", "get_track_record", {"universe": "ind49", "horizon": 3}),
                tool_block("t2", "get_regime", {"as_of": ""}),
            ),
            response(
                "end_turn",
                text_block("The 3-month model has an IC of 0.065 and is right 54% of the time."),
            ),
        ]
    )
    kit = StubKit()
    ans = Assistant(client, kit).ask("How good is the industry model?")  # type: ignore[arg-type]
    assert ans.ok
    assert ans.grounding.grounded
    assert not ans.repaired
    assert [c.name for c in ans.tool_calls] == ["get_track_record", "get_regime"]
    assert ans.tool_calls[1].is_error
    # both tool results went back in ONE user message
    second = client.requests[1]["messages"]
    results = second[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["tool_choice"] == {"type": "auto"}
    assert req["output_config"] == {"effort": "medium"}
    assert req["fallbacks"] == "default"
    assert req["betas"] == [FALLBACK_BETA]
    assert all(t["strict"] for t in req["tools"])
    assert "thinking" not in req  # cannot be disabled on Opus 5.5; left adaptive


def test_assistant_repairs_unsupported_numbers_once() -> None:
    client = FakeClient(
        [
            response(
                "tool_use",
                tool_block("t1", "get_track_record", {"universe": "ind49", "horizon": 3}),
            ),
            response("end_turn", text_block("Right 56% of the time.")),
            response("end_turn", text_block("Right 54% of the time.")),
        ]
    )
    ans = Assistant(client, StubKit()).ask("hit rate?")  # type: ignore[arg-type]
    assert ans.repaired
    assert ans.grounding.grounded
    assert "56%" in client.requests[2]["messages"][-1]["content"]

    stubborn = FakeClient(
        [
            response("end_turn", text_block("Right 56% of the time.")),
            response("end_turn", text_block("Still 56%.")),
        ]
    )
    ans = Assistant(stubborn, StubKit()).ask("hit rate?")  # type: ignore[arg-type]
    assert ans.repaired
    assert not ans.ok
    assert [c.text for c in ans.grounding.unsupported] == ["56%"]


def test_assistant_handles_refusal_and_step_limit() -> None:
    ans = Assistant(FakeClient([response("refusal")]), StubKit()).ask("x")  # type: ignore[arg-type]
    assert ans.stop_reason == "refusal"
    assert not ans.ok
    loop = [
        response("tool_use", tool_block(f"t{i}", "get_regime", {"as_of": ""})) for i in range(3)
    ]
    ans = Assistant(FakeClient(loop), StubKit(), max_steps=3, fallbacks=False).ask("x")  # type: ignore[arg-type]
    assert ans.stop_reason == "tool_use"
    assert "step limit" in ans.text


def test_cli_ask_without_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    import httpx2

    from atlas.cli import app
    from atlas.settings import get_settings

    class NoAuth:
        def __init__(self, *a: Any, **k: Any) -> None:
            req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            resp = httpx2.Response(401, request=req)
            raise anthropic.AuthenticationError("no key", response=resp, body=None)

    monkeypatch.setattr(anthropic, "Anthropic", NoAuth)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        res = CliRunner().invoke(app, ["ask", "What is the view on Japan?"])
        assert res.exit_code != 0
        assert "ANTHROPIC_API_KEY" in res.output
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    ("env_value", "expected"), [("sk-ant-test-123", "sk-ant-test-123"), ("", None)]
)
def test_cli_ask_reads_key_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_value: str, expected: str | None
) -> None:
    import anthropic

    from atlas.cli import app
    from atlas.settings import get_settings

    seen: dict[str, Any] = {}

    class Capture:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)
            self.beta = SimpleNamespace(
                messages=SimpleNamespace(
                    create=lambda **_: response("end_turn", text_block("No numbers."))
                )
            )

    (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={env_value}\n")
    monkeypatch.setenv("ATLAS_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ATLAS_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", Capture)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        res = CliRunner().invoke(app, ["ask", "hello"])
        assert res.exit_code == 0, res.output
        assert seen.get("api_key") == expected  # empty entry -> SDK's own resolution
    finally:
        get_settings.cache_clear()
