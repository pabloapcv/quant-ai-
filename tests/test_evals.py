from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from atlas.agent.assistant import Answer, ToolCall
from atlas.agent.evals import (
    Case,
    EvalRun,
    Judge,
    Judgement,
    compare,
    jargon_in,
    load_cases,
    parse_judgement,
    run_eval,
    score_case,
)
from atlas.agent.grounding import check
from atlas.agent.tools import TOOL_SPECS

REPO = Path(__file__).resolve().parents[1]


def test_golden_set_is_well_formed() -> None:
    cases = load_cases(REPO / "config" / "evals" / "golden.yaml")
    assert len(cases) == 176
    tools = {t["name"] for t in TOOL_SPECS}
    for c in cases:
        assert set(c.expect_tools) <= tools, c.id
    cats = {c.category for c in cases}
    assert {"advice", "unavailable", "adversarial", "scenario", "analogs", "documents"} <= cats
    assert all(c.must_refuse_advice for c in cases if c.category == "advice")


def _answer(text: str, results: list[dict[str, Any]], tools: list[str]) -> Answer:
    outs = [json.dumps(r) for r in results]
    return Answer(
        text=text,
        tool_calls=[ToolCall(t, {}, False) for t in tools],
        tool_results=outs,
        grounding=check(text, outs),
    )


def test_score_case_weights() -> None:
    case = Case(id="v-1", category="views", question="q", expect_tools=["get_views"])
    good = _answer("Probability 0.53.", [{"p": 0.53}], ["get_views"])
    full = Judgement(correctness=4, calibration=2, policy=2, clarity=2)
    assert score_case(case, good, full).score == 100.0
    bad = _answer("Probability 0.91.", [{"p": 0.53}], ["get_regime"])
    r = score_case(case, bad, Judgement(correctness=2, calibration=1, policy=2, clarity=1))
    assert not r.tools_ok
    assert r.unsupported == ["0.91"]
    assert r.score == 30.0  # grounding 0, tools 0, judge 6/10 of 50
    no_tools = Case(id="a-1", category="advice", question="q", must_refuse_advice=True)
    assert score_case(no_tools, _answer("No.", [], []), None).score == 50.0


def test_parse_judgement() -> None:
    j = parse_judgement(
        'Sure.\n{"correctness": 3, "calibration": 2, "policy": 2, "clarity": 1, '
        '"rationale": "uses {braces} fine"}'
    )
    assert j.share == 0.8
    assert "braces" in j.rationale
    assert parse_judgement("no json").share == 0.0
    over = Judgement(correctness=9, calibration=-1, policy=2, clarity=2)
    assert over.share == 0.8  # clipped to the rubric's range


class _JudgeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        reply = '{"correctness": 4, "calibration": 2, "policy": 2, "clarity": 2, "rationale": "ok"}'
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)])


def test_run_eval_and_compare(tmp_path: Path) -> None:
    cases = [
        Case(id="v-1", category="views", question="view?", expect_tools=["get_views"]),
        Case(id="v-2", category="views", question="boom"),
    ]

    def ask(q: str) -> Answer:
        if q == "boom":
            raise RuntimeError("tool exploded")
        return _answer("Probability 0.53.", [{"p": 0.53}], ["get_views"])

    client = _JudgeClient()
    run = run_eval(ask, Judge(client), cases, model="m")
    assert [r.score for r in run.results] == [100.0, 0.0]
    assert "tool exploded" in run.results[1].error
    assert "0.53" in client.calls[0]["messages"][0]["content"]  # judge sees tool results
    assert run.summary()["overall"]["score"] == 50.0
    path = run.save(tmp_path)
    again = EvalRun.model_validate_json(path.read_text())
    assert compare(again, run)["delta"] == 0.0
    worse = again.model_copy(deep=True)
    worse.results[0].score = 90.0
    cmp = compare(worse, run)
    assert cmp["regression"]
    assert cmp["biggest_drops"] == [{"id": "v-1", "delta": -10.0}]


def test_run_stops_after_repeated_identical_errors() -> None:
    cases = [Case(id=f"c-{i}", category="views", question="q") for i in range(10)]

    def ask(q: str) -> Answer:
        raise RuntimeError("credit balance is too low")

    run = run_eval(ask, None, cases, model="m")
    assert len(run.results) == 3
    assert "credit balance" in run.stopped_early


def test_executive_cases_penalize_jargon_and_use_their_own_assistant() -> None:
    case = Case(id="e-1", category="executive", question="q", audience="executive")
    full = Judgement(correctness=4, calibration=2, policy=2, clarity=2)
    plain = score_case(case, _answer("A slight lean, 0.53.", [{"p": 0.53}], []), full)
    assert (plain.score, plain.jargon) == (100.0, [])
    jargon = score_case(case, _answer("Rank IC is 0.53 in basis points.", [{"p": 0.53}], []), full)
    assert jargon.jargon == ["IC", "rank IC", "basis points"]
    assert jargon.score == 90.0
    analyst = Case(id="v-1", category="views", question="q")
    assert score_case(analyst, _answer("IC 0.53.", [{"p": 0.53}], []), full).jargon == []
    assert jargon_in("icy, stance-free, IC-like") == []

    asked: list[str] = []

    def ask(q: str) -> Answer:
        asked.append("analyst")
        return _answer("x", [], [])

    def ask_exec(q: str) -> Answer:
        asked.append("executive")
        return _answer("x", [], [])

    run_eval(ask, None, [case, analyst], model="m", ask_executive=ask_exec)
    assert asked == ["executive", "analyst"]
