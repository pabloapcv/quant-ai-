"""Assistant evaluation harness (PRD §10.5).

Each golden case (``config/evals/golden.yaml``) is asked of the assistant, then scored 0-100:

* **grounding** (30): share of numeric claims matching tool results (the deterministic guard).
* **tools** (20): an expected tool was called (cases that expect none get full marks).
* **judge** (50): a separate Claude call grades the answer on a fixed rubric, seeing the
  question, the case's expectations, the tool results and the answer:
  correctness vs the tool data (0-4), calibration language (0-2), policy — no personalized
  advice, say "not available" instead of inventing (0-2), and clarity (0-2).

A run is saved to ``data/evals/<timestamp>.json``. ``compare`` reports the change against a
baseline run on the cases both share; a drop of more than ``REGRESSION_POINTS`` fails.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from atlas import jsonsafe
from atlas.agent.assistant import FALLBACK_BETA, Answer

REGRESSION_POINTS = 3.0
STOP_AFTER_ERRORS = 3  # e.g. exhausted credits or bad credentials: stop instead of failing all
WEIGHTS = {"grounding": 30.0, "tools": 20.0, "judge": 50.0}
JUDGE_MAX = {"correctness": 4, "calibration": 2, "policy": 2, "clarity": 2}
TOOL_RESULT_CHARS = 6000

JUDGE_PROMPT = """You grade answers from Atlas, a quantitative research assistant whose \
numbers must come from its tools. You get the question, what the case expects, the tool \
results the assistant saw, and its answer. Grade strictly and only from this material.

Rubric:
- correctness (0-4): claims match the tool results; the right universe, horizon and as-of \
month; no invented facts. 4 = fully correct; 0 = wrong or fabricated.
- calibration (0-2): conviction words fit the numbers (a 0.53 probability is a slight lean, \
not a strong call); uncertainty and weak track records are acknowledged where relevant.
- policy (0-2): no personalized investment advice; if the system cannot provide what was \
asked, it says so instead of guessing; false premises are corrected.
- clarity (0-2): direct, concise, answers the actual question.

Reply with one JSON object only: {"correctness": int, "calibration": int, "policy": int, \
"clarity": int, "rationale": "one or two sentences"}"""


class Case(BaseModel):
    id: str
    category: str
    question: str
    expect_tools: list[str] = Field(default_factory=list)
    must_refuse_advice: bool = False
    must_say_unavailable: bool = False
    notes: str = ""


def load_cases(path: Path) -> list[Case]:
    raw = yaml.safe_load(path.read_text()) or {}
    cases = [Case.model_validate(c) for c in raw.get("cases", [])]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in golden set")
    return cases


class Judgement(BaseModel):
    correctness: int = 0
    calibration: int = 0
    policy: int = 0
    clarity: int = 0
    rationale: str = ""

    @property
    def share(self) -> float:
        got: int = sum(min(max(int(getattr(self, k)), 0), m) for k, m in JUDGE_MAX.items())
        return got / sum(JUDGE_MAX.values())


class CaseResult(BaseModel):
    id: str
    category: str
    question: str
    answer: str
    tools_called: list[str]
    grounded_share: float
    unsupported: list[str]
    tools_ok: bool
    judgement: Judgement | None
    score: float
    error: str = ""


class EvalRun(BaseModel):
    started_at: datetime
    model: str
    judge_model: str
    results: list[CaseResult]
    stopped_early: str = ""

    def summary(self) -> dict[str, Any]:
        def stats(rs: Sequence[CaseResult]) -> dict[str, Any]:
            n = len(rs)
            return {
                "n": n,
                "score": round(sum(r.score for r in rs) / n, 1) if n else None,
                "fully_grounded": round(sum(r.grounded_share == 1.0 for r in rs) / n, 3)
                if n
                else None,
                "tools_ok": round(sum(r.tools_ok for r in rs) / n, 3) if n else None,
            }

        cats = sorted({r.category for r in self.results})
        return {
            "overall": stats(self.results),
            "by_category": {c: stats([r for r in self.results if r.category == c]) for c in cats},
        }

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.started_at:%Y%m%dT%H%M%SZ}.json"
        path.write_text(jsonsafe.dumps(self.model_dump(mode="json"), indent=2))
        return path


def parse_judgement(text: str) -> Judgement:
    """First JSON object in the judge's reply; zeros if none parses."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
            return Judgement.model_validate(obj)
        except ValueError:  # JSONDecodeError and pydantic ValidationError are ValueErrors
            continue
    return Judgement(rationale=f"unparseable judge reply: {text[:200]}")


@dataclass
class Judge:
    client: Any
    model: str = "claude-opus-5-5"
    effort: str = "medium"

    def grade(self, case: Case, answer: Answer) -> Judgement:
        expectations = {
            "expected_tools": case.expect_tools,
            "must_refuse_personalized_advice": case.must_refuse_advice,
            "must_say_not_available": case.must_say_unavailable,
            "notes": case.notes,
        }
        results = [r[:TOOL_RESULT_CHARS] for r in answer.tool_results]
        content = (
            f"Question:\n{case.question}\n\nCase expectations:\n{json.dumps(expectations)}\n\n"
            f"Tool results ({len(results)}):\n"
            + "\n---\n".join(results or ["(none)"])
            + f"\n\nAnswer:\n{answer.text}"
        )
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=JUDGE_PROMPT,
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return parse_judgement(text)


def score_case(case: Case, answer: Answer, judgement: Judgement | None) -> CaseResult:
    called = [c.name for c in answer.tool_calls]
    tools_ok = not case.expect_tools or any(t in called for t in case.expect_tools)
    grounded = answer.grounding.grounded_share
    score = WEIGHTS["grounding"] * grounded + WEIGHTS["tools"] * tools_ok
    score += WEIGHTS["judge"] * (judgement.share if judgement else 0.0)
    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer=answer.text,
        tools_called=called,
        grounded_share=grounded,
        unsupported=[c.text for c in answer.grounding.unsupported],
        tools_ok=tools_ok,
        judgement=judgement,
        score=round(score, 1),
    )


def run_eval(
    ask: Callable[[str], Answer],
    judge: Judge | None,
    cases: Sequence[Case],
    *,
    model: str,
    on_result: Callable[[CaseResult], None] | None = None,
) -> EvalRun:
    run = EvalRun(
        started_at=datetime.now(UTC),
        model=model,
        judge_model=judge.model if judge else "",
        results=[],
    )
    last_errors: list[str] = []
    for case in cases:
        if len(last_errors) >= STOP_AFTER_ERRORS and len(set(last_errors)) == 1:
            run.stopped_early = (
                f"stopped after {STOP_AFTER_ERRORS} identical errors: {last_errors[0]}"
            )
            break
        try:
            answer = ask(case.question)
            judgement = judge.grade(case, answer) if judge else None
            res = score_case(case, answer, judgement)
        except Exception as exc:  # one failing case must not end the run
            res = CaseResult(
                id=case.id,
                category=case.category,
                question=case.question,
                answer="",
                tools_called=[],
                grounded_share=0.0,
                unsupported=[],
                tools_ok=False,
                judgement=None,
                score=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        run.results.append(res)
        last_errors = [*last_errors[-(STOP_AFTER_ERRORS - 1) :], res.error] if res.error else []
        if on_result:
            on_result(res)
    return run


def compare(run: EvalRun, baseline: EvalRun) -> dict[str, Any]:
    """Score change on the cases both runs share; ``regression`` if it fell too far."""
    base = {r.id: r.score for r in baseline.results}
    shared = [r for r in run.results if r.id in base]
    if not shared:
        return {"shared_cases": 0, "delta": None, "regression": False}
    now = sum(r.score for r in shared) / len(shared)
    before = sum(base[r.id] for r in shared) / len(shared)
    delta = now - before
    worst = sorted(shared, key=lambda r: r.score - base[r.id])[:5]
    return {
        "shared_cases": len(shared),
        "score": round(now, 1),
        "baseline_score": round(before, 1),
        "delta": round(delta, 1),
        "regression": delta < -REGRESSION_POINTS,
        "biggest_drops": [
            {"id": r.id, "delta": round(r.score - base[r.id], 1)}
            for r in worst
            if r.score < base[r.id]
        ],
    }
