"""Weekly research report (PRD §10.3, user story #1).

Pipeline:

1. **Facts** — gathered deterministically from the engine (no LLM): macro regime, current
   views and their track records for the headline universes, model portfolios, the signal
   gate summary, and what changed since the previous report.
2. **Writer** — Claude writes the report from those facts only.
3. **Red team** — a separate Claude pass challenges the main calls with the same facts.
4. **Grounding** — both texts must have every number traceable to the facts; each gets one
   repair turn, and anything still unsupported is listed in the report.

Output: ``data/reports/<date>/report.md`` plus ``facts.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from atlas import jsonsafe
from atlas.agent.assistant import DEFAULT_MODEL, DISCLAIMER, FALLBACK_BETA
from atlas.agent.grounding import GroundingReport, check
from atlas.agent.tools import ToolKit

HEADLINE_UNIVERSES = ("ind12", "ctry_dev_lcl")

WRITER_PROMPT = """You are Atlas, writing the weekly quantitative research note. You get a \
JSON object of facts computed by the Atlas engine. Write the note from these facts only.

Rules:
- Every number must come from the facts (you may round, and may write fractions as \
percentages). Do not estimate, extrapolate or add outside market knowledge.
- Be plain about conviction: most views are low-conviction leans; say so. Mention when a \
track record is weak.
- Research views, not investment advice; no instructions to buy or sell.
- Markdown. Use exactly these sections, in order, each a few sentences or a short list:
  ## Summary
  ## Macro regime
  ## US sectors
  ## Developed markets
  ## Model portfolios
  ## What changed since last week
  ## Model quality
- Name the as-of month for every view. No preamble before ## Summary."""

RED_TEAM_PROMPT = """You are the red team for the Atlas weekly note. You get the same JSON \
facts and the draft note. Find the weakest points: claims the facts support only \
weakly, conviction that is overstated, track records that undercut a call, regime readings \
that conflict (for example the quadrant vs the HMM), sector calls that the historical \
analogs contradict (what sectors did after the closest past macro states; a small sample, \
so say how many analogs agree), and data that is stale or missing.

Write a section starting with exactly "## Risks and counterpoints": 3 to 6 bullets, each \
naming the claim and why it may be wrong. Use only numbers from the facts. No preamble."""

REPAIR = (
    "These numbers do not appear in the facts: {nums}. Rewrite the full text using only "
    "numbers from the facts (or remove the unsupported ones). Reply with the full text only."
)


@dataclass
class Report:
    as_of: date
    facts: dict[str, Any]
    body: str = ""
    red_team: str = ""
    grounding: list[GroundingReport] = field(default_factory=list)
    model: str = ""

    @property
    def unsupported(self) -> list[str]:
        return [c.text for g in self.grounding for c in g.unsupported]

    def markdown(self) -> str:
        parts = [
            f"# Atlas weekly research note — {self.as_of.isoformat()}",
            "",
            self.body.strip() or "_No narrative generated (dry run)._",
        ]
        if self.red_team.strip():
            parts += ["", self.red_team.strip()]
        if self.unsupported:
            parts += [
                "",
                f"> Grounding: not traceable to engine facts: {', '.join(self.unsupported)}",
            ]
        parts += ["", "---", f"_{DISCLAIMER}_", ""]
        return "\n".join(parts)


# ---------------------------------------------------------------------------- facts
def _changes(current: dict[str, Any], previous: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Stance changes and the biggest score moves per universe since the previous report."""
    if not previous:
        return []
    out = []
    for universe in HEADLINE_UNIVERSES:
        cur = {v["asset"]: v for v in current.get("views", {}).get(universe, {}).get("views", [])}
        old = {v["asset"]: v for v in previous.get("views", {}).get(universe, {}).get("views", [])}
        for asset, v in cur.items():
            if asset not in old:
                continue
            if v["stance"] != old[asset]["stance"] or abs(v["score"] - old[asset]["score"]) >= 0.5:
                out.append(
                    {
                        "universe": universe,
                        "asset": asset,
                        "stance_before": old[asset]["stance"],
                        "stance_now": v["stance"],
                        "score_before": old[asset]["score"],
                        "score_now": v["score"],
                    }
                )
    return out


def previous_facts(reports_dir: Path, before: date) -> dict[str, Any] | None:
    earlier = sorted(
        p for p in reports_dir.glob("*/facts.json") if p.parent.name < before.isoformat()
    )
    return json.loads(earlier[-1].read_text()) if earlier else None


def gather_facts(toolkit: ToolKit, as_of: date, reports_dir: Path) -> dict[str, Any]:
    def get(name: str, **args: Any) -> dict[str, Any]:
        out, _ = toolkit.call(name, args)
        result: dict[str, Any] = json.loads(out)
        return result

    facts: dict[str, Any] = {"report_date": as_of.isoformat()}
    facts["macro_regime"] = get("get_regime", as_of="")
    facts["views"] = {u: get("get_views", universe=u, horizon=3) for u in HEADLINE_UNIVERSES}
    facts["track_records"] = {
        u: get("get_track_record", universe=u, horizon=3) for u in HEADLINE_UNIVERSES
    }
    facts["portfolios"] = {}
    for u in HEADLINE_UNIVERSES:
        pf = get("get_portfolio", universe=u)
        if "weights" in pf:  # keep the note focused: biggest tilts only
            pf["largest_tilts"] = sorted(pf.pop("weights"), key=lambda w: -abs(w["active"]))[:6]
        facts["portfolios"][u] = pf
    facts["signal_gates"] = {}
    for u in HEADLINE_UNIVERSES:
        g = get("get_signal_gates", universe=u)
        if "signals" in g:
            g["signals"] = [
                {k: s[k] for k in ("signal", "ic_tstat", "gates_passed_of_6", "failed_gates")}
                for s in g["signals"]
            ]
        facts["signal_gates"][u] = g
    # US macro analogs vs the US sector views; the red team checks calls against them.
    facts["historical_analogs"] = get("find_analogs", universe="ind12", as_of="", k=10)
    facts["changes_since_last_report"] = _changes(facts, previous_facts(reports_dir, as_of))
    return facts


# ---------------------------------------------------------------------------- writing
class ReportWriter:
    def __init__(
        self,
        client: Any,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        fallbacks: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks

    def _call(self, system: str, messages: list[dict[str, Any]]) -> tuple[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "output_config": {"effort": self.effort},
            "messages": messages,
        }
        if self.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        resp = self.client.beta.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return "", resp
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text, resp

    def _grounded(self, system: str, user: str, evidence: str) -> tuple[str, GroundingReport, str]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        text, resp = self._call(system, messages)
        report = check(text, [evidence])
        if text and not report.grounded:
            nums = ", ".join(c.text for c in report.unsupported)
            messages += [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": REPAIR.format(nums=nums)},
            ]
            revised, resp = self._call(system, messages)
            if revised:
                text, report = revised, check(revised, [evidence])
        return text, report, getattr(resp, "model", self.model)

    def write(self, facts: dict[str, Any], as_of: date) -> Report:
        evidence = jsonsafe.dumps(facts)
        body, g1, model = self._grounded(WRITER_PROMPT, f"Facts:\n{evidence}", evidence)
        red, g2, _ = self._grounded(
            RED_TEAM_PROMPT, f"Facts:\n{evidence}\n\nDraft note:\n{body}", evidence
        )
        return Report(
            as_of=as_of, facts=facts, body=body, red_team=red, grounding=[g1, g2], model=model
        )


def save(report: Report, reports_dir: Path) -> Path:
    out = reports_dir / report.as_of.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "facts.json").write_text(jsonsafe.dumps(report.facts, indent=2))
    (out / "report.md").write_text(report.markdown())
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": report.model,
        "unsupported_numbers": report.unsupported,
    }
    (out / "meta.json").write_text(jsonsafe.dumps(meta, indent=2))
    return out / "report.md"
