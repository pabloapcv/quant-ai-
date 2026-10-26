"""Monthly model review (PRD §15, P8): a fixed-format check of the live paper track.

Deterministic, like the briefing: every figure is read from recorded state (frozen views and
their scores, the holdout log, pipeline run reports, eval runs, the token ledger), and the
decisions follow written rules rather than judgement:

* keep the champion unless ``atlas.views.models.compare`` says a challenger qualifies
  (at least 12 shared scored months and a paired t-stat of 2 or more);
* investigate a universe when its live IC drift monitor flags it;
* fix the data when a pipeline step failed on more than two runs in the month;
* re-run the assistant eval when its latest score is more than 3 points below baseline.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from atlas.backtest.holdout import HoldoutLog

EVAL_DROP = 3.0
FAILED_RUNS = 2


def _month_runs(data_dir: Path, month: str) -> list[dict[str, Any]]:
    key = month.replace("-", "")
    runs = sorted((data_dir / "runs").glob(f"{key}*.json"))
    return [json.loads(p.read_text()) for p in runs]


def _drift(ic_monitor: dict[str, Any], sigmas: float = 2.0, window: int = 6) -> list[str]:
    """Universes whose last ``window`` live ICs sit below backtest mean - sigmas * s.e."""
    flagged = []
    for key, m in sorted(ic_monitor.items()):
        live = [x for x in m.get("live_ics", []) if x is not None][-window:]
        bt = m.get("backtest") or {}
        if len(live) < 3 or not bt.get("sd"):
            continue
        floor = bt["mean"] - sigmas * bt["sd"] / np.sqrt(len(live))
        if float(np.mean(live)) < floor:
            flagged.append(key)
    return flagged


def _eval_scores(data_dir: Path) -> dict[str, Any]:
    folder = data_dir / "evals"
    base = folder / "baseline.json"
    runs = sorted(p for p in folder.glob("2*.json"))
    out: dict[str, Any] = {"baseline": None, "latest": None, "latest_at": None, "latest_n": 0}

    def score(path: Path) -> tuple[float | None, int]:
        rs = json.loads(path.read_text()).get("results", [])
        return (round(sum(r["score"] for r in rs) / len(rs), 1) if rs else None), len(rs)

    if base.exists():
        out["baseline"], _ = score(base)
    full = [p for p in runs if score(p)[1] >= 100]  # whole-set runs only, not spot checks
    if full:
        out["latest"], out["latest_n"] = score(full[-1])
        out["latest_at"] = full[-1].stem
    return out


def _usage(data_dir: Path, month: str) -> dict[str, Any]:
    by: dict[str, float] = {}
    calls = 0
    for p in sorted((data_dir / "usage").glob(f"{month}-*.jsonl")):
        for line in p.read_text().splitlines():
            if line:
                r = json.loads(line)
                purpose = r.get("purpose", "other")
                by[purpose] = by.get(purpose, 0.0) + float(r.get("weighted", 0))
                calls += 1
    return {"calls": calls, "weighted_tokens": round(sum(by.values())), "by_purpose": dict(by)}


def monthly_review(
    data_dir: Path,
    month: str,
    track: dict[str, Any],
    challengers: dict[str, dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    per_universe: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the review for ``month`` (YYYY-MM) from recorded state.

    ``track`` / ``challengers`` are ``TrackRecord.summary()`` dicts, ``verdicts`` the output
    of ``compare`` per challenger, ``per_universe`` the scored months (universe, horizon,
    as_of, rank_ic, hit_rate); the CLI computes these from the frozen views.
    """
    runs = _month_runs(data_dir, month)
    failed: Counter[str] = Counter()
    for r in runs:
        for s in r.get("steps", []):
            if not s.get("ok", True):
                failed[s["name"]] += 1
    latest = runs[-1] if runs else {}
    drift = _drift(latest.get("ic_monitor") or {})
    holdout = {}
    for e in HoldoutLog(data_dir).entries():
        if e.get("family") == "ensemble_v1":
            ens = e.get("ensemble") or {}
            holdout[e["universe"]] = {
                "ic": ens.get("ic_mean"),
                "t": ens.get("ic_tstat"),
                "hit_rate": ens.get("hit_rate"),
                "clean": not e.get("contaminated", True),
            }
    evals = _eval_scores(data_dir)
    decisions = []
    promote = [v for v, out in verdicts.items() if "qualifies" in str(out.get("verdict", ""))]
    if promote:
        decisions.append(
            f"Challenger {', '.join(promote)} qualifies: publish it as a new model version "
            "after review (rule: 12+ shared months, paired t >= 2)."
        )
    else:
        decisions.append("Keep the champion: no challenger qualifies yet.")
    decisions += [f"Investigate {k}: live IC is below its backtest range." for k in drift]
    decisions += [
        f"Fix '{name}': it failed on {n} runs this month."
        for name, n in failed.most_common()
        if n > FAILED_RUNS
    ]
    if (
        evals["baseline"] is not None
        and evals["latest"] is not None
        and evals["latest"] < evals["baseline"] - EVAL_DROP
    ):
        decisions.append("Re-run the assistant eval: the latest full run is below baseline.")
    if len(decisions) == 1 and not drift:
        decisions.append("No other action needed.")
    return {
        "month": month,
        "runs": len(runs),
        "runs_ok": sum(1 for r in runs if r.get("ok")),
        "failed_steps": dict(failed),
        "stale_series": latest.get("stale", []),
        "track": track,
        "challengers": challengers,
        "verdicts": verdicts,
        "scored_months": per_universe,
        "drift_flags": drift,
        "holdout": holdout,
        "evals": evals,
        "usage": _usage(data_dir, month),
        "decisions": decisions,
    }


def _f(x: Any, fmt: str = "{:+.3f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def review_markdown(r: dict[str, Any], today: date) -> str:
    t = r["track"]
    lines = [
        f"# Ramon monthly model review, {r['month']}",
        "",
        f"_Written {today.isoformat()} from recorded state; rules in `atlas/review.py`._",
        "",
        "## Decisions",
        "",
        *[f"- {d}" for d in r["decisions"]],
        "",
        "## Live paper track (champion)",
        "",
        f"- Months scored: {t.get('months_scored', 0)}; pending: {t.get('months_pending', 0)}",
        f"- Mean rank IC: {_f(t.get('mean_rank_ic'))}; hit rate: "
        f"{_f(t.get('hit_rate'), '{:.0%}')}; coverage: {_f(t.get('coverage'), '{:.0%}')}",
    ]
    if r["scored_months"]:
        lines += [
            "",
            "| Universe | Horizon | Month | Rank IC | Hit rate |",
            "|---|---|---|---|---|",
        ]
        for m in r["scored_months"]:
            lines.append(
                f"| {m['universe']} | {m['horizon']} | {m['as_of']} | {_f(m['rank_ic'])} | "
                f"{_f(m.get('hit_rate'), '{:.0%}')} |"
            )
    lines += ["", "## Challengers", ""]
    for v, summ in r["challengers"].items():
        verdict = r["verdicts"].get(v, {}).get("verdict", "n/a")
        lines.append(f"- **{v}**: {summ.get('months_scored', 0)} months scored; {verdict}")
    lines += ["", "## 2021-2025 holdout (ensemble, 3 months)", ""]
    lines += ["| Universe | IC | t-stat | Hit rate | Opening |", "|---|---|---|---|---|"]
    for u, h in sorted(r["holdout"].items()):
        lines.append(
            f"| {u} | {_f(h['ic'])} | {_f(h['t'], '{:+.2f}')} | {_f(h['hit_rate'], '{:.0%}')} | "
            f"{'clean' if h['clean'] else 'contaminated'} |"
        )
    lines += [
        "",
        "## Operations",
        "",
        f"- Pipeline runs: {r['runs']} ({r['runs_ok']} fully ok)",
        "- Failed steps: "
        + (", ".join(f"{k} x{n}" for k, n in r["failed_steps"].items()) or "none"),
        f"- Stale series at the last run: {', '.join(r['stale_series']) or 'none'}",
        f"- Live IC drift flags: {', '.join(r['drift_flags']) or 'none'}",
        f"- Assistant eval: latest full run {_f(r['evals']['latest'], '{:.1f}')} "
        f"({r['evals']['latest_n']} questions, {r['evals']['latest_at'] or 'n/a'}) vs baseline "
        f"{_f(r['evals']['baseline'], '{:.1f}')}",
        f"- Claude usage: {r['usage']['calls']} calls, {r['usage']['weighted_tokens']:,} weighted "
        "tokens ("
        + ", ".join(f"{k} {round(v):,}" for k, v in sorted(r["usage"]["by_purpose"].items()))
        + ")",
        "",
    ]
    return "\n".join(lines)


def write_review(data_dir: Path, returns: dict[str, Any], month: str, today: date) -> Path:
    """Score the frozen views, compare challengers, and write ``reports/review-<month>.md``.

    ``returns``: universe -> research panel (date, asset, ret), as the pipeline builds it.
    """
    from atlas import jsonsafe
    from atlas.views.models import CHALLENGERS, compare, views_root
    from atlas.views.track import score_all

    champ = score_all(data_dir, returns)
    challengers, verdicts = {}, {}
    for version in CHALLENGERS:
        ch = score_all(data_dir, returns, root=views_root(data_dir, version))
        challengers[version] = ch.summary()
        verdicts[version] = compare(champ, ch)
    scored = [
        {
            "universe": m.universe,
            "horizon": m.horizon,
            "as_of": m.as_of.isoformat()[:7],
            "rank_ic": m.rank_ic,
            "hit_rate": m.hit_rate,
        }
        for m in sorted(champ.scored, key=lambda m: (m.as_of, m.universe))
    ]
    r = monthly_review(data_dir, month, champ.summary(), challengers, verdicts, scored)
    out = data_dir / "reports" / f"review-{month}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(review_markdown(r, today))
    out.with_suffix(".json").write_text(jsonsafe.dumps(r, indent=2))
    return out


def review_due(data_dir: Path, today: date) -> str | None:
    """The previous month, if its review is missing and we are in the first week (else None)."""
    if today.day > 7:
        return None
    prev = date(today.year - (today.month == 1), (today.month - 2) % 12 + 1, 1)
    month = prev.isoformat()[:7]
    return None if (data_dir / "reports" / f"review-{month}.md").exists() else month
