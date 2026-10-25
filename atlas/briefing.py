"""Plain-language executive briefing, built deterministically from the engine's data.

No language model is involved: fixed rules turn each number into words, so the briefing is
instant, free, reproducible and cannot state anything the data does not. Every sentence
carries the numbers it rests on (``facts``) so a reader can drill down.

Wording rules (thresholds on the engine's own scales):

* growth / inflation composites (z-scores vs trend): above +0.5 strong, 0..+0.5 modestly
  above trend, -0.5..0 slightly below trend, below -0.5 weak;
* financial conditions (above 0 = tighter): below -0.5 easy, -0.5..+0.5 normal, above tight;
* 12-month recession probability: below 15% low, 15-30% moderate, 30-50% elevated, 50%+ high;
* central-bank stance (-2..+2): +2 strongly hawkish, +1 hawkish, 0 balanced, -1 dovish,
  -2 strongly dovish;
* the regime call is "clear" when its probability is 60%+, otherwise "mixed signals".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


def _safe(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """A section with no data says so instead of failing the whole briefing."""
    try:
        return fn()
    except Exception as exc:
        return {"error": str(exc)}


REGIME_WORDS = {
    "goldilocks": ("Healthy growth, contained inflation", "good"),
    "overheating": ("Strong growth, rising inflation", "warn"),
    "stagflation": ("Soft growth, elevated inflation", "warn"),
    "deflationary_slowdown": ("Slowing growth, easing inflation", "warn"),
}


def _level(score: float | None, *, hot: str, cold: str) -> str:
    if score is None:
        return "unclear"
    if score > 0.5:
        return f"strong ({hot})" if hot else "strong"
    if score > 0:
        return "modestly above trend"
    if score > -0.5:
        return "slightly below trend"
    return f"weak ({cold})" if cold else "weak"


def _conditions(score: float | None) -> str:
    if score is None:
        return "unclear"
    if score < -0.5:
        return "easy (credit is readily available, markets calm)"
    if score > 0.5:
        return "tight (credit harder to get, markets stressed)"
    return "normal"


def _recession(prob: float | None) -> tuple[str, str]:
    if prob is None:
        return "unknown", "neutral"
    if prob < 0.15:
        return "low", "good"
    if prob < 0.30:
        return "moderate", "warn"
    if prob < 0.50:
        return "elevated", "bad"
    return "high", "bad"


STANCE_WORDS = {
    2: "strongly hawkish: focused on bringing inflation down, open to further increases",
    1: "hawkish: leaning toward tighter policy",
    0: "balanced: no clear lean",
    -1: "dovish: leaning toward supporting growth",
    -2: "strongly dovish: focused on supporting growth, open to cuts",
}
ACTION_WORDS = {"hike": "raised rates", "cut": "cut rates", "hold": "held rates", "other": "acted"}


def _bank(name: str, st: dict[str, Any]) -> dict[str, Any]:
    if "error" in st:
        return {"bank": name, "headline": f"No {name} statement available yet.", "facts": {}}
    r = st.get("extracted_reading") or {}
    action = r.get("action")
    bp = r.get("change_bp")
    move = ACTION_WORDS.get(action or "", "met")
    size = f" by {abs(bp) / 100:.2f} percentage point" if bp else ""
    headline = f"{name} {move}{size}."
    stance = r.get("stance")
    tone = STANCE_WORDS.get(stance, "") if isinstance(stance, int) else ""
    return {
        "bank": name,
        "headline": headline,
        "tone": tone,
        "summary": r.get("summary"),
        "meeting_date": st["meeting_date"],
        "url": st.get("url"),
        "facts": {"action": action, "change_bp": bp, "stance": r.get("stance")},
    }


def _leans(views: dict[str, Any], label: str) -> dict[str, Any]:
    if "error" in views:
        return {"label": label, "favored": [], "least": [], "as_of": None}
    ranked = sorted(views["views"], key=lambda v: -v["score"])

    def pick(vs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"asset": v["asset"], "chance_beat_peers": v["prob_beat_median"], "lean": v["stance"]}
            for v in vs
        ]

    return {
        "label": label,
        "as_of": views["as_of_month"],
        "favored": pick(ranked[:3]),
        "least": pick(ranked[-3:][::-1]),
        "all_neutral": all(v["stance"] == "neutral" for v in ranked),
    }


# Prices an executive watches: (series id, plain name, "pct" price or "pp" rate). Monthly IMF
# commodity averages and daily/weekly FRED series alike: changes are measured against the
# latest observation at least 3 / 12 months older than the newest one.
PRICES: tuple[tuple[str, str, str], ...] = (
    ("CMDTY.OIL.BRENT", "Crude oil (Brent)", "pct"),
    ("CMDTY.GAS.US", "Natural gas (US)", "pct"),
    ("CMDTY.GAS.EU", "Natural gas (Europe)", "pct"),
    ("CMDTY.COPPER", "Copper", "pct"),
    ("CMDTY.ALUMINUM", "Aluminum", "pct"),
    ("CMDTY.WHEAT", "Wheat", "pct"),
    ("CMDTY.GOLD", "Gold", "pct"),
    ("US.USD.BROAD", "US dollar vs trading partners", "pct"),
    ("US.RATES.10Y", "10-year US Treasury yield", "pp"),
    ("US.MORTGAGE.30Y", "30-year US mortgage rate", "pp"),
    ("US.CREDIT.BAA10Y", "Extra yield on corporate bonds (Baa)", "pp"),
)
BIG_MOVE = {"pct": (0.15, 0.25), "pp": (0.5, 1.0)}  # (3 months, 12 months)


def _change(obs: list[tuple[date, float]], days: int, kind: str) -> float | None:
    last_d, last_v = obs[-1]
    base = [v for d, v in obs if d <= last_d - timedelta(days=days)]
    if not base or (kind == "pct" and base[-1] == 0):
        return None
    return last_v / base[-1] - 1 if kind == "pct" else last_v - base[-1]


def price_moves(series: dict[str, list[tuple[date, float]]]) -> list[dict[str, Any]]:
    """3- and 12-month moves of ``PRICES``, flagging the big ones (``BIG_MOVE``)."""
    out = []
    for sid, name, kind in PRICES:
        obs = sorted(series.get(sid) or [])
        if not obs:
            continue
        c3, c12 = _change(obs, 89, kind), _change(obs, 360, kind)
        t3, t12 = BIG_MOVE[kind]
        out.append(
            {
                "series_id": sid,
                "name": name,
                "kind": kind,
                "as_of": obs[-1][0].isoformat(),
                "level": obs[-1][1] if kind == "pp" else None,
                "change_3m": c3,
                "change_12m": c12,
                "big": (c3 is not None and abs(c3) >= t3) or (c12 is not None and abs(c12) >= t12),
            }
        )
    return out


def _price_series(kit: Any) -> dict[str, list[tuple[date, float]]]:
    df = kit.store.as_of([sid for sid, _, _ in PRICES], datetime.now(UTC))
    out: dict[str, list[tuple[date, float]]] = {}
    for sid, d, v in df.select("series_id", "event_time", "value").drop_nulls().iter_rows():
        out.setdefault(sid, []).append((d, v))
    return out


HIDDEN_ALERTS = {"data_quality", "live_ic_drift"}  # operational, not for this audience


def _plain_alert(a: dict[str, Any]) -> str:
    """Rewrite an alert in everyday words; unknown kinds keep their original message."""
    rule, key, v = a.get("rule", ""), str(a.get("key", "")), a.get("value")
    parts = key.split(":")
    if rule == "regime_prob_cross" and len(parts) == 4:
        words = REGIME_WORDS.get(parts[1], (parts[1].replace("_", " "), ""))[0].lower()
        way = "rose above" if parts[3] == "up" else "fell below"
        return f"The odds of a '{words}' economy {way} {float(parts[2]):.0%}."
    if rule == "recession_prob" and isinstance(v, int | float):
        return f"Recession risk over the next 12 months rose to {v:.0%}."
    if rule == "hy_oas_jump":
        return "Borrowing costs for riskier companies moved sharply in the past week."
    if rule == "vix_term_inversion":
        return "Stock-market fear spiked: investors are paying up for short-term protection."
    if rule == "view_flip_3m":
        return "A market lean flipped direction: " + str(a.get("message", ""))
    return str(a.get("message", ""))


def _economy(reg: dict[str, Any]) -> dict[str, Any] | None:
    if "error" in reg:
        return None
    g, i = reg["growth"]["score"], reg["inflation"]["score"]
    fc = reg["financial_conditions"]["score"]
    probs = reg["quadrant_probs"]
    top = max(probs, key=probs.get) if probs else reg["regime"]
    words, light = REGIME_WORDS.get(top, (str(top), "neutral"))
    clear = (probs.get(top) or 0) >= 0.6
    rec = reg["recession"]["prob_12m"] if reg.get("recession") else None
    rec_word, rec_light = _recession(rec)
    sentence = (
        f"Growth is {_level(g, hot='', cold='')}, inflation is "
        f"{_level(i, hot='running hot', cold='subdued')}, and financial conditions are "
        f"{_conditions(fc).split(' (')[0]}. Recession risk over the next year is {rec_word}"
        + (f" ({rec:.0%})" if rec is not None else "")
        + "."
    )
    return {
        "regime": words,
        "regime_light": light,
        "regime_confidence": "clear" if clear else "mixed signals",
        "regime_probability": probs.get(top),
        "sentence": sentence,
        "growth": _level(g, hot="", cold=""),
        "inflation": _level(i, hot="running hot", cold="subdued"),
        "financial_conditions": _conditions(fc),
        "recession": {"probability": rec, "word": rec_word, "light": rec_light},
        "gdp_nowcast": reg.get("gdp_nowcast") or None,
        "data_month": reg["data_month"],
        "facts": {"growth_score": g, "inflation_score": i, "financial_conditions_score": fc},
    }


def _what_changed(
    now: dict[str, Any] | None,
    before: dict[str, Any] | None,
    banks: list[dict[str, Any]],
    since: date,
) -> list[str]:
    """Plain sentences on what moved since ``since`` (the earlier reading is point-in-time)."""
    out: list[str] = []
    if now and before:
        if now["regime"] != before["regime"]:
            out.append(
                f"The overall picture shifted from '{before['regime'].lower()}' to "
                f"'{now['regime'].lower()}'."
            )
        else:
            p0, p1 = before["regime_probability"], now["regime_probability"]
            if p0 is not None and p1 is not None and abs(p1 - p0) >= 0.05:
                way = "firmed" if p1 > p0 else "weakened"
                out.append(
                    f"The '{now['regime'].lower()}' reading {way}: its odds went from "
                    f"{p0:.0%} to {p1:.0%}."
                )
        r0, r1 = before["recession"]["probability"], now["recession"]["probability"]
        if r0 is not None and r1 is not None and abs(r1 - r0) >= 0.03:
            way = "rose" if r1 > r0 else "fell"
            out.append(f"Recession risk {way} from {r0:.0%} to {r1:.0%}.")
        elif r0 is not None and r1 is not None:
            out.append(f"Recession risk was little changed ({r0:.0%} a month ago, {r1:.0%} now).")
        for key, name in (
            ("growth", "Growth"),
            ("inflation", "Inflation"),
            ("financial_conditions", "Financial conditions"),
        ):
            a, b = before[key].split(" (")[0], now[key].split(" (")[0]
            if a != b:
                out.append(f"{name} went from {a} to {b}.")
    for bk in banks:
        md = bk.get("meeting_date")
        if md and md >= since.isoformat():
            out.append(f"{bk['headline'][:-1]} on {md}.")
    return out


def build_briefing(
    kit: Any, alerts: list[dict[str, Any]], today: date | None = None
) -> dict[str, Any]:
    today = today or datetime.now(UTC).date()
    since = today - timedelta(days=30)
    out: dict[str, Any] = {"economy": None, "central_banks": [], "leans": [], "alerts": []}
    out["economy"] = _economy(_safe(lambda: kit.get_regime("")))
    before = _economy(_safe(lambda: kit.get_regime(since.isoformat())))
    out["central_banks"] = [
        _bank("Federal Reserve", _safe(lambda: kit.get_fomc_statement(""))),
        _bank("ECB", _safe(lambda: kit.get_ecb_statement(""))),
    ]
    out["leans"] = [
        _leans(_safe(lambda: kit.get_views("ind12", 3)), "US sectors"),
        _leans(_safe(lambda: kit.get_views("ctry_dev_lcl", 3)), "Developed stock markets"),
    ]
    out["alerts"] = [
        {"date": str(a.get("at", ""))[:10], "message": _plain_alert(a)}
        for a in alerts
        if a.get("rule") not in HIDDEN_ALERTS
    ][:6]
    try:
        out["prices"] = price_moves(_price_series(kit))
    except Exception:  # no store (tests) or data not ingested: the section is left out
        out["prices"] = []
    out["what_changed"] = {
        "since": since.isoformat(),
        "items": _what_changed(out["economy"], before, out["central_banks"], since),
    }
    out["trust_note"] = (
        "How far to trust the market leans: over 2021-2025, data the models never saw, "
        "their sector and country rankings did not reliably beat a simple average. Use them "
        "as context for discussion, not as calls. The economy and central-bank readings above "
        "are descriptions of published data, not forecasts."
    )
    return out


def recent_alerts(data_dir: Path, days: int, today: date | None = None) -> list[dict[str, Any]]:
    """Alerts logged in the last ``days`` days, newest first."""
    cutoff = ((today or datetime.now(UTC).date()) - timedelta(days=days)).isoformat()
    rows: list[dict[str, Any]] = []
    for f in sorted((data_dir / "alerts").glob("*.jsonl")):
        if f.stem >= cutoff:
            rows += [json.loads(line) for line in f.read_text().splitlines() if line]
    return rows[::-1]


def _quarter(q: str) -> str:
    """'2026Q3' -> 'Q3 2026'."""
    return f"{q[4:]} {q[:4]}" if len(q) == 6 and q[4] == "Q" else q


def _fmt_move(x: float | None, kind: str) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.0f}%" if kind == "pct" else f"{x:+.2f} pts"


def to_markdown(b: dict[str, Any], as_of: date) -> str:
    """The briefing as a short note to read or forward (same content as the web page)."""
    lines = [f"# Ramon briefing, {as_of:%B} {as_of.day}, {as_of.year}", ""]
    e = b.get("economy")
    if e:
        lines += ["## The economy", "", e["sentence"], ""]
        lines.append(f"- **Overall picture:** {e['regime']} ({e['regime_confidence']})")
        r = e["recession"]
        if r["probability"] is not None:
            lines.append(
                f"- **Recession risk, next 12 months:** {r['word']} ({r['probability']:.0%})"
            )
        g = e.get("gdp_nowcast")
        if g:
            lines.append(
                f"- **Growth right now:** {g['gdp_saar_pct']:.1f}% annual pace "
                f"(Atlanta Fed estimate for {_quarter(g['quarter'])}, updated {g['updated']})"
            )
        lines += [f"- **Borrowing and market conditions:** {e['financial_conditions']}", ""]
    wc = b.get("what_changed") or {}
    lines += [f"## What changed since {wc.get('since', '')}", ""]
    lines += [f"- {t}" for t in wc.get("items", [])] or ["- No significant changes."]
    if b.get("prices"):
        lines += [
            "",
            "## Prices that matter",
            "",
            "| | Past 3 months | Past year |",
            "|---|---|---|",
        ]
        for pr in b["prices"]:
            big = " **(big move)**" if pr["big"] else ""
            m3, m12 = (
                _fmt_move(pr["change_3m"], pr["kind"]),
                _fmt_move(pr["change_12m"], pr["kind"]),
            )
            lines.append(f"| {pr['name']}{big} | {m3} | {m12} |")
    lines += ["", "## Central banks", ""]
    for c in b.get("central_banks", []):
        tone = f" Tone: {c['tone']}." if c.get("tone") else ""
        lines.append(f"- **{c['bank']}** ({c.get('meeting_date', 'n/a')}): {c['headline']}{tone}")
    lines += ["", "## Where the models lean (next 3 months)", ""]
    for lean in b.get("leans", []):
        if not lean.get("as_of"):
            continue

        def names(vs: list[dict[str, Any]]) -> str:
            return ", ".join(f"{v['asset']} ({v['chance_beat_peers']:.0%})" for v in vs)

        lines.append(
            f"- **{lean['label']}:** slightly favored {names(lean['favored'])}; "
            f"least favored {names(lean['least'])}. Percentages are the chance of beating peers."
        )
    lines += ["", f"_{b['trust_note']}_", ""]
    if b.get("alerts"):
        lines += ["## Recent warnings", ""]
        lines += [f"- {a['date']}: {a['message']}" for a in b["alerts"]]
        lines.append("")
    lines.append("_Research based on public data and historical patterns, not investment advice._")
    return "\n".join(lines) + "\n"
