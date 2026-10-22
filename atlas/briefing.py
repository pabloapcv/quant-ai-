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

from collections.abc import Callable
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


def build_briefing(kit: Any, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    reg = _safe(lambda: kit.get_regime(""))
    out: dict[str, Any] = {"economy": None, "central_banks": [], "leans": [], "alerts": []}
    if "error" not in reg:
        g, i = reg["growth"]["score"], reg["inflation"]["score"]
        fc = reg["financial_conditions"]["score"]
        probs = reg["quadrant_probs"]
        top = max(probs, key=probs.get) if probs else reg["regime"]
        words, light = REGIME_WORDS.get(top, (str(top), "neutral"))
        clear = (probs.get(top) or 0) >= 0.6
        rec = reg["recession"]["prob_12m"] if reg.get("recession") else None
        rec_word, rec_light = _recession(rec)
        nowcast = reg.get("gdp_nowcast") or {}
        sentence = (
            f"Growth is {_level(g, hot='', cold='')}, inflation is "
            f"{_level(i, hot='running hot', cold='subdued')}, and financial conditions are "
            f"{_conditions(fc).split(' (')[0]}. Recession risk over the next year is {rec_word}"
            + (f" ({rec:.0%})" if rec is not None else "")
            + "."
        )
        out["economy"] = {
            "regime": words,
            "regime_light": light,
            "regime_confidence": "clear" if clear else "mixed signals",
            "regime_probability": probs.get(top),
            "sentence": sentence,
            "growth": _level(g, hot="", cold=""),
            "inflation": _level(i, hot="running hot", cold="subdued"),
            "financial_conditions": _conditions(fc),
            "recession": {"probability": rec, "word": rec_word, "light": rec_light},
            "gdp_nowcast": nowcast or None,
            "data_month": reg["data_month"],
            "facts": {"growth_score": g, "inflation_score": i, "financial_conditions_score": fc},
        }
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
    out["trust_note"] = (
        "How far to trust the market leans: over 2021-2025, data the models never saw, "
        "their sector and country rankings did not reliably beat a simple average. Use them "
        "as context for discussion, not as calls. The economy and central-bank readings above "
        "are descriptions of published data, not forecasts."
    )
    return out
