"""'Your industry' lens: what the engine says about one US sector, in plain words.

Deterministic, like the briefing: the model lean for the sector, how it has historically moved
in each what-if event, and what followed in the past periods most like today. No language
model; every sentence carries the numbers it rests on.
"""

from __future__ import annotations

from typing import Any

UNIVERSE = "ind12"

# One-click what-if events (shared with the web app via /v1/presets). Units follow the scenario
# engine: yields and spreads in percentage points, dollar and oil as fractions, VIX in points.
PRESETS: list[dict[str, Any]] = [
    {
        "id": "oil_up",
        "when": "oil prices jumped 30%",
        "label": "Oil prices jump 30%",
        "shocks": {"oil": 0.3},
    },
    {
        "id": "oil_down",
        "when": "oil prices fell 30%",
        "label": "Oil prices fall 30%",
        "shocks": {"oil": -0.3},
    },
    {
        "id": "rates_up",
        "when": "interest rates rose 1 point",
        "label": "Interest rates rise 1 point",
        "shocks": {"us10y": 1, "us2y": 1},
    },
    {
        "id": "rates_down",
        "when": "interest rates fell 1 point",
        "label": "Interest rates fall 1 point",
        "shocks": {"us10y": -1, "us2y": -1},
    },
    {
        "id": "panic",
        "when": "markets panicked",
        "label": "Market panic",
        "shocks": {"vix": 15, "credit_spread": 1},
        "note": "fear index up 15 points, corporate borrowing spreads up 1 point",
    },
    {
        "id": "usd_up",
        "when": "the dollar strengthened 10%",
        "label": "Dollar strengthens 10%",
        "shocks": {"usd": 0.1},
    },
    {
        "id": "inflation_fears",
        "when": "inflation fears rose",
        "label": "Inflation fears rise",
        "shocks": {"breakeven10y": 0.5},
        "note": "expected inflation up 0.5 points",
    },
]

SIGNAL_WORDS = {
    "mom_12_1": "its price trend over the past year",
    "resmom_ff3_36": "its own trend, after removing broad market moves",
    "value_log_bm": "its valuation (how cheap it is relative to its book value)",
    "low_vol_36": "how steady its returns have been",
    "ltr_60_12": "its five-year performance (long runs tend to partly reverse)",
    "reversal_1m": "last month's move (short moves tend to partly reverse)",
}


def _lean_words(p: float) -> str:
    if p >= 0.6:
        return "lean clearly toward"
    if p > 0.52:
        return "lean slightly toward"
    if p >= 0.48:
        return "show no real lean on"
    if p > 0.4:
        return "lean slightly against"
    return "lean clearly against"


def industries(kit: Any) -> list[str]:
    return [v["asset"] for v in kit.get_views(UNIVERSE, 3)["views"]]


def build_lens(kit: Any, asset: str) -> dict[str, Any]:
    views = kit.get_views(UNIVERSE, 3)
    ranked = sorted(views["views"], key=lambda v: -v["score"])
    names = [v["asset"] for v in ranked]
    if asset not in names:
        raise KeyError(asset)
    v = ranked[names.index(asset)]
    drivers = [
        {
            "what": SIGNAL_WORDS.get(d["signal"], d["signal"]),
            "effect": "helps" if d["contribution"] > 0 else "weighs on it",
            "signal": d["signal"],
            "contribution": d["contribution"],
        }
        for d in v.get("top_drivers", [])[:2]
        if d["contribution"] != 0
    ]
    lean = {
        "as_of": views["as_of_month"],
        "chance_beat_peers": v["prob_beat_median"],
        "words": _lean_words(v["prob_beat_median"]),
        "rank": names.index(asset) + 1,
        "of": len(names),
        "confidence": v["confidence"],
        "drivers": drivers,
    }

    events = []
    for p in PRESETS:
        shocks = [{"factor": f, "value": x} for f, x in p["shocks"].items()]
        r = kit.run_scenario(UNIVERSE, shocks, False)
        if "error" in r:
            continue
        rows = sorted(r["excess_vs_average"], key=lambda x: -x["expected"])
        mine = next((x for x in rows if x["asset"] == asset), None)
        if mine is None:
            continue
        events.append(
            {
                "id": p["id"],
                "label": p["label"],
                "when": p["when"],
                "vs_average": mine["expected"],
                "range_p10": mine["range_p10"],
                "range_p90": mine["range_p90"],
                "rank": rows.index(mine) + 1,
                "of": len(rows),
                "months": r["fit"]["months"],
            }
        )
    by_effect = sorted(events, key=lambda e: -e["vs_average"])
    helps = [e for e in by_effect if e["vs_average"] > 0.002][:2]
    hurts = [e for e in by_effect[::-1] if e["vs_average"] < -0.002][:2]

    history = None
    an = kit.find_analogs(UNIVERSE, "", 10)
    if "error" not in an:
        fwd = next((x for x in an["forward_excess_by_asset"]["12m"] if x["asset"] == asset), None)
        if fwd:
            history = {
                "median_vs_average": fwd["median_excess"],
                "share_beat_average": fwd["share_of_analogs_positive"],
                "periods": fwd["analogs_counted"],
                "months": [a["month"][:7] for a in an["analogs"]],
            }

    def pct(x: float) -> str:
        return f"{x * 100:+.1f}%"

    sentences = [
        f"The models {lean['words']} {asset} for the next 3 months "
        f"({lean['chance_beat_peers']:.0%} chance of beating the other sectors; "
        f"{lean['rank']} of {lean['of']})."
    ]
    if helps:
        sentences.append(
            "Historically it held up best when "
            + " and ".join(
                f"{e['when']} ({pct(e['vs_average'])} vs the average sector)" for e in helps
            )
            + "."
        )
    if hurts:
        sentences.append(
            "It was most exposed when "
            + " and ".join(f"{e['when']} ({pct(e['vs_average'])})" for e in hurts)
            + "."
        )
    if history:
        way = "beat" if history["median_vs_average"] > 0 else "lagged"
        sentences.append(
            f"In the {history['periods']} past periods most like today, it typically {way} the "
            f"average sector over the next year ({pct(history['median_vs_average'])}, ahead in "
            f"{history['share_beat_average']:.0%} of them)."
        )
    return {
        "asset": asset,
        "summary": sentences,
        "lean": lean,
        "events": events,
        "helps": [e["id"] for e in helps],
        "hurts": [e["id"] for e in hurts],
        "history": history,
        "trust_note": (
            "The lean is weak evidence: over 2021-2025 the sector rankings did not reliably beat "
            "a simple average. The event and history figures describe how this sector moved in "
            "the past, not a forecast."
        ),
    }
