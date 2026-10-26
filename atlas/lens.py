"""'Your business' lens: what the engine says about one sector, country or currency, in plain
words.

Deterministic, like the briefing: the model lean and what drives it, how the asset moved in
each what-if event, and what followed in the past periods most like today. No language model;
every sentence carries the numbers it rests on. The per-market work (views, every what-if run,
analogs) is shared by all assets of a market, so it is computed once (``market_facts``) and
each lens is then cheap (``build_lens``).
"""

from __future__ import annotations

from typing import Any

MARKETS: dict[str, dict[str, str]] = {
    "ind12": {
        "label": "US sector",
        "noun": "sector",
        "peers": "the other sectors",
        "avg": "the average sector",
        "prompt": "My business is closest to",
    },
    "ctry_usd": {
        "label": "Stock market abroad",
        "noun": "market",
        "peers": "the other markets",
        "avg": "the average market",
        "prompt": "A country we sell to or invest in",
    },
    "fx_g10": {
        "label": "Major currency",
        "noun": "currency",
        "peers": "the other major currencies",
        "avg": "the average currency",
        "prompt": "A currency we are exposed to (against the US dollar)",
    },
    "fx_em": {
        "label": "Emerging-market currency",
        "noun": "currency",
        "peers": "the other emerging-market currencies",
        "avg": "the average emerging-market currency",
        "prompt": "An emerging-market currency we are exposed to (against the US dollar)",
        "trust": "Read this lean with care. It ranks the total return of holding the currency, "
        "including the interest it pays, which is known in advance. Over 2021-2025, data the "
        "models never saw, that ranking was right in 72% of months. On the exchange rate alone "
        "(what matters if you only convert money), higher-interest currencies did not do better "
        "before 2021 and did only in 2021-2025, a period that suited them. The event and history "
        "figures describe the past, not a forecast.",
    },
    "cmdty": {
        "label": "Commodity",
        "noun": "commodity",
        "peers": "the other commodities",
        "avg": "the average commodity",
        "prompt": "An input cost or product price we depend on",
        "trust": "The lean is early evidence: over 2021-2025, data the models never saw, the "
        "commodity rankings were right in 60% of months, but not by enough to be conclusive. "
        "Prices are monthly averages of world benchmark prices in US dollars, not what any one "
        "buyer pays. The event and history figures describe the past, not a forecast.",
    },
}

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

# What a component's contribution means, in words: (when it helps, when it weighs on the asset).
# Ensemble weights are non-negative, so the contribution's sign is the signal's own direction.
SIGNAL_WORDS: dict[str, tuple[str, str]] = {
    "mom_12_1": (
        "a strong price trend over the past year",
        "a weak price trend over the past year",
    ),
    "mom_6_1": (
        "a strong price trend over the past six months",
        "a weak price trend over the past six months",
    ),
    "mom_3_0": (
        "a strong move over the past three months",
        "a weak move over the past three months",
    ),
    "resmom_ff3_36": (
        "a strong trend of its own, after removing broad market moves",
        "a weak trend of its own, after removing broad market moves",
    ),
    "value_log_bm": (
        "it looks cheap relative to its book value",
        "it looks expensive relative to its book value",
    ),
    "low_vol_36": ("its returns have been steady", "its returns have been volatile"),
    "ltr_60_12": (
        "it has lagged over the past five years (long runs tend to partly reverse)",
        "it has run up over the past five years (long runs tend to partly reverse)",
    ),
    "reversal_1m": (
        "it fell last month (short moves tend to partly reverse)",
        "it rose last month (short moves tend to partly reverse)",
    ),
    "yield_fall_12": (
        "its bond yields fell over the past year",
        "its bond yields rose over the past year",
    ),
    "cli_chg_6": (
        "its economy's leading indicator improved over six months",
        "its economy's leading indicator worsened over six months",
    ),
    "cli_level": (
        "its economy's leading indicator is high",
        "its economy's leading indicator is low",
    ),
    "carry": (
        "it pays more interest than other currencies",
        "it pays less interest than other currencies",
    ),
    "carry_chg_12": (
        "its interest-rate advantage grew over the past year",
        "its interest-rate advantage shrank over the past year",
    ),
    "term_spread": ("its yield curve is steep", "its yield curve is flat"),
    "yield_vs_60m": (
        "its bond yield is high versus its five-year average",
        "its bond yield is low versus its five-year average",
    ),
    "cli_fall_6": (
        "its economy's leading indicator slowed",
        "its economy's leading indicator picked up",
    ),
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


def market_facts(kit: Any, market: str) -> dict[str, Any]:
    """Everything the lenses of one market need, computed once."""
    if market not in MARKETS:
        raise KeyError(market)
    scenarios = {}
    for p in PRESETS:
        shocks = [{"factor": f, "value": x} for f, x in p["shocks"].items()]
        scenarios[p["id"]] = kit.run_scenario(market, shocks, False)
    prices = {}
    if market == "cmdty":
        from atlas.signals.commodity import COMMODITY_NAMES, PREFIX

        for cid, name in COMMODITY_NAMES.items():
            obs = kit.get_series(PREFIX + cid, 13, "").get("observations") or []
            if len(obs) >= 13:
                last = obs[-1]
                prices[name] = {
                    "month": last["period"],
                    "change_3m": last["value"] / obs[-4]["value"] - 1,
                    "change_12m": last["value"] / obs[-13]["value"] - 1,
                }
    return {
        "market": market,
        "prices": prices,
        "views": kit.get_views(market, 3),
        "scenarios": scenarios,
        "analogs": kit.find_analogs(market, "", 10),
    }


def assets(facts: dict[str, Any]) -> list[str]:
    return [v["asset"] for v in facts["views"]["views"]]


def build_lens(facts: dict[str, Any], asset: str) -> dict[str, Any]:
    m = MARKETS[facts["market"]]
    views = facts["views"]
    ranked = sorted(views["views"], key=lambda v: -v["score"])
    names = [v["asset"] for v in ranked]
    if asset not in names:
        raise KeyError(asset)
    v = ranked[names.index(asset)]
    drivers = [
        {
            "what": SIGNAL_WORDS.get(d["signal"], (d["signal"], d["signal"]))[
                0 if d["contribution"] > 0 else 1
            ],
            "effect": "helps" if d["contribution"] > 0 else "weighs on it",
            "signal": d["signal"],
            "contribution": d["contribution"],
        }
        for d in v.get("top_drivers", [])[:2]
        if d["contribution"] != 0
    ]
    pro = [d["what"] for d in drivers if d["effect"] == "helps"]
    con = [d["what"] for d in drivers if d["effect"] != "helps"]
    drivers_sentence = " ".join(
        part
        for part in (
            f"Working for it: {' and '.join(pro)}." if pro else "",
            f"Working against it: {' and '.join(con)}." if con else "",
        )
        if part
    )
    lean = {
        "as_of": views["as_of_month"],
        "chance_beat_peers": v["prob_beat_median"],
        "words": _lean_words(v["prob_beat_median"]),
        "rank": names.index(asset) + 1,
        "of": len(names),
        "confidence": v["confidence"],
        "drivers": drivers,
        "drivers_sentence": drivers_sentence,
    }

    events = []
    for p in PRESETS:
        r = facts["scenarios"].get(p["id"]) or {"error": "not run"}
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
    an = facts["analogs"]
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

    price = facts.get("prices", {}).get(asset)
    sentences = []
    if price:

        def move(x: float) -> str:
            return f"{'up' if x >= 0 else 'down'} {abs(x):.0%}"

        sentences.append(
            f"{asset} prices are {move(price['change_12m'])} over the past year and "
            f"{move(price['change_3m'])} over the past three months (monthly average, "
            f"{price['month'][:7]})."
        )
    sentences += [
        f"The models {lean['words']} {asset} for the next 3 months "
        f"({lean['chance_beat_peers']:.0%} chance of beating {m['peers']}; "
        f"{lean['rank']} of {lean['of']})."
    ]
    if helps:
        sentences.append(
            "Historically it held up best when "
            + " and ".join(f"{e['when']} ({pct(e['vs_average'])} vs {m['avg']})" for e in helps)
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
            f"In the {history['periods']} past periods most like today, it typically {way} "
            f"{m['avg']} over the next year ({pct(history['median_vs_average'])}, ahead in "
            f"{history['share_beat_average']:.0%} of them)."
        )
    return {
        "asset": asset,
        "market": facts["market"],
        "noun": m["noun"],
        "avg": m["avg"],
        "summary": sentences,
        "lean": lean,
        "events": events,
        "helps": [e["id"] for e in helps],
        "hurts": [e["id"] for e in hurts],
        "history": history,
        "price": price,
        "trust_note": m.get("trust")
        or (
            f"The lean is weak evidence: over 2021-2025 these {m['noun']} rankings did not "
            f"reliably beat a simple average. The event and history figures describe how this "
            f"{m['noun']} moved in the past, not a forecast."
        ),
    }
