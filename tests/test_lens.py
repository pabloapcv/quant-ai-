from typing import Any

import pytest

from atlas.lens import PRESETS, assets, build_lens, market_facts


class Kit:
    def get_views(self, u: str, h: int) -> dict[str, Any]:
        rows = [("Energy", 2.0, 0.58), ("Retail", 0.0, 0.47), ("Tech", -1.0, 0.42)]
        return {
            "as_of_month": "2026-09-01",
            "views": [
                {
                    "asset": a,
                    "score": sc,
                    "prob_beat_median": p,
                    "confidence": "low",
                    "top_drivers": [{"signal": "mom_12_1", "contribution": -0.1}],
                }
                for a, sc, p in rows
            ],
        }

    def run_scenario(self, u: str, shocks: list[dict[str, Any]], same: bool) -> dict[str, Any]:
        f = shocks[0]["factor"]
        retail = {"oil": -0.02 * shocks[0]["value"] / 0.3, "usd": 0.04}.get(f, 0.0)
        return {
            "fit": {"months": 200},
            "excess_vs_average": [
                {"asset": "Energy", "expected": 0.05, "range_p10": 0, "range_p90": 0},
                {"asset": "Retail", "expected": retail, "range_p10": 0, "range_p90": 0},
            ],
        }

    def find_analogs(self, u: str, as_of: str, k: int) -> dict[str, Any]:
        return {
            "analogs": [{"month": "2024-03-01"}],
            "forward_excess_by_asset": {
                "12m": [
                    {
                        "asset": "Retail",
                        "median_excess": 0.032,
                        "share_of_analogs_positive": 0.7,
                        "analogs_counted": 10,
                    }
                ]
            },
        }


def test_lens_sentences() -> None:
    facts = market_facts(Kit(), "ind12")
    assert assets(facts) == ["Energy", "Retail", "Tech"]
    b = build_lens(facts, "Retail")
    assert b["lean"]["rank"] == 2
    assert b["lean"]["drivers"][0]["effect"] == "weighs on it"
    assert len(b["events"]) == len(PRESETS)
    assert b["helps"] == ["usd_up", "oil_down"]
    assert b["hurts"] == ["oil_up"]
    assert b["summary"] == [
        "The models lean slightly against Retail for the next 3 months "
        "(47% chance of beating the other sectors; 2 of 3).",
        "Historically it held up best when the dollar strengthened 10% (+4.0% vs the average "
        "sector) and oil prices fell 30% (+2.0% vs the average sector).",
        "It was most exposed when oil prices jumped 30% (-2.0%).",
        "In the 10 past periods most like today, it typically beat the average sector over the "
        "next year (+3.2%, ahead in 70% of them).",
    ]
    with pytest.raises(KeyError):
        build_lens(facts, "Nope")
    with pytest.raises(KeyError):
        market_facts(Kit(), "nope")


def test_currency_wording() -> None:
    b = build_lens(market_facts(Kit(), "fx_g10"), "Retail")
    assert "chance of beating the other major currencies" in b["summary"][0]
    assert "vs the average currency" in b["summary"][1]
    assert "currency rankings" in b["trust_note"]
