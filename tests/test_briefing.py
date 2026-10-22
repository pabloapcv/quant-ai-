from typing import Any

from atlas.briefing import _bank, _conditions, _level, _recession, build_briefing


def test_wording_thresholds() -> None:
    assert _level(0.8, hot="running hot", cold="") == "strong (running hot)"
    assert _level(0.2, hot="", cold="") == "modestly above trend"
    assert _level(-0.2, hot="", cold="") == "slightly below trend"
    assert _level(-0.9, hot="", cold="subdued") == "weak (subdued)"
    assert _conditions(-0.7).startswith("easy")
    assert _conditions(0.7).startswith("tight")
    assert _conditions(0.1) == "normal"
    assert _recession(0.1) == ("low", "good")
    assert _recession(0.6) == ("high", "bad")


def test_bank_headline() -> None:
    st = {
        "meeting_date": "2026-09-16",
        "extracted_reading": {"action": "cut", "change_bp": -50, "stance": -1, "summary": "s"},
    }
    b = _bank("Federal Reserve", st)
    assert b["headline"] == "Federal Reserve cut rates by 0.50 percentage point."
    assert b["tone"].startswith("dovish")
    assert _bank("ECB", {"error": "none"})["headline"].startswith("No ECB")


class Kit:
    def get_regime(self, as_of: str) -> dict[str, Any]:
        return {
            "growth": {"score": 0.6},
            "inflation": {"score": -0.1},
            "financial_conditions": {"score": 0.0},
            "quadrant_probs": {"goldilocks": 0.7, "stagflation": 0.3},
            "regime": "goldilocks",
            "recession": {"prob_12m": 0.2},
            "data_month": "2026-09-01",
        }

    def get_fomc_statement(self, d: str) -> dict[str, Any]:
        return {"error": "x"}

    get_ecb_statement = get_fomc_statement

    def get_views(self, u: str, h: int) -> dict[str, Any]:
        vs = [
            {"asset": a, "score": s, "prob_beat_median": 0.5, "stance": "neutral"}
            for a, s in zip("ABCDEFG", range(7), strict=True)
        ]
        return {"as_of_month": "2026-08-01", "views": vs}


def test_build_briefing() -> None:
    alerts = [
        {"rule": "data_quality", "message": "q"},
        {
            "rule": "regime_prob_cross",
            "at": "2026-10-01T00",
            "message": "m",
            "key": "regime_prob_cross:stagflation:0.5:up",
        },
    ]
    b = build_briefing(Kit(), alerts)
    e = b["economy"]
    assert e["regime_confidence"] == "clear"
    assert e["recession"]["word"] == "moderate"
    assert e["sentence"].startswith("Growth is strong, inflation is slightly below trend")
    assert [v["asset"] for v in b["leans"][0]["favored"]] == ["G", "F", "E"]
    assert [v["asset"] for v in b["leans"][0]["least"]] == ["A", "B", "C"]
    assert b["alerts"] == [
        {
            "date": "2026-10-01",
            "message": "The odds of a 'soft growth, elevated inflation' economy rose above 50%.",
        }
    ]
