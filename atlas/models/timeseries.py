"""Time-series views for credit and crypto (PRD §3.1, Phase 5): beat cash or Treasuries?

Credit and crypto each have two assets here, too few for the cross-sectional rankings the
other universes use, so each asset gets its own forecast: the probability that its next
3-month excess return is positive.

Assets and monthly excess returns (month-end values):

* ``CREDIT.AAA`` / ``CREDIT.BAA``: Moody's Aaa / Baa corporate bonds over 10-year Treasuries,
  from the spread: ``ret_t = s_{t-1}/1200 - D * (s_t - s_{t-1}) / 100`` with spread duration
  ``D = 7`` (FRED AAA10Y / BAA10Y, daily since 1983 / 1986).
* ``CRYPTO.BTC`` / ``CRYPTO.ETH``: price change minus cash (fed funds / 12), Coinbase prices
  via FRED (2014 / 2016 onwards): a short history, so these forecasts rest on little data.

Signals, chosen from the literature before any test was run (no search over alternatives):

* trend: the past 12 months' cumulative excess return (time-series momentum; Moskowitz, Ooi
  & Pedersen 2012), for every asset;
* value (credit only): the spread minus its 5-year average (wide spreads tend to compensate;
  Asness, Moskowitz & Pedersen 2013).

Model: a logistic regression of "next 3-month excess > 0" on the standardized signals, fitted
walk-forward on months whose outcome was known at the forecast date, refitted every 12
months, with at least 60 months of history. Stances follow the PRD §3.3 probability
thresholds (no model-agreement term: one model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from atlas.pit.store import PITStore

HORIZON = 3
MIN_TRAIN = 60
REFIT_EVERY = 12
SPREAD_DURATION = 7.0

SLEEVES: dict[str, dict[str, Any]] = {
    "CREDIT.AAA": {
        "name": "US high-grade corporate bonds (Aaa) vs Treasuries",
        "series": "US.CREDIT.AAA10Y",
        "kind": "spread",
        "asset_class": "credit",
    },
    "CREDIT.BAA": {
        "name": "US investment-grade corporate bonds (Baa) vs Treasuries",
        "series": "US.CREDIT.BAA10Y",
        "kind": "spread",
        "asset_class": "credit",
    },
    "CRYPTO.BTC": {
        "name": "Bitcoin vs cash",
        "series": "CRYPTO.BTC",
        "kind": "price",
        "asset_class": "crypto",
    },
    "CRYPTO.ETH": {
        "name": "Ether vs cash",
        "series": "CRYPTO.ETH",
        "kind": "price",
        "asset_class": "crypto",
    },
}


def _month_end(store: PITStore, sid: str, at: datetime) -> pl.DataFrame:
    df = store.as_of([sid], at).select("event_time", "value").drop_nulls().sort("event_time")
    return (
        df.with_columns(pl.col("event_time").dt.truncate("1mo").alias("date"))
        .group_by("date")
        .agg(pl.col("value").last())
        .sort("date")
    )


def sleeve_panel(store: PITStore, at: datetime | None = None) -> pl.DataFrame:
    """date, asset, ret (excess), trend, value: one row per asset-month, as known at ``at``."""
    at = at or datetime.now(UTC)
    cash = _month_end(store, "US.RATES.FF", at).rename({"value": "ff"})
    frames = []
    for asset, spec in SLEEVES.items():
        lv = _month_end(store, spec["series"], at)
        if lv.height < 24:
            continue
        if spec["kind"] == "spread":
            s = pl.col("value")
            df = lv.with_columns(
                (s.shift(1) / 1200 - SPREAD_DURATION * (s - s.shift(1)) / 100).alias("ret"),
                (s - s.rolling_mean(60, min_samples=60)).alias("value_sig"),
            )
        else:
            df = lv.join(cash, on="date", how="left").with_columns(
                (
                    pl.col("value") / pl.col("value").shift(1) - 1 - pl.col("ff").shift(1) / 1200
                ).alias("ret"),
                pl.lit(None, dtype=pl.Float64).alias("value_sig"),
            )
        df = df.with_columns(
            pl.col("ret").log1p().rolling_sum(12, min_samples=12).alias("trend"),
            pl.lit(asset).alias("asset"),
        )
        frames.append(
            df.select("date", "asset", "ret", "trend", pl.col("value_sig").alias("value"))
        )
    if not frames:
        raise ValueError("no credit or crypto data — run `atlas ingest fred --all`")
    this_month = at.date().replace(day=1)  # still running: its "month end" is not final
    out = pl.concat(frames).filter(pl.col("date") < this_month).sort(["asset", "date"])
    fwd = pl.col("ret").shift(-1)
    for k in range(2, HORIZON + 1):
        fwd = fwd + pl.col("ret").shift(-k)
    return out.with_columns(fwd.over("asset").alias("fwd"))


@dataclass
class SleeveForecast:
    asset: str
    date: date
    prob: float
    trend: float | None
    value: float | None
    fwd: float | None


def _features(asset: str) -> list[str]:
    return ["trend", "value"] if SLEEVES[asset]["kind"] == "spread" else ["trend"]


def walk_forward(panel: pl.DataFrame) -> list[SleeveForecast]:
    """Monthly out-of-sample probabilities that the next 3-month excess return is positive."""
    out: list[SleeveForecast] = []
    for asset in sorted(panel["asset"].unique().to_list()):
        feats = _features(asset)
        df = panel.filter(pl.col("asset") == asset).drop_nulls(feats).sort("date")
        rows = df.to_dicts()
        model: LogisticRegression | None = None
        fallback: float | None = None  # all past outcomes alike: smoothed base rate instead
        mu = sd = np.zeros(len(feats))
        last_fit = -REFIT_EVERY
        for i, r in enumerate(rows):
            train = [x for x in rows[: max(0, i - HORIZON + 1)] if x["fwd"] is not None]
            if len(train) < MIN_TRAIN:
                continue
            if (model is None and fallback is None) or i - last_fit >= REFIT_EVERY:
                x = np.array([[t[f] for f in feats] for t in train], dtype=float)
                y = np.array([t["fwd"] > 0 for t in train])
                last_fit = i
                if y.all() or not y.any():  # nothing to separate: Laplace-smoothed rate
                    model, fallback = None, float((y.sum() + 1) / (len(y) + 2))
                else:
                    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-12
                    model, fallback = LogisticRegression(C=1.0).fit((x - mu) / sd, y), None
            if model is not None:
                z = (np.array([[r[f] for f in feats]], dtype=float) - mu) / sd
                prob = float(model.predict_proba(z)[0, 1])
            else:
                prob = float(fallback if fallback is not None else 0.5)
            out.append(
                SleeveForecast(
                    asset=asset,
                    date=r["date"],
                    prob=prob,
                    trend=r["trend"],
                    value=r.get("value"),
                    fwd=r["fwd"],
                )
            )
    return out


def score(forecasts: list[SleeveForecast], start: date, end: date) -> dict[str, Any]:
    """Brier vs the base rate known at each date, hit rate of calls, per asset and pooled."""
    res: dict[str, Any] = {}
    for asset in [*sorted({f.asset for f in forecasts}), "all"]:
        fs = [f for f in forecasts if (asset == "all" or f.asset == asset) and f.fwd is not None]
        hist = {a: [f for f in forecasts if f.asset == a and f.fwd is not None] for a in SLEEVES}
        win = [f for f in fs if start <= f.date <= end]
        if not win:
            continue
        brier, base = [], []
        for f in win:
            past = [h.fwd > 0 for h in hist[f.asset] if h.date < f.date and h.fwd is not None]
            rate = float(np.mean(past)) if past else 0.5
            hit = 1.0 if (f.fwd or 0) > 0 else 0.0
            brier.append((f.prob - hit) ** 2)
            base.append((rate - hit) ** 2)
        calls = [f for f in win if abs(f.prob - 0.5) >= 0.05]
        right = [f for f in calls if (f.prob > 0.5) == ((f.fwd or 0) > 0)]
        res[asset] = {
            "months": len(win),
            "brier": float(np.mean(brier)),
            "brier_base_rate": float(np.mean(base)),
            "brier_improvement": float(np.mean(base) - np.mean(brier)),
            "calls": len(calls),
            "hit_rate": len(right) / len(calls) if calls else None,
        }
    return res


def stance_for(prob: float) -> str:
    if prob >= 0.65:
        return "strong_overweight"
    if prob >= 0.55:
        return "overweight"
    if prob <= 0.35:
        return "strong_underweight"
    if prob <= 0.45:
        return "underweight"
    return "neutral"


def latest_views(forecasts: list[SleeveForecast]) -> list[dict[str, Any]]:
    """The newest forecast per asset, as plain records."""
    latest: dict[str, SleeveForecast] = {}
    for f in forecasts:
        if f.asset not in latest or f.date > latest[f.asset].date:
            latest[f.asset] = f
    out = []
    for a, f in sorted(latest.items()):
        spec = SLEEVES[a]
        out.append(
            {
                "asset_id": a,
                "asset_name": spec["name"],
                "asset_class": spec["asset_class"],
                "as_of": f.date.isoformat(),
                "horizon": f"{HORIZON}M",
                "prob_positive_excess": round(f.prob, 3),
                "stance": stance_for(f.prob),
                "benchmark": "10-year Treasuries" if spec["kind"] == "spread" else "cash",
                "drivers": {
                    "trend_12m": None if f.trend is None else round(f.trend, 4),
                    "spread_vs_5y_avg": None
                    if f.value is None or math.isnan(f.value)
                    else round(f.value, 3),
                },
            }
        )
    return out


FAMILY = "ts_v1"
UNIVERSE = "sleeves"


def open_sleeve_holdout(
    data_dir: Any, store: PITStore, *, reason: str, code_sha: str
) -> dict[str, Any]:
    """Score 2021-2025 once and log it (clean: the design was fixed before any test)."""
    from atlas.backtest.holdout import (
        HOLDOUT_END,
        HOLDOUT_START,
        HoldoutAlreadyOpened,
        HoldoutLog,
    )

    log = HoldoutLog(data_dir)
    previous = log.openings(UNIVERSE, FAMILY)
    if previous:
        raise HoldoutAlreadyOpened(
            f"holdout already opened for {UNIVERSE}/{FAMILY} on {previous[0]['opened_at']}"
        )
    fc = walk_forward(sleeve_panel(store))
    scores = score(fc, HOLDOUT_START, HOLDOUT_END)
    pooled = scores.get("all", {})
    entry = {
        "opened_at": datetime.now(UTC).isoformat(),
        "universe": UNIVERSE,
        "family": FAMILY,
        "window": [HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()],
        "reopening": False,
        "contaminated": False,
        "reason": reason,
        "code_sha": code_sha,
        "config_version": FAMILY,
        "ensemble": {
            "months": pooled.get("months"),
            "brier": pooled.get("brier"),
            "brier_baseline": pooled.get("brier_base_rate"),
            "brier_improvement": pooled.get("brier_improvement"),
            "hit_rate": pooled.get("hit_rate"),
        },
        "assets": {k: v for k, v in scores.items() if k != "all"},
        "research_window": score(fc, date(1900, 1, 1), date(2020, 12, 31)),
    }
    log.record(entry)
    return entry


def freeze_sleeves(data_dir: Any, store: PITStore) -> tuple[bool, str]:
    """Write this month's views once to gold/sleeves/3M/<month>.json (never overwritten)."""
    import json

    from atlas import jsonsafe

    try:
        panel = sleeve_panel(store)
    except ValueError as exc:  # no credit / crypto series ingested
        return True, f"skipped: {exc}"
    views = latest_views(walk_forward(panel))
    if not views:
        return True, "no forecasts yet"
    month = max(v["as_of"] for v in views)
    folder = data_dir / "gold" / "sleeves" / f"{HORIZON}M"
    path = folder / f"{month}.json"
    if path.exists():
        same = json.loads(path.read_text()) == json.loads(jsonsafe.dumps(views))
        return True, f"{month[:7]} frozen earlier; {'unchanged' if same else 'kept first'}"
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonsafe.dumps(views, indent=2))
    return True, f"{month[:7]} frozen -> gold/sleeves/{HORIZON}M/{path.name}"
