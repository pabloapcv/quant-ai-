"""Tools the assistant can call: thin, cached, JSON-safe wrappers over the quant engine.

PRD §10.1: the language model is an analyst, not a calculator — every number it states
must come from one of these tools. Each tool returns a plain dict; errors (missing data,
unknown universe) come back as ``{"error": ...}`` so the model can explain them instead of
guessing. Results are cached per argument set: a conversation re-asking the same thing
does not re-run a 20-second gate suite.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from atlas import jsonsafe
from atlas.backtest.holdout import HoldoutLog
from atlas.backtest.trials import TrialLog
from atlas.catalog import Catalog
from atlas.models.expected_returns.ensemble import EnsembleConfig, EnsembleRun, walk_forward
from atlas.models.nowcast import gdp_nowcast
from atlas.models.portfolio.optimizer import PortfolioConfig, backtest_portfolio
from atlas.models.regime.analogs import find_analogs as search_analogs
from atlas.models.regime.analogs import forward_excess, summarize_forward
from atlas.models.regime.engine import RegimeEngine, data_month_for
from atlas.models.regime.market import market_regimes
from atlas.models.regime.quadrant import classify_expr
from atlas.models.scenario import FACTOR_NAMES, FACTORS, factor_changes, fit_scenario_model
from atlas.pit.store import PITStore
from atlas.signals.research import ResearchContext, run_gate_suite
from atlas.signals.universes import UNIVERSES, get_universe
from atlas.text.documents import DocStore, statement_changes
from atlas.text.search import BM25Index
from atlas.text.tone import load_tones
from atlas.views.publish import build_views, evaluate

UNIVERSE_ENUM = sorted(UNIVERSES)


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


_UNIVERSE = {"type": "string", "enum": UNIVERSE_ENUM, "description": "Research universe id"}
_HORIZON = {"type": "integer", "enum": [1, 3], "description": "Forecast horizon in months"}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_universes",
        "description": "List the research universes (industries, countries) with what each covers.",
        "input_schema": _schema({}, []),
    },
    {
        "name": "get_views",
        "description": (
            "Current model views for every asset in a universe: stance, calibrated probability "
            "of beating the median asset, expected excess return vs the equal-weight average "
            "with its 80% range, signal agreement, confidence and the top drivers."
        ),
        "input_schema": _schema(
            {"universe": _UNIVERSE, "horizon": _HORIZON}, ["universe", "horizon"]
        ),
    },
    {
        "name": "get_track_record",
        "description": (
            "Walk-forward track record of the view model since 1970: rank IC, ICIR, Brier "
            "score vs 50%, interval coverage, directional hit rate, overweight-minus-underweight; "
            "plus the locked 2021-2025 holdout result when it has been opened."
        ),
        "input_schema": _schema(
            {"universe": _UNIVERSE, "horizon": _HORIZON}, ["universe", "horizon"]
        ),
    },
    {
        "name": "get_signal_gates",
        "description": (
            "Every signal in a universe checked against the PRD promotion gates (IC t-stat, "
            "deflated Sharpe, family PBO, regime consistency, ensemble contribution, rationale). "
            "Slow on first call (about 20-30 seconds)."
        ),
        "input_schema": _schema({"universe": _UNIVERSE}, ["universe"]),
    },
    {
        "name": "get_portfolio",
        "description": (
            "Model portfolio for a universe: latest weights vs the equal-weight benchmark and "
            "the walk-forward backtest since 1970 (net active return, tracking error, "
            "information ratio)."
        ),
        "input_schema": _schema({"universe": _UNIVERSE}, ["universe"]),
    },
    {
        "name": "search_catalog",
        "description": "Find data series by keyword ('japan yield', 'cpi', 'euro'); returns ids.",
        "input_schema": _schema({"query": {"type": "string"}}, ["query"]),
    },
    {
        "name": "get_series",
        "description": (
            "Recent observations of one data series as known at a date (point-in-time). "
            "Returns up to 'last_n' periods."
        ),
        "input_schema": _schema(
            {
                "series_id": {"type": "string"},
                "last_n": {"type": "integer", "description": "1-120"},
                "as_of": {"type": "string", "description": "YYYY-MM-DD; empty for latest"},
            },
            ["series_id", "last_n", "as_of"],
        ),
    },
    {
        "name": "get_regime",
        "description": (
            "US macro regime as of a date: growth/inflation/financial-conditions scores, "
            "quadrant probabilities, 12-month recession probability, and the current-quarter "
            "real GDP nowcast (Atlanta Fed GDPNow, as known at the date). Needs FRED data."
        ),
        "input_schema": _schema(
            {"as_of": {"type": "string", "description": "YYYY-MM-DD; empty for today"}},
            ["as_of"],
        ),
    },
    {
        "name": "find_analogs",
        "description": (
            "Historical analogs: the k past months (since 1971) whose US macro state "
            "(growth, inflation, financial conditions and their 3-month changes) is closest "
            "to the as-of month, and what each asset in a universe returned over the next "
            "3 and 12 months vs the equal-weight average. Descriptive, not a forecast."
        ),
        "input_schema": _schema(
            {
                "universe": _UNIVERSE,
                "as_of": {"type": "string", "description": "YYYY-MM-DD; empty for today"},
                "k": {"type": "integer", "description": "Number of analogs, 3-20"},
            },
            ["universe", "as_of", "k"],
        ),
    },
    {
        "name": "run_scenario",
        "description": (
            "Scenario engine: how each asset in a universe moved, historically, in a month "
            "with the given factor shocks. Factors not shocked move by their conditional "
            "expectation (reported). Units: us10y, us2y, breakeven10y, credit_spread in "
            "percentage points (0.5 = +50bp); usd and oil as fractions (-0.2 = -20%); vix in "
            "points. Returns excess vs the equal-weight average with an 80% bootstrap range."
        ),
        "input_schema": _schema(
            {
                "universe": _UNIVERSE,
                "shocks": {
                    "type": "array",
                    "items": _schema(
                        {
                            "factor": {"type": "string", "enum": list(FACTOR_NAMES)},
                            "value": {"type": "number"},
                        },
                        ["factor", "value"],
                    ),
                },
                "same_regime": {
                    "type": "boolean",
                    "description": "Fit only on months in the current macro quadrant",
                },
            },
            ["universe", "shocks", "same_regime"],
        ),
    },
    {
        "name": "search_documents",
        "description": (
            "Search central-bank text for passages matching a query, as published by the "
            "as-of date: FOMC statements and minutes (2006 onward) and ECB monetary policy "
            "statements and press-conference Q&A transcripts (1998 onward). Returns the best "
            "passages with bank, document, meeting date and link. Use it for what central "
            "banks said, not for numbers."
        ),
        "input_schema": _schema(
            {
                "query": {"type": "string"},
                "bank": {"type": "string", "enum": ["any", "fomc", "ecb"]},
                "kind": {
                    "type": "string",
                    "enum": ["any", "statement", "minutes", "press_conference"],
                },
                "as_of": {"type": "string", "description": "YYYY-MM-DD; empty for today"},
                "k": {"type": "integer", "description": "Passages to return, 1-12"},
            },
            ["query", "bank", "kind", "as_of", "k"],
        ),
    },
    {
        "name": "get_ecb_statement",
        "description": (
            "The ECB monetary policy statement in force at a date (the latest on or before "
            "it): full text, sentences added and removed since the previous meeting, and the "
            "opening of the press-conference Q&A."
        ),
        "input_schema": _schema(
            {"date": {"type": "string", "description": "YYYY-MM-DD; empty for the latest"}},
            ["date"],
        ),
    },
    {
        "name": "get_fomc_statement",
        "description": (
            "The FOMC statement in force at a date (the latest released on or before it): "
            "full text, the sentences added and removed since the previous statement, and an "
            "extracted reading (action, size, hawkish/dovish stance -2..+2, dissents)."
        ),
        "input_schema": _schema(
            {"date": {"type": "string", "description": "YYYY-MM-DD; empty for the latest"}},
            ["date"],
        ),
    },
]


class ToolKit:
    def __init__(self, data_dir: Path, config_dir: Path) -> None:
        self.data_dir = data_dir
        self.store = PITStore(data_dir)
        self.catalog = Catalog.load(config_dir / "series.yaml")
        self._cache: dict[str, dict[str, Any]] = {}
        self._runs: dict[tuple[str, int], tuple[ResearchContext, EnsembleRun]] = {}
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "list_universes": self.list_universes,
            "get_views": self.get_views,
            "get_track_record": self.get_track_record,
            "get_signal_gates": self.get_signal_gates,
            "get_portfolio": self.get_portfolio,
            "search_catalog": self.search_catalog,
            "get_series": self.get_series,
            "get_regime": self.get_regime,
            "find_analogs": self.find_analogs,
            "run_scenario": self.run_scenario,
            "search_documents": self.search_documents,
            "get_fomc_statement": self.get_fomc_statement,
            "get_ecb_statement": self.get_ecb_statement,
        }
        self._text_index: BM25Index | None = None

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool; returns (JSON text, is_error). Never raises for tool-level failures."""
        key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        if key not in self._cache:
            handler = self._handlers.get(name)
            if handler is None:
                return jsonsafe.dumps({"error": f"unknown tool {name!r}"}), True
            try:
                self._cache[key] = handler(**arguments)
            except (ValueError, KeyError, TypeError) as exc:
                return jsonsafe.dumps({"error": str(exc)}), True
        result = self._cache[key]
        return jsonsafe.dumps(result), "error" in result

    # ------------------------------------------------------------------ helpers
    def _context(self, universe: str, *, regimes: bool = False) -> ResearchContext:
        uni = get_universe(universe)
        reg = None
        if regimes:
            try:
                reg = market_regimes(self.store).select("date", "regime")
            except ValueError:
                reg = None
        return ResearchContext(
            panel=uni.build_panel(self.store),
            universe=universe,
            log=TrialLog(self.data_dir),
            regimes=reg,
            n_quantiles=3,
            signals=uni.signals,
            families=uni.families,
        )

    def _run(self, universe: str, horizon: int) -> tuple[ResearchContext, EnsembleRun]:
        if (universe, horizon) not in self._runs:
            ctx = self._context(universe)
            cfg = EnsembleConfig(
                horizon=horizon, components=get_universe(universe).ensemble_components
            )
            self._runs[(universe, horizon)] = (ctx, walk_forward(ctx, cfg))
        return self._runs[(universe, horizon)]

    # ------------------------------------------------------------------ tools
    def list_universes(self) -> dict[str, Any]:
        return {
            "universes": [{"id": u.name, "description": u.description} for u in UNIVERSES.values()]
        }

    def get_views(self, universe: str, horizon: int) -> dict[str, Any]:
        _, run = self._run(universe, horizon)
        views = build_views(run, universe)
        return {
            "universe": universe,
            "as_of_month": views[0].as_of.isoformat(),
            "horizon": views[0].horizon,
            "benchmark": views[0].expected_excess_return.benchmark,
            "views": [
                {
                    "asset": v.asset_name,
                    "stance": v.stance.value,
                    "score": round(v.score, 2),
                    "prob_beat_median": None
                    if v.prob_outperform is None
                    else round(v.prob_outperform, 3),
                    "expected_excess_return": round(v.expected_excess_return.mean, 4),
                    "range_p10": v.expected_excess_return.p10
                    and round(v.expected_excess_return.p10, 4),
                    "range_p90": v.expected_excess_return.p90
                    and round(v.expected_excess_return.p90, 4),
                    "agreement": None if v.agreement is None else round(v.agreement, 2),
                    "confidence": v.confidence.value,
                    "top_drivers": [
                        {"signal": d.name, "contribution": round(d.contribution, 3)}
                        for d in v.drivers[:3]
                    ],
                }
                for v in views
            ],
            "range_coverage": 0.8,  # range_p10..range_p90 is an 80% interval
            "note": "Returns are fractions (0.012 = 1.2%) over the horizon. Not investment advice.",
        }

    def get_track_record(self, universe: str, horizon: int) -> dict[str, Any]:
        _, run = self._run(universe, horizon)
        ev = evaluate(run, date(1970, 1, 1))
        summary = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.summary().items()}
        out: dict[str, Any] = {
            "universe": universe,
            "horizon_months": horizon,
            "since": "1970-01",
            **summary,
        }
        opened = HoldoutLog(self.data_dir).openings(universe, "ensemble_v1")
        if opened and horizon == 3:
            # the latest entry: corrections are appended, the log is never edited
            first = opened[-1]
            ens = first.get("ensemble", {})
            out["holdout_2021_2025"] = {
                "ic_mean": round(ens.get("ic_mean", float("nan")), 4),
                "ic_tstat": round(ens.get("ic_tstat", float("nan")), 2),
                # entries before 2026-10-22 recorded no call count: their hit rate is unreliable
                "hit_rate": round(ens["hit_rate"], 3)
                if "directional_calls" in ens and "hit_rate" in ens
                else None,
                "directional_calls": ens.get("directional_calls"),
                "months": ens.get("months"),
                "contaminated": first.get("contaminated", True),
                "note": "The locked 2021-2025 holdout, opened once. Contaminated means the "
                "model was designed on data that included those years.",
            }
        return out

    def get_signal_gates(self, universe: str) -> dict[str, Any]:
        suite = run_gate_suite(self._context(universe, regimes=True), n_splits=8)
        return {
            "universe": universe,
            "configurations_tried": suite.n_trials,
            "signals": [
                {
                    "signal": g.signal,
                    "description": suite.signals[g.signal].description,
                    "ic_tstat": round(g.ic_tstat, 2),
                    "deflated_sharpe": round(g.dsr, 3),
                    "family_pbo": None if g.pbo is None else round(g.pbo, 3),
                    "regimes_positive_of_4": g.regimes_positive,
                    "ensemble_contribution": None
                    if g.marginal_icir is None
                    else round(g.marginal_icir, 3),
                    "gates_passed_of_6": sum(g.checks.values()),
                    "failed_gates": g.failed,
                }
                for g in suite.gates.values()
            ],
        }

    def get_portfolio(self, universe: str) -> dict[str, Any]:
        ctx, run = self._run(universe, 3)
        cap = 0.05 if universe.startswith("ind") else 0.08
        panel = ctx.panel.select("date", "asset", "ret")
        latest = run.forecasts.drop_nulls("exp_excess")["date"].max()
        filler = panel.filter(pl.col("date") == latest).with_columns(
            pl.col("date").dt.offset_by("1mo"), pl.lit(0.0).alias("ret")
        )
        bt = backtest_portfolio(
            run.forecasts.filter(pl.col("date") >= date(1970, 1, 1)),
            pl.concat([panel, filler]),
            horizon=3,
            cfg=PortfolioConfig(max_active=cap),
        )
        hist = bt.monthly.filter(pl.col("date") < latest)
        summary = type(bt)(monthly=hist, weights=bt.weights).summary()
        w = bt.weights.filter(pl.col("date") == latest)
        uni = get_universe(universe)
        return {
            "universe": universe,
            "as_of_month": str(latest),
            "constraints": {
                "tracking_error_budget": 0.04,
                "max_active_weight": cap,
                "long_only": True,
            },
            "weights": [
                {
                    "asset": uni.asset_name(r["asset"]),
                    "weight": round(r["weight"], 4),
                    "benchmark": round(r["bench"], 4),
                    "active": round(r["weight"] - r["bench"], 4),
                }
                for r in w.sort("weight", descending=True).iter_rows(named=True)
            ],
            "backtest_since_1970": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()
            },
        }

    def search_catalog(self, query: str) -> dict[str, Any]:
        words = [w for w in query.lower().split() if w]
        hits = [
            {
                "id": s.id,
                "description": s.description,
                "frequency": s.frequency.value,
                "source": s.source,
            }
            for s in self.catalog
            if all(w in f"{s.id} {s.description} {' '.join(s.tags)}".lower() for w in words)
        ]
        return {"query": query, "matches": hits[:25], "total": len(hits)}

    def get_series(self, series_id: str, last_n: int, as_of: str) -> dict[str, Any]:
        spec = self.catalog.get(series_id)
        at: date | datetime = date.fromisoformat(as_of) if as_of else datetime.now(UTC)
        df = self.store.as_of([series_id], at)
        if df.is_empty():
            return {"error": f"no data for {series_id} as of {as_of or 'now'}; ingest it first"}
        n = max(1, min(int(last_n), 120))
        tail = df.sort("event_time").tail(n)
        return {
            "series_id": series_id,
            "description": spec.description,
            "as_of": as_of or "latest",
            "observations": [
                {
                    "period": r["event_time"].isoformat(),
                    "value": r["value"],
                    "approx_vintage": r["is_approx_vintage"],
                }
                for r in tail.iter_rows(named=True)
            ],
        }

    def get_regime(self, as_of: str) -> dict[str, Any]:
        cube = self.store.load_cube()
        if "US.IP" not in cube:
            return {
                "error": "macro regime needs FRED data: set ATLAS_FRED_API_KEY and run "
                "`atlas ingest fred --all`"
            }
        at = date.fromisoformat(as_of) if as_of else datetime.now(UTC).date()
        rep = RegimeEngine(cube).run(at, with_hmm=True)  # second opinion on the quadrant
        out: dict[str, Any] = json.loads(rep.model_dump_json())
        out["gdp_nowcast"] = gdp_nowcast(cube, at)
        return out

    def find_analogs(self, universe: str, as_of: str, k: int) -> dict[str, Any]:
        cube = self.store.load_cube()
        if "US.IP" not in cube:
            return {"error": "analogs need FRED data: run `atlas ingest fred --all`"}
        at = date.fromisoformat(as_of) if as_of else datetime.now(UTC).date()
        comps, _ = RegimeEngine(cube).composites(at)
        res = search_analogs(comps, current_month=data_month_for(at), k=max(3, min(int(k), 20)))
        if res is None:
            return {"error": f"no complete macro state for {data_month_for(at)}"}
        uni = get_universe(universe)
        panel = uni.build_panel(self.store)
        months = [a.month for a in res.analogs]
        forward: dict[str, Any] = {}
        per_analog: dict[int, dict[date, pl.DataFrame]] = {}
        for h in (3, 12):
            ex = forward_excess(panel, months, h)
            per_analog[h] = {m: ex.filter(pl.col("month") == m) for m in months}
            forward[f"{h}m"] = [
                {
                    "asset": uni.asset_name(r["asset"]),
                    "mean_excess": round(r["mean"], 4),
                    "median_excess": round(r["median"], 4),
                    "share_of_analogs_positive": round(r["hit_rate"], 2),
                    "analogs_counted": r["n"],
                }
                for r in summarize_forward(ex).iter_rows(named=True)
            ]

        def best_worst(h: int, m: date) -> dict[str, Any]:
            ex = per_analog[h][m].sort("excess", descending=True)
            if ex.is_empty():
                return {}
            top, bottom = ex.row(0, named=True), ex.row(-1, named=True)
            return {
                "best": uni.asset_name(top["asset"]),
                "best_excess": round(top["excess"], 4),
                "worst": uni.asset_name(bottom["asset"]),
                "worst_excess": round(bottom["excess"], 4),
            }

        return {
            "universe": universe,
            "current_month": res.current_month.isoformat(),
            "current_state": {c: round(v, 2) for c, v in res.current_state.items()},
            "analogs": [
                {
                    "month": a.month.isoformat(),
                    "distance": round(a.distance, 2),
                    "state": {c: round(v, 2) for c, v in a.state.items()},
                    "next_12m": best_worst(12, a.month),
                }
                for a in res.analogs
            ],
            "forward_excess_by_asset": forward,
            "note": (
                "Distance is Mahalanobis on the macro state (lower = closer). Excess returns "
                "are fractions vs the equal-weight universe average. A small-sample "
                "description of past episodes, not a backtested forecast or advice."
            ),
        }

    def run_scenario(
        self, universe: str, shocks: list[dict[str, Any]], same_regime: bool
    ) -> dict[str, Any]:
        cube = self.store.load_cube()
        if any(f.series_id not in cube for f in FACTORS):
            return {"error": "scenario factors need FRED data: run `atlas ingest fred --all`"}
        today = datetime.now(UTC).date()
        uni = get_universe(universe)
        panel = uni.build_panel(self.store)
        months = None
        regime = None
        if same_regime:
            comps, _ = RegimeEngine(cube).composites(today)
            quad = comps.select(
                "event_time", classify_expr(pl.col("growth"), pl.col("inflation")).alias("q")
            ).drop_nulls()
            regime = quad["q"][-1]
            months = quad.filter(pl.col("q") == regime)["event_time"].to_list()
        model = fit_scenario_model(factor_changes(self.store, today), panel, months=months)
        res = model.run({sh["factor"]: float(sh["value"]) for sh in shocks})

        def row(a: Any) -> dict[str, Any]:
            return {
                "expected": round(a.expected, 4),
                "range_p10": round(a.p10, 4),
                "range_p90": round(a.p90, 4),
            }

        units = {f.name: f.unit for f in FACTORS}
        return {
            "universe": universe,
            "shocks": [{"factor": f, "value": v, "unit": units[f]} for f, v in res.shocks.items()],
            "implied_moves_of_other_factors": {
                f: {"value": round(v, 4), "unit": units[f]}
                for f, v in res.implied_factor_moves.items()
            },
            "fit": {
                "months": res.n_months,
                "from": res.window[0].isoformat(),
                "to": res.window[1].isoformat(),
                "regime_filter": regime,
            },
            "range_coverage": 0.8,  # range_p10..range_p90 is an 80% bootstrap band
            "equal_weight_average_return": row(res.average),
            "excess_vs_average": [{"asset": uni.asset_name(a.asset), **row(a)} for a in res.assets],
            "note": (
                "Same-month historical response (betas on monthly factor changes), not a "
                "forecast. Returns are fractions. Not investment advice."
            ),
        }

    def _as_of_end(self, value: str) -> datetime:
        if not value:
            return datetime.now(UTC)
        d = date.fromisoformat(value)
        return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=UTC)

    def search_documents(
        self, query: str, bank: str, kind: str, as_of: str, k: int
    ) -> dict[str, Any]:
        if self._text_index is None:
            docs = list(DocStore(self.data_dir).iter())
            if not docs:
                return {"error": "no documents stored: run `atlas text sync`"}
            self._text_index = BM25Index(docs)
        hits = self._text_index.search(
            query,
            k=max(1, min(int(k), 12)),
            as_of=self._as_of_end(as_of),
            kind=None if kind == "any" else kind,
            source=None if bank == "any" else bank,
        )
        return {
            "query": query,
            "as_of": as_of or "latest",
            "passages": [
                {
                    "bank": h.passage.doc.source,
                    "document": h.passage.doc.doc_id,
                    "kind": h.passage.doc.kind,
                    "meeting_date": h.passage.doc.meeting_date.isoformat(),
                    "published": h.passage.doc.published_at.date().isoformat(),
                    "url": h.passage.doc.url,
                    "text": h.passage.text[:900],
                    "relevance": round(h.score, 2),
                }
                for h in hits
            ],
            "note": "Keyword (BM25) search. Quote the Fed's words; numbers in passages are "
            "the Fed's, as of publication.",
        }

    def get_fomc_statement(self, date: str) -> dict[str, Any]:
        docs = list(
            DocStore(self.data_dir).iter(
                source="fomc", kind="statement", as_of=self._as_of_end(date)
            )
        )
        if not docs:
            return {"error": f"no FOMC statement stored on or before {date or 'today'}"}
        cur, prev = docs[-1], (docs[-2] if len(docs) > 1 else None)
        tone = load_tones(self.data_dir).get(cur.doc_id)
        return {
            "meeting_date": cur.meeting_date.isoformat(),
            "published": cur.published_at.isoformat(),
            "url": cur.url,
            "text": cur.paragraphs,
            "previous_meeting_date": prev.meeting_date.isoformat() if prev else None,
            "changes_since_previous": statement_changes(prev, cur) if prev else None,
            "extracted_reading": None
            if tone is None
            else tone.model_dump(exclude={"doc_id", "model", "text_change_bp"}),
        }

    def get_ecb_statement(self, date: str) -> dict[str, Any]:
        store = DocStore(self.data_dir)
        at = self._as_of_end(date)
        docs = list(store.iter(source="ecb", kind="statement", as_of=at))
        if not docs:
            return {"error": f"no ECB statement stored on or before {date or 'today'}"}
        cur, prev = docs[-1], (docs[-2] if len(docs) > 1 else None)
        qa = [d for d in store.iter(source="ecb", kind="press_conference", as_of=at)]
        qa_doc = next((d for d in reversed(qa) if d.meeting_date == cur.meeting_date), None)
        return {
            "meeting_date": cur.meeting_date.isoformat(),
            "url": cur.url,
            "text": cur.paragraphs,
            "previous_meeting_date": prev.meeting_date.isoformat() if prev else None,
            "changes_since_previous": statement_changes(prev, cur) if prev else None,
            "extracted_reading": None
            if (tone := load_tones(self.data_dir, "ecb").get(cur.doc_id)) is None
            else tone.model_dump(exclude={"doc_id", "model", "text_change_bp"}),
            "press_conference_opening": qa_doc.paragraphs[:6] if qa_doc else None,
            "press_conference_paragraphs": len(qa_doc.paragraphs) if qa_doc else 0,
        }
