"""``atlas`` command-line interface."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from atlas import jsonsafe
from atlas.backtest.trials import TrialLog
from atlas.catalog import Catalog
from atlas.connectors.base import RawStore
from atlas.connectors.fred import FredConnector
from atlas.connectors.kenfrench import KenFrenchConnector
from atlas.connectors.sdmx import AGENCIES, SdmxConnector
from atlas.ingest import IngestReport, ingest_series
from atlas.log import configure_logging
from atlas.models.expected_returns.ensemble import EnsembleConfig
from atlas.models.expected_returns.ensemble import walk_forward as ensemble_walk_forward
from atlas.models.portfolio.optimizer import PortfolioConfig, backtest_portfolio
from atlas.models.regime.engine import RegimeEngine, RegimeReport
from atlas.models.regime.evaluate import walk_forward
from atlas.models.regime.market import market_regimes
from atlas.pit.snapshot import create_snapshot, git_sha, load_snapshot, verify_snapshot
from atlas.pit.store import PITStore
from atlas.quality.contracts import find_stale
from atlas.settings import get_settings
from atlas.signals.cards import write_cards
from atlas.signals.research import ResearchContext, family_pbo, run_gate_suite, run_signal
from atlas.signals.universes import UNIVERSES, get_universe
from atlas.views.publish import build_views, evaluate, publish

app = typer.Typer(no_args_is_help=True, help="Atlas quant research assistant.")
ingest_app = typer.Typer(no_args_is_help=True, help="Pull data from sources into the PIT store.")
pit_app = typer.Typer(no_args_is_help=True, help="Query the point-in-time store.")
snapshot_app = typer.Typer(no_args_is_help=True, help="Create and verify reproducible snapshots.")
catalog_app = typer.Typer(no_args_is_help=True, help="Inspect the series catalog.")
regime_app = typer.Typer(no_args_is_help=True, help="Macro regime engine (Phase 2).")
views_app = typer.Typer(no_args_is_help=True, help="Industry views: ensemble forecasts (Phase 4).")
portfolio_app = typer.Typer(
    no_args_is_help=True, help="Model portfolio vs equal-weight benchmark (Phase 4)."
)
research_app = typer.Typer(no_args_is_help=True, help="Signal research & backtests (Phase 3).")
pipeline_app = typer.Typer(
    no_args_is_help=True, help="Scheduled pipelines and the live track record."
)
report_app = typer.Typer(no_args_is_help=True, help="Research reports (Phase 6).")
app.add_typer(ingest_app, name="ingest")
app.add_typer(pit_app, name="pit")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(catalog_app, name="catalog")
app.add_typer(regime_app, name="regime")
app.add_typer(views_app, name="views")
app.add_typer(portfolio_app, name="portfolio")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(report_app, name="report")
app.add_typer(research_app, name="research")


@app.callback()
def _main() -> None:
    s = get_settings()
    configure_logging(s.log_level, json=s.log_json)


def _catalog() -> Catalog:
    return Catalog.load(get_settings().config_dir / "series.yaml")


def _store() -> PITStore:
    return PITStore(get_settings().data_dir)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


# ---------------------------------------------------------------------------- catalog
@catalog_app.command("list")
def catalog_list(
    source: Annotated[str | None, typer.Option(help="Filter by source")] = None,
) -> None:
    cat = _catalog()
    specs = cat.by_source(source) if source else list(cat)
    for s in specs:
        typer.echo(
            f"{s.id:<24} {s.source}:{s.source_id:<14} {s.frequency.value:<3} {s.description}"
        )
    typer.echo(f"\n{len(specs)} series")


# ---------------------------------------------------------------------------- ingest
@ingest_app.command("fred")
def ingest_fred(
    series: Annotated[
        list[str] | None, typer.Argument(help="Catalog ids, e.g. US.CPI.CORE")
    ] = None,
    all_series: Annotated[
        bool, typer.Option("--all", help="Every fred series in the catalog")
    ] = False,
    vintages: Annotated[
        bool, typer.Option("--vintages/--no-vintages", help="ALFRED history")
    ] = True,
    concurrency: Annotated[int, typer.Option(min=1, max=16)] = 4,
) -> None:
    """Ingest FRED/ALFRED series into the PIT store."""
    settings = get_settings()
    if settings.fred_api_key is None:
        raise typer.BadParameter("set ATLAS_FRED_API_KEY (free key at fredaccount.stlouisfed.org)")
    cat = _catalog()
    if all_series:
        specs = cat.by_source("fred")
    elif series:
        specs = [cat.get(s) for s in series]
    else:
        raise typer.BadParameter("pass series ids or --all")

    async def run() -> int:
        async with FredConnector(
            settings.fred_api_key.get_secret_value() if settings.fred_api_key else "",
            raw_store=RawStore(settings.data_dir),
            max_calls_per_minute=settings.fred_max_calls_per_minute,
            timeout_s=settings.http_timeout_s,
        ) as fred:
            reports = await ingest_series(
                specs, fred, _store(), vintages=vintages, concurrency=concurrency
            )
        return _echo_reports(reports)

    failures = asyncio.run(run())
    raise typer.Exit(code=1 if failures else 0)


def _echo_reports(reports: list[IngestReport]) -> int:
    for r in reports:
        status = "ok " if r.ok else "ERR"
        detail = r.error or (
            f"fetched={r.rows_fetched} new={r.rows_new} restated={r.rows_conflicting}"
        )
        typer.echo(f"[{status}] {r.series_id:<24} {detail}")
    return sum(not r.ok for r in reports)


@ingest_app.command("kenfrench")
def ingest_kenfrench(
    series: Annotated[
        list[str] | None, typer.Argument(help="Catalog ids, e.g. FF.IND12.HLTH")
    ] = None,
    all_series: Annotated[bool, typer.Option("--all", help="Every kenfrench series")] = False,
) -> None:
    """Ingest Fama-French factors and industry portfolios (no API key needed)."""
    settings = get_settings()
    cat = _catalog()
    if all_series:
        specs = cat.by_source("kenfrench")
    elif series:
        specs = [cat.get(s) for s in series]
    else:
        raise typer.BadParameter("pass series ids or --all")

    async def run() -> int:
        async with KenFrenchConnector(
            raw_store=RawStore(settings.data_dir), timeout_s=settings.http_timeout_s
        ) as kf:
            # one connector instance downloads each zip once; serialize to share the cache
            reports = await ingest_series(specs, kf, _store(), concurrency=1)
        return _echo_reports(reports)

    raise typer.Exit(code=1 if asyncio.run(run()) else 0)


@ingest_app.command("sdmx")
def ingest_sdmx(
    agency: Annotated[
        list[str] | None, typer.Option("--agency", help="ecb | oecd | bis | imf (repeatable)")
    ] = None,
    series: Annotated[list[str] | None, typer.Argument(help="Catalog ids")] = None,
) -> None:
    """Ingest global statistics (ECB, OECD, BIS, IMF) over SDMX. No keys needed."""
    settings = get_settings()
    cat = _catalog()
    if series:
        specs = [cat.get(s) for s in series]
    else:
        wanted = agency or sorted(AGENCIES)
        unknown = set(wanted) - set(AGENCIES)
        if unknown:
            raise typer.BadParameter(f"unknown agency: {sorted(unknown)}")
        specs = [s for a in wanted for s in cat.by_source(a)]

    async def run() -> int:
        failures = 0
        for name in sorted({s.source for s in specs}):
            async with SdmxConnector(
                name,
                raw_store=RawStore(settings.data_dir),
                timeout_s=max(90.0, settings.http_timeout_s),
            ) as conn:
                group = [s for s in specs if s.source == name]
                # serial: entries sharing a query reuse one cached download
                failures += _echo_reports(await ingest_series(group, conn, _store(), concurrency=1))
        return failures

    raise typer.Exit(code=1 if asyncio.run(run()) else 0)


# ---------------------------------------------------------------------------- pit
@pit_app.command("as-of")
def pit_as_of(
    series_id: str,
    at: Annotated[str, typer.Option("--at", help="YYYY-MM-DD (end of day New York)")],
    tail: Annotated[int, typer.Option(help="Show the last N periods")] = 12,
    exclude_approx: Annotated[bool, typer.Option("--exclude-approx")] = False,
) -> None:
    """Show a series exactly as it was known on a date."""
    df = _store().as_of([series_id], _parse_date(at), include_approx=not exclude_approx)
    with pl.Config(tbl_rows=tail, tbl_cols=8):
        typer.echo(df.select("event_time", "value", "knowledge_time", "vintage_id").tail(tail))


@pit_app.command("history")
def pit_history(series_id: str, event_time: str) -> None:
    """Revision path of one observation (all vintages)."""
    df = _store().history(series_id, _parse_date(event_time))
    typer.echo(df.select("knowledge_time", "value", "vintage_id", "is_approx_vintage"))


@pit_app.command("status")
def pit_status() -> None:
    """Series in the store and staleness against their frequency."""
    store = _store()
    last = store.last_knowledge_times()
    cat = _catalog()
    stale = {
        s.series_id
        for s in find_stale(
            last, [(s.id, s.frequency, s.publication_lag_days) for s in cat], datetime.now(UTC)
        )
    }
    for spec in cat:
        mark = "STALE" if spec.id in stale else "ok"
        typer.echo(f"{spec.id:<24} last_known={last.get(spec.id)!s:<32} {mark}")


# ---------------------------------------------------------------------------- snapshot
@snapshot_app.command("create")
def snapshot_create() -> None:
    s = get_settings()
    snap = create_snapshot(s.data_dir, s.config_dir)
    code = f"code={snap.code_sha} dirty={snap.code_dirty}"
    typer.echo(f"{snap.snapshot_id}  files={len(snap.files)}  {code}")


@snapshot_app.command("verify")
def snapshot_verify(snapshot_id: str) -> None:
    data_dir = get_settings().data_dir
    problems = verify_snapshot(data_dir, load_snapshot(data_dir, snapshot_id))
    for p in problems:
        typer.echo(p)
    typer.echo("ok" if not problems else f"{len(problems)} problems")
    raise typer.Exit(code=1 if problems else 0)


# ---------------------------------------------------------------------------- regime
def _print_report(rep: RegimeReport) -> None:
    typer.echo(f"Regime as of {rep.as_of} (data through {rep.data_month:%Y-%m})")
    typer.echo(f"  regime: {rep.regime.value if rep.regime else 'n/a'}")
    for q, p in sorted(rep.quadrant_probs.items(), key=lambda kv: -kv[1]):
        typer.echo(f"    {q.value:<24} {p:6.1%}")
    for name in ("growth", "inflation", "financial_conditions"):
        ps = getattr(rep, name)
        score = "n/a" if ps.score is None else f"{ps.score:+.2f}"
        chg = "" if ps.change_3m is None else f"  3m chg {ps.change_3m:+.2f}"
        prob = "" if ps.prob_above_trend is None else f"  P(>trend) {ps.prob_above_trend:.0%}"
        typer.echo(f"  {name:<22} {score}{chg}{prob}  [{ps.n_features} inputs]")
        for feat, c in sorted(ps.contributions.items(), key=lambda kv: -abs(kv[1])):
            typer.echo(f"      {feat:<20} {c:+.2f}")
    if rep.hmm:
        top = rep.hmm.most_likely
        typer.echo(f"  HMM ({rep.hmm.n_states} states): {top.label} p={top.prob:.0%}")
    if rep.recession:
        typer.echo(
            f"  recession within 12m: {rep.recession.prob_12m:.0%} "
            f"(probit, n={rep.recession.n_obs}, trained through {rep.recession.train_end:%Y-%m})"
        )
    if rep.missing_features:
        typer.echo(f"  missing inputs: {', '.join(rep.missing_features)}")


@regime_app.command("now")
def regime_now(
    at: Annotated[str | None, typer.Option("--at", help="YYYY-MM-DD; default today")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    hmm: Annotated[bool, typer.Option("--hmm/--no-hmm")] = True,
) -> None:
    """Current (or historical, point-in-time) macro regime."""
    when = _parse_date(at) if at else datetime.now(UTC).date()
    rep = RegimeEngine(_store().load_cube()).run(when, with_hmm=hmm)
    if as_json:
        typer.echo(rep.model_dump_json(indent=2))
    else:
        _print_report(rep)


@regime_app.command("analogs")
def regime_analogs(
    universe: Annotated[
        str, typer.Option(help="ind12 | ind49 | ctry_usd | ctry_dev_lcl")
    ] = "ind12",
    at: Annotated[str | None, typer.Option("--at", help="YYYY-MM-DD; default today")] = None,
    k: Annotated[int, typer.Option(help="Number of analogs")] = 10,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Past months whose macro state is closest to now, and what each asset did next."""
    from atlas.agent.tools import ToolKit

    s = get_settings()
    out = ToolKit(s.data_dir, s.config_dir).find_analogs(universe, at or "", k)
    if "error" in out:
        typer.echo(out["error"], err=True)
        raise typer.Exit(1)
    if as_json:
        typer.echo(jsonsafe.dumps(out, indent=2))
        return
    state = "  ".join(f"{c}={v:+.2f}" for c, v in out["current_state"].items())
    typer.echo(f"Macro state {out['current_month']}: {state}\n")
    typer.echo(
        f"{'month':<12}{'dist':>6}  {'growth':>7}{'infl':>7}{'fincond':>8}  next 12m best / worst"
    )
    for a in out["analogs"]:
        st, nx = a["state"], a["next_12m"]
        bw = (
            f"{nx['best']} {nx['best_excess']:+.1%} / {nx['worst']} {nx['worst_excess']:+.1%}"
            if nx
            else "n/a"
        )
        typer.echo(
            f"{a['month']:<12}{a['distance']:>6.2f}  {st['growth']:>+7.2f}{st['inflation']:>+7.2f}"
            f"{st['financial_conditions']:>+8.2f}  {bw}"
        )
    for h, rows in out["forward_excess_by_asset"].items():
        typer.echo(f"\nForward {h} excess vs equal-weight (mean / median / share positive):")
        for r in rows:
            typer.echo(
                f"  {r['asset']:<28}{r['mean_excess']:>+8.2%}{r['median_excess']:>+8.2%}"
                f"{r['share_of_analogs_positive']:>7.0%}  (n={r['analogs_counted']})"
            )
    typer.echo(f"\n{out['note']}")


@regime_app.command("backtest")
def regime_backtest(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "1995-01-01",
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD; default today")] = None,
    out: Annotated[str | None, typer.Option(help="Write monthly results to this CSV")] = None,
) -> None:
    """Walk-forward: real-time regime vs ex-post regime, and recession-probit scoring."""
    stop = _parse_date(end) if end else datetime.now(UTC).date()
    bt = walk_forward(RegimeEngine(_store().load_cube()), _parse_date(start), stop)
    for k, v in bt.summary().items():
        typer.echo(f"{k:<26} {v if not isinstance(v, float) else round(v, 4)}")
    typer.echo(bt.confusion)
    if out:
        bt.monthly.write_csv(out)
        typer.echo(f"wrote {out}")


# ---------------------------------------------------------------------------- research
def _fmt(v: object) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _context(universe: str, start: str | None, cost_bps: float, quantiles: int) -> ResearchContext:
    try:
        uni = get_universe(universe)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    store = _store()
    try:
        regimes: pl.DataFrame | None = market_regimes(store).select("date", "regime")
    except ValueError:
        regimes = None  # market factor not ingested: regime-conditional IC is skipped
    return ResearchContext(
        panel=uni.build_panel(store),
        universe=universe,
        log=TrialLog(get_settings().data_dir),
        regimes=regimes,
        n_quantiles=quantiles,
        cost_bps=cost_bps,
        start=_parse_date(start) if start else None,
        signals=uni.signals,
        families=uni.families,
    )


UniverseOpt = Annotated[str, typer.Option(help="ind12 | ind49 | ctry_usd | ctry_dev_lcl")]
StartOpt = Annotated[str | None, typer.Option(help="YYYY-MM-DD")]
CostOpt = Annotated[float, typer.Option(help="Cost per unit one-way turnover, bps")]
QuantOpt = Annotated[int, typer.Option(min=2, max=10)]


@research_app.command("list")
def research_list(universe: UniverseOpt = "ind49") -> None:
    """Available signals, families and universes."""
    uni = get_universe(universe)
    for s in uni.signals.values():
        typer.echo(f"{s.name:<16} {s.family:<20} {s.description}")
        typer.echo(f"{'':<16} {'':<20} ref: {s.reference}")
    typer.echo(f"\nfamilies: {', '.join(uni.families)}   universes: {', '.join(UNIVERSES)}")


@research_app.command("signal")
def research_signal(
    name: str,
    universe: UniverseOpt = "ind49",
    start: StartOpt = None,
    cost_bps: CostOpt = 10.0,
    quantiles: QuantOpt = 5,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Tearsheet for one signal: IC, decay, quantile spread, net Sharpe, PSR/DSR, regimes."""
    ctx = _context(universe, start, cost_bps, quantiles)
    if name not in ctx.signals:
        raise typer.BadParameter(f"unknown signal {name!r}; see `atlas research list`")
    rep = run_signal(ctx.signals[name], ctx)
    summary = rep.summary()
    if as_json:
        extra = {
            "ic_decay": rep.ic_decay,
            "quantile_mean_annual": rep.quantile_mean_annual,
            "regime_ic": rep.regime_ic,
        }
        typer.echo(jsonsafe.dumps({**summary, **extra}))
        return
    for k, v in summary.items():
        typer.echo(f"{k:<22} {_fmt(v)}")
    decay = "  ".join(f"{k}m:{v:+.3f}" for k, v in rep.ic_decay.items())
    quint = "  ".join(f"Q{k}:{v:+.2%}" for k, v in rep.quantile_mean_annual.items())
    regimes = "  ".join(f"{k}:{v:+.3f}" for k, v in sorted(rep.regime_ic.items()))
    typer.echo(f"{'ic_decay':<22} {decay}")
    typer.echo(f"{'quantile_ann_return':<22} {quint}")
    typer.echo(f"{'ic_by_market_regime':<22} {regimes or 'n/a'}")


@research_app.command("family")
def research_family(
    family: str,
    universe: UniverseOpt = "ind49",
    start: StartOpt = None,
    cost_bps: CostOpt = 10.0,
    quantiles: QuantOpt = 5,
    splits: Annotated[int, typer.Option(help="CSCV blocks (even)")] = 16,
) -> None:
    """Backtest every configuration in a family, then estimate PBO across them."""
    ctx = _context(universe, start, cost_bps, quantiles)
    if family not in ctx.families:
        raise typer.BadParameter(f"unknown family {family!r}: {', '.join(ctx.families)}")
    reports = [run_signal(s, ctx) for s in ctx.families[family]]
    pbo, months = family_pbo(reports, n_splits=splits)
    typer.echo(f"{'config':<16} {'IC':>7} {'t(NW)':>7} {'Sharpe':>7} {'DSR':>6}")
    for rep in sorted(reports, key=lambda r: -r.sharpe_net):
        typer.echo(
            f"{rep.name:<16} {rep.ic_mean:>7.3f} {rep.ic_tstat:>7.2f} "
            f"{rep.sharpe_net:>7.2f} {rep.dsr:>6.2f}"
        )
    typer.echo(f"\nPBO (CSCV, {splits} blocks, {months} common months): {pbo:.2f}")


@research_app.command("gates")
def research_gates(
    universe: UniverseOpt = "ind49",
    start: StartOpt = None,
    cost_bps: CostOpt = 10.0,
    quantiles: QuantOpt = 5,
    splits: Annotated[int, typer.Option(help="CSCV blocks (even)")] = 16,
    out: Annotated[str | None, typer.Option(help="Write full results as JSON")] = None,
) -> None:
    """Run every family, then check each signal against the PRD promotion gates."""
    ctx = _context(universe, start, cost_bps, quantiles)
    suite = run_gate_suite(ctx, n_splits=splits)
    results = list(suite.gates.values())
    reports = suite.reports
    ens_icir = suite.ensemble_icir

    typer.echo(f"{'signal':<16} {'t':>6} {'DSR':>5} {'PBO':>5} {'reg':>4} {'marg':>6}  result")
    for g in results:
        pbo_s = "n/a" if g.pbo is None else f"{g.pbo:.2f}"
        marg_s = "n/a" if g.marginal_icir is None else f"{g.marginal_icir:+.2f}"
        verdict = "PASS" if g.passed else "fail: " + ", ".join(g.failed)
        typer.echo(
            f"{g.signal:<16} {g.ic_tstat:>6.2f} {g.dsr:>5.2f} {pbo_s:>5} "
            f"{g.regimes_positive}/{g.regimes_total:<2} {marg_s:>6}  {verdict}"
        )
    typer.echo(
        f"\nensemble ICIR {ens_icir:.2f}; trials logged on {universe}: "
        f"{ctx.log.stats(universe=universe).n_trials}"
    )
    if out:
        payload = [
            {
                **asdict(g),
                "passed": g.passed,
                "summary": reports[g.signal].summary(),
                "regime_ic": reports[g.signal].regime_ic,
            }
            for g in results
        ]
        Path(out).write_text(jsonsafe.dumps({"ensemble_icir": ens_icir, "gates": payload}))
        typer.echo(f"wrote {out}")


@research_app.command("cards")
def research_cards(
    universe: UniverseOpt = "ind49",
    cost_bps: CostOpt = 10.0,
    quantiles: QuantOpt = 5,
    out: Annotated[str, typer.Option(help="Directory for the Markdown cards")] = "docs/signals",
) -> None:
    """Regenerate one signal card per signal (PRD §14) from a full gate-suite run."""
    ctx = _context(universe, None, cost_bps, quantiles)
    suite = run_gate_suite(ctx)
    paths = write_cards(
        suite, Path(out) / universe, generated=datetime.now(UTC).date(), cost_bps=cost_bps
    )
    typer.echo(f"wrote {len(paths)} files to {Path(out) / universe}")


# ---------------------------------------------------------------------------- views
HorizonOpt = Annotated[int, typer.Option(min=1, max=12, help="Forecast horizon, months")]


GateTOpt = Annotated[
    float | None,
    typer.Option(help="Walk-forward gate: components need a past IC t-stat >= this (e.g. 2)"),
]
GateRegOpt = Annotated[
    int | None,
    typer.Option(
        help="Walk-forward gate: positive past IC in >= this many market regimes (e.g. 3)"
    ),
]


def _ensemble_config(
    universe: str, horizon: int, gate_tstat: float | None, gate_regimes: int | None
) -> EnsembleConfig:
    gated = gate_tstat is not None or gate_regimes is not None
    return EnsembleConfig(
        horizon=horizon,
        components=get_universe(universe).ensemble_components,
        gate_tstat=gate_tstat,
        gate_regimes=gate_regimes,
        version="ens_v1_gated" if gated else "ens_v1",
    )


@views_app.command("build")
def views_build(
    universe: UniverseOpt = "ind12",
    horizon: HorizonOpt = 3,
    gate_tstat: GateTOpt = None,
    gate_regimes: GateRegOpt = None,
    as_of: Annotated[str | None, typer.Option("--as-of", help="YYYY-MM-01; default latest")] = None,
    do_publish: Annotated[
        bool, typer.Option("--publish", help="Freeze to data/gold/views")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Current views: stance, calibrated probability, expected excess return and its range."""
    ctx = _context(universe, None, 10.0, 3)
    run = ensemble_walk_forward(ctx, _ensemble_config(universe, horizon, gate_tstat, gate_regimes))
    views = build_views(
        run,
        universe,
        _parse_date(as_of) if as_of else None,
        code_sha=git_sha(),
        n_trials=ctx.log.stats(universe=universe).n_trials,
    )
    if as_json:
        typer.echo(jsonsafe.dumps([v.model_dump(mode="json") for v in views], indent=2))
    else:
        typer.echo(f"Views as of {views[0].as_of:%Y-%m} | horizon {views[0].horizon} | {universe}")
        typer.echo(
            f"{'asset':<28} {'stance':<20} {'score':>6} {'P(beat)':>8} {'E[excess]':>10} "
            f"{'p10':>8} {'p90':>8} {'agree':>6}  conf"
        )
        for v in views:
            e = v.expected_excess_return
            p10 = "n/a" if e.p10 is None else f"{e.p10:+.1%}"
            p90 = "n/a" if e.p90 is None else f"{e.p90:+.1%}"
            prob = "n/a" if v.prob_outperform is None else f"{v.prob_outperform:.2f}"
            agree = "n/a" if v.agreement is None else f"{v.agreement:.2f}"
            typer.echo(
                f"{v.asset_name:<28} {v.stance.value:<20} {v.score:>+6.2f} {prob:>8} "
                f"{e.mean:>+10.2%} {p10:>8} {p90:>8} {agree:>6}  {v.confidence.value}"
            )
    if do_publish:
        path = publish(views, get_settings().data_dir, universe)
        typer.echo(f"published {path}")


@views_app.command("evaluate")
def views_evaluate(
    universe: UniverseOpt = "ind12",
    horizon: HorizonOpt = 3,
    gate_tstat: GateTOpt = None,
    gate_regimes: GateRegOpt = None,
    start: StartOpt = "1970-01-01",
) -> None:
    """Walk-forward track record of the view model: IC, calibration, coverage, hit rate."""
    ctx = _context(universe, None, 10.0, 3)
    run = ensemble_walk_forward(ctx, _ensemble_config(universe, horizon, gate_tstat, gate_regimes))
    ev = evaluate(run, _parse_date(start) if start else None)
    for k, v in ev.summary().items():
        typer.echo(f"{k:<22} {_fmt(v)}")
    typer.echo("\nreliability (predicted vs realized share beating the median):")
    typer.echo(ev.reliability)


# ---------------------------------------------------------------------------- portfolio
@portfolio_app.command("backtest")
def portfolio_backtest(
    universe: UniverseOpt = "ind49",
    horizon: HorizonOpt = 3,
    gate_tstat: GateTOpt = None,
    gate_regimes: GateRegOpt = None,
    start: StartOpt = "1970-01-01",
    te: Annotated[float, typer.Option(help="Tracking-error budget, annual")] = 0.04,
    max_active: Annotated[float, typer.Option(help="Max active weight per asset")] = 0.05,
    cost_bps: CostOpt = 10.0,
    out: Annotated[str | None, typer.Option(help="Write monthly results to this CSV")] = None,
) -> None:
    """Walk-forward model portfolio: views -> constrained tilts -> net active return."""
    ctx = _context(universe, None, cost_bps, 3)
    run = ensemble_walk_forward(ctx, _ensemble_config(universe, horizon, gate_tstat, gate_regimes))
    fc = run.forecasts
    if start:
        fc = fc.filter(pl.col("date") >= _parse_date(start))
    cfg = PortfolioConfig(te_annual=te, max_active=max_active, cost_bps=cost_bps)
    bt = backtest_portfolio(fc, ctx.panel.select("date", "asset", "ret"), horizon=horizon, cfg=cfg)
    for k, v in bt.summary().items():
        typer.echo(f"{k:<28} {_fmt(v)}")
    if out:
        bt.monthly.write_csv(out)
        typer.echo(f"wrote {out}")


@portfolio_app.command("now")
def portfolio_now(
    universe: UniverseOpt = "ind12",
    horizon: HorizonOpt = 3,
    gate_tstat: GateTOpt = None,
    gate_regimes: GateRegOpt = None,
    te: Annotated[float, typer.Option(help="Tracking-error budget, annual")] = 0.04,
    max_active: Annotated[float, typer.Option(help="Max active weight per asset")] = 0.05,
) -> None:
    """Latest model-portfolio weights against the equal-weight benchmark."""
    ctx = _context(universe, None, 10.0, 3)
    run = ensemble_walk_forward(ctx, _ensemble_config(universe, horizon, gate_tstat, gate_regimes))
    latest = run.forecasts.drop_nulls("exp_excess")["date"].max()
    # include the latest month: backtest needs a next-month return, so append a zero row
    panel = ctx.panel.select("date", "asset", "ret")
    filler = panel.filter(pl.col("date") == latest).with_columns(
        pl.col("date").dt.offset_by("1mo"), pl.lit(0.0).alias("ret")
    )
    fc = run.forecasts.filter(pl.col("date") >= pl.lit(latest).dt.offset_by("-24mo"))
    bt = backtest_portfolio(
        fc,
        pl.concat([panel, filler]),
        horizon=horizon,
        cfg=PortfolioConfig(te_annual=te, max_active=max_active),
    )
    w = (
        bt.weights.filter(pl.col("date") == latest)
        .with_columns((pl.col("weight") - pl.col("bench")).alias("active"))
        .sort("active", descending=True)
    )
    ex_ante = bt.monthly.filter(pl.col("date") == latest)["ex_ante_te"].item()
    typer.echo(f"Model portfolio as of {latest!s:.7} | {universe} | ex-ante TE {ex_ante:.1%}")
    for r in w.iter_rows(named=True):
        typer.echo(
            f"{r['asset']:<8} weight {r['weight']:6.1%}  bench {r['bench']:6.1%}  "
            f"active {r['active']:+6.1%}"
        )


# ---------------------------------------------------------------------------- assistant
@app.command("ask")
def ask(
    question: str,
    model: Annotated[str | None, typer.Option(help="Claude model id")] = None,
    effort: Annotated[str | None, typer.Option(help="low | medium | high | xhigh | max")] = None,
    show_tools: Annotated[
        bool, typer.Option("--show-tools", help="List the tool calls made")
    ] = False,
) -> None:
    """Ask the research assistant a question; every number is checked against tool results."""
    import anthropic

    from atlas.agent.assistant import DISCLAIMER, Assistant
    from atlas.agent.tools import ToolKit

    s = get_settings()
    try:
        key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else ""
        # an empty .env entry means "not set": let the SDK fall back to its own resolution
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        assistant = Assistant(
            client,
            ToolKit(s.data_dir, s.config_dir),
            model=model or s.assistant_model,
            effort=effort or s.assistant_effort,
        )
        answer = assistant.ask(question)
    except anthropic.AuthenticationError:
        raise typer.BadParameter(
            "Anthropic credentials missing or invalid: "
            "set ANTHROPIC_API_KEY or run `ant auth login`"
        ) from None
    except anthropic.APIConnectionError:
        typer.echo("Could not reach the Anthropic API; check the network and retry.")
        raise typer.Exit(code=1) from None
    typer.echo(answer.text)
    if answer.grounding.unsupported:
        nums = ", ".join(c.text for c in answer.grounding.unsupported)
        typer.echo(f"\n[grounding] not traceable to tool results: {nums}")
    if show_tools:
        for c in answer.tool_calls:
            typer.echo(f"[tool] {c.name} {c.arguments}{' (error)' if c.is_error else ''}")
    typer.echo(f"\n{DISCLAIMER}")


# ---------------------------------------------------------------------------- pipeline
@pipeline_app.command("daily")
def pipeline_daily(
    skip_ingest: Annotated[bool, typer.Option("--skip-ingest", help="Reuse stored data")] = False,
) -> None:
    """Ingest every source, check staleness, snapshot, freeze views, score the live track."""
    from atlas.pipeline import run_daily

    report = asyncio.run(run_daily(get_settings(), ingest=not skip_ingest))
    for st in report.steps:
        typer.echo(f"[{'ok ' if st.ok else 'ERR'}] {st.name:<24} {st.seconds:>6.1f}s  {st.detail}")
    if report.stale:
        typer.echo(
            f"stale: {', '.join(report.stale[:12])}{' ...' if len(report.stale) > 12 else ''}"
        )
    raise typer.Exit(code=0 if report.ok else 1)


@pipeline_app.command("track")
def pipeline_track() -> None:
    """Live paper track record: frozen views scored once their horizon has elapsed."""
    from atlas.views.track import score_all

    s = get_settings()
    store = _store()
    universes = {p.name for p in (s.data_dir / "gold" / "views").glob("*") if p.is_dir()}
    returns = {
        u: get_universe(u).build_panel(store).select("date", "asset", "ret")
        for u in sorted(universes)
        if u in UNIVERSES
    }
    rec = score_all(s.data_dir, returns)
    for k, v in rec.summary().items():
        typer.echo(f"{k:<20} {_fmt(v)}")
    for m in rec.scored:
        hit = "n/a" if m.hit_rate is None else f"{m.hit_rate:.0%}"
        typer.echo(
            f"  {m.universe:<13} {m.horizon} {m.as_of:%Y-%m}  IC {m.rank_ic:+.3f}  "
            f"calls {m.n_calls} hit {hit}  coverage {m.coverage:.0%}"
        )
    for u, h, d in rec.pending:
        typer.echo(f"  {u:<13} {h} {d:%Y-%m}  pending (outcome not yet realized)")


# ---------------------------------------------------------------------------- report
@report_app.command("weekly")
def report_weekly(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Gather facts only; no Claude calls")
    ] = False,
    model: Annotated[str | None, typer.Option(help="Claude model id")] = None,
) -> None:
    """Weekly research note: engine facts -> Claude writer -> red team -> grounding check."""
    from atlas.agent.report import Report, ReportWriter, gather_facts, save
    from atlas.agent.tools import ToolKit

    s = get_settings()
    reports_dir = s.data_dir / "reports"
    today = datetime.now(UTC).date()
    facts = gather_facts(ToolKit(s.data_dir, s.config_dir), today, reports_dir)
    if dry_run:
        path = save(Report(as_of=today, facts=facts), reports_dir)
        typer.echo(f"facts gathered (dry run): {path.parent / 'facts.json'}")
        return

    import anthropic

    key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else ""
    try:
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        report = ReportWriter(client, model=model or s.assistant_model).write(facts, today)
    except anthropic.AuthenticationError:
        raise typer.BadParameter(
            "Anthropic credentials missing or invalid: "
            "set ANTHROPIC_API_KEY or run `ant auth login`"
        ) from None
    path = save(report, reports_dir)
    typer.echo(f"wrote {path}")
    if report.unsupported:
        typer.echo(f"[grounding] not traceable to engine facts: {', '.join(report.unsupported)}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
