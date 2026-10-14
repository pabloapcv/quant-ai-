"""``atlas`` command-line interface."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from atlas.backtest.gates import evaluate_gates
from atlas.backtest.trials import TrialLog
from atlas.catalog import Catalog
from atlas.connectors.base import RawStore
from atlas.connectors.fred import FredConnector
from atlas.connectors.kenfrench import KenFrenchConnector
from atlas.ingest import IngestReport, ingest_series
from atlas.log import configure_logging
from atlas.models.regime.engine import RegimeEngine, RegimeReport
from atlas.models.regime.evaluate import walk_forward
from atlas.models.regime.market import market_regimes
from atlas.pit.snapshot import create_snapshot, load_snapshot, verify_snapshot
from atlas.pit.store import PITStore
from atlas.quality.contracts import find_stale
from atlas.settings import get_settings
from atlas.signals.industry import FAMILIES, SIGNALS, UNIVERSES, research_panel
from atlas.signals.research import (
    ResearchContext,
    ensemble_marginal_icir,
    family_pbo,
    run_signal,
    signal_panel,
)

app = typer.Typer(no_args_is_help=True, help="Atlas quant research assistant.")
ingest_app = typer.Typer(no_args_is_help=True, help="Pull data from sources into the PIT store.")
pit_app = typer.Typer(no_args_is_help=True, help="Query the point-in-time store.")
snapshot_app = typer.Typer(no_args_is_help=True, help="Create and verify reproducible snapshots.")
catalog_app = typer.Typer(no_args_is_help=True, help="Inspect the series catalog.")
regime_app = typer.Typer(no_args_is_help=True, help="Macro regime engine (Phase 2).")
research_app = typer.Typer(no_args_is_help=True, help="Signal research & backtests (Phase 3).")
app.add_typer(ingest_app, name="ingest")
app.add_typer(pit_app, name="pit")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(catalog_app, name="catalog")
app.add_typer(regime_app, name="regime")
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
        s.series_id for s in find_stale(last, [(s.id, s.frequency) for s in cat], datetime.now(UTC))
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
    store = _store()
    try:
        regimes: pl.DataFrame | None = market_regimes(store).select("date", "regime")
    except ValueError:
        regimes = None  # market factor not ingested: regime-conditional IC is skipped
    return ResearchContext(
        panel=research_panel(store, universe),
        universe=universe,
        log=TrialLog(get_settings().data_dir),
        regimes=regimes,
        n_quantiles=quantiles,
        cost_bps=cost_bps,
        start=_parse_date(start) if start else None,
    )


UniverseOpt = Annotated[str, typer.Option(help="ind12 | ind49")]
StartOpt = Annotated[str | None, typer.Option(help="YYYY-MM-DD")]
CostOpt = Annotated[float, typer.Option(help="Cost per unit one-way turnover, bps")]
QuantOpt = Annotated[int, typer.Option(min=2, max=10)]


@research_app.command("list")
def research_list() -> None:
    """Available signals, families and universes."""
    for s in SIGNALS.values():
        typer.echo(f"{s.name:<16} {s.family:<20} {s.description}")
        typer.echo(f"{'':<16} {'':<20} ref: {s.reference}")
    typer.echo(f"\nfamilies: {', '.join(FAMILIES)}   universes: {', '.join(UNIVERSES)}")


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
    if name not in SIGNALS:
        raise typer.BadParameter(f"unknown signal {name!r}; see `atlas research list`")
    rep = run_signal(SIGNALS[name], _context(universe, start, cost_bps, quantiles))
    summary = rep.summary()
    if as_json:
        extra = {
            "ic_decay": rep.ic_decay,
            "quantile_mean_annual": rep.quantile_mean_annual,
            "regime_ic": rep.regime_ic,
        }
        typer.echo(json.dumps({**summary, **extra}, default=str))
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
    if family not in FAMILIES:
        raise typer.BadParameter(f"unknown family {family!r}: {', '.join(FAMILIES)}")
    ctx = _context(universe, start, cost_bps, quantiles)
    reports = [run_signal(s, ctx) for s in FAMILIES[family]]
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
    pbos = {
        f: family_pbo([run_signal(s, ctx) for s in members], n_splits=splits)[0]
        for f, members in FAMILIES.items()
    }
    reports = {n: run_signal(s, ctx) for n, s in SIGNALS.items()}
    # Overlays share their base signal's ranking, so only distinct rankings enter the ensemble.
    rankings = {n: s for n, s in SIGNALS.items() if s.ranking_of is None}
    ens_icir, marginal = ensemble_marginal_icir(
        {n: signal_panel(s, ctx) for n, s in rankings.items()}
    )
    results = []
    for n, s in SIGNALS.items():
        base = s.ranking_of or n
        results.append(
            evaluate_gates(s, reports[n], pbo=pbos.get(s.family), marginal_icir=marginal[base])
        )

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
        Path(out).write_text(json.dumps({"ensemble_icir": ens_icir, "gates": payload}, default=str))
        typer.echo(f"wrote {out}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
