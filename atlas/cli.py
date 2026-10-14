"""``atlas`` command-line interface."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Annotated

import polars as pl
import typer

from atlas.catalog import Catalog
from atlas.connectors.base import RawStore
from atlas.connectors.fred import FredConnector
from atlas.ingest import ingest_series
from atlas.log import configure_logging
from atlas.pit.snapshot import create_snapshot, load_snapshot, verify_snapshot
from atlas.pit.store import PITStore
from atlas.quality.contracts import find_stale
from atlas.settings import get_settings

app = typer.Typer(no_args_is_help=True, help="Atlas quant research assistant.")
ingest_app = typer.Typer(no_args_is_help=True, help="Pull data from sources into the PIT store.")
pit_app = typer.Typer(no_args_is_help=True, help="Query the point-in-time store.")
snapshot_app = typer.Typer(no_args_is_help=True, help="Create and verify reproducible snapshots.")
catalog_app = typer.Typer(no_args_is_help=True, help="Inspect the series catalog.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(pit_app, name="pit")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(catalog_app, name="catalog")


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
        for r in reports:
            status = "ok " if r.ok else "ERR"
            detail = (
                r.error
                or f"fetched={r.rows_fetched} new={r.rows_new} restated={r.rows_conflicting}"
            )
            typer.echo(f"[{status}] {r.series_id:<24} {detail}")
        return sum(not r.ok for r in reports)

    failures = asyncio.run(run())
    raise typer.Exit(code=1 if failures else 0)


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


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
