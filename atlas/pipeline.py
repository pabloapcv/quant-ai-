"""Daily pipeline (PRD §9): ingest -> text -> quality -> snapshot -> views -> track -> alerts.

Each step runs independently: a failing source is recorded and the run continues, so one
agency's outage never stops the rest. Every run writes ``data/runs/<timestamp>.json``.

Run it every day: sources that only publish their current values (Ken French, ECB, OECD,
BIS, IMF) get true point-in-time history only from the days Atlas records them, and the
live track record exists only for views frozen before their outcome.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from atlas import jsonsafe
from atlas.agent.usage import BudgetExceeded, make_client
from atlas.alerts import run_alerts
from atlas.catalog import Catalog
from atlas.connectors.base import RawStore
from atlas.connectors.fred import FredConnector
from atlas.connectors.kenfrench import KenFrenchConnector
from atlas.connectors.sdmx import AGENCIES, SdmxConnector
from atlas.ingest import IngestReport, ingest_series
from atlas.models.regime.engine import RegimeEngine, data_month_for
from atlas.pit.snapshot import create_snapshot, git_sha
from atlas.pit.store import PITStore
from atlas.quality.contracts import find_stale
from atlas.quality.drift import backtest_ic_stats, feature_psi
from atlas.settings import Settings
from atlas.text.ecb import EcbTextConnector
from atlas.text.fed import FedTextConnector
from atlas.text.tone import run_tone
from atlas.views.publish import ViewExistsError, publish, views_dir
from atlas.views.reproduce import compute_views
from atlas.views.track import TrackRecord, score_all

DEFAULT_VIEWS: tuple[tuple[str, int], ...] = (
    ("ind12", 3),
    ("ind49", 3),
    ("ctry_dev_lcl", 3),
    ("ctry_usd", 3),
    ("fx_g10", 3),
    ("rates_dev", 3),
)


@dataclass
class Step:
    name: str
    ok: bool
    detail: str
    seconds: float


@dataclass
class PipelineReport:
    started_at: datetime
    steps: list[Step] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    snapshot_id: str | None = None
    track: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    # "universe:horizon" -> {"backtest": {mean, sd, n}, "live_ics": [...] (oldest first)}
    ic_monitor: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_psi: dict[str, float] = field(default_factory=dict)  # diagnostic, top inputs

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    def to_json(self) -> str:
        return jsonsafe.dumps({**asdict(self), "ok": self.ok}, indent=2)


INCREMENTAL_BUFFER = timedelta(days=3)
RETRY_DELAY_S = 120.0  # pause before re-trying failed series once
BACKTEST_IC_START = date(1970, 1, 1)  # same window as the reported walk-forward track record


def incremental_since(store: PITStore, series_ids: Sequence[str]) -> dict[str, date]:
    """Per series, the date to resume vintage fetching from (last release minus a buffer).

    Series with no stored history are left out, so they get a full fetch.
    """
    last = store.last_knowledge_times()
    return {
        sid: (t.date() - INCREMENTAL_BUFFER)
        for sid in series_ids
        if (t := last.get(sid)) is not None
    }


def _summarize(reports: list[IngestReport]) -> tuple[bool, str]:
    failed = [r for r in reports if not r.ok]
    new = sum(r.rows_new for r in reports)
    detail = f"{len(reports) - len(failed)}/{len(reports)} series ok, {new} new rows"
    if failed:
        detail += "; failed: " + ", ".join(r.series_id for r in failed[:8])
    return not failed, detail


async def _timed(
    report: PipelineReport, name: str, fn: Callable[[], Awaitable[tuple[bool, str]]]
) -> None:
    t0 = time.monotonic()
    try:
        ok, detail = await fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    report.steps.append(Step(name, ok, detail, round(time.monotonic() - t0, 1)))


async def run_daily(
    settings: Settings,
    *,
    ingest: bool = True,
    views: Sequence[tuple[str, int]] = DEFAULT_VIEWS,
    now: datetime | None = None,
) -> PipelineReport:
    started = now or datetime.now(UTC)
    report = PipelineReport(started_at=started)
    data_dir, config_dir = settings.data_dir, settings.config_dir
    cat = Catalog.load(config_dir / "series.yaml")
    raw = RawStore(data_dir)

    failed: dict[str, list[str]] = {}  # ingest step -> series that failed in it

    def summarize(step: str, reports: list[IngestReport]) -> tuple[bool, str]:
        bad = [r.series_id for r in reports if not r.ok]
        if bad:
            failed[step] = bad
        return _summarize(reports)

    def fred_key() -> str:
        return settings.fred_api_key.get_secret_value() if settings.fred_api_key else ""

    def connector(source: str) -> Any:
        if source == "fred":
            since = incremental_since(PITStore(data_dir), [x.id for x in cat.by_source("fred")])
            return FredConnector(
                fred_key(),
                raw_store=raw,
                max_calls_per_minute=settings.fred_max_calls_per_minute,
                since=since,
            )
        if source == "kenfrench":
            return KenFrenchConnector(raw_store=raw)
        return SdmxConnector(source, raw_store=raw)

    if ingest:

        async def fred() -> tuple[bool, str]:
            key = settings.fred_api_key.get_secret_value() if settings.fred_api_key else ""
            if not key:
                return False, "skipped: no FRED key (ATLAS_FRED_API_KEY)"
            async with connector("fred") as conn:
                return summarize(
                    "ingest:fred",
                    await ingest_series(cat.by_source("fred"), conn, PITStore(data_dir)),
                )

        async def kenfrench() -> tuple[bool, str]:
            async with connector("kenfrench") as conn:
                reports = await ingest_series(
                    cat.by_source("kenfrench"), conn, PITStore(data_dir), concurrency=1
                )
            return summarize("ingest:kenfrench", reports)

        async def fed_text() -> tuple[bool, str]:
            async with FedTextConnector(data_dir) as conn:
                out = await conn.sync()
            detail = f"{out['new']} new documents"
            if out["failed"]:
                detail += f"; failed: {', '.join(out['failed'][:3])}"
            return not out["failed"], detail

        async def ecb_text() -> tuple[bool, str]:
            year = started.year
            async with EcbTextConnector(data_dir) as conn:  # recent years: a few requests
                out = await conn.sync(years=[year, year - 1])
            detail = f"{out['new']} new statements"
            if out["failed"]:
                detail += f"; failed: {', '.join(out['failed'][:3])}"
            return not out["failed"], detail

        await _timed(report, "ingest:fred", fred)
        await _timed(report, "ingest:fed_text", fed_text)
        await _timed(report, "ingest:ecb_text", ecb_text)
        await _timed(report, "ingest:kenfrench", kenfrench)
        for agency in sorted(AGENCIES):

            async def sdmx(agency: str = agency) -> tuple[bool, str]:
                async with connector(agency) as conn:
                    reports = await ingest_series(
                        cat.by_source(agency), conn, PITStore(data_dir), concurrency=1
                    )
                return summarize(f"ingest:{agency}", reports)

            await _timed(report, f"ingest:{agency}", sdmx)

        async def retry() -> tuple[bool, str]:
            """One more try for series that failed (sources have transient 5xx errors)."""
            if not failed:
                return True, "nothing to retry"
            await asyncio.sleep(RETRY_DELAY_S)
            recovered: list[str] = []
            still: list[str] = []
            for step_name, ids in failed.items():
                specs = [cat.get(i) for i in ids]
                async with connector(specs[0].source) as conn:
                    reps = await ingest_series(specs, conn, PITStore(data_dir), concurrency=1)
                ok_ids = [r.series_id for r in reps if r.ok]
                bad = [r.series_id for r in reps if not r.ok]
                recovered += ok_ids
                still += bad
                step = next(st for st in report.steps if st.name == step_name)
                if not bad:
                    step.ok = True
                    step.detail += f"; recovered on retry: {', '.join(ok_ids)}"
            detail = f"{len(recovered)} recovered"
            if still:
                detail += f"; still failing: {', '.join(still[:8])}"
            return not still, detail

        await _timed(report, "ingest:retry", retry)

    store = PITStore(data_dir)

    if ingest:

        async def tone() -> tuple[bool, str]:
            try:
                out = run_tone(data_dir, make_client(settings, "extract"))
            except BudgetExceeded as exc:
                return False, str(exc)
            except Exception as exc:  # no Anthropic credentials: skip, data still flows
                return True, f"skipped: {type(exc).__name__}"
            return out["failed"] == 0, f"{out['extracted']} statements read"

        await _timed(report, "text:tone", tone)

    async def quality() -> tuple[bool, str]:
        stale = find_stale(
            store.last_knowledge_times(),
            [(s.id, s.frequency, s.publication_lag_days) for s in cat],
            started,
        )
        report.stale = sorted(s.series_id for s in stale)
        # staleness is a warning, not a failure: monthly data legitimately pauses
        return True, f"{len(report.stale)} stale of {len(cat)} series"

    await _timed(report, "quality:staleness", quality)

    async def snapshot() -> tuple[bool, str]:
        snap = create_snapshot(data_dir, config_dir, now=started)
        report.snapshot_id = snap.snapshot_id
        return True, f"{snap.snapshot_id}: {len(snap.files)} files"

    await _timed(report, "snapshot", snapshot)

    returns: dict[str, Any] = {}
    for universe, horizon in views:

        async def freeze(universe: str = universe, horizon: int = horizon) -> tuple[bool, str]:
            run, vs, panel = compute_views(
                store,
                data_dir,
                universe,
                horizon,
                code_sha=git_sha(),
                snapshot_id=report.snapshot_id,
            )
            returns[universe] = panel.select("date", "asset", "ret")
            report.ic_monitor[f"{universe}:{horizon}M"] = {
                "backtest": backtest_ic_stats(run.forecasts, BACKTEST_IC_START),
                "live_ics": [],
            }
            target = (
                views_dir(data_dir, universe, vs[0].horizon) / f"{vs[0].as_of.isoformat()}.json"
            )
            existed = target.exists()
            try:
                publish(vs, data_dir, universe)
            except ViewExistsError:
                return (
                    True,
                    f"{vs[0].as_of:%Y-%m} frozen earlier; today's recompute differs (kept first)",
                )
            if existed:
                return True, f"{vs[0].as_of:%Y-%m} frozen earlier; unchanged"
            return True, f"{vs[0].as_of:%Y-%m} frozen -> {target.relative_to(data_dir)}"

        await _timed(report, f"views:{universe}:{horizon}M", freeze)

    async def track() -> tuple[bool, str]:
        rec: TrackRecord = score_all(data_dir, returns)
        report.track = rec.summary()
        for m in sorted(rec.scored, key=lambda m: m.as_of):
            mon = report.ic_monitor.setdefault(f"{m.universe}:{m.horizon}", {"live_ics": []})
            mon.setdefault("live_ics", []).append(m.rank_ic)
        return (
            True,
            f"{report.track['months_scored']} scored, {report.track['months_pending']} pending",
        )

    await _timed(report, "track:score", track)

    async def drift() -> tuple[bool, str]:
        cube = store.load_cube()
        if "US.IP" not in cube:
            return True, "skipped: no macro data"
        _, matrix = RegimeEngine(cube).composites(started)
        values = feature_psi(matrix, data_month_for(started))
        report.feature_psi = dict(sorted(values.items(), key=lambda kv: -kv[1])[:5])
        top = ", ".join(f"{k} {v:.2f}" for k, v in report.feature_psi.items())
        return True, f"PSI (36m vs prior 10y, diagnostic): {top}"

    await _timed(report, "quality:drift", drift)

    async def alerts() -> tuple[bool, str]:
        failed = [st.name for st in report.steps if not st.ok]
        fired = run_alerts(
            data_dir, config_dir, failed_steps=failed, now=started, ic_monitor=report.ic_monitor
        )
        report.alerts = [a.model_dump() for a in fired]
        sent = sum(a.status == "sent" for a in fired)
        return True, f"{len(fired)} triggered, {sent} sent"

    await _timed(report, "alerts", alerts)

    out = data_dir / "runs" / f"{started:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json())
    return report


def latest_run(data_dir: Path) -> Path | None:
    runs = sorted((data_dir / "runs").glob("*.json"))
    return runs[-1] if runs else None
