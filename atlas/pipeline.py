"""Daily pipeline (PRD §9): ingest -> quality -> snapshot -> views -> track -> alerts.

Each step runs independently: a failing source is recorded and the run continues, so one
agency's outage never stops the rest. Every run writes ``data/runs/<timestamp>.json``.

Run it every day: sources that only publish their current values (Ken French, ECB, OECD,
BIS, IMF) get true point-in-time history only from the days Atlas records them, and the
live track record exists only for views frozen before their outcome.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from atlas import jsonsafe
from atlas.alerts import run_alerts
from atlas.backtest.trials import TrialLog
from atlas.catalog import Catalog
from atlas.connectors.base import RawStore
from atlas.connectors.fred import FredConnector
from atlas.connectors.kenfrench import KenFrenchConnector
from atlas.connectors.sdmx import AGENCIES, SdmxConnector
from atlas.ingest import IngestReport, ingest_series
from atlas.models.expected_returns.ensemble import EnsembleConfig, walk_forward
from atlas.pit.snapshot import create_snapshot, git_sha
from atlas.pit.store import PITStore
from atlas.quality.contracts import find_stale
from atlas.settings import Settings
from atlas.signals.research import ResearchContext
from atlas.signals.universes import get_universe
from atlas.views.publish import ViewExistsError, build_views, publish, views_dir
from atlas.views.track import TrackRecord, score_all

DEFAULT_VIEWS: tuple[tuple[str, int], ...] = (
    ("ind12", 3),
    ("ind49", 3),
    ("ctry_dev_lcl", 3),
    ("ctry_usd", 3),
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

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    def to_json(self) -> str:
        return jsonsafe.dumps({**asdict(self), "ok": self.ok}, indent=2)


INCREMENTAL_BUFFER = timedelta(days=3)


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

    if ingest:

        async def fred() -> tuple[bool, str]:
            key = settings.fred_api_key.get_secret_value() if settings.fred_api_key else ""
            if not key:
                return False, "skipped: no FRED key (ATLAS_FRED_API_KEY)"
            since = incremental_since(PITStore(data_dir), [x.id for x in cat.by_source("fred")])
            async with FredConnector(
                key,
                raw_store=raw,
                max_calls_per_minute=settings.fred_max_calls_per_minute,
                since=since,
            ) as conn:
                return _summarize(
                    await ingest_series(cat.by_source("fred"), conn, PITStore(data_dir))
                )

        async def kenfrench() -> tuple[bool, str]:
            async with KenFrenchConnector(raw_store=raw) as conn:
                reports = await ingest_series(
                    cat.by_source("kenfrench"), conn, PITStore(data_dir), concurrency=1
                )
            return _summarize(reports)

        await _timed(report, "ingest:fred", fred)
        await _timed(report, "ingest:kenfrench", kenfrench)
        for agency in sorted(AGENCIES):

            async def sdmx(agency: str = agency) -> tuple[bool, str]:
                async with SdmxConnector(agency, raw_store=raw) as conn:
                    reports = await ingest_series(
                        cat.by_source(agency), conn, PITStore(data_dir), concurrency=1
                    )
                return _summarize(reports)

            await _timed(report, f"ingest:{agency}", sdmx)

    store = PITStore(data_dir)

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
            uni = get_universe(universe)
            ctx = ResearchContext(
                panel=uni.build_panel(store),
                universe=universe,
                log=TrialLog(data_dir),
                n_quantiles=3,
                signals=uni.signals,
                families=uni.families,
            )
            returns[universe] = ctx.panel.select("date", "asset", "ret")
            run = walk_forward(
                ctx, EnsembleConfig(horizon=horizon, components=uni.ensemble_components)
            )
            vs = build_views(run, universe, code_sha=git_sha(), snapshot_id=report.snapshot_id)
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
        return (
            True,
            f"{report.track['months_scored']} scored, {report.track['months_pending']} pending",
        )

    await _timed(report, "track:score", track)

    async def alerts() -> tuple[bool, str]:
        failed = [st.name for st in report.steps if not st.ok]
        fired = run_alerts(data_dir, config_dir, failed_steps=failed, now=started)
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
