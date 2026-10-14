"""Ingest runner: connector → contract → append-only PIT store."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from atlas.catalog import SeriesSpec
from atlas.connectors.base import Connector, ConnectorError
from atlas.pit.store import PITStore
from atlas.quality.contracts import DataQualityError

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class IngestReport:
    series_id: str
    rows_fetched: int = 0
    rows_new: int = 0
    rows_conflicting: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def ingest_series(
    specs: Sequence[SeriesSpec],
    connector: Connector,
    store: PITStore,
    *,
    vintages: bool = True,
    concurrency: int = 4,
) -> list[IngestReport]:
    """Ingest each series independently; one failing series never blocks the others."""
    sem = asyncio.Semaphore(concurrency)

    async def one(spec: SeriesSpec) -> IngestReport:
        async with sem:
            try:
                df = await connector.fetch(spec, vintages=vintages)
                results = store.write(df)
            except (ConnectorError, DataQualityError) as exc:
                log.error("ingest.failed", series=spec.id, error=str(exc))
                return IngestReport(spec.id, error=str(exc))
        report = IngestReport(
            spec.id,
            rows_fetched=df.height,
            rows_new=sum(r.rows_new for r in results),
            rows_conflicting=sum(r.rows_conflicting for r in results),
        )
        if report.rows_conflicting:
            log.warning("ingest.restated_vintages", series=spec.id, rows=report.rows_conflicting)
        log.info("ingest.done", series=spec.id, fetched=report.rows_fetched, new=report.rows_new)
        return report

    return list(await asyncio.gather(*(one(s) for s in specs)))
