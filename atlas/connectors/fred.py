"""FRED / ALFRED connector (St. Louis Fed).

FRED and ALFRED share one API. ALFRED is FRED queried across real-time periods: each
returned row is a value plus the ``[realtime_start, realtime_end]`` window during which
FRED published it. ``realtime_start`` is therefore the vintage date — the day the value
became public — which, combined with the series' usual release time, gives
``knowledge_time``.

Two modes:

* ``vintages=True`` (ALFRED): full revision history. Required for anything backtested.
* ``vintages=False`` (FRED): current values only. Every row is stamped with today's vintage,
  so nothing leaks into the past; useful for dashboards, useless for backtests.

Observations older than ALFRED's vintage coverage would only become "known" on the first
archived vintage (often decades late). When ``backfill_pre_vintage`` is on, those periods
also get a synthetic row at ``period_end + publication_lag`` carrying the earliest archived
value, flagged ``is_approx_vintage=True`` so consumers can exclude it.

API docs: https://fred.stlouisfed.org/docs/api/fred/
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import polars as pl
import structlog

from atlas.catalog import SeriesSpec
from atlas.connectors.base import ConnectorError, HttpClient, RawStore, RetryPolicy
from atlas.pit.backfill import approx_first_releases
from atlas.pit.clock import NEW_YORK, et_wall_time_to_utc
from atlas.pit.schema import conform

log = structlog.get_logger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/"
REALTIME_MIN = date(1776, 7, 4)  # FRED's documented "beginning of time"
REALTIME_MAX = date(9999, 12, 31)
OBSERVATIONS_PAGE_LIMIT = 100_000
VINTAGE_DATES_PAGE_LIMIT = 10_000
RELEASE_DATES_PAGE_LIMIT = 1_000
# FRED rejects observation requests spanning too many vintages; chunk well below the cap.
VINTAGES_PER_CHUNK = 1_500

_RAW_COLUMNS = {
    "realtime_start": pl.String,
    "realtime_end": pl.String,
    "date": pl.String,
    "value": pl.String,
}


class FredError(ConnectorError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"FRED error {code}: {message}")
        self.code = code
        self.message = message

    @property
    def too_many_vintages(self) -> bool:
        return self.code == 400 and "vintage" in self.message.lower()


@dataclass(frozen=True)
class FredSeriesInfo:
    id: str
    title: str
    frequency_short: str
    units: str
    seasonal_adjustment_short: str
    last_updated: str


@dataclass(frozen=True)
class Release:
    release_id: int
    name: str
    date: date


class FredConnector:
    source = "fred"

    def __init__(
        self,
        api_key: str,
        *,
        raw_store: RawStore | None = None,
        run_id: str | None = None,
        max_calls_per_minute: int = 100,
        timeout_s: float = 30.0,
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        today: date | None = None,
        since: dict[str, date] | None = None,
    ) -> None:
        """``since``: per catalog id, fetch only vintages from that date (incremental runs)."""
        if not api_key:
            raise ValueError("FRED api_key is required (set ATLAS_FRED_API_KEY)")
        self._api_key = api_key
        self._raw = raw_store
        self.run_id = run_id or f"run_{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        self._today = today
        self._since = since or {}
        self._http = HttpClient(
            base_url=FRED_BASE_URL,
            max_calls=max_calls_per_minute,
            period_s=60.0,
            timeout_s=timeout_s,
            retry=retry,
            secrets=[api_key],
            transport=transport,
        )

    async def __aenter__(self) -> FredConnector:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ low level
    async def _get(self, endpoint: str, **params: str | int) -> dict[str, Any]:
        public_params: dict[str, str | int] = {**params, "file_type": "json"}
        resp = await self._http.get(endpoint, {**public_params, "api_key": self._api_key})
        try:
            payload: dict[str, Any] = resp.json()
        except json.JSONDecodeError:
            raise FredError(resp.status_code, f"non-JSON response from {endpoint}") from None
        if "error_code" in payload or resp.status_code >= 400:
            raise FredError(
                int(payload.get("error_code", resp.status_code)),
                str(payload.get("error_message", "unknown error")),
            )
        if self._raw is not None:
            self._raw.put(
                source=self.source,
                run_id=self.run_id,
                request={"endpoint": endpoint, "params": public_params},
                body=resp.content,
            )
        return payload

    def _today_et(self) -> date:
        return self._today or datetime.now(NEW_YORK).date()

    # ------------------------------------------------------------------ metadata
    async def series_info(self, source_id: str) -> FredSeriesInfo:
        payload = await self._get("series", series_id=source_id)
        s = payload["seriess"][0]
        return FredSeriesInfo(
            id=s["id"],
            title=s["title"],
            frequency_short=s["frequency_short"],
            units=s["units"],
            seasonal_adjustment_short=s["seasonal_adjustment_short"],
            last_updated=s["last_updated"],
        )

    async def vintage_dates(self, source_id: str) -> list[date]:
        dates: list[date] = []
        offset = 0
        while True:
            payload = await self._get(
                "series/vintagedates",
                series_id=source_id,
                realtime_start=REALTIME_MIN.isoformat(),
                realtime_end=REALTIME_MAX.isoformat(),
                limit=VINTAGE_DATES_PAGE_LIMIT,
                offset=offset,
            )
            batch = payload.get("vintage_dates", [])
            dates.extend(date.fromisoformat(d) for d in batch)
            offset += len(batch)
            if not batch or offset >= int(payload.get("count", 0)):
                return dates

    async def release_calendar(self, start: date, end: date) -> list[Release]:
        """Scheduled release dates for all FRED releases in ``[start, end]``."""
        releases: list[Release] = []
        offset = 0
        while True:
            payload = await self._get(
                "releases/dates",
                realtime_start=start.isoformat(),
                realtime_end=end.isoformat(),
                include_release_dates_with_no_data="true",
                sort_order="asc",
                limit=RELEASE_DATES_PAGE_LIMIT,
                offset=offset,
            )
            batch = payload.get("release_dates", [])
            releases.extend(
                Release(
                    int(r["release_id"]), r.get("release_name", ""), date.fromisoformat(r["date"])
                )
                for r in batch
            )
            offset += len(batch)
            if not batch or offset >= int(payload.get("count", 0)):
                return releases

    # ------------------------------------------------------------------ observations
    async def _observations_window(
        self, source_id: str, realtime_start: date, realtime_end: date
    ) -> pl.DataFrame:
        rows: list[dict[str, str]] = []
        offset = 0
        while True:
            payload = await self._get(
                "series/observations",
                series_id=source_id,
                realtime_start=realtime_start.isoformat(),
                realtime_end=realtime_end.isoformat(),
                sort_order="asc",
                limit=OBSERVATIONS_PAGE_LIMIT,
                offset=offset,
            )
            batch = payload.get("observations", [])
            rows.extend(batch)
            offset += len(batch)
            if not batch or offset >= int(payload.get("count", 0)):
                break
        return pl.DataFrame(
            [{k: r[k] for k in _RAW_COLUMNS} for r in rows], schema=_RAW_COLUMNS, orient="row"
        )

    async def _vintages_from(self, source_id: str, since: date) -> pl.DataFrame:
        try:
            return await self._observations_window(source_id, since, REALTIME_MAX)
        except FredError as exc:
            if not exc.too_many_vintages:
                raise
        return await self._all_vintages(source_id)  # window too wide: fall back to chunking

    async def _all_vintages(self, source_id: str) -> pl.DataFrame:
        try:
            return await self._observations_window(source_id, REALTIME_MIN, REALTIME_MAX)
        except FredError as exc:
            if not exc.too_many_vintages:
                raise
        vintages = await self.vintage_dates(source_id)
        log.info("fred.chunking_vintages", series=source_id, vintages=len(vintages))
        frames = []
        for i in range(0, len(vintages), VINTAGES_PER_CHUNK):
            lo = vintages[i]
            nxt = i + VINTAGES_PER_CHUNK
            hi = vintages[nxt] - timedelta(days=1) if nxt < len(vintages) else REALTIME_MAX
            frames.append(await self._observations_window(source_id, lo, hi))
        return pl.concat(frames)

    async def fetch(
        self,
        spec: SeriesSpec,
        *,
        vintages: bool = True,
        backfill_pre_vintage: bool = True,
    ) -> pl.DataFrame:
        """Fetch one series into the canonical silver schema."""
        if spec.source != self.source:
            raise ValueError(f"{spec.id} is a {spec.source!r} series, not fred")
        info = await self.series_info(spec.source_id)
        if info.frequency_short != spec.frequency.value:
            log.warning(
                "fred.frequency_mismatch",
                series=spec.id,
                catalog=spec.frequency.value,
                fred=info.frequency_short,
            )

        since = self._since.get(spec.id)
        if vintages and since is not None:
            # Incremental: only vintages published from `since`. History is already stored,
            # so no pre-vintage backfill (the window start would look like a late release).
            raw = await self._vintages_from(spec.source_id, since)
            backfill_pre_vintage = False
        elif vintages:
            raw = await self._all_vintages(spec.source_id)
        else:
            today = self._today_et()
            raw = await self._observations_window(spec.source_id, today, today)

        return self._to_silver(raw, spec, info, backfill_pre_vintage=backfill_pre_vintage)

    def _to_silver(
        self,
        raw: pl.DataFrame,
        spec: SeriesSpec,
        info: FredSeriesInfo,
        *,
        backfill_pre_vintage: bool,
    ) -> pl.DataFrame:
        df = raw.select(
            pl.col("date").str.to_date("%Y-%m-%d").alias("event_time"),
            pl.col("realtime_start").str.to_date("%Y-%m-%d").alias("vintage_date"),
            # "." is FRED's missing-value marker
            pl.when(pl.col("value") == ".")
            .then(None)
            .otherwise(pl.col("value"))
            .cast(pl.Float64)
            .alias("value"),
        )
        # FRED fills market holidays with the prior close and posts the row on the trading
        # day *before* the holiday, so its vintage precedes the date it describes. The value
        # was public, but "the value for day d" cannot be known before d: clamp to the date.
        early = df.filter(pl.col("vintage_date") < pl.col("event_time")).height
        if early:
            log.info("fred.clamped_early_vintages", series=spec.id, rows=early)
            df = df.with_columns(
                pl.max_horizontal("vintage_date", "event_time").alias("vintage_date")
            )
        df = _collapse_unchanged(df).with_columns(pl.lit(False).alias("is_approx_vintage"))

        if backfill_pre_vintage and df.height:
            approx = approx_first_releases(df, spec.frequency, spec.publication_lag_days)
            df = pl.concat([df, approx])

        now = datetime.now(UTC)
        out = df.with_columns(
            pl.lit(spec.id).alias("series_id"),
            pl.lit(self.source).alias("source"),
            pl.lit(spec.source_id).alias("source_series_id"),
            et_wall_time_to_utc(pl.col("vintage_date"), spec.release_time_et).alias(
                "knowledge_time"
            ),
            pl.lit(info.units).alias("unit"),
            pl.lit(spec.frequency.value).alias("frequency"),
            pl.when(pl.col("is_approx_vintage"))
            .then(pl.lit("approx:") + pl.col("vintage_date").cast(pl.String))
            .otherwise(pl.col("vintage_date").cast(pl.String))
            .alias("vintage_id"),
            pl.lit(info.seasonal_adjustment_short != "NSA").alias("sa_flag"),
            pl.lit(self.run_id).alias("ingest_run_id"),
            pl.lit(now).alias("ingested_at"),
        )
        return conform(out).sort(["event_time", "knowledge_time"])


def _collapse_unchanged(df: pl.DataFrame) -> pl.DataFrame:
    """Drop vintages that repeat the previous vintage's value for the same period.

    Chunked real-time windows split one published value into several rows; this restores
    one row per actual change (null-safe, so a withdrawal is kept as a change).
    """
    df = df.sort(["event_time", "vintage_date"])
    first_in_period = pl.int_range(pl.len()).over("event_time") == 0
    changed = pl.col("value").ne_missing(pl.col("value").shift(1).over("event_time"))
    return df.filter(first_in_period | changed)


__all__ = ["FredConnector", "FredError", "FredSeriesInfo", "Release"]
