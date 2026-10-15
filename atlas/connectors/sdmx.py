"""SDMX connector for the official global statistics agencies (PRD §5.2, Phase 5).

One parser covers ECB, OECD, BIS and IMF: each serves SDMX-CSV with ``TIME_PERIOD`` and
``OBS_VALUE`` columns plus one column per dimension.

Catalog ``source_id``: ``"<flow>|<key>|<selector>"``

* ``flow``      dataflow id as the agency names it (``EXR``, ``OECD.SDD.STES,DSD_STES@DF_CLI,4.1``,
                ``WS_EER``, ``IMF.STA,CPI``)
* ``key``       SDMX key; may use ``+`` to fetch several series in one request
* ``selector``  ``DIM=VALUE[,DIM=VALUE]`` picking this series' rows out of that response
                (empty when the key already names exactly one series)

Every spec sharing a ``(flow, key)`` reuses one download, so a whole dataset costs one
request — important for OECD, which rate-limits hard.

Point-in-time: these endpoints serve the current vintage only. Rows are stamped at fetch
time; history gets flagged approx first releases at ``period_end + publication_lag``
(``atlas.pit.backfill``). Heavily revised series (OECD CLIs) carry revision look-ahead in
their approx rows; strict backtests exclude them with ``include_approx=False``.
"""

from __future__ import annotations

import csv
import io
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx
import polars as pl

from atlas.catalog import SeriesSpec
from atlas.connectors.base import ConnectorError, HttpClient, RawStore, RetryPolicy
from atlas.pit.backfill import approx_first_releases
from atlas.pit.clock import NEW_YORK, et_wall_time_to_utc
from atlas.pit.schema import UTC_DATETIME, conform


class SdmxError(ConnectorError):
    pass


@dataclass(frozen=True)
class Agency:
    source: str
    base_url: str
    path: str  # format string with {flow} and {key}
    params: tuple[tuple[str, str], ...]
    max_calls_per_minute: int


AGENCIES: dict[str, Agency] = {
    "ecb": Agency(
        "ecb",
        "https://data-api.ecb.europa.eu/service/",
        "data/{flow}/{key}",
        (("format", "csvdata"),),
        30,
    ),
    "oecd": Agency(
        "oecd",
        "https://sdmx.oecd.org/public/rest/",
        "data/{flow}/{key}",
        (("format", "csvfile"),),
        10,
    ),
    "bis": Agency(
        "bis",
        "https://stats.bis.org/api/v2/",
        "data/dataflow/BIS/{flow}/1.0/{key}",
        (("format", "csv"),),
        30,
    ),
    "imf": Agency(
        "imf",
        "https://api.imf.org/external/sdmx/2.1/",
        "data/{flow}/{key}",
        (),
        20,
    ),
}


@dataclass(frozen=True)
class SdmxSource:
    flow: str
    key: str
    selector: tuple[tuple[str, str], ...]

    @classmethod
    def parse(cls, source_id: str) -> SdmxSource:
        parts = source_id.split("|")
        if len(parts) not in (2, 3):
            raise ValueError(f"bad sdmx source_id {source_id!r}: want 'flow|key[|DIM=V,...]'")
        sel: list[tuple[str, str]] = []
        if len(parts) == 3 and parts[2]:
            for item in parts[2].split(","):
                dim, _, val = item.partition("=")
                if not dim or not val:
                    raise ValueError(f"bad selector {item!r} in {source_id!r}")
                sel.append((dim.strip(), val.strip()))
        return cls(parts[0], parts[1], tuple(sel))


_PERIOD: list[tuple[re.Pattern[str], Callable[[re.Match[str]], date]]] = [
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"^(\d{4})-(\d{2})$"), lambda m: date(int(m[1]), int(m[2]), 1)),
    (re.compile(r"^(\d{4})-M(\d{2})$"), lambda m: date(int(m[1]), int(m[2]), 1)),
    (re.compile(r"^(\d{4})-Q([1-4])$"), lambda m: date(int(m[1]), 3 * int(m[2]) - 2, 1)),
    (re.compile(r"^(\d{4})-S([12])$"), lambda m: date(int(m[1]), 6 * int(m[2]) - 5, 1)),
    (re.compile(r"^(\d{4})(?:-A1)?$"), lambda m: date(int(m[1]), 1, 1)),
]


def parse_period(text: str) -> date:
    """SDMX TIME_PERIOD -> period start date (daily, monthly, quarterly, half-year, annual)."""
    t = text.strip()
    for pattern, build in _PERIOD:
        m = pattern.match(t)
        if m:
            return build(m)
    raise SdmxError(f"unrecognized TIME_PERIOD {text!r}")


def parse_csv(text: str, selector: tuple[tuple[str, str], ...] = ()) -> pl.DataFrame:
    """SDMX-CSV -> ``event_time``, ``value`` for the rows matching ``selector``.

    Raises if the selection still mixes several series (ambiguous catalog entry).
    """
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if "TIME_PERIOD" not in fields or "OBS_VALUE" not in fields:
        raise SdmxError(f"not SDMX-CSV (columns: {fields[:8]})")
    for dim, _ in selector:
        if dim not in fields:
            raise SdmxError(f"selector dimension {dim!r} not in response")
    skip = {"TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "OBS_CONF", "OBS_PRE_BREAK", "OBS_COM"}
    dims = [f for f in fields if f.isupper() and f not in skip and not f.startswith("TITLE")]
    rows: dict[date, float | None] = {}
    series_keys: set[tuple[str, ...]] = set()
    for rec in reader:
        if any(rec.get(d) != v for d, v in selector):
            continue
        series_keys.add(tuple(rec.get(d, "") for d in dims if d in _SERIES_DIMS))
        raw = (rec.get("OBS_VALUE") or "").strip()
        value = None if raw in ("", "NaN", "NA") else float(raw)
        rows[parse_period(rec["TIME_PERIOD"])] = value
    if len(series_keys) > 1:
        raise SdmxError(f"selection matches {len(series_keys)} series; tighten the selector")
    if not rows:
        raise SdmxError("no observations matched")
    items = sorted(rows.items())
    return pl.DataFrame(
        {"event_time": [d for d, _ in items], "value": [v for _, v in items]},
        schema={"event_time": pl.Date, "value": pl.Float64},
    )


# Dimensions that identify a series (as opposed to per-row attributes) across agencies.
_SERIES_DIMS = {
    "KEY",
    "FREQ",
    "FREQUENCY",
    "REF_AREA",
    "COUNTRY",
    "CURRENCY",
    "CURRENCY_DENOM",
    "EXR_TYPE",
    "EXR_SUFFIX",
    "MEASURE",
    "UNIT_MEASURE",
    "ACTIVITY",
    "ADJUSTMENT",
    "TRANSFORMATION",
    "TIME_HORIZ",
    "METHODOLOGY",
    "EER_TYPE",
    "EER_BASKET",
    "INDEX_TYPE",
    "COICOP_1999",
    "TYPE_OF_TRANSFORMATION",
    "PROVIDER_FM",
    "INSTRUMENT_FM",
    "PROVIDER_FM_ID",
    "DATA_TYPE_FM",
}


class SdmxConnector:
    """One instance per agency; caches each (flow, key) response for the run."""

    def __init__(
        self,
        agency: str,
        *,
        raw_store: RawStore | None = None,
        run_id: str | None = None,
        timeout_s: float = 90.0,
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        fetched_at: datetime | None = None,
        start_period: str | None = None,
    ) -> None:
        if agency not in AGENCIES:
            raise ValueError(f"unknown agency {agency!r}: {sorted(AGENCIES)}")
        self.agency = AGENCIES[agency]
        self.source = self.agency.source
        self._raw = raw_store
        self.run_id = run_id or f"run_{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        self._fetched_at = fetched_at
        self._start = start_period
        self._http = HttpClient(
            base_url=self.agency.base_url,
            max_calls=self.agency.max_calls_per_minute,
            period_s=60.0,
            timeout_s=timeout_s,
            retry=retry,
            accept="text/csv",
            transport=transport,
        )
        self._cache: dict[tuple[str, str], str] = {}
        self._failed: dict[tuple[str, str], SdmxError] = {}

    async def __aenter__(self) -> SdmxConnector:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _text(self, flow: str, key: str) -> str:
        ck = (flow, key)
        if ck in self._failed:
            raise self._failed[ck]
        if ck not in self._cache:
            path = self.agency.path.format(flow=flow, key=key)
            params = dict(self.agency.params)
            if self._start:
                params["startPeriod"] = self._start
            resp = await self._http.get(path, params)
            if resp.status_code != 200:
                err = SdmxError(
                    f"{self.source} HTTP {resp.status_code} for {flow}/{key}: "
                    f"{resp.text[:160].strip()}"
                )
                self._failed[ck] = err
                raise err
            text = resp.text
            if self._raw is not None:
                self._raw.put(
                    source=self.source,
                    run_id=self.run_id,
                    request={"path": path, "params": params},
                    body=text.encode("utf-8"),
                )
            self._cache[ck] = text
        return self._cache[ck]

    async def fetch(self, spec: SeriesSpec, *, vintages: bool = True) -> pl.DataFrame:
        """``vintages`` controls the approx first-release backfill (no true vintages here)."""
        if spec.source != self.source:
            raise ValueError(f"{spec.id} is a {spec.source!r} series, not {self.source}")
        src = SdmxSource.parse(spec.source_id)
        table = parse_csv(await self._text(src.flow, src.key), src.selector)

        fetched = (self._fetched_at or datetime.now(UTC)).replace(microsecond=0)
        today = fetched.astimezone(NEW_YORK).date()
        df = table.filter(pl.col("value").is_not_null()).with_columns(
            pl.lit(today).alias("vintage_date"), pl.lit(False).alias("is_approx_vintage")
        )
        if vintages and df.height:
            approx = approx_first_releases(df, spec.frequency, spec.publication_lag_days)
            df = pl.concat([df, approx.select(df.columns)])
        out = df.with_columns(
            pl.lit(spec.id).alias("series_id"),
            pl.lit(self.source).alias("source"),
            pl.lit(spec.source_id).alias("source_series_id"),
            pl.when(pl.col("is_approx_vintage"))
            .then(et_wall_time_to_utc(pl.col("vintage_date"), spec.release_time_et))
            .otherwise(pl.lit(fetched, dtype=UTC_DATETIME))
            .alias("knowledge_time"),
            pl.lit(None, dtype=pl.String).alias("unit"),
            pl.lit(spec.frequency.value).alias("frequency"),
            pl.when(pl.col("is_approx_vintage"))
            .then(pl.lit("approx:") + pl.col("vintage_date").cast(pl.String))
            .otherwise(pl.col("vintage_date").cast(pl.String))
            .alias("vintage_id"),
            pl.lit(None, dtype=pl.Boolean).alias("sa_flag"),
            pl.lit(self.run_id).alias("ingest_run_id"),
            pl.lit(datetime.now(UTC)).alias("ingested_at"),
        )
        return conform(out).sort(["event_time", "knowledge_time"])
