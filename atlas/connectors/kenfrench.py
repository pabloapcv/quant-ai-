"""Kenneth French Data Library connector (Fama-French factors, industry portfolios).

Free, long history (industries from 1926-07), the standard academic source. Each file is a
zipped CSV holding several titled sections (e.g. "Average Value Weighted Returns --
Monthly"); rows are ``YYYYMM`` followed by percent returns, with ``-99.99`` / ``-999`` as
missing.

Point-in-time: the library publishes only the current snapshot (it is rebuilt from the
latest CRSP database and history can shift slightly). Rows are stamped with today's
vintage, and history gets approx first-release rows at ``period_end + publication_lag``
(see ``atlas.pit.backfill``). Note that the returns themselves are market data — knowable
at each month-end from prices; the lag is the library's distribution delay.

Catalog ``source_id`` format: ``"<dataset>|<section title or empty>|<column>"``.
"""

from __future__ import annotations

import io
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx
import polars as pl

from atlas.catalog import SeriesSpec
from atlas.connectors.base import ConnectorError, HttpClient, RawStore, RetryPolicy
from atlas.pit.backfill import approx_first_releases
from atlas.pit.clock import NEW_YORK, et_wall_time_to_utc
from atlas.pit.schema import UTC_DATETIME, Frequency, conform

KF_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
MISSING = {-99.99, -999.0}


class KenFrenchError(ConnectorError):
    pass


@dataclass(frozen=True)
class KFSource:
    dataset: str
    section: str  # "" = first table in the file
    column: str
    # Shift applied to annual rows' year. Fama-French BE/ME row t is formed in June t from
    # fiscal year t-1 accounting data, so it describes year t-1 (offset -1).
    year_offset: int = 0

    @classmethod
    def parse(cls, source_id: str) -> KFSource:
        parts = source_id.split("|")
        if len(parts) not in (3, 4):
            raise ValueError(
                f"bad kenfrench source_id {source_id!r}: want 'dataset|section|column[|offset]'"
            )
        offset = int(parts[3]) if len(parts) == 4 else 0
        return cls(parts[0], parts[1], parts[2], offset)


def parse_section(text: str, section: str = "", *, annual: bool = False) -> pl.DataFrame:
    """Parse one table: ``event_time`` (Date) + one Float64 column per header field.

    Monthly tables have ``YYYYMM`` rows; ``annual=True`` reads ``YYYY`` rows (dated Jan 1).
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    if section:
        try:
            start = next(i for i, ln in enumerate(lines) if ln.strip() == section) + 1
        except StopIteration:
            raise KenFrenchError(f"section {section!r} not found") from None
    else:
        start = next((i for i, ln in enumerate(lines) if ln.startswith(",")), -1)
        if start < 0:
            raise KenFrenchError("no table header found")
    header = [h.strip() for h in lines[start].split(",")[1:]]

    rows: list[list[float | None]] = []
    dates: list[date] = []
    for ln in lines[start + 1 :]:
        fields = [f.strip() for f in ln.split(",")]
        if not fields[0].isdigit():
            break  # blank line or next section
        if len(fields[0]) != (4 if annual else 6):
            continue  # the other frequency's rows
        month = 1 if annual else int(fields[0][4:])
        dates.append(date(int(fields[0][:4]), month, 1))
        values = [float(f) for f in fields[1 : len(header) + 1]]
        rows.append([None if v in MISSING else v for v in values])
    if not rows:
        kind = "annual" if annual else "monthly"
        raise KenFrenchError(f"section {section or '<first>'!r} has no {kind} rows")
    data: dict[str, list[date] | list[float | None]] = {"event_time": dates}
    for j, name in enumerate(header):
        data[name] = [r[j] for r in rows]
    return pl.DataFrame(data, schema={"event_time": pl.Date, **dict.fromkeys(header, pl.Float64)})


class KenFrenchConnector:
    source = "kenfrench"

    def __init__(
        self,
        *,
        raw_store: RawStore | None = None,
        run_id: str | None = None,
        timeout_s: float = 60.0,
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        fetched_at: datetime | None = None,
    ) -> None:
        self._raw = raw_store
        self.run_id = run_id or f"run_{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        self._fetched_at = fetched_at
        self._http = HttpClient(
            base_url=KF_BASE_URL,
            max_calls=1,
            period_s=1.0,
            timeout_s=timeout_s,
            retry=retry,
            transport=transport,
            accept="*/*",  # the library's server answers 406 to JSON-only Accept headers
        )
        self._files: dict[str, str] = {}
        self._failed: dict[str, KenFrenchError] = {}

    async def __aenter__(self) -> KenFrenchConnector:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _csv(self, dataset: str) -> str:
        """Download and unzip ``<dataset>_CSV.zip`` once per connector instance."""
        if dataset in self._failed:
            raise self._failed[dataset]
        if dataset not in self._files:
            try:
                self._files[dataset] = await self._download(dataset)
            except KenFrenchError as exc:
                self._failed[dataset] = exc  # don't re-download once per series
                raise
        return self._files[dataset]

    async def _download(self, dataset: str) -> str:
        path = f"{dataset}_CSV.zip"
        resp = await self._http.get(path, {})
        if resp.status_code != 200:
            raise KenFrenchError(f"HTTP {resp.status_code} fetching {path}")
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if len(names) != 1:
                    raise KenFrenchError(f"{path}: expected one CSV, found {names}")
                text = zf.read(names[0]).decode("latin-1")
        except zipfile.BadZipFile:
            raise KenFrenchError(f"{path} is not a zip file") from None
        if self._raw is not None:
            self._raw.put(
                source=self.source,
                run_id=self.run_id,
                request={"path": path},
                body=text.encode("utf-8"),
            )
        return text

    async def fetch(self, spec: SeriesSpec, *, vintages: bool = True) -> pl.DataFrame:
        """``vintages`` controls the approx first-release backfill (no true vintages exist)."""
        if spec.source != self.source:
            raise ValueError(f"{spec.id} is a {spec.source!r} series, not kenfrench")
        src = KFSource.parse(spec.source_id)
        annual = spec.frequency == Frequency.ANNUAL
        table = parse_section(await self._csv(src.dataset), src.section, annual=annual)
        if src.year_offset:
            table = table.with_columns(pl.col("event_time").dt.offset_by(f"{src.year_offset}y"))
        if src.column not in table.columns:
            raise KenFrenchError(f"column {src.column!r} not in {src.dataset} ({table.columns})")

        # A snapshot is known from the moment we fetched it (not end of day).
        fetched = (self._fetched_at or datetime.now(UTC)).replace(microsecond=0)
        today = fetched.astimezone(NEW_YORK).date()
        df = table.select(
            "event_time",
            pl.lit(today).alias("vintage_date"),
            pl.col(src.column).alias("value"),
            pl.lit(False).alias("is_approx_vintage"),
        ).filter(pl.col("value").is_not_null())
        if vintages and df.height:
            approx = approx_first_releases(df, spec.frequency, spec.publication_lag_days)
            df = pl.concat([df, approx])

        out = df.with_columns(
            pl.lit(spec.id).alias("series_id"),
            pl.lit(self.source).alias("source"),
            pl.lit(spec.source_id).alias("source_series_id"),
            pl.when(pl.col("is_approx_vintage"))
            .then(et_wall_time_to_utc(pl.col("vintage_date"), spec.release_time_et))
            .otherwise(pl.lit(fetched, dtype=UTC_DATETIME))
            .alias("knowledge_time"),
            pl.lit("ratio" if annual else "percent_return").alias("unit"),
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
