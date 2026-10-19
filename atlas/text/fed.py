"""Federal Reserve text connector: FOMC statements and minutes from federalreserve.gov.

The Board publishes an index of every press release (``/json/ne-press.json``: date, title,
type, link). FOMC statements are the "FOMC statement" releases (2006 onward on the site);
minutes are the "Minutes of the Federal Open Market Committee" releases, whose page links to
the full minutes document. Only pages not yet stored are fetched, at most one request per
second; raw HTML goes to the raw zone so parsing can be re-run.

Release times: the index gives an Eastern-time timestamp (older entries a date only, which
is taken as 14:00 ET, the usual statement time); ``published_at`` is stored in UTC.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from atlas.connectors.base import ConnectorError, HttpClient, RawStore
from atlas.text.documents import DocStore, Document, Kind

BASE = "https://www.federalreserve.gov"
ET = ZoneInfo("America/New_York")
STATEMENT_TITLE = re.compile(r"^(Federal Reserve issues )?FOMC statement\b", re.I)
MINUTES_TITLE = re.compile(r"^Minutes of the Federal Open Market Committee", re.I)
MINUTES_LINK = re.compile(r'href="(/monetarypolicy/fomcminutes(\d{8})\.htm)"')
_DROP = re.compile(
    r"^(For media inquiries|For release at|For immediate release|Implementation Note|"
    r"Last Update|Share$|Return to top)",
    re.I,
)


class _ArticleText(HTMLParser):
    """Paragraph text inside ``<div id=container>`` (headings included)."""

    def __init__(self, container: str = "article") -> None:
        super().__init__(convert_charrefs=True)
        self.container = container
        self.paragraphs: list[str] = []
        self._depth = 0  # div nesting inside the article; 0 = outside
        self._buf: list[str] = []
        self._in_block = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "div":
            if self._depth:
                self._depth += 1
            elif dict(attrs).get("id") == self.container:
                self._depth = 1
        elif self._depth and tag in ("p", "h3", "h4", "h5", "li"):
            self._flush()
            self._in_block = True
        elif self._depth and tag in ("script", "style"):
            self._in_block = False

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._depth:
            self._depth -= 1
            if not self._depth:
                self._flush()
        elif self._depth and tag in ("p", "h3", "h4", "h5", "li"):
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._depth and self._in_block:
            self._buf.append(data)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._buf)).replace("\u2011", "-").strip()
        text = text.removesuffix("Return to text").strip()
        if text and not _DROP.match(text):
            self.paragraphs.append(text)
        self._buf = []
        self._in_block = False


def article_paragraphs(html: str) -> list[str]:
    """Body paragraphs; pages from before the 2016 redesign use ``div#content``."""
    for container in ("article", "content"):
        parser = _ArticleText(container)
        parser.feed(html)
        parser.close()
        if len(parser.paragraphs) >= 2:
            return parser.paragraphs
    return []


def _release_time(stamp: str) -> datetime:
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y"):
        try:
            local = datetime.strptime(stamp.strip(), fmt)  # noqa: DTZ007 - ET, attached below
        except ValueError:
            continue
        if fmt == "%m/%d/%Y":
            local = local.replace(hour=14)
        return local.replace(tzinfo=ET).astimezone(UTC)
    raise ValueError(f"unparseable release time {stamp!r}")


def select_releases(index: list[dict[str, Any]]) -> list[tuple[Kind, dict[str, Any]]]:
    out: list[tuple[Kind, dict[str, Any]]] = []
    for item in index:
        title = str(item.get("t", "")).strip()
        if "d" not in item or "l" not in item:
            continue
        if STATEMENT_TITLE.match(title):
            out.append(("statement", item))
        elif MINUTES_TITLE.match(title):
            out.append(("minutes", item))
    return out


class FedTextConnector:
    def __init__(
        self,
        data_dir: Path,
        *,
        http: HttpClient | None = None,
        run_id: str | None = None,
    ) -> None:
        self.store = DocStore(data_dir)
        self.raw = RawStore(data_dir)
        self.http = http or HttpClient(
            base_url=BASE, max_calls=1, period_s=1.0, accept="text/html,application/json"
        )
        self.run_id = run_id or datetime.now(UTC).strftime("fed-%Y%m%dT%H%M%S")
        self._minutes_seen: set[datetime] = set()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> FedTextConnector:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _get(self, path: str) -> str:
        resp = await self.http.get(path, {})
        if resp.status_code != 200:
            raise ConnectorError(f"HTTP {resp.status_code} for {path}")
        try:
            text = resp.content.decode("utf-8-sig")
        except UnicodeDecodeError:  # some 2010-2011 pages are Windows-1252
            text = resp.content.decode("cp1252", errors="replace")
        # archived as UTF-8 text (the raw zone stores bodies as UTF-8)
        self.raw.put(source="fed", run_id=self.run_id, request={"path": path}, body=text.encode())
        return text

    async def index(self) -> list[dict[str, Any]]:
        data = json.loads(await self._get("/json/ne-press.json"))
        return [x for x in data if isinstance(x, dict)]

    async def sync(self, *, since: date | None = None, limit: int | None = None) -> dict[str, Any]:
        """Fetch every statement / minutes not stored yet. Returns counts and failures."""
        releases = select_releases(await self.index())
        self._minutes_seen = {
            d.published_at for d in self.store.iter(source="fomc", kind="minutes")
        }
        new, skipped, failed = 0, 0, []
        for kind, item in sorted(releases, key=lambda r: _release_time(r[1]["d"])):
            published = _release_time(item["d"])
            if since and published.date() < since:
                continue
            if limit is not None and new >= limit:
                break
            try:
                doc = await self._document(kind, item, published)
            except (ConnectorError, ValueError) as exc:
                failed.append(f"{item['l']}: {exc}")
                continue
            if doc is None:
                skipped += 1
                continue
            new += self.store.save(doc)
        return {"releases": len(releases), "new": new, "skipped": skipped, "failed": failed}

    async def _document(
        self, kind: Kind, item: dict[str, Any], published: datetime
    ) -> Document | None:
        link = str(item["l"])
        if kind == "statement":
            meeting = published.astimezone(ET).date()
            doc_id = f"fomc-statement-{meeting.isoformat()}"
            if self.store.exists("fomc", kind, doc_id):
                return None
            paragraphs = article_paragraphs(await self._get(link))
            url = BASE + link
            body = [p for p in paragraphs if p.lower() != str(item["t"]).strip().lower()]
        else:
            m = re.search(r"(\d{8})", link)
            if m is None:
                raise ValueError("no date in minutes link")
            release_page = await self._get(link) if not self._known_minutes(item) else ""
            found = MINUTES_LINK.search(release_page)
            if found is None:
                return None  # older format (or already stored)
            path, ymd = found.group(1), found.group(2)
            meeting = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
            doc_id = f"fomc-minutes-{meeting.isoformat()}"
            if self.store.exists("fomc", kind, doc_id):
                return None
            body = article_paragraphs(await self._get(path))
            url = BASE + path
        body = [p for p in body if not re.fullmatch(r"[A-Z][a-z]+ \d{1,2}, \d{4}", p)]
        if len(body) < 2:
            raise ValueError("no article text found")
        return Document(
            doc_id=doc_id,
            source="fomc",
            kind=kind,
            title=str(item["t"]).strip(),
            url=url,
            published_at=published,
            meeting_date=meeting,
            paragraphs=body,
        )

    def _known_minutes(self, item: dict[str, Any]) -> bool:
        """True if the minutes released by ``item`` are already stored (skip the fetch)."""
        return _release_time(item["d"]) in self._minutes_seen
