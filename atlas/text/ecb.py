"""ECB monetary policy statements and press-conference transcripts (PRD §6.6).

Source: ecb.europa.eu. The statement archive lazy-loads one listing per year
(``/press/press_conference/monetary-policy-statement/<year>/html/index_include.en.html``),
each linking the English page of every meeting (1998 onward). A page holds the introductory
statement and, below it, the transcript of the questions and answers; they are stored as two
documents, kinds ``statement`` and ``press_conference``.

Politeness: the ECB's robots.txt asks for a 5-second crawl delay, so requests are spaced
5 seconds apart; stored documents are never refetched. Release times: the statement is read
at the press conference (14:30 CET until 2022, 14:45 since); ``published_at`` is set to
15:00 Frankfurt time for both documents, conservatively after either start.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from atlas.connectors.base import ConnectorError, HttpClient, RawStore
from atlas.text.documents import DocStore, Document

BASE = "https://www.ecb.europa.eu"
LIST = "/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html"
FRANKFURT = ZoneInfo("Europe/Berlin")
FIRST_YEAR = 1998
_LINK = re.compile(
    r'href="(/press/press_conference/monetary-policy-statement/\d{4}/html/'
    r'(?:ecb\.)?is(\d{6})[^"]*\.en\.html)"'
)
_QA_START = re.compile(r"^(question|q:)", re.I)
_QA_INVITE = re.compile(r"(ready to take your questions|at your disposal for questions)", re.I)
_DROP = re.compile(r"^(\*\s*)+$|^Jump to the transcript|^Corrects? ", re.I)


class _StatementPage(HTMLParser):
    """Paragraphs of every ``div.section`` inside ``<main>``; notes the Q&A anchor."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.qa_at: int | None = None
        self._in_main = False
        self._depth = 0  # div depth inside the section; 0 = outside
        self._buf: list[str] = []
        self._block = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "main":
            self._in_main = True
        elif tag == "div" and self._in_main:
            if self._depth:
                self._depth += 1
            elif "section" in (a.get("class") or "").split():
                self._depth = 1
        elif self._depth and tag in ("p", "h2", "h3", "li"):
            self._flush()
            self._block = True
        elif self._depth and tag == "a" and a.get("id") == "qa" and self.qa_at is None:
            self.qa_at = len(self.paragraphs)
        elif self._depth and tag == "br":
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "main":
            self._in_main = False
        if tag == "div" and self._depth:
            self._depth -= 1
            if not self._depth:
                self._flush()
        elif self._depth and tag in ("p", "h2", "h3", "li"):
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._depth and self._block:
            self._buf.append(data)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if text and not _DROP.match(text):
            self.paragraphs.append(text)
        self._buf, self._block = [], False


def split_page(html: str) -> tuple[list[str], list[str]]:
    """(statement paragraphs, Q&A paragraphs)."""
    p = _StatementPage()
    p.feed(html)
    p.close()
    paras = p.paragraphs
    cut = p.qa_at
    if cut is None:
        cut = next((i for i, t in enumerate(paras) if _QA_START.match(t)), None)
    if cut is None:
        invite = next((i for i, t in enumerate(paras) if _QA_INVITE.search(t)), None)
        cut = None if invite is None else invite + 1
    if cut is None:
        return paras, []
    return paras[:cut], paras[cut:]


def listing(html: str) -> list[tuple[date, str]]:
    """(meeting date, English page path) per statement; the date is in the file name."""
    seen: dict[str, date] = {}
    for path, ymd in _LINK.findall(html):
        yy, mm, dd = int(ymd[:2]), int(ymd[2:4]), int(ymd[4:])
        seen.setdefault(path, date(1900 + yy if yy >= 90 else 2000 + yy, mm, dd))
    return sorted(((d, p) for p, d in seen.items()), key=lambda x: x[0])


def _published(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 15, 0, tzinfo=FRANKFURT).astimezone(UTC)


class EcbTextConnector:
    def __init__(
        self, data_dir: Path, *, http: HttpClient | None = None, run_id: str | None = None
    ) -> None:
        self.store = DocStore(data_dir)
        self.raw = RawStore(data_dir)
        self.http = http or HttpClient(base_url=BASE, max_calls=1, period_s=5.0, accept="text/html")
        self.run_id = run_id or datetime.now(UTC).strftime("ecb-%Y%m%dT%H%M%S")

    async def __aenter__(self) -> EcbTextConnector:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.http.aclose()

    async def _get(self, path: str) -> str:
        resp = await self.http.get(path, {})
        if resp.status_code != 200:
            raise ConnectorError(f"HTTP {resp.status_code} for {path}")
        text = resp.content.decode("utf-8", errors="replace")
        self.raw.put(
            source="ecb_text", run_id=self.run_id, request={"path": path}, body=text.encode()
        )
        return text

    async def sync(
        self, *, years: list[int] | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """Fetch statements not stored yet; recent years first (default: all since 1998)."""
        this_year = datetime.now(UTC).year
        new, failed, pages = 0, [], 0
        for year in years or list(range(this_year, FIRST_YEAR - 1, -1)):
            try:
                entries = listing(await self._get(LIST.format(year=year)))
            except ConnectorError as exc:
                failed.append(f"{year}: {exc}")
                continue
            for day, path in entries:
                if limit is not None and new >= limit:
                    return {"new": new, "pages": pages, "failed": failed}
                sid = f"ecb-statement-{day.isoformat()}"
                if self.store.exists("ecb", "statement", sid):
                    continue
                try:
                    statement, qa = split_page(await self._get(path))
                except ConnectorError as exc:
                    failed.append(f"{path}: {exc}")
                    continue
                pages += 1
                if len(statement) < 2:
                    failed.append(f"{path}: no statement text found")
                    continue
                common = {
                    "source": "ecb",
                    "url": BASE + path,
                    "published_at": _published(day),
                    "meeting_date": day,
                }
                new += self.store.save(
                    Document(
                        doc_id=sid,
                        kind="statement",
                        title="ECB monetary policy statement",
                        paragraphs=statement,
                        **common,
                    )
                )
                if qa:
                    self.store.save(
                        Document(
                            doc_id=f"ecb-press_conference-{day.isoformat()}",
                            kind="press_conference",
                            title="ECB press conference, questions and answers",
                            paragraphs=qa,
                            **common,
                        )
                    )
        return {"new": new, "pages": pages, "failed": failed}
