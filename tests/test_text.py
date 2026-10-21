from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from atlas.agent.tools import ToolKit
from atlas.connectors.base import HttpClient
from atlas.text.documents import DocStore, Document, statement_changes
from atlas.text.fed import BASE, FedTextConnector, article_paragraphs, select_releases
from atlas.text.search import BM25Index
from atlas.text.tone import extract, parse_tone, text_change_bp, text_target

REPO = Path(__file__).resolve().parents[1]


def _page(*paras: str, title: str = "Federal Reserve issues FOMC statement") -> str:
    body = "".join(f"<p>{p}</p>" for p in paras)
    return (
        "<html><body><div id='nav'><p>Menu</p></div><div id='article'>"
        f"<p class='article__time'>March 18, 2008</p><h3 class='title'>{title}</h3>"
        "<p class='releaseTime'>For release at 2:00 p.m. EDT</p><div><p>Share</p></div>"
        f"<div class='col'>{body}<P>For media inquiries, call 202-452-2955</P></div>"
        "</div><footer><p>Footer</p></footer></body></html>"
    )


def test_article_paragraphs_keeps_only_body() -> None:
    paras = article_paragraphs(_page("The Committee decided today.", "Inflation has eased."))
    assert paras == [
        "March 18, 2008",
        "Federal Reserve issues FOMC statement",
        "The Committee decided today.",
        "Inflation has eased.",
    ]


def test_legacy_layout_and_encoding() -> None:
    legacy = (
        "<div id='leftnav'><p>Nav</p></div><div id='content'><p>Minutes of the meeting</p>"
        "<p>Participants agreed \u2014 broadly.</p></div>"
    )
    assert article_paragraphs(legacy) == [
        "Minutes of the meeting",
        "Participants agreed \u2014 broadly.",
    ]
    assert article_paragraphs("<div id='x'><p>a</p></div>") == []


def test_select_releases() -> None:
    idx = [
        {"d": "9/16/2026 2:00:00 PM", "t": "Federal Reserve issues FOMC statement", "l": "/a.htm"},
        {"d": "3/18/2008", "t": "FOMC statement", "l": "/b.htm"},
        {
            "d": "10/7/2026 2:00:00 PM",
            "t": "Minutes of the Federal Open Market Committee, X",
            "l": "/c.htm",
        },
        {"d": "10/8/2026", "t": "Enforcement action", "l": "/d.htm"},
        {"t": "no date"},
    ]
    assert [(k, x["l"]) for k, x in select_releases(idx)] == [
        ("statement", "/a.htm"),
        ("statement", "/b.htm"),
        ("minutes", "/c.htm"),
    ]


def _doc(day: date, paras: list[str], kind: str = "statement") -> Document:
    return Document(
        doc_id=f"fomc-{kind}-{day}",
        source="fomc",
        kind=kind,
        title="t",
        url="u",
        published_at=datetime(day.year, day.month, day.day, 18, tzinfo=UTC),
        meeting_date=day,
        paragraphs=paras,
    )


def test_statement_changes_ignores_votes() -> None:
    a = _doc(
        date(2025, 7, 30),
        ["Growth moderated. Inflation remains elevated.", "Voting for the action were A; B."],
    )
    b = _doc(
        date(2025, 9, 17),
        ["Growth moderated. Job gains have slowed.", "Voting for the action were A; C."],
    )
    assert statement_changes(a, b) == {
        "added": ["Job gains have slowed."],
        "removed": ["Inflation remains elevated."],
    }


def test_bm25_point_in_time_and_kind() -> None:
    old = _doc(
        date(2008, 3, 18), ["The Committee lowered the target rate amid financial market strains."]
    )
    new = _doc(
        date(2022, 6, 15), ["Inflation remains elevated; the Committee raised the target range."]
    )
    mins = _doc(
        date(2022, 6, 15), ["Participants discussed inflation expectations at length."], "minutes"
    )
    idx = BM25Index([old, new, mins])
    hits = idx.search("inflation target", k=5)
    assert {h.passage.doc.doc_id for h in hits} == {new.doc_id, old.doc_id, mins.doc_id}
    assert hits[0].passage.doc.doc_id == new.doc_id
    past = idx.search("inflation target", as_of=datetime(2010, 1, 1, tzinfo=UTC))
    assert [h.passage.doc.doc_id for h in past] == [old.doc_id]
    assert [h.passage.doc.kind for h in idx.search("inflation", kind="minutes")] == ["minutes"]
    assert idx.search("zzzz") == []


def test_tone_parsing_and_size_check() -> None:
    d = date(2022, 1, 1)
    prev = _doc(
        d,
        [
            "The Committee decided to maintain the target range for the federal funds rate at 3/4 to 1 percent."
        ],
    )
    hike = _doc(
        d,
        [
            "The Committee decided to raise the target range for the federal funds rate to 1-1/2 to 1-3/4 percent."
        ],
    )
    old = _doc(
        d,
        [
            "The Committee decided today to lower its target for the federal funds rate 75 basis points to 2-1/4 percent."
        ],
    )
    zlb = _doc(
        d,
        [
            "The Committee decided to establish a target range for the federal funds rate of 0 to 1/4 percent."
        ],
    )
    assert text_target(prev) == 1.0
    assert text_target(hike) == 1.75
    assert text_target(old) == 2.25
    assert text_target(zlb) == 0.25
    assert text_change_bp(hike, prev) == 75
    assert text_change_bp(hike, None) is None
    assert text_target(_doc(d, ["Inflation remains elevated."])) is None

    t = parse_tone(
        'ok {"action": "hike", "change_bp": 75, "stance": 2, "balance_sheet": "shrinking", "dissents": 1, "summary": "s"}',
        "id",
        "m",
    )
    assert t is not None
    assert t.stance == 2
    assert parse_tone('{"action": "hike", "stance": 9}', "id", "m") is None

    reply = '{"action": "hike", "change_bp": 50, "stance": 1, "balance_sheet": "steady", "dissents": 0, "summary": "s"}'
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    model="h", content=[SimpleNamespace(type="text", text=reply)]
                )
            )
        )
    )
    tone = extract(client, hike, prev)
    assert tone is not None
    assert tone.text_change_bp == 75
    assert tone.consistent is False  # the model said 50bp, the stated levels say 75bp


async def test_connector_sync_and_tools(tmp_path: Path) -> None:
    index = [
        {
            "d": "3/18/2008",
            "t": "FOMC statement",
            "l": "/newsevents/pressreleases/monetary20080318a.htm",
        },
        {
            "d": "4/30/2008 2:15:00 PM",
            "t": "FOMC statement",
            "l": "/newsevents/pressreleases/monetary20080430a.htm",
        },
        {
            "d": "4/8/2008 2:00:00 PM",
            "t": "Minutes of the Federal Open Market Committee, March 18, 2008",
            "l": "/newsevents/pressreleases/monetary20080408a.htm",
        },
    ]
    pages = {
        "/json/ne-press.json": json.dumps(index),
        "/newsevents/pressreleases/monetary20080318a.htm": _page(
            "The Committee decided today to lower its target for the federal funds rate 75 basis points.",
            "Inflation has been elevated.",
        ),
        "/newsevents/pressreleases/monetary20080430a.htm": _page(
            "The Committee decided today to lower its target for the federal funds rate 25 basis points.",
            "Inflation has been elevated.",
        ),
        "/newsevents/pressreleases/monetary20080408a.htm": '<a href="/monetarypolicy/fomcminutes20080318.htm">Minutes</a>',
        "/monetarypolicy/fomcminutes20080318.htm": _page(
            "Participants discussed strains in financial markets.",
            "Members agreed.",
            title="Minutes",
        ),
    }
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        body = pages.get(req.url.path)
        return httpx.Response(200, text=body) if body else httpx.Response(404)

    http = HttpClient(
        base_url=BASE, max_calls=100, period_s=1.0, transport=httpx.MockTransport(handler)
    )
    data = tmp_path / "data"
    async with FedTextConnector(data, http=http) as conn:
        out = await conn.sync()
    assert out == {"releases": 3, "new": 3, "skipped": 0, "failed": []}
    docs = list(DocStore(data).iter())
    assert [d.doc_id for d in docs] == [
        "fomc-statement-2008-03-18",
        "fomc-minutes-2008-03-18",
        "fomc-statement-2008-04-30",
    ]
    assert docs[0].published_at == datetime(2008, 3, 18, 18, tzinfo=UTC)  # 14:00 EDT
    (data / "text" / "fomc" / "tone").mkdir()
    tone = {
        "doc_id": "fomc-statement-2008-04-30", "action": "cut", "change_bp": -25, "stance": -1,
        "balance_sheet": "not_mentioned", "dissents": 2, "summary": "s",
    }  # fmt: skip
    (data / "text" / "fomc" / "tone" / "fomc-statement-2008-04-30.json").write_text(
        json.dumps(tone)
    )  # derived data next to the documents is not read as a document
    assert len(list(DocStore(data).iter())) == 3
    n = len(calls)
    http2 = HttpClient(
        base_url=BASE, max_calls=100, period_s=1.0, transport=httpx.MockTransport(handler)
    )
    async with FedTextConnector(data, http=http2) as conn:
        again = await conn.sync()
    assert again["new"] == 0
    assert len(calls) == n + 1  # only the index: stored documents are not refetched

    kit = ToolKit(data, REPO / "config")
    st = kit.get_fomc_statement("2008-04-15")
    assert st["meeting_date"] == "2008-03-18"
    assert st["previous_meeting_date"] is None
    latest = kit.get_fomc_statement("")
    assert any("25 basis points" in s for s in latest["changes_since_previous"]["added"])
    assert latest["extracted_reading"]["stance"] == -1
    res: Any = kit.search_documents("financial markets strains", "fomc", "any", "2008-04-10", 5)
    assert res["passages"][0]["kind"] == "minutes"
    assert all(p["published"] <= "2008-04-10" for p in res["passages"])
