from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from atlas.agent.tools import ToolKit
from atlas.connectors.base import HttpClient
from atlas.text.documents import DocStore
from atlas.text.ecb import BASE, LIST, EcbTextConnector, listing, split_page

REPO = Path(__file__).resolve().parents[1]

NEW = (
    "<html><body><div class='section'><p>Nav</p></div><main><div class='title'><h1>PRESS</h1></div>"
    "<div class='section'><h2>Christine Lagarde</h2><p>Good afternoon.</p>"
    "<p>We decided today to keep the three key ECB interest rates unchanged.</p>"
    "<p>We are now ready to take your questions.</p><p><strong>* * *</strong></p>"
    "<p><a id='qa'></a><strong>Are we in a good place?</strong></p><p>Yes, we are.</p></div>"
    "<div class='related-topics'><p>Related topics</p></div></main><footer><p>F</p></footer></body></html>"
)
OLD = (
    "<main><div class='section'><p>Introductory statement</p><p>Ladies and gentlemen.</p>"
    "<p>We are now at your disposal for questions.</p></div>"
    "<div class='section'><p>Question: Will rates rise?</p><p>Duisenberg: We never pre-commit.</p></div></main>"
)


def test_split_new_and_old_layouts() -> None:
    st, qa = split_page(NEW)
    assert st == [
        "Christine Lagarde",
        "Good afternoon.",
        "We decided today to keep the three key ECB interest rates unchanged.",
        "We are now ready to take your questions.",
    ]
    assert qa == ["Are we in a good place?", "Yes, we are."]
    st2, qa2 = split_page(OLD)
    assert st2[-1] == "We are now at your disposal for questions."
    assert qa2 == ["Question: Will rates rise?", "Duisenberg: We never pre-commit."]


def test_listing_dates_from_file_names() -> None:
    html = (
        '<dt isoDate="2025-03-06"><a href="/press/press_conference/monetary-policy-statement/2025/'
        'shared/pdf/ecb.ds250306.en.pdf">pdf</a></dt>'
        '<a href="/press/press_conference/monetary-policy-statement/2025/html/ecb.is250130~ab.en.html">x</a>'
        '<a href="/press/press_conference/monetary-policy-statement/2025/html/ecb.is250130~ab.de.html">de</a>'
        '<a href="/press/press_conference/monetary-policy-statement/1999/html/is990107.en.html">old</a>'
    )
    assert listing(html) == [
        (
            date(1999, 1, 7),
            "/press/press_conference/monetary-policy-statement/1999/html/is990107.en.html",
        ),
        (
            date(2025, 1, 30),
            "/press/press_conference/monetary-policy-statement/2025/html/ecb.is250130~ab.en.html",
        ),
    ]


async def test_sync_and_tools(tmp_path: Path) -> None:
    p1 = "/press/press_conference/monetary-policy-statement/2025/html/ecb.is250130~a.en.html"
    p2 = "/press/press_conference/monetary-policy-statement/2025/html/ecb.is250306~b.en.html"
    second = NEW.replace(
        "keep the three key ECB interest rates unchanged",
        "lower the deposit facility rate by 25 basis points",
    )
    pages = {
        LIST.format(year=2025): f'<a href="{p1}">a</a><a href="{p2}">b</a>',
        p1: NEW,
        p2: second,
    }
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        body = pages.get(req.url.path)
        return httpx.Response(200, text=body) if body else httpx.Response(404)

    def http() -> HttpClient:
        return HttpClient(
            base_url=BASE, max_calls=100, period_s=1.0, transport=httpx.MockTransport(handler)
        )

    data = tmp_path / "data"
    async with EcbTextConnector(data, http=http()) as conn:
        out = await conn.sync(years=[2025])
    assert out == {"new": 2, "pages": 2, "failed": []}
    docs = list(DocStore(data).iter(source="ecb"))
    assert {d.kind for d in docs} == {"statement", "press_conference"}
    assert docs[0].published_at == datetime(2025, 1, 30, 14, 0, tzinfo=UTC)  # 15:00 CET
    n = len(calls)
    async with EcbTextConnector(data, http=http()) as conn:
        assert (await conn.sync(years=[2025]))["new"] == 0
    assert len(calls) == n + 1  # only the listing: stored meetings are not refetched

    kit = ToolKit(data, REPO / "config")
    st = kit.get_ecb_statement("")
    assert st["meeting_date"] == "2025-03-06"
    assert any("deposit facility" in x for x in st["changes_since_previous"]["added"])
    assert st["press_conference_opening"] == ["Are we in a good place?", "Yes, we are."]
    assert kit.get_ecb_statement("2025-01-01")["error"]
    res = kit.search_documents("deposit facility rate", "ecb", "statement", "", 3)
    assert res["passages"][0]["bank"] == "ecb"
    assert kit.search_documents("deposit facility rate", "fomc", "any", "", 3)["passages"] == []
