from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from atlas.api import create_app
from atlas.settings import Settings
from tests.test_country import _country_store

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("apidata")
    _country_store(d)
    views = d / "gold" / "views" / "ctry_dev_lcl" / "3M"
    views.mkdir(parents=True)
    (views / "2026-09-01.json").write_text(
        json.dumps([{"view_id": "v-123", "asset_id": "COUNTRY.JPN", "stance": "neutral"}])
    )
    (d / "alerts").mkdir()
    (d / "alerts" / "2999-01-01.jsonl").write_text(json.dumps({"rule": "r", "message": "m"}) + "\n")
    return d


@pytest.fixture(scope="module")
def client(data: Path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=data, config_dir=REPO / "config")))


def test_health_and_views(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/v1/views", params={"universe": "ctry_dev_lcl", "horizon": 3})
    assert r.status_code == 200
    assert len(r.json()["views"]) == 9


def test_errors_map_to_status_codes(client: TestClient) -> None:
    assert client.get("/v1/views", params={"universe": "nope"}).status_code == 422
    assert client.get("/v1/regime").status_code == 404  # no FRED data in this store
    assert client.get("/v1/series/NOT.A.SERIES").status_code == 404
    assert client.get("/v1/snapshots/..%2Fsecret").status_code in {404, 422}
    r = client.post("/v1/scenario", json={"universe": "ctry_dev_lcl", "shocks": {"us10y": 1}})
    assert r.status_code == 404  # scenario factors need FRED data


def test_frozen_views_and_alerts(client: TestClient) -> None:
    r = client.get("/v1/views/frozen", params={"universe": "ctry_dev_lcl"})
    assert r.json()["months"][0]["as_of"] == "2026-09-01"
    assert client.get("/v1/views/v-123").json()["asset_id"] == "COUNTRY.JPN"
    assert client.get("/v1/views/missing").status_code == 404
    assert client.get("/v1/alerts").json()["alerts"] == [{"rule": "r", "message": "m"}]


def test_bearer_token(data: Path) -> None:
    app = create_app(
        Settings(data_dir=data, config_dir=REPO / "config", api_token=SecretStr("s3cret"))
    )
    c = TestClient(app)
    assert c.get("/health").status_code == 200  # health stays open
    assert c.get("/v1/universes").status_code == 401
    assert c.get("/v1/universes", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = c.get("/v1/universes", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert ok.json()["universes"]


def test_chat_streams_tool_and_answer_events(data: Path) -> None:
    from tests.test_agent import FakeClient, response, text_block, tool_block

    fake = FakeClient(
        [
            response("tool_use", tool_block("t1", "list_universes", {})),
            response("end_turn", text_block("There are 4 universes.")),
        ]
    )
    app = create_app(Settings(data_dir=data, config_dir=REPO / "config"), llm_client=fake)
    with TestClient(app).stream("POST", "/v1/chat", json={"question": "What universes?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [blk.split("\n") for blk in body.strip().split("\n\n")]
    names = [e[0].removeprefix("event: ") for e in events]
    assert names == ["tool", "answer"]
    answer = json.loads(events[1][1].removeprefix("data: "))
    assert answer["grounded"] is True
    assert "not investment advice" in answer["disclaimer"]


def test_chat_budget_error(data: Path, tmp_path: Path) -> None:
    from atlas.agent.usage import MeteredClient, UsageLedger
    from tests.test_agent import FakeClient

    ledger = UsageLedger(tmp_path)
    metered = MeteredClient(FakeClient([]), ledger, daily_budget=1, purpose="chat")
    ledger.record("chat", "m", type("U", (), {"input_tokens": 5})())
    app = create_app(Settings(data_dir=data, config_dir=REPO / "config"), llm_client=metered)
    with TestClient(app).stream("POST", "/v1/chat", json={"question": "hi"}) as r:
        body = "".join(r.iter_text())
    assert body.startswith("event: error")
    assert '"kind": "budget"' in body


def test_web_app_and_central_banks(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/app/"
    page = client.get("/app/")
    assert page.status_code == 200
    assert "<title>Ramon</title>" in page.text
    assert client.get("/app/app.js").status_code == 200
    cb = client.get("/v1/central-banks").json()
    assert set(cb) == {"fomc", "ecb"}  # errors (no documents stored) are reported per bank


def test_chat_passes_history(data: Path) -> None:
    from tests.test_agent import FakeClient, response, text_block

    fake = FakeClient([response("end_turn", text_block("No numbers here."))])
    app = create_app(Settings(data_dir=data, config_dir=REPO / "config"), llm_client=fake)
    body = {
        "question": "And Japan?",
        "history": [{"question": "View on Germany?", "answer": "Neutral."}],
    }
    with TestClient(app).stream("POST", "/v1/chat", json=body) as r:
        "".join(r.iter_text())
    msgs = fake.requests[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "View on Germany?"
    assert msgs[-1]["content"] == "And Japan?"


def test_chat_executive_style_and_briefing(data: Path) -> None:
    from tests.test_agent import FakeClient, response, text_block

    fake = FakeClient([response("end_turn", text_block("No numbers here."))])
    app = create_app(Settings(data_dir=data, config_dir=REPO / "config"), llm_client=fake)
    client = TestClient(app)
    with client.stream("POST", "/v1/chat", json={"question": "Hi", "style": "executive"}) as r:
        "".join(r.iter_text())
    system = fake.requests[0]["system"]
    assert len(system) == 2
    assert "executive" in system[1]["text"]
    assert client.post("/v1/chat", json={"question": "Hi", "style": "pirate"}).status_code == 422
    b = client.get("/v1/briefing").json()
    assert {"economy", "central_banks", "leans", "alerts", "trust_note"} <= set(b)


def test_presets(data: Path) -> None:
    app = create_app(Settings(data_dir=data, config_dir=REPO / "config"))
    ps = TestClient(app).get("/v1/presets").json()["presets"]
    assert {p["id"] for p in ps} >= {"oil_up", "rates_up", "panic"}
    assert all(p["when"] and p["shocks"] for p in ps)


def test_briefing_cached_on_disk_across_restarts(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    import atlas.briefing

    data = shutil.copytree(data, tmp_path / "d", ignore=shutil.ignore_patterns("cache"))

    calls: list[int] = []

    def fake(kit: object, alerts: object, today: object = None) -> dict[str, object]:
        calls.append(1)
        return {"economy": None, "n": len(calls)}

    monkeypatch.setattr(atlas.briefing, "build_briefing", fake)
    settings = Settings(data_dir=data, config_dir=REPO / "config")
    first = TestClient(create_app(settings)).get("/v1/briefing").json()
    second = TestClient(create_app(settings)).get("/v1/briefing").json()  # a "restart"
    assert first == second == {"economy": None, "n": 1}
    assert len(list((data / "cache" / "web").glob("*_briefing.json"))) == 1
