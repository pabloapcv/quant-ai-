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
