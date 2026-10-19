from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from atlas.alerts import (
    Alert,
    AlertConfig,
    AlertInputs,
    AlertState,
    Channels,
    Rule,
    check_ratio,
    check_regime_cross,
    check_series_change,
    check_view_flips,
    dispatch,
    evaluate,
    load_state,
)

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)


def test_repo_config_loads() -> None:
    cfg = AlertConfig.load(REPO / "config" / "alerts.yaml")
    kinds = {r.kind for r in cfg.rules}
    assert kinds == {
        "regime_prob_cross", "recession_prob", "series_change", "ratio", "view_flip",
        "pipeline_failure",
    }  # fmt: skip
    assert cfg.channels.ntfy_topic_url == ""  # external channels ship disabled


def test_regime_cross_both_directions() -> None:
    rule = Rule(name="x", kind="regime_prob_cross", thresholds=[0.5, 0.7])
    alerts = check_regime_cross(rule, {"a": 0.45, "b": 0.75}, {"a": 0.72, "b": 0.6})
    keys = sorted(a.key for a in alerts)
    assert keys == ["x:a:0.5:up", "x:a:0.7:up", "x:b:0.7:down"]
    assert check_regime_cross(rule, {}, {"a": 0.9}) == []  # first run: nothing to cross from


def test_series_change_and_ratio() -> None:
    d0 = date(2026, 10, 1)
    obs = [(d0 + timedelta(days=i), 3.0 + (0.6 if i >= 8 else 0.0)) for i in range(10)]
    rule = Rule(name="hy", kind="series_change", series_id="HY", days=7, threshold=0.5)
    (a,) = check_series_change(rule, obs)
    assert a.value == pytest.approx(0.6)
    assert a.key == "hy:HY:up"
    assert check_series_change(rule, obs[:5]) == []  # no observation a week back

    r = Rule(name="vix", kind="ratio", numerator="V", denominator="V3", threshold=1.0)
    assert check_ratio(r, [(d0, 25.0)], [(d0, 20.0)])[0].value == pytest.approx(1.25)
    assert check_ratio(r, [(d0, 15.0)], [(d0, 20.0)]) == []


def _view(asset: str, stance: str, as_of: str) -> dict[str, object]:
    return {"asset_id": asset, "asset_name": asset, "stance": stance, "as_of": as_of, "score": 1.0}


def test_view_flips() -> None:
    old = [_view("A", "overweight", "2026-08-01"), _view("B", "neutral", "2026-08-01")]
    new = [_view("A", "underweight", "2026-09-01"), _view("B", "overweight", "2026-09-01")]
    strict = Rule(name="f", kind="view_flip")
    assert [a.key for a in check_view_flips(strict, "u", [old, new])] == ["f:u:A:2026-09-01"]
    loose = Rule(name="f", kind="view_flip", include_neutral=True)
    assert len(check_view_flips(loose, "u", [old, new])) == 2
    assert check_view_flips(strict, "u", [new]) == []


def test_evaluate_and_dispatch_cooldown_rate_limit(tmp_path: Path) -> None:
    cfg = AlertConfig(
        cooldown_days=7,
        max_per_run=1,
        rules=[
            Rule(name="rec", kind="recession_prob", threshold=0.4),
            Rule(name="dq", kind="pipeline_failure"),
        ],
    )
    inputs = AlertInputs(recession_prob=0.55, failed_steps=["ingest:fred"])
    alerts = evaluate(cfg, inputs, {}, NOW.date())
    assert [a.rule for a in alerts] == ["rec", "dq"]
    state = AlertState()
    out = dispatch(cfg, alerts, state, tmp_path, NOW)
    assert [a.status for a in out] == ["sent", "rate_limited"]
    # Next day: recession alert is in cooldown, data-quality alert (new key per day) is sent.
    again = evaluate(cfg, inputs, {}, (NOW + timedelta(days=1)).date())
    out2 = dispatch(cfg, again, load_state(tmp_path), tmp_path, NOW + timedelta(days=1))
    assert [a.status for a in out2] == ["cooldown", "sent"]
    # After the cooldown it fires again.
    later = NOW + timedelta(days=8)
    out3 = dispatch(
        cfg, evaluate(cfg, inputs, {}, later.date()), load_state(tmp_path), tmp_path, later
    )
    assert out3[0].status == "sent"
    assert (tmp_path / "alerts" / f"{NOW.date()}.jsonl").read_text().count("\n") == 2


def test_delivery_channels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(200 if "ntfy" in str(req.url) else 500)

    monkeypatch.setenv("TEST_SLACK_URL", "https://hooks.example.test/x")
    cfg = AlertConfig(
        channels=Channels(
            ntfy_topic_url="https://ntfy.example.test/t", slack_webhook_env="TEST_SLACK_URL"
        )
    )
    a = Alert(rule="r", key="k", message="hello")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        (out,) = dispatch(cfg, [a], AlertState(), tmp_path, NOW, client=client)
    assert out.delivery == {"ntfy": "ok", "slack": "http 500"}
    assert len(seen) == 2
