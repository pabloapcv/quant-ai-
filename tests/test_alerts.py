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
        "pipeline_failure", "live_ic_drift", "cb_decision", "price_move",
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


def test_live_ic_rule() -> None:
    cfg = AlertConfig(rules=[Rule(name="ic", kind="live_ic_drift", window=6, sigmas=2.0)])
    monitor = {
        "ind12:3M": {"backtest": {"mean": 0.05, "sd": 0.2, "n": 600}, "live_ics": [-0.2] * 6},
        "ind49:3M": {"backtest": {"mean": 0.05, "sd": 0.2, "n": 600}, "live_ics": [0.1] * 6},
        "ctry:3M": {"live_ics": [-0.5] * 6},  # no backtest stats: never flagged
    }
    alerts = evaluate(cfg, AlertInputs(ic_monitor=monitor), {}, NOW.date())
    assert [a.key for a in alerts] == ["ic:ind12:3M"]
    assert "below the backtest" in alerts[0].message


def test_cb_decision_rule(tmp_path: Path) -> None:
    from atlas.alerts import recent_cb_statements
    from atlas.text.documents import DocStore, Document, decision

    def doc(source: str, day: date, text: str) -> Document:
        return Document(
            doc_id=f"{source}-statement-{day}", source=source, kind="statement", title="t", url="u",
            published_at=datetime(day.year, day.month, day.day, 18, tzinfo=UTC), meeting_date=day,
            paragraphs=[text, "Voting for the action were A."],
        )  # fmt: skip

    store = DocStore(tmp_path)
    old = doc(
        "fomc",
        date(2026, 9, 16),
        "The Committee decided to maintain the target range for the federal funds rate at 3-1/2 to 3-3/4 percent.",
    )
    new = doc(
        "fomc",
        date(2026, 10, 28),
        "The Committee decided to raise the target range for the federal funds rate to 3-3/4 to 4 percent.",
    )
    ecb = doc(
        "ecb",
        date(2026, 10, 29),
        "The Governing Council today decided to keep the three key ECB interest rates unchanged.",
    )
    guide = doc(
        "ecb",
        date(2026, 10, 30),
        "We expect to raise the key ECB interest rates in December. Inflation is high.",
    )
    for d in (old, new, ecb, guide):
        store.save(d)
    assert decision(new, old) == ("raise", "federal funds target 4.0% (+25bp)")
    ecb_decision = decision(ecb)
    assert ecb_decision is not None
    assert ecb_decision[0] == "hold"
    assert decision(guide) is None  # forward guidance is not a decision

    now = datetime(2026, 10, 30, 23, tzinfo=UTC)
    recent = recent_cb_statements(tmp_path, now, days=3)
    assert [r["doc_id"] for r in recent] == [
        "fomc-statement-2026-10-28",
        "ecb-statement-2026-10-29",
    ]
    cfg = AlertConfig(rules=[Rule(name="cb", kind="cb_decision", days=3)])
    alerts = evaluate(cfg, AlertInputs(cb_statements=recent), {}, now.date())
    assert [a.key for a in alerts] == ["cb:fomc-statement-2026-10-28"]  # holds stay quiet
    assert "FOMC 2026-10-28" in alerts[0].message
    holds = AlertConfig(rules=[Rule(name="cb", kind="cb_decision", days=3, include_holds=True)])
    assert len(evaluate(holds, AlertInputs(cb_statements=recent), {}, now.date())) == 2


def test_price_move_alert() -> None:
    from atlas.alerts import Rule, check_price_move

    rule = Rule(name="price_move_3m", kind="price_move", days=92, threshold=0.2)
    obs = [(date(2026, m, 1), v) for m, v in ((5, 100.0), (6, 100.0), (7, 110.0), (9, 125.0))]
    (a,) = check_price_move(rule, "CMDTY.WHEAT", "Wheat", obs)
    assert a.message == "Wheat prices rose 25% over the past 3 months (latest September 2026)."
    assert a.key == "price_move_3m:CMDTY.WHEAT:up"
    assert check_price_move(rule, "CMDTY.WHEAT", "Wheat", obs[:-1]) == []


def test_send_text_posts_to_ntfy_and_slack(monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.alerts import Channels, send_text

    seen: list[tuple[str, bytes, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.content, dict(request.headers)))
        return httpx.Response(200)

    monkeypatch.setenv("RAMON_TEST_SLACK", "https://hooks.example/abc")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    ch = Channels(ntfy_topic_url="https://ntfy.example/ramon", slack_webhook_env="RAMON_TEST_SLACK")
    out = send_text(ch, "hello", title="Ramon briefing", client=client)
    assert out == {"ntfy": "ok", "slack": "ok"}
    assert seen[0][0] == "https://ntfy.example/ramon"
    assert seen[0][1] == b"hello"
    assert seen[0][2]["title"] == "Ramon briefing"
    assert b"*Ramon briefing*" in seen[1][1]
    assert send_text(Channels(slack_webhook_env=""), "x") == {}
