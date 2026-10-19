"""Alert evaluator (PRD §11.3): rules from ``config/alerts.yaml``, run after the pipeline.

Each rule kind is a small pure check over inputs gathered once per run (regime report,
point-in-time series, frozen views, the pipeline report). Alerts carry a ``key`` (rule +
subject + direction); a key that fired within ``cooldown_days`` is not sent again, and at most
``max_per_run`` are sent per run. State (previous quadrant probabilities, recent keys) lives
in ``data/alerts/state.json``; every alert, sent or suppressed, is appended to
``data/alerts/<date>.jsonl``.

Channels: the log file always; ntfy and Slack only when configured. Delivery failures are
recorded on the alert, never raised.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import polars as pl
import yaml
from pydantic import BaseModel, Field

from atlas import jsonsafe

Kind = Literal[
    "regime_prob_cross", "recession_prob", "series_change", "ratio", "view_flip",
    "pipeline_failure", "live_ic_drift",
]  # fmt: skip


class Rule(BaseModel):
    name: str
    kind: Kind
    threshold: float | None = None
    thresholds: list[float] = Field(default_factory=list)
    series_id: str | None = None
    days: int = 7
    numerator: str | None = None
    denominator: str | None = None
    horizon: int = 3
    include_neutral: bool = False
    window: int = 6
    sigmas: float = 2.0


class Channels(BaseModel):
    ntfy_topic_url: str = ""
    slack_webhook_env: str = "ATLAS_SLACK_WEBHOOK_URL"


class AlertConfig(BaseModel):
    cooldown_days: int = 7
    max_per_run: int = 10
    channels: Channels = Field(default_factory=Channels)
    rules: list[Rule] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> AlertConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})


class Alert(BaseModel):
    rule: str
    key: str
    message: str
    value: float | None = None
    status: Literal["new", "sent", "cooldown", "rate_limited"] = "new"
    delivery: dict[str, str] = Field(default_factory=dict)


class AlertInputs(BaseModel):
    """Everything the rules look at, gathered once per run."""

    quadrant_probs: dict[str, float] = Field(default_factory=dict)
    recession_prob: float | None = None
    series: dict[str, list[tuple[date, float]]] = Field(default_factory=dict)
    views: dict[str, list[list[dict[str, Any]]]] = Field(default_factory=dict)  # newest last
    failed_steps: list[str] = Field(default_factory=list)
    ic_monitor: dict[str, dict[str, Any]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------- checks
def check_regime_cross(
    rule: Rule, prev: Mapping[str, float], cur: Mapping[str, float]
) -> list[Alert]:
    out = []
    for quad, p in cur.items():
        q = prev.get(quad)
        if q is None:
            continue
        for t in rule.thresholds:
            if q < t <= p or p < t <= q:
                up = p >= t
                out.append(
                    Alert(
                        rule=rule.name,
                        key=f"{rule.name}:{quad}:{t}:{'up' if up else 'down'}",
                        message=f"{quad} probability {'rose above' if up else 'fell below'} "
                        f"{t:.0%}: {q:.0%} -> {p:.0%}",
                        value=p,
                    )
                )
    return out


def check_recession(rule: Rule, prob: float | None) -> list[Alert]:
    t = rule.threshold if rule.threshold is not None else 0.4
    if prob is None or prob <= t:
        return []
    return [
        Alert(
            rule=rule.name,
            key=f"{rule.name}:above:{t}",
            message=f"12-month recession probability {prob:.0%} is above {t:.0%}",
            value=prob,
        )
    ]


def check_series_change(rule: Rule, obs: Sequence[tuple[date, float]]) -> list[Alert]:
    if not obs or rule.threshold is None:
        return []
    obs = sorted(obs)
    last_d, last_v = obs[-1]
    base = [v for d, v in obs if d <= last_d - timedelta(days=rule.days)]
    if not base:
        return []
    chg = last_v - base[-1]
    if abs(chg) < rule.threshold:
        return []
    up = chg > 0
    return [
        Alert(
            rule=rule.name,
            key=f"{rule.name}:{rule.series_id}:{'up' if up else 'down'}",
            message=f"{rule.series_id} moved {chg:+.2f} in {rule.days} days "
            f"(to {last_v:.2f} on {last_d})",
            value=chg,
        )
    ]


def check_ratio(
    rule: Rule, num: Sequence[tuple[date, float]], den: Sequence[tuple[date, float]]
) -> list[Alert]:
    if not num or not den or rule.threshold is None:
        return []
    common = sorted(set(dict(num)) & set(dict(den)))
    if not common:
        return []
    d = common[-1]
    ratio = dict(num)[d] / dict(den)[d]
    if ratio <= rule.threshold:
        return []
    return [
        Alert(
            rule=rule.name,
            key=f"{rule.name}:above",
            message=f"{rule.numerator}/{rule.denominator} = {ratio:.2f} on {d} "
            f"(above {rule.threshold:.2f})",
            value=ratio,
        )
    ]


def _sign(stance: str) -> int:
    return 1 if "overweight" in stance else -1 if "underweight" in stance else 0


def check_view_flips(
    rule: Rule, universe: str, files: Sequence[Sequence[Mapping[str, Any]]]
) -> list[Alert]:
    if len(files) < 2:
        return []
    old = {v["asset_id"]: v for v in files[-2]}
    out = []
    for v in files[-1]:
        o = old.get(v["asset_id"])
        if o is None:
            continue
        a, b = _sign(o["stance"]), _sign(v["stance"])
        flipped = a != b if rule.include_neutral else a * b == -1
        if flipped:
            out.append(
                Alert(
                    rule=rule.name,
                    key=f"{rule.name}:{universe}:{v['asset_id']}:{v['as_of']}",
                    message=f"{universe} {v['asset_name']}: {o['stance']} -> {v['stance']} "
                    f"({rule.horizon}M view, {v['as_of']})",
                    value=float(v["score"]),
                )
            )
    return out


def check_pipeline(rule: Rule, failed: Sequence[str], today: date) -> list[Alert]:
    if not failed:
        return []
    return [
        Alert(
            rule=rule.name,
            key=f"{rule.name}:{today}",
            message=f"pipeline steps failed: {', '.join(failed)}",
            value=float(len(failed)),
        )
    ]


def check_live_ic(rule: Rule, monitor: Mapping[str, Mapping[str, Any]]) -> list[Alert]:
    from atlas.quality.drift import live_ic_drift

    out = []
    for key, mon in sorted(monitor.items()):
        hit = live_ic_drift(
            mon.get("live_ics", []), mon.get("backtest", {}), window=rule.window, sigmas=rule.sigmas
        )
        if hit:
            out.append(
                Alert(
                    rule=rule.name,
                    key=f"{rule.name}:{key}",
                    message=f"{key} live rank IC {hit['live_mean']:+.3f} over the last "
                    f"{hit['n']} scored months is below the backtest's {hit['backtest_mean']:+.3f} "
                    f"minus {rule.sigmas:g} standard errors ({hit['floor']:+.3f})",
                    value=hit["live_mean"],
                )
            )
    return out


def evaluate(
    cfg: AlertConfig, inputs: AlertInputs, prev_probs: Mapping[str, float], today: date
) -> list[Alert]:
    alerts: list[Alert] = []
    for rule in cfg.rules:
        if rule.kind == "regime_prob_cross":
            alerts += check_regime_cross(rule, prev_probs, inputs.quadrant_probs)
        elif rule.kind == "recession_prob":
            alerts += check_recession(rule, inputs.recession_prob)
        elif rule.kind == "series_change" and rule.series_id:
            alerts += check_series_change(rule, inputs.series.get(rule.series_id, []))
        elif rule.kind == "ratio" and rule.numerator and rule.denominator:
            alerts += check_ratio(
                rule,
                inputs.series.get(rule.numerator, []),
                inputs.series.get(rule.denominator, []),
            )
        elif rule.kind == "view_flip":
            for universe, files in sorted(inputs.views.items()):
                alerts += check_view_flips(rule, universe, files)
        elif rule.kind == "pipeline_failure":
            alerts += check_pipeline(rule, inputs.failed_steps, today)
        elif rule.kind == "live_ic_drift":
            alerts += check_live_ic(rule, inputs.ic_monitor)
    return alerts


# ---------------------------------------------------------------------------- inputs
def series_needed(cfg: AlertConfig) -> list[str]:
    ids = {r.series_id for r in cfg.rules} | {r.numerator for r in cfg.rules}
    ids |= {r.denominator for r in cfg.rules}
    return sorted(i for i in ids if i)


def load_views(
    data_dir: Path, horizon: int, keep: int = 2
) -> dict[str, list[list[dict[str, Any]]]]:
    out: dict[str, list[list[dict[str, Any]]]] = {}
    for d in sorted((data_dir / "gold" / "views").glob(f"*/{horizon}M")):
        files = sorted(d.glob("*.json"))[-keep:]
        if files:
            out[d.parent.name] = [json.loads(f.read_text()) for f in files]
    return out


def frame_to_obs(df: pl.DataFrame, lookback_days: int = 60) -> list[tuple[date, float]]:
    df = df.drop_nulls("value").sort("event_time")
    if df.is_empty():
        return []
    start = df["event_time"].max() - timedelta(days=lookback_days)  # type: ignore[operator]
    tail = df.filter(pl.col("event_time") >= start)
    return list(zip(tail["event_time"].to_list(), tail["value"].to_list(), strict=True))


# ---------------------------------------------------------------------------- dispatch
class AlertState(BaseModel):
    quadrant_probs: dict[str, float] = Field(default_factory=dict)
    fired: dict[str, datetime] = Field(default_factory=dict)  # key -> last sent


def _state_path(data_dir: Path) -> Path:
    return data_dir / "alerts" / "state.json"


def load_state(data_dir: Path) -> AlertState:
    p = _state_path(data_dir)
    return AlertState.model_validate_json(p.read_text()) if p.exists() else AlertState()


def dispatch(
    cfg: AlertConfig,
    alerts: list[Alert],
    state: AlertState,
    data_dir: Path,
    now: datetime,
    *,
    client: httpx.Client | None = None,
) -> list[Alert]:
    """Apply cooldown and rate limit, deliver, log, and update ``state`` in place."""
    cutoff = now - timedelta(days=cfg.cooldown_days)
    state.fired = {k: t for k, t in state.fired.items() if t > cutoff}
    sent = 0
    for a in alerts:
        if a.key in state.fired:
            a.status = "cooldown"
        elif sent >= cfg.max_per_run:
            a.status = "rate_limited"
        else:
            a.status = "sent"
            a.delivery = _deliver(cfg.channels, a, client)
            state.fired[a.key] = now
            sent += 1
    folder = data_dir / "alerts"
    folder.mkdir(parents=True, exist_ok=True)
    if alerts:
        with (folder / f"{now.date().isoformat()}.jsonl").open("a") as fh:
            for a in alerts:
                fh.write(jsonsafe.dumps({"at": now.isoformat(), **a.model_dump()}) + "\n")
    _state_path(data_dir).write_text(state.model_dump_json(indent=2))
    return alerts


def _deliver(channels: Channels, alert: Alert, client: httpx.Client | None) -> dict[str, str]:
    targets: dict[str, tuple[str, dict[str, Any]]] = {}
    if channels.ntfy_topic_url:
        targets["ntfy"] = (channels.ntfy_topic_url, {"content": alert.message.encode()})
    slack = os.environ.get(channels.slack_webhook_env, "") if channels.slack_webhook_env else ""
    if slack:
        targets["slack"] = (slack, {"json": {"text": f"Atlas: {alert.message}"}})
    if not targets:
        return {}
    own = client is None
    http = client or httpx.Client(timeout=10.0)
    out: dict[str, str] = {}
    try:
        for name, (url, body) in targets.items():
            try:
                r = http.post(url, **body)
                out[name] = "ok" if r.is_success else f"http {r.status_code}"
            except httpx.HTTPError as exc:
                out[name] = f"error: {type(exc).__name__}"
    finally:
        if own:
            http.close()
    return out


def run_alerts(
    data_dir: Path,
    config_dir: Path,
    *,
    failed_steps: Sequence[str] = (),
    now: datetime | None = None,
    client: httpx.Client | None = None,
    ic_monitor: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Alert]:
    """Gather inputs from the store, evaluate every rule, dispatch.

    ``ic_monitor`` comes from the pipeline run; when absent (``atlas alerts check``) the latest
    saved run's is used.
    """
    from atlas.models.regime.engine import RegimeEngine
    from atlas.pit.store import PITStore

    now = now or datetime.now(UTC)
    path = config_dir / "alerts.yaml"
    if not path.exists():
        return []
    cfg = AlertConfig.load(path)
    store = PITStore(data_dir)
    cube = store.load_cube()
    if ic_monitor is None:
        runs = sorted((data_dir / "runs").glob("*.json"))
        ic_monitor = json.loads(runs[-1].read_text()).get("ic_monitor", {}) if runs else {}
    inputs = AlertInputs(
        failed_steps=list(failed_steps), ic_monitor={k: dict(v) for k, v in ic_monitor.items()}
    )
    if "US.IP" in cube:
        rep = RegimeEngine(cube).run(now.date(), with_hmm=False)
        inputs.quadrant_probs = {str(k): float(v) for k, v in rep.quadrant_probs.items()}
        inputs.recession_prob = rep.recession.prob_12m if rep.recession else None
    for sid in series_needed(cfg):
        if sid in cube:
            inputs.series[sid] = frame_to_obs(cube.as_of(sid, now))
    for h in sorted({r.horizon for r in cfg.rules if r.kind == "view_flip"}):
        inputs.views.update(load_views(data_dir, h))

    state = load_state(data_dir)
    alerts = evaluate(cfg, inputs, state.quadrant_probs, now.date())
    if inputs.quadrant_probs:
        state.quadrant_probs = inputs.quadrant_probs
    return dispatch(cfg, alerts, state, data_dir, now, client=client)
