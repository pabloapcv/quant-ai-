"""Operational metrics in Prometheus text format (PRD §12 observability).

Everything is read from what Ramon already records (store knowledge times, the latest
pipeline run, the usage ledger, alert logs, report metadata, the document store), so a
scrape costs a few file reads and no computation. Served at ``GET /metrics`` by the API and
printed by ``atlas metrics``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.catalog import Catalog


@dataclass
class Metric:
    name: str
    help: str
    kind: str  # gauge | counter
    samples: list[tuple[dict[str, str], float]] = field(default_factory=list)

    def add(self, value: float | None, **labels: str) -> None:
        if value is not None and math.isfinite(float(value)):
            self.samples.append((labels, float(value)))


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render(metrics: Iterable[Metric]) -> str:
    lines: list[str] = []
    for m in metrics:
        if not m.samples:
            continue
        lines += [f"# HELP {m.name} {m.help}", f"# TYPE {m.name} {m.kind}"]
        for labels, value in m.samples:
            lab = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
            lines.append(f"{m.name}{{{lab}}} {value:.15g}" if lab else f"{m.name} {value:.15g}")
    return "\n".join(lines) + "\n"


def _latest_json(folder: Path, pattern: str) -> dict[str, Any] | None:
    files = sorted(folder.glob(pattern))
    return json.loads(files[-1].read_text()) if files else None


def collect(
    data_dir: Path, config_dir: Path, *, daily_budget: int, now: datetime | None = None
) -> list[Metric]:
    from atlas.agent.usage import UsageLedger
    from atlas.pit.store import PITStore

    now = now or datetime.now(UTC)
    out: list[Metric] = []

    # data freshness per source
    fresh = Metric(
        "atlas_source_data_age_seconds",
        "Seconds since the newest observation of any series from the source became known",
        "gauge",
    )
    catalog = Catalog.load(config_dir / "series.yaml")
    source_of = {s.id: s.source for s in catalog}
    newest: dict[str, datetime] = {}
    for sid, t in PITStore(data_dir).last_knowledge_times().items():
        src = source_of.get(sid)
        if src and t is not None and (src not in newest or t > newest[src]):
            newest[src] = t
    for src, t in sorted(newest.items()):
        fresh.add((now - t).total_seconds(), source=src)
    out.append(fresh)

    run = _latest_json(data_dir / "runs", "2*.json")
    if run:
        started = datetime.fromisoformat(run["started_at"])
        out.append(
            m := Metric(
                "atlas_pipeline_last_run_timestamp_seconds",
                "Start time of the latest daily pipeline run",
                "gauge",
            )
        )
        m.add(started.timestamp())
        out.append(
            m := Metric(
                "atlas_pipeline_last_run_ok", "1 if every step of the latest run succeeded", "gauge"
            )
        )
        m.add(1.0 if run.get("ok") else 0.0)
        ok = Metric("atlas_pipeline_step_ok", "1 if the step succeeded in the latest run", "gauge")
        secs = Metric(
            "atlas_pipeline_step_seconds", "Duration of the step in the latest run", "gauge"
        )
        for st in run.get("steps", []):
            ok.add(1.0 if st["ok"] else 0.0, step=st["name"])
            secs.add(st["seconds"], step=st["name"])
        out += [ok, secs]
        out.append(
            m := Metric(
                "atlas_stale_series", "Series past their staleness limit in the latest run", "gauge"
            )
        )
        m.add(len(run.get("stale", [])))
        track = run.get("track", {})
        out.append(
            m := Metric(
                "atlas_live_track_months", "Frozen view-months by model and status", "gauge"
            )
        )
        models = {"ens_v1": track, **run.get("track_challengers", {})}
        for model, tr in models.items():
            m.add(tr.get("months_scored"), model=model, status="scored")
            m.add(tr.get("months_pending"), model=model, status="pending")
        ic = Metric("atlas_live_track_rank_ic", "Mean rank IC of scored live view-months", "gauge")
        for model, tr in models.items():
            ic.add(tr.get("mean_rank_ic"), model=model)
        out.append(ic)
        psi = Metric(
            "atlas_feature_psi",
            "Population stability index of the most-shifted regime inputs (diagnostic)",
            "gauge",
        )
        for feat, v in run.get("feature_psi", {}).items():
            psi.add(v, feature=feat)
        out.append(psi)

    usage = UsageLedger(data_dir).totals(now.date())
    out.append(
        m := Metric(
            "atlas_llm_weighted_tokens_today", "Claude usage today in weighted tokens", "gauge"
        )
    )
    m.add(usage["weighted"])
    out.append(m := Metric("atlas_llm_calls_today", "Claude calls today", "gauge"))
    m.add(usage["calls"])
    out.append(
        m := Metric(
            "atlas_llm_daily_budget_tokens", "Daily weighted-token budget (0 = none)", "gauge"
        )
    )
    m.add(daily_budget)
    out.append(
        m := Metric(
            "atlas_llm_cache_hit_ratio",
            "Share of input tokens read from the prompt cache today",
            "gauge",
        )
    )
    m.add(usage["cache_hit_rate"])

    alerts = Metric("atlas_alerts_today", "Alerts triggered today by status", "gauge")
    log = data_dir / "alerts" / f"{now.date().isoformat()}.jsonl"
    counts: dict[str, int] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            if line:
                status = json.loads(line).get("status", "unknown")
                counts[status] = counts.get(status, 0) + 1
    for status, n in sorted(counts.items()):
        alerts.add(n, status=status)
    out.append(alerts)

    reports = sorted(p for p in (data_dir / "reports").glob("2*/meta.json"))
    if reports:
        meta = json.loads(reports[-1].read_text())
        out.append(
            m := Metric(
                "atlas_report_unsupported_numbers",
                "Numbers in the latest weekly note not traceable to engine facts",
                "gauge",
            )
        )
        m.add(len(meta.get("unsupported_numbers", [])))

    docs = Metric("atlas_documents", "Stored central-bank documents by kind", "gauge")
    for kind in ("statement", "minutes"):
        folder = data_dir / "text" / "fomc" / kind
        docs.add(len(list(folder.glob("*.json"))) if folder.exists() else 0, kind=kind)
    out.append(docs)
    return out
