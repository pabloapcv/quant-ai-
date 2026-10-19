"""LLM cost controls (PRD §10.6): a usage ledger and a daily token budget with a hard stop.

Every Claude call made through ``MeteredClient`` is appended to
``data/usage/<UTC date>.jsonl`` with its purpose (ask, report, eval, chat), model and token
counts. Before each call the day's total is compared with the budget; once it is reached,
calls raise ``BudgetExceeded`` until the next UTC day.

The budget counts *weighted* tokens, roughly in proportion to price: input and cache-write
tokens count 1x, cache reads 0.1x, output tokens 5x. A cached system prompt and tool list are
therefore nearly free on repeat calls, as they are on the bill.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

WEIGHTS = {"input": 1.0, "cache_write": 1.0, "cache_read": 0.1, "output": 5.0}


class BudgetExceeded(RuntimeError):
    pass


class UsageLedger:
    def __init__(self, data_dir: Path) -> None:
        self.folder = data_dir / "usage"
        self._lock = threading.Lock()

    def _path(self, day: date) -> Path:
        return self.folder / f"{day.isoformat()}.jsonl"

    def record(
        self, purpose: str, model: str, usage: Any, now: datetime | None = None
    ) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        counts = {
            "input": int(getattr(usage, "input_tokens", 0) or 0),
            "output": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_write": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            "cache_read": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        }
        weighted = sum(counts[k] * w for k, w in WEIGHTS.items())
        row: dict[str, Any] = {
            "at": now.isoformat(),
            "purpose": purpose,
            "model": model,
            **counts,
            "weighted": weighted,
        }
        with self._lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            with self._path(now.date()).open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return row

    def rows(self, day: date) -> list[dict[str, Any]]:
        p = self._path(day)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line]

    def totals(self, day: date) -> dict[str, Any]:
        rows = self.rows(day)
        out: dict[str, Any] = {k: sum(r[k] for r in rows) for k in (*WEIGHTS, "weighted")}
        out["calls"] = len(rows)
        out["by_purpose"] = {}
        for r in rows:
            out["by_purpose"][r["purpose"]] = out["by_purpose"].get(r["purpose"], 0) + r["weighted"]
        cached = out["cache_read"] + out["cache_write"] + out["input"]
        out["cache_hit_rate"] = out["cache_read"] / cached if cached else None
        return out


class MeteredClient:
    """Drop-in for ``anthropic.Anthropic`` where only ``.beta.messages.create`` is used."""

    def __init__(self, client: Any, ledger: UsageLedger, daily_budget: int, purpose: str) -> None:
        self.client = client
        self.ledger = ledger
        self.daily_budget = daily_budget
        self.purpose = purpose
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def spent_today(self) -> float:
        return float(self.ledger.totals(datetime.now(UTC).date())["weighted"])

    def _create(self, **kwargs: Any) -> Any:
        spent = self.spent_today()
        if self.daily_budget > 0 and spent >= self.daily_budget:
            raise BudgetExceeded(
                f"daily LLM budget reached ({spent:,.0f} of {self.daily_budget:,} weighted "
                "tokens); it resets at 00:00 UTC (ATLAS_LLM_DAILY_TOKEN_BUDGET)"
            )
        resp = self.client.beta.messages.create(**kwargs)
        self.ledger.record(
            self.purpose,
            getattr(resp, "model", kwargs.get("model", "")),
            getattr(resp, "usage", None),
        )
        return resp


def make_client(settings: Any, purpose: str) -> MeteredClient:
    """Anthropic client (key from settings, else the SDK's own resolution), metered."""
    import anthropic

    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    raw = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    return MeteredClient(
        raw, UsageLedger(settings.data_dir), settings.llm_daily_token_budget, purpose
    )
