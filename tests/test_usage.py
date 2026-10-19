from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.agent.assistant import Assistant
from atlas.agent.usage import BudgetExceeded, MeteredClient, UsageLedger
from tests.test_agent import FakeClient, response, text_block, tool_block


def _usage(inp: int, out: int, cw: int = 0, cr: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cw,
        cache_read_input_tokens=cr,
    )


def test_ledger_weights_and_totals(tmp_path: Path) -> None:
    led = UsageLedger(tmp_path)
    now = datetime(2026, 10, 19, 12, tzinfo=UTC)
    row = led.record("ask", "m", _usage(100, 10, cw=1000, cr=5000), now)
    assert row["weighted"] == 100 + 50 + 1000 + 500
    led.record("report", "m", _usage(10, 0), now)
    led.record("ask", "m", None, now)  # a response without usage counts as zero
    t = led.totals(now.date())
    assert t["calls"] == 3
    assert t["by_purpose"] == {"ask": 1650.0, "report": 10.0}
    assert t["cache_hit_rate"] == pytest.approx(5000 / 6110)


def test_budget_hard_stop(tmp_path: Path) -> None:
    class Raw:
        def __init__(self) -> None:
            self.n = 0
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

        def create(self, **kw: Any) -> Any:
            self.n += 1
            return SimpleNamespace(model="m", usage=_usage(600, 0))

    raw = Raw()
    client = MeteredClient(raw, UsageLedger(tmp_path), daily_budget=1000, purpose="chat")
    client.beta.messages.create(model="m")
    client.beta.messages.create(model="m")  # 600 spent < 1000: allowed
    with pytest.raises(BudgetExceeded, match="daily LLM budget"):
        client.beta.messages.create(model="m")
    assert raw.n == 2
    unlimited = MeteredClient(raw, UsageLedger(tmp_path), daily_budget=0, purpose="chat")
    unlimited.beta.messages.create(model="m")
    assert raw.n == 3


class _Kit:
    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        return '{"p": 0.53}', False


def test_cache_breakpoints_and_tool_callback() -> None:
    client = FakeClient(
        [
            response(
                "tool_use", tool_block("t1", "get_views", {"universe": "ind12", "horizon": 3})
            ),
            response("tool_use", tool_block("t2", "get_regime", {"as_of": ""})),
            response("end_turn", text_block("Probability 0.53.")),
        ]
    )
    seen: list[str] = []
    ans = Assistant(client, _Kit()).ask("q", on_tool=lambda c: seen.append(c.name))  # type: ignore[arg-type]
    assert seen == ["get_views", "get_regime"]
    assert ans.grounding.grounded
    first = client.requests[0]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    last_msgs = client.requests[-1]["messages"]
    marked = [
        blk
        for m in last_msgs
        if isinstance(m["content"], list)
        for blk in m["content"]
        if isinstance(blk, dict) and "cache_control" in blk
    ]
    assert len(marked) == 1  # only the newest tool results carry a breakpoint
    assert marked[0]["tool_use_id"] == "t2"
