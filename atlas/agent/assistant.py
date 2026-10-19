"""The Atlas research assistant: Claude + the quant engine as tools (PRD §10).

Flow for one question:

1. Claude decides which tools to call (``tool_choice: auto``); all tool calls in a turn
   run and their results go back in a single user message.
2. When Claude answers, the numeric grounding guard checks every number against the tool
   results of the conversation.
3. If anything is unsupported, Claude gets one repair turn naming those numbers. If the
   revised answer still has unsupported numbers, they are reported with the answer.

Defaults: ``claude-opus-5-5`` at ``medium`` effort (thinking stays on — it cannot be
disabled on this model), and server-side refusal fallbacks (``fallbacks: "default"``) so a
safety decline is re-run on Anthropic's recommended fallback model instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas.agent.grounding import GroundingReport, check
from atlas.agent.tools import TOOL_SPECS, ToolKit

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DISCLAIMER = (
    "Research output from model views and historical backtests, not investment advice. "
    "Past performance does not guarantee future results."
)

SYSTEM_PROMPT = """You are Atlas, a quantitative research assistant. You answer questions \
about markets using the Atlas quant engine, which you reach through tools: model views on \
US industries and country equity markets, their walk-forward track records, promotion-gate \
results for each signal, model portfolios, point-in-time data series, the macro regime, \
historical macro analogs and a factor-shock scenario engine.

How to answer:
- Get every number you state from a tool result in this conversation, at the precision the \
tool gave it (you may round, and you may write fractions as percentages). If no tool \
provides a number, say that it is not available rather than estimating it.
- Name the universe, horizon and as-of month that a view refers to. Use the 3-month \
horizon unless the question asks for 1 month: it is the model's primary, stronger horizon.
- Be plain about uncertainty. Most views are low-conviction; a probability of 0.53 is a \
slight lean, not a call. Mention track-record quality when it matters to the answer.
- Explain what drives a view (the top signals) when asked why.
- These are research views, not personalized investment advice. Do not tell anyone what \
to buy or sell for their own portfolio; describe what the models say.
- Keep answers short: a few sentences or a compact list. When you use a tool, you may say \
a brief sentence first. If no tool can express what was asked, say so instead of guessing. \
Do not include internal or system XML tags in your response."""

REPAIR_PROMPT = (
    "These numbers in your answer do not match any tool result in this conversation: {nums}. "
    "Revise the answer: use tool results for every number (call tools if needed), or remove "
    "numbers you cannot support. Reply with the full revised answer only, written for the user "
    "as if it were your first answer: do not mention this check, a revision or 'flagged' figures."
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    is_error: bool


@dataclass
class Answer:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)  # JSON text, same order as calls
    grounding: GroundingReport = field(default_factory=GroundingReport)
    repaired: bool = False
    stop_reason: str = ""
    model: str = ""

    @property
    def ok(self) -> bool:
        return self.stop_reason not in ("refusal", "max_tokens") and self.grounding.grounded


class Assistant:
    def __init__(
        self,
        client: Any,  # anthropic.Anthropic or anything with .beta.messages.create(**kw)
        toolkit: ToolKit,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_steps: int = 10,
        fallbacks: bool = True,
    ) -> None:
        self.client = client
        self.toolkit = toolkit
        self.model = model
        self.effort = effort
        self.max_steps = max_steps
        self.fallbacks = fallbacks

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "tools": [{**t, "strict": True} for t in TOOL_SPECS],
            "tool_choice": {"type": "auto"},
            "output_config": {"effort": self.effort},
            "messages": messages,
        }
        if self.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return self.client.beta.messages.create(**kwargs)

    def ask(self, question: str) -> Answer:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        calls: list[ToolCall] = []
        results: list[str] = []
        repaired = False
        response: Any = None
        for _ in range(self.max_steps):
            response = self._create(messages)
            stop = response.stop_reason
            if stop == "refusal":
                return Answer(
                    text="The request was declined by the model's safety system.",
                    tool_calls=calls,
                    tool_results=results,
                    stop_reason=stop,
                    model=getattr(response, "model", ""),
                )
            if stop == "max_tokens":
                break
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if stop == "tool_use" and tool_uses:
                messages.append({"role": "assistant", "content": response.content})
                batch = []
                for block in tool_uses:
                    out, is_error = self.toolkit.call(block.name, dict(block.input))
                    calls.append(ToolCall(block.name, dict(block.input), is_error))
                    results.append(out)
                    batch.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": out,
                            "is_error": is_error,
                        }
                    )
                messages.append({"role": "user", "content": batch})  # all results, one message
                continue

            text = "".join(b.text for b in response.content if b.type == "text").strip()
            report = check(text, results)
            if report.grounded or repaired:
                return Answer(
                    text=text,
                    tool_calls=calls,
                    tool_results=results,
                    grounding=report,
                    repaired=repaired,
                    stop_reason=stop,
                    model=getattr(response, "model", ""),
                )
            repaired = True
            nums = ", ".join(c.text for c in report.unsupported)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": REPAIR_PROMPT.format(nums=nums)})

        text = ""
        if response is not None:
            text = "".join(
                b.text for b in response.content if getattr(b, "type", "") == "text"
            ).strip()
        return Answer(
            text=text or "No complete answer within the step limit.",
            tool_calls=calls,
            tool_results=results,
            grounding=check(text, results),
            repaired=repaired,
            stop_reason=getattr(response, "stop_reason", "") or "step_limit",
            model=getattr(response, "model", ""),
        )
