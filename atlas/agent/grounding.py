"""Deterministic numeric grounding (PRD §10.4, goal G6).

Every number in the assistant's answer must match a value returned by a tool in the same
conversation, at the precision the answer displays it. A tool's fraction ``0.0647`` supports
``0.065``, ``6.5%`` and ``6.47%``; a value of ``12.81`` supports ``12.8``; a
percentage-point value ``0.69`` supports ``69bp``. Anything else is
reported as unsupported.

Not treated as claims (they carry no model output):

* integers 0-24 written without decimals — counts, ranks, horizons ("3 of 4 regimes",
  "12-1 momentum", "3-month") — and integers that appear verbatim in a tool result;
* years 1900-2100 and dates that appear in a tool result;
* numbers inside identifiers (``mom_12_1``, ``FX.EURUSD``, ``Q5``).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# A number not glued to letters/underscores/dots on either side (identifiers are skipped).
_NUM = re.compile(
    r"(?<![\w.])(?P<sign>[-+\u2212]?)(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<pct>\s?%|\s?bp\b|\s?pp\b)?(?![\w])"
)
_DATE = re.compile(r"\b(19|20)\d{2}-\d{2}(?:-\d{2})?\b")
SMALL_INT_MAX = 24


@dataclass(frozen=True)
class Claim:
    text: str
    value: float
    decimals: int
    unit: str  # "", "%", "bp", "pp"


@dataclass(frozen=True)
class GroundingReport:
    claims: list[Claim] = field(default_factory=list)
    unsupported: list[Claim] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.unsupported

    @property
    def grounded_share(self) -> float:
        return 1.0 if not self.claims else 1 - len(self.unsupported) / len(self.claims)


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def evidence(tool_results: Iterable[str]) -> tuple[list[float], set[str]]:
    """All numeric values and all string tokens found in the tool results."""
    numbers: list[float] = []
    strings: set[str] = set()
    for text in tool_results:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        for leaf in _walk(payload):
            if isinstance(leaf, bool):
                continue
            if isinstance(leaf, int | float) and math.isfinite(leaf):
                numbers.append(float(leaf))
            elif isinstance(leaf, str):
                strings.add(leaf)
                for m in _NUM.finditer(leaf):
                    numbers.append(float(m.group("num").replace(",", "")))
    return numbers, strings


def extract_claims(answer: str) -> list[Claim]:
    claims = []
    cleaned = _DATE.sub(" ", answer)
    for m in _NUM.finditer(cleaned):
        raw = m.group("num")
        sign = -1.0 if m.group("sign") in ("-", "\u2212") else 1.0
        num = raw.replace(",", "")
        decimals = len(num.split(".")[1]) if "." in num else 0
        unit = (m.group("pct") or "").strip()
        claims.append(
            Claim(text=m.group(0).strip(), value=sign * float(num), decimals=decimals, unit=unit)
        )
    return claims


def _matches(claim: Claim, t: float) -> bool:
    tol = 0.5 * 10 ** (-claim.decimals) + 1e-9
    candidates = [t]
    if claim.unit in ("%", "pp"):
        candidates.append(t * 100)
    elif claim.unit == "bp":
        candidates += [t * 1e4, t * 100]  # from a fraction, or from percentage points
    else:
        candidates.append(t * 100)  # "6.5" written for a 0.065 fraction without a % sign
    return any(abs(abs(c) - abs(claim.value)) <= tol for c in candidates)


def check(answer: str, tool_results: Iterable[str]) -> GroundingReport:
    results = list(tool_results)
    numbers, strings = evidence(results)
    joined = " ".join(strings)
    claims = extract_claims(answer)
    unsupported = []
    for c in claims:
        if c.decimals == 0 and c.unit == "":
            if abs(c.value) <= SMALL_INT_MAX or 1900 <= c.value <= 2100:
                continue
            if c.text.lstrip("+-\u2212") in joined or any(n == abs(c.value) for n in numbers):
                continue
        if any(_matches(c, t) for t in numbers):
            continue
        unsupported.append(c)
    return GroundingReport(claims=claims, unsupported=unsupported)
