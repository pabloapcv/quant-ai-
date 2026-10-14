"""Promotion gates: when a signal may enter the production ensemble (PRD §8.4).

| Gate                                  | Threshold                                      |
|---------------------------------------|------------------------------------------------|
| IC t-stat (Newey-West)                | >= 2.0 with a published prior, else >= 3.0     |
| Deflated Sharpe ratio                 | >= 0.95 (i.e. p < 0.05 after multiple testing) |
| PBO across the signal's family        | < 0.20                                         |
| Positive IC in >= 3 of 4 regimes      | required                                       |
| Marginal contribution to ensemble ICIR| > 0                                            |
| Economic rationale documented         | literature reference present                   |
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from atlas.backtest.crosssection import SignalReport
from atlas.signals.industry import SignalDef

T_PUBLISHED = 2.0
T_NOVEL = 3.0
DSR_MIN = 0.95
PBO_MAX = 0.20
REGIMES_REQUIRED = 3


@dataclass(frozen=True)
class GateResult:
    signal: str
    family: str
    ic_tstat: float
    dsr: float
    pbo: float | None
    regimes_positive: int
    regimes_total: int
    marginal_icir: float | None
    reference: str
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def failed(self) -> list[str]:
        return [k for k, ok in self.checks.items() if not ok]


def _ok(x: float | None, test: bool) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and test


def evaluate_gates(
    sig: SignalDef,
    report: SignalReport,
    *,
    pbo: float | None,
    marginal_icir: float | None,
) -> GateResult:
    t = report.ic_tstat
    t_needed = T_PUBLISHED if sig.reference else T_NOVEL
    regime_ic = report.regime_ic
    positive = sum(1 for v in regime_ic.values() if v > 0)
    checks = {
        "ic_tstat": _ok(t, t >= t_needed),
        "dsr": _ok(report.dsr, report.dsr >= DSR_MIN),
        "pbo": _ok(pbo, pbo is not None and pbo < PBO_MAX),
        "regimes": len(regime_ic) >= 4 and positive >= REGIMES_REQUIRED,
        "marginal": _ok(marginal_icir, marginal_icir is not None and marginal_icir > 0),
        "rationale": bool(sig.reference and sig.description),
    }
    return GateResult(
        signal=sig.name,
        family=sig.family,
        ic_tstat=t,
        dsr=report.dsr,
        pbo=pbo,
        regimes_positive=positive,
        regimes_total=len(regime_ic),
        marginal_icir=marginal_icir,
        reference=sig.reference,
        checks=checks,
    )
