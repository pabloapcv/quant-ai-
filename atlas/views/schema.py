"""The canonical recommendation object (PRD §3.2) and stance rules (PRD §3.3)."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Stance(StrEnum):
    STRONG_UNDERWEIGHT = "strong_underweight"
    UNDERWEIGHT = "underweight"
    NEUTRAL = "neutral"
    OVERWEIGHT = "overweight"
    STRONG_OVERWEIGHT = "strong_overweight"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExpectedReturn(BaseModel):
    mean: float
    p10: float | None
    p90: float | None
    benchmark: str


class Driver(BaseModel):
    name: str
    contribution: float  # weight x z-score inside the ensemble score
    value: float  # the component's cross-sectional z-score
    unit: str = "z"


class Provenance(BaseModel):
    model_version: str
    code_sha: str | None
    data_through: date
    n_trials: int | None = None
    snapshot_id: str | None = None  # data snapshot the view was computed from


class View(BaseModel):
    model_config = ConfigDict(frozen=True)

    view_id: str
    created_at: datetime
    as_of: date  # month the signals are formed at (end of month)
    asset_id: str
    asset_name: str
    asset_class: str
    horizon: str
    stance: Stance
    score: float = Field(ge=-3, le=3)
    expected_excess_return: ExpectedReturn
    prob_outperform: float | None = Field(default=None, ge=0, le=1)
    agreement: float | None = Field(default=None, ge=0, le=1)
    confidence: Confidence
    drivers: list[Driver]
    invalidation: str
    provenance: Provenance


# ---------------------------------------------------------------------------- stance rules
HYSTERESIS = 0.02


def raw_stance(prob: float | None, agreement: float | None) -> Stance:
    """PRD §3.3 thresholds without hysteresis."""
    if prob is None:
        return Stance.NEUTRAL
    a = agreement if agreement is not None else 0.0
    if a < 0.5 and 0.35 < prob < 0.65:
        return Stance.NEUTRAL
    if prob >= 0.65 and a >= 0.75:
        return Stance.STRONG_OVERWEIGHT
    if prob >= 0.55:
        return Stance.OVERWEIGHT
    if prob <= 0.35 and a >= 0.75:
        return Stance.STRONG_UNDERWEIGHT
    if prob <= 0.45:
        return Stance.UNDERWEIGHT
    return Stance.NEUTRAL


_ORDER = list(Stance)


def stance_with_hysteresis(
    prob: float | None, agreement: float | None, previous: Stance | None
) -> Stance:
    """Change stance only when the probability clears the new threshold by ``HYSTERESIS``.

    Stops flip-flopping when a probability hovers at a boundary: the new stance must still
    hold after nudging the probability back toward the previous stance by 0.02.
    """
    new = raw_stance(prob, agreement)
    if previous is None or new == previous or prob is None:
        return new
    toward_prev = -HYSTERESIS if _ORDER.index(new) > _ORDER.index(previous) else HYSTERESIS
    confirmed = raw_stance(min(1.0, max(0.0, prob + toward_prev)), agreement)
    return new if confirmed == new else previous


def confidence_for(prob: float | None, agreement: float | None) -> Confidence:
    if prob is None or agreement is None:
        return Confidence.LOW
    edge = abs(prob - 0.5)
    if edge >= 0.1 and agreement >= 0.75:
        return Confidence.HIGH
    if edge >= 0.05 and agreement >= 0.5:
        return Confidence.MEDIUM
    return Confidence.LOW
