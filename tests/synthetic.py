"""A synthetic economy with known regimes, for end-to-end tests of the regime engine.

Two latent factors drive everything: growth ``g`` (8-year cycle) and inflation ``f``
(5-year cycle). Each catalog input is built so its engine feature tracks the right factor
(growth-type rates for YoY features, levels for level features, an inverted curve that
leads growth). Every observation is first released with noise and revised to the true
value a year later, so real-time and final vintages genuinely differ.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from numpy.typing import NDArray

from atlas.pit.schema import OBSERVATION_SCHEMA, conform

START = date(1970, 1, 1)
N_MONTHS = 56 * 12  # through 2025-12
G_PERIOD, F_PERIOD = 96, 60


def months() -> list[date]:
    return [date(START.year + (i // 12), i % 12 + 1, 1) for i in range(N_MONTHS)]


def growth_factor(t: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sin(2 * math.pi * t / G_PERIOD)


def inflation_factor(t: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sin(2 * math.pi * t / F_PERIOD + 1.0)


def true_regime(m: int) -> tuple[float, float]:
    """Trailing-12m means of the factors at month index ``m`` (what YoY features measure)."""
    window = np.arange(m - 11, m + 1, dtype=float)
    return float(growth_factor(window).mean()), float(inflation_factor(window).mean())


def _levels(seed: int = 3) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    t = np.arange(N_MONTHS, dtype=float)
    g, f = growth_factor(t), inflation_factor(t)

    def rate_level(factor: NDArray[np.float64], base: float, amp: float) -> NDArray[np.float64]:
        # monthly log growth -> level whose YoY ≈ base + amp * trailing mean(factor)
        return 100 * np.exp(np.cumsum((base + amp * factor) / 1200))

    def noise(s: float) -> NDArray[np.float64]:
        return rng.normal(0, s, N_MONTHS)

    return {
        "US.IP": rate_level(g, 2, 4),
        "US.LABOR.NFP": rate_level(g, 1.5, 2),
        "US.RETAIL": rate_level(g, 4, 5),
        "US.HOUSING.PERMITS": rate_level(g, 0, 15),
        "US.LABOR.CLAIMS.INIT": rate_level(-g, 0, 15),
        "US.LABOR.UNRATE": 5 + np.cumsum(-g * 0.15),
        "US.SENTIMENT.UMICH": 85 + 10 * g + noise(1),
        "US.CPI.CORE": rate_level(f, 3, 2),
        "US.CPI.HEADLINE": rate_level(f, 3, 3),
        "US.PCE.CORE": rate_level(f, 2.5, 1.5),
        "US.RATES.5Y5Y": 2.3 + 0.4 * f + noise(0.05),
        "CMDTY.OIL.WTI": rate_level(f, 2, 30),
        "US.FCI.NFCI": -0.6 * g + noise(0.1),
        "US.VOL.VIX": 18 - 6 * g + noise(1),
        "US.FCI.STLFSI": -0.8 * g + noise(0.1),
        # leads growth by 12 months: inverts ahead of downturns
        "US.CURVE.10Y3M": 1.2 + 1.5 * growth_factor(t + 12) + noise(0.1),
        "US.LABOR.SAHM": np.clip(-0.6 * g, 0, None) + noise(0.02),
        "US.NBER.RECESSION": (g < -0.75).astype(float),
    }


def build_observations(*, revise: bool = True, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 1)
    ms = months()
    rows = []
    ingested = datetime(2026, 1, 1, tzinfo=UTC)
    for series_id, values in _levels(seed).items():
        is_nber = series_id == "US.NBER.RECESSION"
        for ev, true_v in zip(ms, values, strict=True):
            period_end = date(ev.year + (ev.month // 12), ev.month % 12 + 1, 1) - timedelta(days=1)
            lag = 365 if is_nber else 15
            first_kt = datetime.combine(period_end + timedelta(days=lag), datetime.min.time(), UTC)
            first_kt += timedelta(hours=13, minutes=30)
            vintages = [(first_kt, float(true_v))]
            if revise and not is_nber:
                noisy = float(true_v) * (1 + rng.normal(0, 0.002)) + rng.normal(0, 0.02)
                vintages = [(first_kt, noisy), (first_kt + timedelta(days=365), float(true_v))]
            for kt, v in vintages:
                rows.append(
                    {
                        "series_id": series_id,
                        "source": "synthetic",
                        "source_series_id": series_id,
                        "event_time": ev,
                        "knowledge_time": kt,
                        "value": v,
                        "unit": "synthetic",
                        "frequency": "M",
                        "vintage_id": kt.date().isoformat(),
                        "sa_flag": True,
                        "is_approx_vintage": False,
                        "ingest_run_id": "synthetic",
                        "ingested_at": ingested,
                    }
                )
    return conform(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA))
