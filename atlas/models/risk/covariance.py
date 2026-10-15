"""Walk-forward covariance estimates (PRD §7.3).

Monthly returns only (daily data arrives with Phase 1). The estimate at month ``t`` uses
returns through ``t``:

* Ledoit-Wolf shrinkage on the trailing ``window`` months — stable with few observations
  per asset (60 months vs 49 industries would be near-singular as a sample covariance);
* an EWMA covariance (half-life ``halflife`` months) — reacts to volatility regimes;
* blended 50/50. Both are positive semi-definite, so the blend is too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RiskConfig:
    window: int = 60
    halflife: float = 12.0
    blend: float = 0.5  # weight on Ledoit-Wolf
    min_obs: int = 36


def ewma_covariance(x: FloatArray, halflife: float) -> FloatArray:
    t = x.shape[0]
    decay = 0.5 ** (1.0 / halflife)
    w = decay ** np.arange(t - 1, -1, -1, dtype=np.float64)
    w /= w.sum()
    mu = w @ x
    xc = x - mu
    return np.asarray((xc * w[:, None]).T @ xc, dtype=np.float64)


def covariance(x: FloatArray, cfg: RiskConfig | None = None) -> FloatArray:
    """Blended covariance of the rows of ``x`` (T x N, no NaNs)."""
    c = cfg or RiskConfig()
    if x.shape[0] < c.min_obs:
        raise ValueError(f"need >= {c.min_obs} observations, got {x.shape[0]}")
    lw = LedoitWolf().fit(x).covariance_
    ew = ewma_covariance(x, c.halflife)
    cov = c.blend * lw + (1 - c.blend) * ew
    return np.asarray((cov + cov.T) / 2, dtype=np.float64)
