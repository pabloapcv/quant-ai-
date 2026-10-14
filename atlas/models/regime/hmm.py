"""Statistical regimes: Gaussian HMM on the pillar composites (PRD §7.1).

The number of states is chosen by BIC. States are relabelled by their mean growth /
inflation / financial-conditions profile so labels stay comparable across refits.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM
from numpy.typing import NDArray
from pydantic import BaseModel

from atlas.models.regime.quadrant import classify

STRESS_THRESHOLD = 0.5  # mean financial-conditions composite above which a state is "stressed"

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


class HMMState(BaseModel):
    label: str
    prob: float
    means: dict[str, float]
    persistence: float  # P(stay in state next month)


class HMMResult(BaseModel):
    n_states: int
    n_obs: int
    bic: float
    states: list[HMMState]

    @property
    def most_likely(self) -> HMMState:
        return max(self.states, key=lambda s: s.prob)


def _label(means: dict[str, float]) -> str:
    base = classify(means.get("growth", 0.0), means.get("inflation", 0.0)).value
    if means.get("financial_conditions", 0.0) > STRESS_THRESHOLD:
        return f"{base}+stress"
    return base


def fit_regime_hmm(
    x: NDArray[np.float64],
    columns: Sequence[str],
    *,
    n_states_options: Sequence[int] = (2, 3, 4),
    seed: int = 7,
    min_obs: int = 120,
) -> HMMResult | None:
    """Fit on rows of ``x`` (time-ordered, no NaNs). Returns posterior at the last row."""
    if x.shape[0] < min_obs:
        return None
    best: tuple[float, GaussianHMM] | None = None
    for k in n_states_options:
        if x.shape[0] < 10 * k * x.shape[1]:
            continue
        model = GaussianHMM(
            n_components=k, covariance_type="full", n_iter=300, tol=1e-4, random_state=seed
        )
        try:
            with warnings.catch_warnings():
                # hmmlearn 0.3.3 sets ndarray.shape, deprecated in NumPy 2.5 (harmless)
                warnings.simplefilter("ignore", DeprecationWarning)
                model.fit(x)
                bic = float(model.bic(x))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(bic) and (best is None or bic < best[0]):
            best = (bic, model)
    if best is None:
        return None

    bic, model = best
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        posterior = model.predict_proba(x)[-1]  # smoothed at the last row == filtered
    states = []
    for k in range(model.n_components):
        means = {c: float(m) for c, m in zip(columns, model.means_[k], strict=True)}
        states.append(
            HMMState(
                label=_label(means),
                prob=float(posterior[k]),
                means=means,
                persistence=float(model.transmat_[k, k]),
            )
        )
    states.sort(key=lambda s: -s.means.get("growth", 0.0))
    return HMMResult(n_states=model.n_components, n_obs=int(x.shape[0]), bic=bic, states=states)
