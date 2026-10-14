"""Performance and overfitting statistics (PRD §8.3).

Sharpe-family statistics follow Bailey & López de Prado:

* PSR  — probability the true Sharpe exceeds a benchmark, given sample length, skew and
  kurtosis (non-normal returns widen the confidence band).
* DSR  — PSR against the Sharpe you'd *expect from the best of N unskilled trials*. This is
  how multiple testing is charged for: the more configurations tried, the higher the bar.
* PBO  — probability of backtest overfitting via combinatorially symmetric CV (CSCV): how
  often the in-sample winner ranks below median out of sample.

Sharpe inputs/outputs are per period unless named ``*_annual``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329
FloatArray = NDArray[np.float64]


def _clean(x: Sequence[float] | FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    return a[np.isfinite(a)]


def sharpe(returns: Sequence[float] | FloatArray) -> float:
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def annualize_sharpe(sr: float, periods_per_year: int) -> float:
    return sr * math.sqrt(periods_per_year)


def sortino(returns: Sequence[float] | FloatArray) -> float:
    r = _clean(returns)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if r.size else 0.0
    return float(r.mean() / downside) if downside > 0 else float("nan")


def max_drawdown(returns: Sequence[float] | FloatArray) -> float:
    """Worst peak-to-trough loss of the compounded equity curve (negative fraction)."""
    r = _clean(returns)
    if not r.size:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float((equity / peak - 1.0).min())


def newey_west_tstat(x: Sequence[float] | FloatArray, lags: int | None = None) -> float:
    """t-stat of the mean with HAC (Bartlett) standard errors — IC series are autocorrelated."""
    a = _clean(x)
    n = a.size
    if n < 3:
        return float("nan")
    lags = lags if lags is not None else math.floor(4 * (n / 100) ** (2 / 9))
    e = a - a.mean()
    var = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        var += 2 * w * float(e[k:] @ e[:-k]) / n
    se = math.sqrt(var / n) if var > 0 else float("nan")
    return float(a.mean() / se) if se and math.isfinite(se) else float("nan")


def probabilistic_sharpe_ratio(
    returns: Sequence[float] | FloatArray, sr_benchmark: float = 0.0
) -> float:
    r = _clean(returns)
    n = r.size
    sr = sharpe(r)
    if n < 3 or not math.isfinite(sr):
        return float("nan")
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # non-excess
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, trial_sr_variance: float) -> float:
    """E[max per-period Sharpe] across ``n_trials`` skill-less strategies."""
    if n_trials <= 1 or trial_sr_variance <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(trial_sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: Sequence[float] | FloatArray, n_trials: int, trial_sr_variance: float | None = None
) -> float:
    """PSR against the expected maximum Sharpe of ``n_trials`` unskilled trials.

    ``trial_sr_variance``: cross-trial variance of per-period Sharpe ratios. When unknown,
    the variance of the Sharpe *estimator* under the null (≈ 1/T) is used.
    """
    r = _clean(returns)
    var = trial_sr_variance if trial_sr_variance is not None else 1.0 / max(r.size, 1)
    return probabilistic_sharpe_ratio(r, expected_max_sharpe(n_trials, var))


def pbo_cscv(
    performance: FloatArray, n_splits: int = 16, *, max_combinations: int = 5000, seed: int = 0
) -> float:
    """Probability of backtest overfitting (Bailey, Borwein, López de Prado & Zhu 2017).

    ``performance``: T x N matrix of per-period returns for N candidate configurations.
    Rows are cut into ``n_splits`` contiguous blocks; for each half/half combination the
    in-sample best configuration's out-of-sample rank is recorded. PBO = share of
    combinations where that rank is at or below the median (logit <= 0).
    """
    m = np.asarray(performance, dtype=np.float64)
    t, n = m.shape
    if n < 2 or t < 2 * n_splits:
        return float("nan")
    blocks = np.array_split(np.arange(t), n_splits)
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in picks]

    def block_sharpes(idx: FloatArray) -> FloatArray:
        sub = m[idx.astype(int)]
        sd = sub.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, sub.mean(axis=0) / sd, -np.inf)

    logits = []
    for combo in combos:
        is_idx = np.concatenate([blocks[i] for i in combo]).astype(np.float64)
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo]).astype(
            np.float64
        )
        best = int(np.argmax(block_sharpes(is_idx)))
        oos = block_sharpes(oos_idx)
        rank = float((oos < oos[best]).sum() + 1) / (n + 1)  # relative rank in (0, 1)
        logits.append(math.log(rank / (1 - rank)))
    return float(np.mean(np.asarray(logits) <= 0))
