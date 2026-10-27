"""Model portfolio: benchmark-relative tilts from views (PRD §7.4).

Each month: maximize expected active return minus a turnover cost, subject to

* tracking error ``<= te_annual`` (ex ante, from the risk model),
* ``|active weight| <= max_active`` per asset,
* long-only, fully invested (active weights sum to zero).

The benchmark is the equal-weight universe (the reference the views forecast against).
Expected returns come from the views' expected excess returns, either directly or (with
``black_litterman=True``) blended with an equilibrium prior. The benchmark is equal-weight,
so the equilibrium that makes it optimal has zero expected *active* return: the prior is 0,
with covariance ``tau * Sigma``, and each view's uncertainty ``Omega`` comes from its own
80% conformal range. The posterior ``(inv(tau Sigma) + inv(Omega))^-1 inv(Omega) q`` shrinks
uncertain views toward zero and spreads conviction across correlated assets (He & Litterman
1999). A cap-weighted prior needs market caps and arrives with Phase 1 data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import polars as pl
from numpy.typing import NDArray

from atlas.models.risk.covariance import RiskConfig, covariance

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PortfolioConfig:
    te_annual: float = 0.04
    max_active: float = 0.05
    cost_bps: float = 10.0  # per unit of one-way turnover
    turnover_aversion: float = 1.0  # multiplies cost in the objective
    periods_per_year: int = 12
    black_litterman: bool = False
    tau: float = 0.05  # prior uncertainty as a fraction of the return covariance


def optimize_active(
    alpha: FloatArray,
    cov: FloatArray,
    bench: FloatArray,
    prev: FloatArray,
    cfg: PortfolioConfig | None = None,
) -> FloatArray:
    """Portfolio weights (not active weights). Falls back to ``prev`` if infeasible."""
    c = cfg or PortfolioConfig()
    n = alpha.size
    w = cp.Variable(n)
    active = w - bench
    te_var = (c.te_annual**2) / c.periods_per_year
    cost = c.turnover_aversion * c.cost_bps / 1e4 * cp.norm1(w - prev)
    problem = cp.Problem(
        cp.Maximize(alpha @ w - cost),
        [
            cp.sum(w) == 1,
            w >= 0,
            cp.abs(active) <= c.max_active,
            cp.quad_form(active, cp.psd_wrap(cov)) <= te_var,
        ],
    )
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return prev
    if w.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        return prev
    out: FloatArray = np.clip(np.asarray(w.value, dtype=np.float64), 0.0, None)
    return np.asarray(out / out.sum(), dtype=np.float64)


Z80 = 1.2816  # standard-normal quantile of the 80% range's upper edge


def black_litterman(
    q: FloatArray, view_sd: FloatArray, cov: FloatArray, tau: float = 0.05
) -> FloatArray:
    """Posterior expected active returns: zero prior (equal-weight equilibrium), views ``q``."""
    prior_prec = np.linalg.inv(tau * cov + 1e-12 * np.eye(cov.shape[0]))
    omega_inv = np.diag(1.0 / np.maximum(view_sd, 1e-6) ** 2)
    post: FloatArray = np.linalg.solve(prior_prec + omega_inv, omega_inv @ q)
    return post


@dataclass(frozen=True)
class PortfolioBacktest:
    monthly: pl.DataFrame  # date, active_gross, turnover, cost, active_net, ex_ante_te, n_assets
    weights: pl.DataFrame  # date, asset, weight, bench

    def summary(self) -> dict[str, float | int | None]:
        m = self.monthly
        net = m["active_net"].to_numpy()
        gross = m["active_gross"].to_numpy()
        te = float(np.std(net, ddof=1) * math.sqrt(12)) if net.size > 2 else math.nan
        ir = float(np.mean(net) * 12 / te) if te and te > 0 else math.nan

        def _f(x: float) -> float | None:
            return x if math.isfinite(x) else None

        return {
            "months": m.height,
            "active_return_annual_gross": _f(float(np.mean(gross) * 12)),
            "active_return_annual_net": _f(float(np.mean(net) * 12)),
            "tracking_error_realized": _f(te),
            "tracking_error_ex_ante_mean": _f(float(np.mean(m["ex_ante_te"].to_numpy()))),
            "information_ratio_net": _f(ir),
            "turnover_monthly": _f(float(np.mean(m["turnover"].to_numpy()))),
            "hit_rate": _f(float(np.mean(net > 0))),
        }


def backtest_portfolio(
    forecasts: pl.DataFrame,
    returns: pl.DataFrame,
    *,
    horizon: int = 1,
    cfg: PortfolioConfig | None = None,
    risk: RiskConfig | None = None,
) -> PortfolioBacktest:
    """Monthly walk-forward: views at ``t`` -> weights held over ``t+1``.

    ``forecasts``: date, asset, exp_excess (over ``horizon`` months; scaled to monthly).
    ``returns``: date, asset, ret (monthly simple returns, the research panel).
    """
    c = cfg or PortfolioConfig()
    rk = risk or RiskConfig()
    wide = returns.pivot(index="date", on="asset", values="ret", aggregate_function="first").sort(
        "date"
    )
    dates = wide["date"].to_list()
    assets = [a for a in wide.columns if a != "date"]
    rmat = wide.select(assets).to_numpy().astype(np.float64)
    pos = {d: i for i, d in enumerate(dates)}
    fc = forecasts.drop_nulls("exp_excess")
    by_date = {d: part for (d,), part in fc.partition_by("date", as_dict=True).items()}

    rows, wrows = [], []
    prev_w: dict[str, float] = {}
    for d in sorted(by_date):
        i = pos.get(d)
        if i is None or i + 1 >= len(dates) or i + 1 < rk.window:
            continue
        hist = rmat[i + 1 - rk.window : i + 1]
        nxt = rmat[i + 1]
        part = by_date[d]
        alpha_map = dict(zip(part["asset"].to_list(), part["exp_excess"].to_list(), strict=True))
        sd_map: dict[str, float] = {}
        if c.black_litterman and {"p_lo", "p_hi"} <= set(part.columns):
            for a, lo, hi in part.select("asset", "p_lo", "p_hi").iter_rows():
                if lo is not None and hi is not None and hi > lo:
                    # 80% range over the horizon -> monthly standard deviation
                    sd_map[a] = (hi - lo) / (2 * Z80) / math.sqrt(horizon)
        cols = [
            j
            for j, a in enumerate(assets)
            if a in alpha_map and np.isfinite(hist[:, j]).all() and np.isfinite(nxt[j])
        ]
        if len(cols) < 5:
            continue
        names = [assets[j] for j in cols]
        cov = covariance(hist[:, cols], rk)
        bench = np.full(len(cols), 1.0 / len(cols))
        prev = np.array([prev_w.get(a, 0.0) for a in names])
        prev = prev / prev.sum() if prev.sum() > 0 else bench
        alpha = np.array([alpha_map[a] for a in names], dtype=np.float64) / horizon
        if c.black_litterman:
            if not all(a in sd_map for a in names):
                continue  # BL needs every view's range
            sd = np.array([sd_map[a] for a in names], dtype=np.float64)
            alpha = black_litterman(alpha, sd, cov, c.tau)
        w = optimize_active(alpha, cov, bench, prev, c)
        active = w - bench
        turnover = float(np.abs(w - prev).sum())
        gross = float(active @ nxt[cols])
        cost = turnover * c.cost_bps / 1e4
        rows.append(
            {
                "date": d,
                "active_gross": gross,
                "turnover": turnover,
                "cost": cost,
                "active_net": gross - cost,
                "ex_ante_te": float(math.sqrt(max(active @ cov @ active, 0.0) * 12)),
                "n_assets": len(cols),
            }
        )
        wrows += [
            {"date": d, "asset": a, "weight": float(x), "bench": float(b)}
            for a, x, b in zip(names, w, bench, strict=True)
        ]
        # weights drift with returns into next month's starting point
        drifted = w * (1 + nxt[cols])
        prev_w = dict(zip(names, drifted / drifted.sum(), strict=True))
    if not rows:
        raise ValueError("no month had enough history and forecasts to build a portfolio")
    return PortfolioBacktest(monthly=pl.DataFrame(rows), weights=pl.DataFrame(wrows))
