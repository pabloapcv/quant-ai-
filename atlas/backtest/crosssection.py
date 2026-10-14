"""Cross-sectional signal backtest: IC analysis + quantile long/short portfolio (PRD §8.1-8.3).

Input panel columns: ``date`` (Date), ``asset`` (str), ``signal`` (float, known at the end
of ``date``) and ``fwd_ret`` (float, simple return over the *next* period, as a fraction).

Portfolio: each period, equal-weight long the top quantile and short the bottom quantile,
rebalanced fully every period. Costs are charged on one-way traded notional:
``net = gross - sum|Δw| * cost_bps / 1e4``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from atlas.backtest import metrics as m


@dataclass(frozen=True)
class SignalReport:
    name: str
    periods_per_year: int
    ic: pl.DataFrame  # date, ic, n_assets
    quantile_returns: pl.DataFrame  # date, quantile, ret
    long_short: pl.DataFrame  # date, gross, turnover, net
    ic_decay: dict[int, float]
    n_trials: int
    trial_sr_variance: float | None
    regime_ic: dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- IC
    @property
    def ic_mean(self) -> float:
        return float(np.nanmean(self.ic["ic"].to_numpy()))

    @property
    def ic_tstat(self) -> float:
        return m.newey_west_tstat(self.ic["ic"].to_numpy())

    @property
    def icir_annual(self) -> float:
        x = self.ic["ic"].to_numpy()
        sd = float(np.nanstd(x, ddof=1))
        return float(np.nanmean(x) / sd * math.sqrt(self.periods_per_year)) if sd > 0 else math.nan

    @property
    def ic_hit_rate(self) -> float:
        return _num((self.ic["ic"] > 0).mean())

    # ---------------------------------------------------------------- portfolio
    def _ann(self, col: str) -> float:
        return _num(self.long_short[col].mean()) * self.periods_per_year

    @property
    def sharpe_gross(self) -> float:
        return m.annualize_sharpe(
            m.sharpe(self.long_short["gross"].to_numpy()), self.periods_per_year
        )

    @property
    def sharpe_net(self) -> float:
        return m.annualize_sharpe(
            m.sharpe(self.long_short["net"].to_numpy()), self.periods_per_year
        )

    @property
    def breakeven_cost_bps(self) -> float:
        to = _num(self.long_short["turnover"].mean())
        return _num(self.long_short["gross"].mean()) / to * 1e4 if to > 0 else math.inf

    @property
    def psr(self) -> float:
        return m.probabilistic_sharpe_ratio(self.long_short["net"].to_numpy())

    @property
    def dsr(self) -> float:
        return m.deflated_sharpe_ratio(
            self.long_short["net"].to_numpy(), self.n_trials, self.trial_sr_variance
        )

    @property
    def quantile_mean_annual(self) -> dict[int, float]:
        q = self.quantile_returns.group_by("quantile").agg(pl.col("ret").mean()).sort("quantile")
        return {int(k): float(v) * self.periods_per_year for k, v in q.iter_rows()}

    @property
    def monotonicity(self) -> float:
        qm = self.quantile_mean_annual
        if len(qm) < 3:
            return math.nan
        rho = spearmanr(list(qm), list(qm.values())).statistic
        return float(rho)

    def summary(self) -> dict[str, Any]:
        ls = self.long_short
        return {
            "signal": self.name,
            "start": ls["date"].min(),
            "end": ls["date"].max(),
            "periods": ls.height,
            "ic_mean": self.ic_mean,
            "ic_tstat_nw": self.ic_tstat,
            "icir_annual": self.icir_annual,
            "ic_hit_rate": self.ic_hit_rate,
            "ls_return_annual_net": self._ann("net"),
            "ls_vol_annual": _num(ls["net"].std()) * math.sqrt(self.periods_per_year),
            "sharpe_gross": self.sharpe_gross,
            "sharpe_net": self.sharpe_net,
            "max_drawdown": m.max_drawdown(ls["net"].to_numpy()),
            "turnover_mean": _num(ls["turnover"].mean()),
            "breakeven_cost_bps": self.breakeven_cost_bps,
            "monotonicity": self.monotonicity,
            "psr": self.psr,
            "dsr": self.dsr,
            "n_trials": self.n_trials,
        }


def _num(v: object) -> float:
    """Polars scalar aggregate -> float (NaN for empty / non-numeric)."""
    return float(v) if isinstance(v, int | float) else math.nan


def _turnover(weights: pl.DataFrame, dates: Sequence[object]) -> pl.DataFrame:
    wide = (
        weights.pivot(index="date", on="asset", values="w", aggregate_function="first")
        .join(pl.DataFrame({"date": list(dates)}, schema={"date": pl.Date}), on="date", how="right")
        .sort("date")
        .fill_null(0.0)
    )
    w = wide.drop("date").to_numpy()
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    return pl.DataFrame({"date": wide["date"], "turnover": np.abs(w - prev).sum(axis=1)})


def backtest_signal(
    panel: pl.DataFrame,
    name: str = "signal",
    *,
    n_quantiles: int = 3,
    cost_bps: float = 10.0,
    periods_per_year: int = 12,
    min_assets: int = 6,
    decay_lags: Sequence[int] = (1, 2, 3, 6, 12),
    n_trials: int = 1,
    trial_sr_variance: float | None = None,
    regimes: pl.DataFrame | None = None,
) -> SignalReport:
    df = panel.select("date", "asset", "signal", "fwd_ret").drop_nulls(["signal", "fwd_ret"])
    df = df.filter(pl.len().over("date") >= min_assets).sort(["date", "asset"])
    if df.is_empty():
        raise ValueError(f"{name}: no dates with >= {min_assets} assets")

    ic = (
        df.group_by("date")
        .agg(
            pl.corr("signal", "fwd_ret", method="spearman").alias("ic"),
            pl.len().alias("n_assets"),
        )
        .sort("date")
    )

    rank = pl.col("signal").rank("average").over("date")
    count = pl.len().over("date")
    quant = ((rank - 1) * n_quantiles / count).floor().cast(pl.Int32) + 1
    df = df.with_columns(quant.clip(1, n_quantiles).alias("quantile"))
    quantile_returns = (
        df.group_by(["date", "quantile"])
        .agg(pl.col("fwd_ret").mean().alias("ret"))
        .sort(["date", "quantile"])
    )

    legs = df.filter(pl.col("quantile").is_in([1, n_quantiles]))
    side = pl.when(pl.col("quantile") == n_quantiles).then(1.0).otherwise(-1.0)
    weights = legs.with_columns((side / pl.len().over(["date", "quantile"])).alias("w")).select(
        "date", "asset", "w", "fwd_ret"
    )
    gross = weights.group_by("date").agg((pl.col("w") * pl.col("fwd_ret")).sum().alias("gross"))
    dates = ic["date"].to_list()
    long_short = (
        gross.join(_turnover(weights, dates), on="date", how="left")
        .sort("date")
        .with_columns((pl.col("gross") - pl.col("turnover") * cost_bps / 1e4).alias("net"))
    )

    # IC decay: signal at t vs the return k periods later.
    by_asset = df.sort(["asset", "date"])
    decay = {}
    for k in decay_lags:
        shifted = by_asset.with_columns(
            pl.col("fwd_ret").shift(-(k - 1)).over("asset").alias("r_k")
        )
        d = (
            shifted.drop_nulls("r_k")
            .group_by("date")
            .agg(pl.corr("signal", "r_k", method="spearman").alias("ic"))
        )
        decay[k] = _num(d["ic"].mean())

    regime_ic: dict[str, float] = {}
    if regimes is not None:
        joined = ic.join(regimes.select("date", "regime"), on="date", how="inner")
        for reg, val in joined.group_by("regime").agg(pl.col("ic").mean()).iter_rows():
            regime_ic[str(reg)] = float(val)

    return SignalReport(
        name=name,
        periods_per_year=periods_per_year,
        ic=ic,
        quantile_returns=quantile_returns,
        long_short=long_short,
        ic_decay=decay,
        n_trials=n_trials,
        trial_sr_variance=trial_sr_variance,
        regime_ic=regime_ic,
    )


def vol_target(
    report: SignalReport,
    target_annual: float,
    *,
    window: int = 6,
    max_leverage: float = 2.0,
    cost_bps: float = 10.0,
) -> SignalReport:
    """Scale the long-short book to a volatility target (Barroso & Santa-Clara 2015).

    The scale applied to the return realized after ``date`` uses only returns realized up to
    ``date``: trailing ``window`` periods of net returns, shifted one period. Extra trading
    from changing the scale is charged at ``cost_bps``.
    """
    ls = report.long_short.sort("date")
    realized_vol = pl.col("net").rolling_std(window, min_samples=window).shift(1) * math.sqrt(
        report.periods_per_year
    )
    scale = (target_annual / realized_vol).clip(0.0, max_leverage)
    scaled = (
        ls.with_columns(scale.alias("scale"))
        .drop_nulls("scale")
        .with_columns(
            (pl.col("scale").diff().abs().fill_null(0.0) * cost_bps / 1e4).alias("_rescale_cost")
        )
        .with_columns(
            (pl.col("gross") * pl.col("scale")).alias("gross"),
            (pl.col("turnover") * pl.col("scale")).alias("turnover"),
            (pl.col("net") * pl.col("scale") - pl.col("_rescale_cost")).alias("net"),
        )
        .drop("_rescale_cost")
    )
    return replace(report, name=report.name, long_short=scaled)
