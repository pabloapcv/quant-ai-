"""Walk-forward ensemble of cross-sectional signals -> expected returns (PRD §7.2, §7.5).

At every month ``t``, using only information available at the end of ``t``:

1. **Components.** Each signal is z-scored across assets.
2. **Weights.** Each component's weight comes from its *past* information coefficient
   (ICs of dates ``< t``, whose returns are realized by ``t``): positive mean ICs only,
   shrunk ``shrink`` toward equal weight, after ``min_history`` months.
3. **Score.** Weighted sum of component z-scores, re-standardized across assets.
4. **Expected excess return** (vs the equal-weight universe): the score times the average
   past Fama-MacBeth slope of excess returns on the score.
5. **Probability of outperforming**: logistic (Platt) calibration of 1{beats the median
   asset} on the
   score, fitted on past (score, outcome) pairs, refit every ``calib_refit`` months. The PRD
   names isotonic regression; with ICs near 0.04 isotonic is flat in the middle and overfits
   the sparse tails, so a two-parameter monotone fit is used instead.
6. **Interval**: split-conformal — past residuals' quantiles added to the forecast.
7. **Agreement**: share of available components whose sign matches the ensemble's.

Horizon: the target is the compounded return over the next ``horizon`` months. An outcome
for date ``s`` is known only at ``s + horizon``, so every fit at ``t`` uses dates
``s <= t - horizon``.

The output is one row per (date, asset), the input to ``atlas.views``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from atlas.backtest.metrics import newey_west_tstat
from atlas.signals.research import ResearchContext, signal_panel

DEFAULT_COMPONENTS = (
    "mom_12_1",
    "resmom_ff3_36",
    "value_log_bm",
    "low_vol_36",
    "ltr_60_12",
    "reversal_1m",
)


@dataclass(frozen=True)
class EnsembleConfig:
    # None = every distinct ranking in the universe's signal set
    components: tuple[str, ...] | None = DEFAULT_COMPONENTS
    min_history: int = 60  # months of past ICs before a component can get weight
    shrink: float = 0.5  # toward equal weight among positive-IC components
    horizon: int = 1  # months
    calib_refit: int = 12  # months between calibration refits
    calib_min_dates: int = 60
    conformal_window: int = 120  # months of past residuals
    coverage: float = 0.8  # central interval (p10 / p90)
    # Walk-forward promotion gates (PRD §8.4). A component may get weight only while its IC
    # history *up to that date* clears them; re-checked every ``calib_refit`` months.
    gate_tstat: float | None = None  # Newey-West IC t-stat, e.g. 2.0
    gate_regimes: int | None = None  # positive mean IC in >= this many market regimes
    gate_min_regime_obs: int = 12  # months a regime needs before its IC counts
    version: str = "ens_v1"


@dataclass(frozen=True)
class EnsembleRun:
    config: EnsembleConfig
    forecasts: pl.DataFrame  # date, asset, score, exp_excess, p_lo, p_hi, prob, agreement, ...
    weights: pl.DataFrame  # date, <component> weights
    component_list: tuple[str, ...] = ()
    components: dict[str, pl.DataFrame] = field(default_factory=dict)


def _xs_z(col: str) -> pl.Expr:
    c = pl.col(col)
    return ((c - c.mean().over("date")) / c.std().over("date")).fill_nan(None)


def forward_return(horizon: int) -> pl.Expr:
    """Compounded return over months t+1 .. t+horizon (null where not yet realized)."""
    lr = pl.col("ret").log1p()
    ahead = lr.reverse().rolling_sum(window_size=horizon, min_samples=horizon).reverse()
    return (ahead.shift(-1).exp() - 1).over("asset")


def component_names(ctx: ResearchContext, cfg: EnsembleConfig) -> list[str]:
    """Components for this universe: the configured ones it has, else all distinct rankings."""
    distinct = [n for n, s in ctx.signals.items() if s.ranking_of is None]
    if cfg.components is None:
        return distinct
    chosen = [c for c in cfg.components if c in ctx.signals]
    return chosen or distinct


def component_panel(ctx: ResearchContext, names: Sequence[str], horizon: int = 1) -> pl.DataFrame:
    """(date, asset, fwd_ret, excess, z_<name>...); ``fwd_ret`` spans ``horizon`` months."""
    out: pl.DataFrame | None = None
    for name in names:
        p = (
            signal_panel(ctx.signals[name], ctx)
            .sort(["asset", "date"])
            .select(
                "date",
                "asset",
                forward_return(horizon).alias("fwd_ret"),
                _xs_z("signal").alias(f"z_{name}"),
            )
        )
        out = (
            p
            if out is None
            else out.join(p.drop("fwd_ret"), on=["date", "asset"], how="full", coalesce=True)
        )
    assert out is not None
    # excess: vs the equal-weight mean (what return forecasts are relative to)
    # excess_med: vs the median asset; cross-sectional returns are right-skewed, so "beats
    # the mean" happens less than half the time while "beats the median" is exactly 50/50
    return out.with_columns(
        (pl.col("fwd_ret") - pl.col("fwd_ret").mean().over("date")).alias("excess"),
        (pl.col("fwd_ret") - pl.col("fwd_ret").median().over("date")).alias("excess_med"),
    ).sort(["date", "asset"])


def _past_mean(values: np.ndarray, min_count: int, lag: int = 1) -> np.ndarray:
    """Expanding mean of rows ``<= t - lag`` for each row t (NaN-aware), NaN until min_count."""
    ok = np.isfinite(values)
    csum = np.cumsum(np.where(ok, values, 0.0), axis=0)
    ccount = np.cumsum(ok, axis=0)
    pad = np.zeros((lag, values.shape[1]))
    prev_sum = np.vstack([pad, csum[:-lag]])[: values.shape[0]]
    prev_cnt = np.vstack([pad, ccount[:-lag]])[: values.shape[0]]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = prev_sum / prev_cnt
    mean[prev_cnt < min_count] = np.nan
    return np.asarray(mean, dtype=np.float64)


def _eligibility(
    ic_mat: np.ndarray, dates: pl.Series, regimes: pl.DataFrame | None, cfg: EnsembleConfig
) -> np.ndarray:
    """(n_dates x n_components) booleans: may the component get weight at that date?

    Uses only ICs whose outcomes are realized by the date (rows <= t - horizon). Without any
    gate configured, every component is eligible (weights still need a positive past IC).
    """
    n_dates, n_comp = ic_mat.shape
    if cfg.gate_tstat is None and cfg.gate_regimes is None:
        return np.ones((n_dates, n_comp), dtype=bool)
    if cfg.gate_regimes is not None and regimes is None:
        raise ValueError("the regime gate needs ResearchContext.regimes (market_regimes)")
    reg_by_date: dict[object, str] = {}
    if regimes is not None:
        reg_by_date = dict(zip(regimes["date"].to_list(), regimes["regime"].to_list(), strict=True))
    reg_labels = np.array([reg_by_date.get(d) for d in dates.to_list()], dtype=object)
    h = cfg.horizon
    lags = max(h, 3)  # overlapping h-month outcomes make ICs autocorrelated
    out = np.zeros((n_dates, n_comp), dtype=bool)
    current = np.zeros(n_comp, dtype=bool)
    for t in range(n_dates):
        if t % cfg.calib_refit == 0:
            last = t - h + 1
            current = np.zeros(n_comp, dtype=bool)
            for j in range(n_comp):
                past = ic_mat[: max(last, 0), j]
                ok = np.isfinite(past)
                if ok.sum() < cfg.min_history:
                    continue
                passed = True
                if cfg.gate_tstat is not None:
                    passed &= bool(newey_west_tstat(past[ok], lags=lags) >= cfg.gate_tstat)
                if passed and cfg.gate_regimes is not None:
                    labels = reg_labels[: max(last, 0)][ok]
                    vals = past[ok]
                    positive = 0
                    for r in set(labels) - {None}:
                        sel = vals[labels == r]
                        if sel.size >= cfg.gate_min_regime_obs and sel.mean() > 0:
                            positive += 1
                    passed &= positive >= cfg.gate_regimes
                current[j] = passed
        out[t] = current
    return out


def walk_forward(ctx: ResearchContext, config: EnsembleConfig | None = None) -> EnsembleRun:
    cfg = config or EnsembleConfig()
    names = component_names(ctx, cfg)
    zcols = [f"z_{n}" for n in names]
    h = cfg.horizon
    panel = component_panel(ctx, names, h)
    dates = panel.select("date").unique().sort("date")["date"]

    # --- 2. weights from past ICs -------------------------------------------------
    ic = (
        panel.group_by("date")
        .agg(pl.corr(z, "fwd_ret", method="spearman").alias(z) for z in zcols)
        .join(dates.to_frame(), on="date", how="right")
        .sort("date")
    )
    ic_mat = ic.select(zcols).to_numpy().astype(np.float64)
    past_ic = _past_mean(ic_mat, cfg.min_history, lag=h)
    eligible = _eligibility(ic_mat, dates, ctx.regimes, cfg)
    positive = np.where(eligible & np.isfinite(past_ic) & (past_ic > 0), past_ic, 0.0)
    n_pos = (positive > 0).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic_w = positive / positive.sum(axis=1, keepdims=True)
        eq_w = (positive > 0) / n_pos
    w = np.where(n_pos > 0, cfg.shrink * eq_w + (1 - cfg.shrink) * ic_w, 0.0)
    weights = pl.DataFrame({"date": dates, **{n: w[:, i] for i, n in enumerate(names)}})

    # --- 3. score -------------------------------------------------------------------
    wlong = weights.rename({n: f"w_{n}" for n in names})
    scored = panel.join(wlong, on="date", how="left").with_columns(
        pl.sum_horizontal(
            pl.col(z).fill_null(0.0) * pl.col(f"w_{n}") for z, n in zip(zcols, names, strict=True)
        ).alias("raw_score"),
        pl.sum_horizontal(pl.col(f"w_{n}") for n in names).alias("w_total"),
    )
    scored = scored.with_columns(
        pl.when(pl.col("w_total") > 0).then(_xs_z("raw_score")).otherwise(None).alias("score")
    )
    # --- 7. agreement ------------------------------------------------------------------
    agree_num = pl.sum_horizontal(
        ((pl.col(z).sign() == pl.col("score").sign()) & (pl.col(f"w_{n}") > 0)).cast(pl.Float64)
        for z, n in zip(zcols, names, strict=True)
    )
    agree_den = pl.sum_horizontal(
        (pl.col(z).is_not_null() & (pl.col(f"w_{n}") > 0)).cast(pl.Float64)
        for z, n in zip(zcols, names, strict=True)
    )
    scored = scored.with_columns(
        pl.when(agree_den > 0).then(agree_num / agree_den).otherwise(None).alias("agreement")
    )

    # --- 4. Fama-MacBeth slope, past average ------------------------------------------
    slopes = (
        scored.drop_nulls(["score", "excess"])
        .group_by("date")
        .agg((pl.cov("score", "excess") / pl.col("score").var()).alias("slope"))
        .join(dates.to_frame(), on="date", how="right")
        .sort("date")
    )
    past_slope = _past_mean(
        slopes.select("slope").to_numpy().astype(np.float64), cfg.min_history, lag=h
    )
    scored = scored.join(
        pl.DataFrame({"date": dates, "past_slope": past_slope[:, 0]}), on="date", how="left"
    ).with_columns((pl.col("past_slope").fill_nan(None) * pl.col("score")).alias("exp_excess"))

    scored = _conformal_and_calibration(scored.sort(["date", "asset"]), dates, cfg)
    keep = [
        "date",
        "asset",
        "score",
        "exp_excess",
        "p_lo",
        "p_hi",
        "prob",
        "agreement",
        "fwd_ret",
        "excess",
        "excess_med",
        *zcols,
    ]
    return EnsembleRun(
        config=cfg,
        forecasts=scored.select(keep),
        weights=weights,
        component_list=tuple(names),
    )


def _conformal_and_calibration(
    df: pl.DataFrame, dates: pl.Series, cfg: EnsembleConfig
) -> pl.DataFrame:
    """Add conformal bounds and isotonic probabilities using only earlier dates."""
    date_idx = {d: i for i, d in enumerate(dates.to_list())}
    di = np.array([date_idx[d] for d in df["date"].to_list()])
    score = df["score"].to_numpy().astype(np.float64)
    exp_ex = df["exp_excess"].to_numpy().astype(np.float64)
    excess = df["excess"].to_numpy().astype(np.float64)
    beats_median = df["excess_med"].to_numpy().astype(np.float64)
    resid = excess - exp_ex
    n_dates = len(dates)
    alpha = (1 - cfg.coverage) / 2

    p_lo = np.full(df.height, np.nan)
    p_hi = np.full(df.height, np.nan)
    prob = np.full(df.height, np.nan)
    rows_by_date = [np.flatnonzero(di == t) for t in range(n_dates)]
    h = cfg.horizon
    model: LogisticRegression | None = None
    for t in range(n_dates):
        rows = rows_by_date[t]
        if not rows.size:
            continue
        # outcomes for date s are known at s + h, so only dates <= t - h are usable
        last = t - h + 1  # exclusive end of usable dates
        lo_t = max(0, last - cfg.conformal_window)
        past = np.concatenate(rows_by_date[lo_t:last]) if last > lo_t else np.array([], dtype=int)
        past_resid = resid[past][np.isfinite(resid[past])] if past.size else np.array([])
        if past_resid.size >= 50:
            q_lo, q_hi = np.quantile(past_resid, [alpha, 1 - alpha])
            p_lo[rows] = exp_ex[rows] + q_lo
            p_hi[rows] = exp_ex[rows] + q_hi
        if last >= cfg.calib_min_dates and (model is None or t % cfg.calib_refit == 0):
            hist = np.concatenate(rows_by_date[:last])
            # with an odd number of assets one sits exactly on the median: it neither beats
            # nor trails it, and counting it as a loss would bias the base rate below 50%
            ok = (
                np.isfinite(score[hist])
                & np.isfinite(beats_median[hist])
                & (beats_median[hist] != 0)
            )
            y = (beats_median[hist][ok] > 0).astype(int)
            if ok.sum() >= 200 and 0 < y.sum() < y.size:
                model = LogisticRegression(C=1e6)
                model.fit(score[hist][ok].reshape(-1, 1), y)
        if model is not None:
            ok_now = np.isfinite(score[rows])
            prob[rows[ok_now]] = model.predict_proba(score[rows][ok_now].reshape(-1, 1))[:, 1]
    return df.with_columns(
        pl.Series("p_lo", p_lo).fill_nan(None),
        pl.Series("p_hi", p_hi).fill_nan(None),
        pl.Series("prob", prob).fill_nan(None),
    )
