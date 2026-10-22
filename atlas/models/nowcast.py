"""US real GDP nowcast research: mixed-frequency DFM + bridge, evaluated against GDPNow.

Status: research only, **not** a published model output. Ramon reports the Atlanta Fed's
GDPNow (point in time, from ALFRED vintages) as its GDP nowcast; see ``gdpnow_as_of``.

Model: ``statsmodels`` ``DynamicFactorMQ`` (one VAR(2) factor, AR(1) idiosyncratic terms)
on 15 monthly activity series and quarterly real GDP growth. Unpublished months of the
target quarter take the factor model's prediction; monthly growth is aggregated to quarters
with Mariano-Murasawa weights, and a bridge regression maps the quarterly indicators
(real PCE, hours, IP, core capex shipments, construction, exports, imports, trade balance,
inventory acceleration, the DFM's own GDP estimate) to GDP growth. ``gdp_saar`` is the
average of the bridge and DFM estimates (the best of the variants tried).

Real-time evaluation, 2012Q1-2025Q4 excluding 2020 (``atlas nowcast evaluate``), RMSE vs
the BEA advance estimate, % SAAR:

=================  ======  =====  ========  =====================
before release     Ramon   DFM    GDPNow    s.d. of the outcome
=================  ======  =====  ========  =====================
1 hour (52 q)      1.61    1.76   0.86      1.65
30 days (37 q)     1.89    1.95   1.09      1.73
60 days (33 q)     1.88    1.98   1.40      1.81
=================  ======  =====  ========  =====================

The PRD's Phase 2 exit criterion (within 10% of GDPNow) is not met, and the model is no
better than a constant forecast. Even in sample, with revised data and complete quarters,
these monthly aggregates explain about 63% of the advance estimate's variance (RMSE ~1.2);
GDPNow maps BEA source data component by component (detailed deflators, services PCE,
Census advance trade and inventories), which free monthly aggregates cannot reproduce.

Point in time: every input, GDP included, is read from the as-of vintage. Pandemic months
(March-September 2020, 2020Q2-Q3 GDP) are masked when estimating for later quarters.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel

from atlas.pit.cube import VintageCube

GDP_SERIES = "US.GDP.REAL"
GDPNOW_SERIES = "US.GDP.NOWCAST.ATL"


@dataclass(frozen=True)
class NowcastInput:
    series_id: str
    transform: Literal["logdiff", "diff"]
    name: str


INPUTS: tuple[NowcastInput, ...] = (
    NowcastInput("US.LABOR.NFP", "logdiff", "payrolls"),
    NowcastInput("US.LABOR.UNRATE", "diff", "unemployment"),
    NowcastInput("US.IP", "logdiff", "industrial_production"),
    NowcastInput("US.RETAIL", "logdiff", "retail_sales"),
    NowcastInput("US.PCE.REAL", "logdiff", "real_pce"),
    NowcastInput("US.INCOME.REAL.EXTRANS", "logdiff", "real_income_ex_transfers"),
    NowcastInput("US.SALES.MFGTRADE.REAL", "logdiff", "real_mfg_trade_sales"),
    NowcastInput("US.HOUSING.STARTS", "logdiff", "housing_starts"),
    NowcastInput("US.CONSTRUCTION.TOTAL", "logdiff", "construction"),
    NowcastInput("US.TRADE.EXPORTS", "logdiff", "exports"),
    NowcastInput("US.TRADE.IMPORTS", "logdiff", "imports"),
    NowcastInput("US.TRADE.BALANCE", "diff", "trade_balance"),
    NowcastInput("US.CAPEX.SHIPMENTS.CORE", "logdiff", "core_capex_shipments"),
    NowcastInput("US.LABOR.HOURS.AGG", "logdiff", "aggregate_hours"),
    NowcastInput("US.INVENTORIES.REAL", "logdiff", "real_inventories"),
)
# Bridge regressors (quarterly, Mariano-Murasawa aggregates of the filled monthly inputs).
BRIDGE = (
    "real_pce",
    "aggregate_hours",
    "industrial_production",
    "core_capex_shipments",
    "construction",
    "exports",
    "imports",
    "trade_balance",
    "inventory_accel",
)
MM_WEIGHTS = np.array([1, 2, 3, 2, 1]) / 3  # months t-4 .. t for the quarter ending at t
SERIES_IDS = (GDP_SERIES, GDPNOW_SERIES, *(i.series_id for i in INPUTS))
COVID_MONTHS = (pd.Period("2020-03", "M"), pd.Period("2020-09", "M"))
COVID_QUARTERS = (pd.Period("2020Q2", "Q"), pd.Period("2020Q3", "Q"))


class Nowcast(BaseModel):
    quarter: str  # e.g. "2019Q1"
    as_of: datetime
    gdp_saar: float  # average of the bridge and DFM estimates
    dfm_saar: float  # the factor model's own GDP estimate (also a bridge regressor)
    months_observed: dict[str, int]  # per input: months of the target quarter published


def _monthly(cube: VintageCube, at: date | datetime, start: date) -> pd.DataFrame:
    cols = {}
    for inp in INPUTS:
        df = cube.as_of(inp.series_id, at).drop_nulls("value").sort("event_time")
        if df.is_empty():
            continue
        s = pd.Series(
            df["value"].to_numpy(),
            index=pd.PeriodIndex(df["event_time"].to_list(), freq="M"),
        )
        s = s[~s.index.duplicated(keep="last")]
        growth = 100 * np.log(s).diff() if inp.transform == "logdiff" else s.diff()
        cols[inp.name] = growth
    out = pd.DataFrame(cols).sort_index()
    return out[out.index >= pd.Period(start, "M")]


def _gdp_growth(cube: VintageCube, at: date | datetime, start: date) -> pd.Series:
    df = cube.as_of(GDP_SERIES, at).drop_nulls("value").sort("event_time")
    s = pd.Series(
        df["value"].to_numpy(), index=pd.PeriodIndex(df["event_time"].to_list(), freq="Q")
    )
    g = 100 * np.log(s).diff()
    return g[g.index >= pd.Period(start, "Q")].rename("gdp")


def _annualize(q_growth_pct: float) -> float:
    return (math.exp(4 * q_growth_pct / 100) - 1) * 100


def _quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Mariano-Murasawa aggregation of monthly growth to quarterly growth of 3-month averages."""
    agg = monthly.rolling(5).apply(lambda w: float(np.dot(w, MM_WEIGHTS)), raw=True)
    q = agg[agg.index.month % 3 == 0]
    q.index = q.index.asfreq("Q")
    # Inventory investment drives GDP through its change: the acceleration of inventories.
    q["inventory_accel"] = q["real_inventories"].diff()
    return q


def nowcast(
    cube: VintageCube,
    quarter: pd.Period,
    at: datetime,
    *,
    start: date = date(1993, 1, 1),
    start_params: np.ndarray | None = None,
) -> tuple[Nowcast, np.ndarray]:
    """Nowcast of ``quarter`` with data as of ``at``; also returns DFM params (warm start)."""
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    monthly = _monthly(cube, at, start)
    gdp = _gdp_growth(cube, at, start)
    gdp = gdp[gdp.index < quarter]  # the target quarter is what we estimate
    last_month = quarter.asfreq("M", how="end")
    monthly = monthly[monthly.index <= last_month]
    masked = quarter > COVID_QUARTERS[1]
    if masked:
        lo, hi = COVID_MONTHS
        monthly.loc[(monthly.index >= lo) & (monthly.index <= hi), :] = np.nan
        gdp.loc[gdp.index.isin(list(COVID_QUARTERS))] = np.nan

    in_q = monthly[monthly.index.asfreq("Q") == quarter]
    observed = {c: int(in_q[c].notna().sum()) for c in monthly.columns}
    full_index = pd.period_range(monthly.index.min(), last_month, freq="M")
    monthly = monthly.reindex(full_index)

    mod = DynamicFactorMQ(
        monthly,
        endog_quarterly=gdp.to_frame(),
        factors=1,
        factor_orders=2,
        idiosyncratic_ar1=True,
        standardize=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=False, maxiter=300, tolerance=1e-6, start_params=start_params)
    pred = res.predict(start=full_index[0], end=last_month)
    dfm_growth = pred["gdp"]  # quarterly GDP growth implied at each quarter-end month
    dfm_q = dfm_growth[dfm_growth.index.month % 3 == 0]
    dfm_q.index = dfm_q.index.asfreq("Q")

    # Ragged edge: months not yet published take the model's prediction.
    filled = monthly.copy()
    recent = filled.index > last_month - 12
    filled.loc[recent] = filled.loc[recent].fillna(pred.loc[filled.index[recent], monthly.columns])
    q = _quarterly(filled)
    q["dfm"] = dfm_q
    cols = [*BRIDGE, "dfm"]
    train = q.join(gdp, how="inner").dropna(subset=[*cols, "gdp"])
    if masked:
        train = train[~train.index.isin(list(COVID_QUARTERS))]
    x = np.column_stack([np.ones(len(train)), train[cols].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, train["gdp"].to_numpy(), rcond=None)
    row = q.loc[quarter, cols].to_numpy(dtype=float)
    bridge = float(np.concatenate([[1.0], row]) @ beta) if np.all(np.isfinite(row)) else math.nan
    dfm_now = float(dfm_q.loc[quarter])
    return (
        Nowcast(
            quarter=str(quarter),
            as_of=at,
            gdp_saar=_annualize((bridge + dfm_now) / 2 if math.isfinite(bridge) else dfm_now),
            dfm_saar=_annualize(dfm_now),
            months_observed=observed,
        ),
        np.asarray(res.params),
    )


# ---------------------------------------------------------------------------- evaluation
def advance_release(cube: VintageCube, quarter: pd.Period) -> tuple[datetime, float] | None:
    """(time of the advance estimate, its % SAAR growth) for ``quarter``; None if not out."""
    obs = cube._by_series.get(GDP_SERIES)
    if obs is None:
        return None
    q_start = quarter.start_time.date()
    rel = obs.filter((pl.col("event_time") == q_start) & ~pl.col("is_approx_vintage"))
    if rel.is_empty():
        return None
    t = rel["knowledge_time"].min()
    assert isinstance(t, datetime)
    hist = cube.as_of(GDP_SERIES, t).sort("event_time")
    prev = quarter - 1
    levels = dict(zip(hist["event_time"].to_list(), hist["value"].to_list(), strict=True))
    cur, before = levels.get(q_start), levels.get(prev.start_time.date())
    if cur is None or before is None:
        return None
    return t, (math.pow(cur / before, 4) - 1) * 100


def gdpnow_as_of(cube: VintageCube, quarter: pd.Period, at: datetime) -> float | None:
    df = cube.as_of(GDPNOW_SERIES, at).filter(pl.col("event_time") == quarter.start_time.date())
    return None if df.is_empty() else float(df["value"][-1])


def gdp_nowcast(cube: VintageCube, at: date | datetime) -> dict[str, object] | None:
    """Latest GDPNow reading as known at ``at``: the quarter it covers and when it was made."""
    if GDPNOW_SERIES not in cube:
        return None
    df = cube.as_of(GDPNOW_SERIES, at).drop_nulls("value").sort("event_time")
    if df.is_empty():
        return None
    last = df.row(-1, named=True)
    q = pd.Period(last["event_time"], "Q")
    return {
        "quarter": str(q),
        "gdp_saar_pct": round(float(last["value"]), 2),
        "updated": last["knowledge_time"].date().isoformat(),
        "source": "Atlanta Fed GDPNow (via ALFRED)",
    }


class EvalRow(BaseModel):
    quarter: str
    days_before_release: int
    as_of: datetime
    actual_advance: float
    atlas: float
    dfm: float
    gdpnow: float | None


def evaluate(
    cube: VintageCube,
    quarters: Sequence[pd.Period],
    *,
    days_before: Sequence[int] = (1,),
    log: bool = False,
) -> pl.DataFrame:
    """Real-time nowcasts vs GDPNow vs the advance estimate, for each quarter and horizon.

    ``days_before=1`` is one hour before the advance release (GDPNow's final nowcast).
    """
    rows: list[EvalRow] = []
    params: np.ndarray | None = None
    for q in quarters:
        rel = advance_release(cube, q)
        if rel is None:
            continue
        t_rel, actual = rel
        for d in sorted(days_before, reverse=True):
            at = t_rel - (timedelta(hours=1) if d <= 1 else timedelta(days=d))
            try:
                nc, params = nowcast(cube, q, at, start_params=params)
            except (ValueError, np.linalg.LinAlgError):
                params = None
                continue
            row = EvalRow(
                quarter=str(q),
                days_before_release=d,
                as_of=at,
                actual_advance=actual,
                atlas=nc.gdp_saar,
                dfm=nc.dfm_saar,
                gdpnow=gdpnow_as_of(cube, q, at),
            )
            rows.append(row)
            if log:
                print(
                    f"{row.quarter} -{d:>2}d atlas {row.atlas:+6.2f} gdpnow "
                    f"{row.gdpnow if row.gdpnow is not None else float('nan'):+6.2f} "
                    f"actual {row.actual_advance:+6.2f}",
                    flush=True,
                )
    return pl.DataFrame([r.model_dump() for r in rows])


def rmse_summary(df: pl.DataFrame, *, exclude: Sequence[str] = ()) -> pl.DataFrame:
    """RMSE of atlas and GDPNow vs the advance estimate per horizon (common quarters)."""
    d = df.filter(~pl.col("quarter").is_in(list(exclude))).drop_nulls("gdpnow")
    return (
        d.group_by("days_before_release")
        .agg(
            pl.len().alias("quarters"),
            ((pl.col("atlas") - pl.col("actual_advance")) ** 2).mean().sqrt().alias("rmse_atlas"),
            ((pl.col("dfm") - pl.col("actual_advance")) ** 2).mean().sqrt().alias("rmse_dfm_only"),
            ((pl.col("gdpnow") - pl.col("actual_advance")) ** 2).mean().sqrt().alias("rmse_gdpnow"),
        )
        .with_columns((pl.col("rmse_atlas") / pl.col("rmse_gdpnow")).alias("ratio"))
        .sort("days_before_release", descending=True)
    )
