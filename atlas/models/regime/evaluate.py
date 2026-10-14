"""Walk-forward evaluation of the regime engine (PRD G2, §8).

For every month-end in the window the engine is re-run as of that date (real-time data
only). Its quadrant is compared with the *ex-post* quadrant: the same month classified with
the final (latest) vintage. Recession probabilities are scored against realized NBER
recessions in the following 12 months.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
from sklearn.metrics import roc_auc_score

from atlas.models.regime.engine import RECESSION_SERIES, RegimeEngine
from atlas.models.regime.quadrant import classify_expr
from atlas.models.regime.recession import forward_label


@dataclass(frozen=True)
class RegimeBacktest:
    monthly: pl.DataFrame
    balanced_accuracy: float | None
    accuracy: float | None
    confusion: pl.DataFrame
    recession_brier: float | None
    recession_brier_baseline: float | None  # always predicting the training base rate
    recession_auc: float | None

    def summary(self) -> dict[str, float | int | None]:
        return {
            "months": self.monthly.height,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "recession_brier": self.recession_brier,
            "recession_brier_baseline": self.recession_brier_baseline,
            "recession_auc": self.recession_auc,
        }


def month_ends(start: date, end: date) -> list[date]:
    s = pl.date_range(start, end, interval="1mo", eager=True).dt.month_end()
    return [d for d in s.to_list() if start <= d <= end]


def walk_forward(
    engine: RegimeEngine, start: date, end: date, *, with_recession: bool = True
) -> RegimeBacktest:
    rows = []
    for at in month_ends(start, end):
        rep = engine.run(at, with_hmm=False, with_recession=with_recession)
        rows.append(
            {
                "as_of": at,
                "data_month": rep.data_month,
                "rt_growth": rep.growth.score,
                "rt_inflation": rep.inflation.score,
                "rt_regime": None if rep.regime is None else rep.regime.value,
                "recession_prob": None if rep.recession is None else rep.recession.prob_12m,
                "recession_base_rate": None
                if rep.recession is None
                else rep.recession.n_positive / rep.recession.n_obs,
            }
        )
    rt = pl.DataFrame(
        rows,
        schema={
            "as_of": pl.Date,
            "data_month": pl.Date,
            "rt_growth": pl.Float64,
            "rt_inflation": pl.Float64,
            "rt_regime": pl.String,
            "recession_prob": pl.Float64,
            "recession_base_rate": pl.Float64,
        },
    )

    # "Final" = the latest vintage of everything, one month past the newest fact.
    latest = engine.cube.latest_knowledge_date or end
    final_at = pl.select(pl.lit(latest).dt.offset_by("1mo").dt.month_end()).item()
    final, _ = engine.composites(final_at)
    final = final.select(
        pl.col("event_time").alias("data_month"),
        classify_expr(pl.col("growth"), pl.col("inflation")).alias("final_regime"),
    )
    monthly = rt.join(final, on="data_month", how="left")

    if RECESSION_SERIES in engine.cube:
        rec = (
            engine.cube.as_of(RECESSION_SERIES, final_at)
            .select(pl.col("event_time").dt.truncate("1mo").alias("data_month"), "value")
            .group_by("data_month")
            .agg(pl.col("value").max())
            .sort("data_month")
            .with_columns(forward_label(pl.col("value"), 12).alias("recession_next_12m"))
            .select("data_month", "recession_next_12m")
        )
        monthly = monthly.join(rec, on="data_month", how="left")
    else:
        monthly = monthly.with_columns(pl.lit(None, dtype=pl.Float64).alias("recession_next_12m"))

    return _score(monthly.sort("as_of"))


def _mean(s: pl.Series) -> float:
    m = s.cast(pl.Float64).mean()
    return float(m) if isinstance(m, int | float) else float("nan")


def _score(monthly: pl.DataFrame) -> RegimeBacktest:
    labelled = monthly.drop_nulls(["rt_regime", "final_regime"])
    confusion = (
        labelled.group_by(["final_regime", "rt_regime"]).len().sort(["final_regime", "rt_regime"])
    )
    bal_acc = acc = None
    if labelled.height:
        acc = _mean(labelled["rt_regime"] == labelled["final_regime"])
        recalls = labelled.group_by("final_regime").agg(
            (pl.col("rt_regime") == pl.col("final_regime")).mean().alias("recall")
        )
        bal_acc = _mean(recalls["recall"])

    scored = monthly.drop_nulls(["recession_prob", "recession_next_12m"])
    brier = base = auc = None
    if scored.height:
        y = scored["recession_next_12m"]
        brier = _mean((scored["recession_prob"] - y) ** 2)
        base = _mean((scored["recession_base_rate"] - y) ** 2)
        if y.n_unique() == 2:
            auc = float(roc_auc_score(y.to_numpy(), scored["recession_prob"].to_numpy()))
    return RegimeBacktest(monthly, bal_acc, acc, confusion, brier, base, auc)
