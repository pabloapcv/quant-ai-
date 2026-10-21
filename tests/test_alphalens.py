"""Cross-check (PRD §8.1): our IC engine against alphalens-reloaded on the same data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.crosssection import backtest_signal
from atlas.backtest.trials import TrialLog
from atlas.signals.industry import SIGNALS, research_panel
from atlas.signals.research import ResearchContext, signal_panel
from tests.test_research import _store

alphalens = pytest.importorskip("alphalens")


def test_rank_ic_matches_alphalens(tmp_path: Path) -> None:
    import alphalens.performance as perf
    import alphalens.utils as al_utils

    store = _store(tmp_path, n_months=180, n_assets=12, start_year=2000)
    ctx = ResearchContext(
        panel=research_panel(store, "ind12"), universe="ind12", log=TrialLog(tmp_path)
    )
    panel = signal_panel(SIGNALS["mom_12_1"], ctx).drop_nulls(["signal", "fwd_ret"])
    ours = backtest_signal(panel, "mom_12_1", n_quantiles=3)

    # alphalens wants prices and a (date, asset) factor series
    rets = pd.DataFrame(ctx.panel.select("date", "asset", "ret").to_dict(as_series=False))
    wide = rets.pivot(index="date", columns="asset", values="ret").sort_index().fillna(0.0)
    prices = (1 + wide).cumprod()
    prices.index = pd.DatetimeIndex(prices.index)
    # our signal at month t is scored on month t+1's return; alphalens computes the forward
    # return from price[t] to price[t+1], with prices indexed at month t's end
    factor = pd.DataFrame(panel.select("date", "asset", "signal").to_dict(as_series=False))
    factor["date"] = pd.DatetimeIndex(factor["date"])
    factor = factor.set_index(["date", "asset"])["signal"]
    data = al_utils.get_clean_factor_and_forward_returns(
        factor, prices, periods=(1,), quantiles=3, max_loss=1.0
    )
    ic = perf.factor_information_coefficient(data)
    theirs = float(ic.iloc[:, 0].mean())
    assert np.isfinite(theirs)
    assert ours.ic_mean == pytest.approx(theirs, abs=1e-6)
