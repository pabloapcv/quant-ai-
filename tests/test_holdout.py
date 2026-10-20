from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from atlas.backtest.holdout import (
    HOLDOUT_START,
    HoldoutAlreadyOpened,
    HoldoutLog,
    open_holdout,
)
from atlas.backtest.trials import TrialLog
from atlas.models.expected_returns.ensemble import EnsembleConfig
from atlas.signals.industry import SIGNALS, research_panel
from atlas.signals.research import ResearchContext, signal_panel
from tests.test_research import _store


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> ResearchContext:
    path = tmp_path_factory.mktemp("holdout")
    store = _store(path, n_months=360, n_assets=12, start_year=1996)  # 1996-2025
    return ResearchContext(
        panel=research_panel(store, "ind12"), universe="ind12", log=TrialLog(path), n_quantiles=3
    )


def test_research_end_excludes_holdout_outcomes(ctx: ResearchContext) -> None:
    sig = SIGNALS["mom_12_1"]
    p = signal_panel(sig, replace(ctx, end=HOLDOUT_START))
    last = p["date"].max()
    assert last == date(2020, 11, 1)  # its 1-month outcome (Dec 2020) ends before 2021
    full_last = signal_panel(sig, ctx)["date"].max()
    assert isinstance(full_last, date)
    assert full_last > HOLDOUT_START


def test_open_once_logged_then_refused(ctx: ResearchContext, tmp_path: Path) -> None:
    cfg = EnsembleConfig(horizon=3, min_history=36, calib_min_dates=48)
    entry = open_holdout(tmp_path, ctx, cfg, family="ens", reason="test", code_sha="abc")
    assert entry["contaminated"] is True
    assert entry["reopening"] is False
    assert entry["ensemble"]["months"] > 0
    assert "mom_12_1" in entry["signals"]
    with pytest.raises(HoldoutAlreadyOpened):
        open_holdout(tmp_path, ctx, cfg, family="ens", reason="again", code_sha="abc")
    again = open_holdout(
        tmp_path, ctx, cfg, family="ens", reason="r", code_sha="abc", allow_reopen=True, clean=True
    )
    assert again["reopening"] is True
    assert again["contaminated"] is False
    log = HoldoutLog(tmp_path)
    assert len(log.openings("ind12", "ens")) == 2
    assert log.openings("ind12", "other") == []


def test_holdout_scores_only_use_holdout_months(ctx: ResearchContext, tmp_path: Path) -> None:
    from atlas.backtest.holdout import holdout_scores

    cfg = EnsembleConfig(horizon=3, min_history=36, calib_min_dates=48)
    res = holdout_scores(ctx, cfg)
    # 2021-01 .. 2025-09: a 3-month outcome must end by 2025-12
    assert res["ensemble"]["months"] == 57
    assert all(v["months"] <= 59 for v in res["signals"].values())
