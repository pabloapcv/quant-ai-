from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from atlas.pit.store import PITStore
from atlas.signals.industry import (
    FAMILIES,
    SIGNALS,
    long_term_reversal,
    low_volatility,
    momentum,
    returns_panel,
    reversal,
)
from tests.conftest import make_obs_frame, utc

REPO = Path(__file__).resolve().parents[1]


def _toy_panel() -> pl.DataFrame:
    dates = [date(2020, m, 1) for m in range(1, 7)]
    rets = {"A": [0.10, 0.10, 0.10, -0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    return pl.DataFrame(
        [{"date": d, "asset": a, "ret": r[i]} for a, r in rets.items() for i, d in enumerate(dates)]
    ).sort(["asset", "date"])


def test_momentum_skip_and_window() -> None:
    out = momentum(3, 1).compute(_toy_panel()).filter(pl.col("asset") == "A")
    sig = out["signal"].to_list()
    assert sig[:2] == [None, None]
    # at Mar: log1p over Jan..Feb (lookback 3, skip 1)
    assert sig[2] == pytest.approx(2 * math.log1p(0.10))
    # at Apr: Feb..Mar -> excludes April's -10%
    assert sig[3] == pytest.approx(2 * math.log1p(0.10))
    assert out["fwd_ret"].to_list()[:2] == [0.10, 0.10]
    assert out["fwd_ret"].to_list()[-1] is None


def test_other_signals() -> None:
    p = _toy_panel()
    rev = reversal().compute(p).filter(pl.col("asset") == "A")["signal"].to_list()
    assert rev[3] == pytest.approx(0.10)
    lv = low_volatility(12).compute(p)["signal"]
    assert lv.null_count() == p.height  # needs >= 12 months
    ltr = long_term_reversal(3, 1).compute(p).filter(pl.col("asset") == "A")["signal"].to_list()
    assert ltr[2] == pytest.approx(-2 * math.log1p(0.10))
    assert "mom_12_1" in SIGNALS
    assert len(FAMILIES["momentum"]) == 12


def _kf_store(path: Path, n_months: int = 96, n_assets: int = 12) -> PITStore:
    rng = np.random.default_rng(5)
    rows = []
    for i in range(n_assets):
        for t in range(n_months):
            d = date(2010 + t // 12, t % 12 + 1, 1)
            rows.append((f"FF.IND12.X{i}", d, utc(2026, 10, 1), float(rng.normal(1, 5))))
    store = PITStore(path)
    store.write(make_obs_frame(rows))
    return store


def test_returns_panel(tmp_path: Path) -> None:
    store = _kf_store(tmp_path, n_months=24, n_assets=3)
    panel = returns_panel(store, "ind12", at=datetime(2026, 10, 2, tzinfo=UTC))
    assert panel.height == 72
    assert set(panel["asset"].unique().to_list()) == {"X0", "X1", "X2"}
    biggest = panel["ret"].abs().max()
    assert isinstance(biggest, float)
    assert biggest < 1.0  # percent -> fraction
    with pytest.raises(ValueError, match="ingest kenfrench"):
        returns_panel(store, "ind49")


def test_cli_research(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    _kf_store(data)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    runner = CliRunner()
    try:
        res = runner.invoke(app, ["research", "list"])
        assert res.exit_code == 0, res.output
        assert "mom_12_1" in res.output

        res = runner.invoke(
            app, ["research", "signal", "mom_6_1", "--universe", "ind12", "--quantiles", "3"]
        )
        assert res.exit_code == 0, res.output
        assert "ic_tstat_nw" in res.output
        assert "quantile_ann_return" in res.output

        res = runner.invoke(
            app,
            [
                "research",
                "signal",
                "mom_6_1",
                "--universe",
                "ind12",
                "--json",
                "--start",
                "2012-01-01",
            ],
        )
        assert res.exit_code == 0, res.output
        assert '"signal": "mom_6_1"' in res.output

        res = runner.invoke(
            app,
            [
                "research",
                "family",
                "low_volatility",
                "--universe",
                "ind12",
                "--quantiles",
                "3",
                "--splits",
                "4",
            ],
        )
        assert res.exit_code == 0, res.output
        assert "PBO" in res.output

        assert runner.invoke(app, ["research", "signal", "nope"]).exit_code != 0
        assert runner.invoke(app, ["research", "family", "nope"]).exit_code != 0
        assert runner.invoke(app, ["ingest", "kenfrench"]).exit_code != 0
        assert (data / "research" / "trials.jsonl").exists()
    finally:
        get_settings.cache_clear()
