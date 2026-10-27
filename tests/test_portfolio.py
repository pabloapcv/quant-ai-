from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from atlas.models.portfolio.optimizer import PortfolioConfig, backtest_portfolio, optimize_active
from atlas.models.risk.covariance import RiskConfig, covariance, ewma_covariance
from tests.test_research import _store

REPO = Path(__file__).resolve().parents[1]


def test_covariance_is_psd_and_shrunk() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, (60, 49))  # T barely above N: sample cov is ill-conditioned
    cov = covariance(x)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.linalg.cond(cov) < np.linalg.cond(np.cov(x, rowvar=False))
    with pytest.raises(ValueError, match="observations"):
        covariance(x[:10], RiskConfig(min_obs=36))
    ew = ewma_covariance(x, 12.0)
    assert ew.shape == (49, 49)


def test_optimizer_respects_constraints() -> None:
    rng = np.random.default_rng(1)
    n = 12
    cov = covariance(rng.normal(0, 0.05, (120, n)))
    bench = np.full(n, 1 / n)
    alpha = np.linspace(-0.01, 0.01, n)
    cfg = PortfolioConfig(te_annual=0.03, max_active=0.04, cost_bps=0)
    w = optimize_active(alpha, cov, bench, bench, cfg)
    active = w - bench
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert np.abs(active).max() <= 0.04 + 1e-6
    assert np.sqrt(active @ cov @ active * 12) <= 0.03 + 1e-4
    assert active[-1] > 0 > active[0]  # tilts toward the best forecast
    # large turnover aversion keeps the previous portfolio
    sticky = optimize_active(alpha, cov, bench, bench, PortfolioConfig(cost_bps=1e6))
    assert np.allclose(sticky, bench, atol=1e-4)


def test_backtest_portfolio_with_perfect_foresight() -> None:
    rng = np.random.default_rng(2)
    dates = [date(2000 + t // 12, t % 12 + 1, 1) for t in range(120)]
    assets = [f"A{i}" for i in range(10)]
    rets = rng.normal(0.008, 0.05, (120, 10))
    panel = pl.DataFrame(
        [
            {"date": d, "asset": a, "ret": float(rets[i, j])}
            for i, d in enumerate(dates)
            for j, a in enumerate(assets)
        ]
    )
    # forecast = next month's actual excess return: the portfolio must beat the benchmark
    fwd = np.vstack([rets[1:], np.full((1, 10), np.nan)])
    fc = pl.DataFrame(
        [
            {"date": d, "asset": a, "exp_excess": float(fwd[i, j] - np.nanmean(fwd[i]))}
            for i, d in enumerate(dates[:-1])
            for j, a in enumerate(assets)
        ]
    )
    bt = backtest_portfolio(fc, panel, cfg=PortfolioConfig(cost_bps=0))
    s = bt.summary()
    assert s["information_ratio_net"] is not None
    assert s["information_ratio_net"] > 3
    assert bt.weights.group_by("date").agg(pl.col("weight").sum())[
        "weight"
    ].to_numpy() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no month"):
        backtest_portfolio(fc.filter(pl.col("date") < date(2001, 1, 1)), panel)


def test_cli_portfolio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    _store(data, n_months=240, start_year=1995)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        runner = CliRunner()
        out = tmp_path / "pf.csv"
        res = runner.invoke(
            app,
            [
                "portfolio",
                "backtest",
                "--universe",
                "ind12",
                "--start",
                "2005-01-01",
                "--out",
                str(out),
            ],
        )
        assert res.exit_code == 0, res.output
        assert "information_ratio_net" in res.output
        assert pl.read_csv(out).height > 0
        res = runner.invoke(app, ["portfolio", "now", "--universe", "ind12"])
        assert res.exit_code == 0, res.output
        assert "ex-ante TE" in res.output
    finally:
        get_settings.cache_clear()


def test_black_litterman_shrinks_uncertain_views() -> None:
    import numpy as np

    from atlas.models.portfolio.optimizer import black_litterman

    cov = np.diag([0.002, 0.002])
    q = np.array([0.01, 0.01])
    sure = black_litterman(q, np.array([1e-4, 1e-4]), cov)
    unsure = black_litterman(q, np.array([0.2, 0.2]), cov)
    assert np.allclose(sure, q, atol=1e-4)  # confident views pass through
    assert np.all(np.abs(unsure) < 0.001)  # uncertain views shrink toward the zero prior
    mixed = black_litterman(q, np.array([1e-4, 0.2]), cov)
    assert mixed[0] > 10 * mixed[1]
