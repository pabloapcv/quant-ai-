from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from atlas.backtest.trials import TrialLog
from atlas.models.expected_returns.ensemble import EnsembleConfig, component_names, walk_forward
from atlas.pit.store import PITStore
from atlas.signals.country import (
    COUNTRIES,
    COUNTRY_FAMILIES,
    COUNTRY_SIGNALS,
    DEVELOPED,
    cli_change,
    country_panel,
    yield_change,
)
from atlas.signals.research import ResearchContext, run_gate_suite
from atlas.signals.universes import COUNTRY_COMPONENTS, UNIVERSES, get_universe
from atlas.views.publish import build_views, evaluate
from tests.conftest import make_obs_frame, utc

REPO = Path(__file__).resolve().parents[1]
KNOWN = utc(2026, 10, 1)


def _country_store(
    path: Path, n_months: int = 240, start_year: int = 1995, seed: int = 11
) -> PITStore:
    rng = np.random.default_rng(seed)
    months = [date(start_year + t // 12, t % 12 + 1, 1) for t in range(n_months)]
    rows = []
    for c in COUNTRIES:
        level, cli = 100.0, 100.0
        for d in months:
            level *= 1 + rng.normal(0.006, 0.05)
            cli += rng.normal(0, 0.3)
            rows += [
                (f"{c}.EQUITY.SHARE", d, KNOWN, level),
                (f"{c}.CLI", d, KNOWN, cli),
                (f"{c}.RATES.10Y", d, KNOWN, 3 + rng.normal(0, 0.5)),
            ]
    for cur in ("USD", "JPY", "GBP", "CAD", "AUD", "CNY", "INR", "BRL", "KRW", "MXN"):
        x = 1.0
        for d in months:
            x *= 1 + rng.normal(0, 0.02)
            rows.append((f"FX.EUR{cur}.EOP", d, KNOWN, x))
    rows.append(("FF.FACTOR.MKTRF", months[0], KNOWN, 1.0))
    store = PITStore(path)
    store.write(make_obs_frame(rows))
    return store


def test_country_panel_usd_conversion_and_lags(tmp_path: Path) -> None:
    store = _country_store(tmp_path)
    lcl = country_panel(store, usd=False).sort(["asset", "date"])
    usd = country_panel(store, usd=True).sort(["asset", "date"])
    assert set(usd["asset"].unique().to_list()) == set(COUNTRIES)

    # USA in USD == local; Germany (EUR) in USD = local x change in USD per EUR
    us_l = lcl.filter(pl.col("asset") == "USA")["ret"].to_numpy()
    us_u = usd.filter(pl.col("asset") == "USA")["ret"].to_numpy()
    assert np.allclose(us_l[1:], us_u[1:], equal_nan=True)
    fx = dict(store.as_of(["FX.EURUSD.EOP"], KNOWN).select("event_time", "value").iter_rows())
    d, prev = date(1995, 7, 1), date(1995, 6, 1)
    de_l = lcl.filter((pl.col("asset") == "DEU") & (pl.col("date") == d))["ret"].item()
    de_u = usd.filter((pl.col("asset") == "DEU") & (pl.col("date") == d))["ret"].item()
    assert de_u == pytest.approx((1 + de_l) * fx[d] / fx[prev] - 1)

    # CLI/yields are lagged one month (published ~2 weeks after month end)
    raw_cli = dict(store.as_of(["USA.CLI"], KNOWN).select("event_time", "value").iter_rows())
    us = lcl.filter(pl.col("asset") == "USA")
    known_in_july = us.filter(pl.col("date") == date(1995, 7, 1))["cli"].item()
    assert known_in_july == pytest.approx(raw_cli[date(1995, 6, 1)])  # June's CLI, known in July

    dev = country_panel(store, usd=False, countries=DEVELOPED)
    assert set(dev["asset"].unique().to_list()) == set(DEVELOPED)
    with pytest.raises(ValueError, match="share prices"):
        country_panel(PITStore(tmp_path / "empty"))


def test_non_finite_returns_are_dropped(tmp_path: Path) -> None:
    rows = [
        ("BRA.EQUITY.SHARE", date(1990, m, 1), KNOWN, v) for m, v in ((1, 0.0), (2, 0.0), (3, 1.0))
    ]
    rows += [("USA.EQUITY.SHARE", date(1990, m, 1), KNOWN, 100.0 + m) for m in (1, 2, 3)]
    store = PITStore(tmp_path)
    store.write(make_obs_frame(rows))
    p = country_panel(store, usd=False)
    bra = p.filter(pl.col("asset") == "BRA")["ret"].to_list()
    assert all(v is None or np.isfinite(v) for v in bra)


def test_country_signals() -> None:
    p = pl.DataFrame(
        {
            "date": [date(2020, m, 1) for m in range(1, 9)],
            "asset": ["A"] * 8,
            "ret": [0.01] * 8,
            "cli": [None, 99.0, 99.5, 100.0, 100.5, 101.0, 101.5, 102.0],
            "y10": [5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3],
        }
    )
    sig = cli_change(3).compute(p)["signal"].to_list()
    assert sig[4] == pytest.approx(100.5 - 99.0)
    yf = yield_change(6).compute(p)["signal"].to_list()
    assert yf[6] == pytest.approx(0.6)  # yield fell 60bp -> positive
    # no skip-0 momentum: averaged prices fake 1-month momentum
    assert all(s.params.get("skip", 1) >= 1 for s in COUNTRY_FAMILIES["momentum"])
    assert "reversal_1m" not in COUNTRY_SIGNALS


def test_universe_registry_and_components(tmp_path: Path) -> None:
    assert {"ind12", "ind49", "ctry_usd", "ctry_dev_lcl"} <= set(UNIVERSES)
    with pytest.raises(ValueError, match="universe"):
        get_universe("mars")
    u = get_universe("ctry_usd")
    assert u.asset_name("USA") == "United States"
    # pinned: research signals added later never enter the live ensemble implicitly
    assert u.ensemble_components == COUNTRY_COMPONENTS
    ctx = ResearchContext(
        panel=pl.DataFrame(), universe="ctry_usd", log=TrialLog(tmp_path), signals=u.signals
    )
    names = component_names(ctx, EnsembleConfig(components=None))
    assert "cli_chg_6" in names
    # industry components that the country set lacks are ignored, falling back to all
    assert component_names(ctx, EnsembleConfig(components=("value_log_bm",))) == names


def test_country_gates_and_views(tmp_path: Path) -> None:
    store = _country_store(tmp_path)
    uni = get_universe("ctry_dev_lcl")
    ctx = ResearchContext(
        panel=uni.build_panel(store),
        universe=uni.name,
        log=TrialLog(tmp_path),
        n_quantiles=3,
        signals=uni.signals,
        families=uni.families,
    )
    suite = run_gate_suite(ctx, n_splits=4)
    # signals whose inputs the synthetic store lacks (3M rates, oil, US yields) are skipped
    assert {"mom_12_1", "cli_chg_6", "yield_fall_12", "seasonality_10y"} <= set(suite.gates)
    assert set(suite.gates) <= set(COUNTRY_SIGNALS)
    run = walk_forward(
        ctx, EnsembleConfig(horizon=3, components=None, min_history=36, calib_min_dates=48)
    )
    views = build_views(run, uni.name)
    assert len(views) == len(DEVELOPED)
    assert all(v.asset_id.startswith("COUNTRY.") for v in views)
    assert {v.asset_name for v in views} <= set(uni.asset_names.values())
    ev = evaluate(run)
    assert ev.months > 0


def test_median_ties_do_not_bias_probabilities(tmp_path: Path) -> None:
    """Odd asset counts: the median asset must not count as 'not beating' the median."""
    store = _country_store(tmp_path, seed=3)
    uni = get_universe("ctry_dev_lcl")  # 9 assets: one sits on the median every date
    ctx = ResearchContext(
        panel=uni.build_panel(store),
        universe=uni.name,
        log=TrialLog(tmp_path),
        signals=uni.signals,
        families=uni.families,
    )
    run = walk_forward(
        ctx, EnsembleConfig(horizon=1, components=None, min_history=24, calib_min_dates=36)
    )
    probs = run.forecasts["prob"].drop_nulls().to_numpy()
    assert abs(float(np.mean(probs)) - 0.5) < 0.03  # pure-noise data: centred on 50%


def test_cli_country_universe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    _country_store(data)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        runner = CliRunner()
        res = runner.invoke(app, ["research", "list", "--universe", "ctry_usd"])
        assert res.exit_code == 0, res.output
        assert "cli_chg_6" in res.output
        res = runner.invoke(
            app, ["research", "signal", "cli_chg_6", "--universe", "ctry_usd", "--quantiles", "3"]
        )
        assert res.exit_code == 0, res.output
        res = runner.invoke(app, ["views", "build", "--universe", "ctry_dev_lcl"])
        assert res.exit_code == 0, res.output
        assert "United States" in res.output
        assert runner.invoke(app, ["views", "build", "--universe", "mars"]).exit_code != 0
    finally:
        get_settings.cache_clear()
