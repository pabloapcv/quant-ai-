from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from atlas.backtest.crosssection import backtest_signal, vol_target
from atlas.backtest.gates import evaluate_gates
from atlas.backtest.trials import TrialLog
from atlas.models.regime.market import REGIMES, market_regimes
from atlas.pit.store import PITStore
from atlas.signals.cards import write_cards
from atlas.signals.industry import (
    FAMILIES,
    SIGNALS,
    fiscal_year_known,
    research_panel,
    residual_momentum,
    residual_momentum_scores,
    value,
    value_momentum,
    vol_scaled_momentum,
)
from atlas.signals.research import (
    ResearchContext,
    ensemble_marginal_icir,
    family_pbo,
    run_gate_suite,
    run_signal,
    signal_panel,
)
from tests.conftest import make_obs_frame, utc

REPO = Path(__file__).resolve().parents[1]
KNOWN = utc(2026, 10, 1)


def _store(
    path: Path, *, n_months: int = 120, n_assets: int = 12, seed: int = 7, start_year: int = 2010
) -> PITStore:
    """Synthetic Ken French-like store: industry returns, annual BE/ME, market factor."""
    rng = np.random.default_rng(seed)
    rows = []
    months = [date(start_year + t // 12, t % 12 + 1, 1) for t in range(n_months)]
    for i in range(n_assets):
        drift = rng.normal(0.8, 0.3)
        for d in months:
            rows.append((f"FF.IND12.X{i}", d, KNOWN, float(rng.normal(drift, 5))))
        for fy in range(start_year - 2, start_year + n_months // 12):
            rows.append(
                (f"FF.IND12BM.X{i}", date(fy, 1, 1), KNOWN, float(abs(rng.normal(0.6, 0.2)) + 0.05))
            )
    for d in months:
        rows.append(("FF.FACTOR.MKTRF", d, KNOWN, float(rng.normal(0.7, 4.5))))
        rows.append(("FF.FACTOR.SMB", d, KNOWN, float(rng.normal(0.2, 3.0))))
        rows.append(("FF.FACTOR.HML", d, KNOWN, float(rng.normal(0.3, 3.0))))
        rows.append(("FF.FACTOR.RF", d, KNOWN, 0.3))
        rows.append(("FF.FACTOR.RMW", d, KNOWN, float(rng.normal(0.2, 2.0))))
        rows.append(("FF.FACTOR.CMA", d, KNOWN, float(rng.normal(0.2, 2.0))))
    store = PITStore(path)
    store.write(make_obs_frame(rows))
    return store


def test_fiscal_year_timing() -> None:
    df = pl.DataFrame(
        {"d": [date(2020, 5, 1), date(2020, 6, 1), date(2020, 12, 1), date(2021, 1, 1)]}
    )
    # Fama-French: FY y usable from end of June y+1
    assert df.select(fiscal_year_known(pl.col("d")))["d"].to_list() == [2018, 2019, 2019, 2019]


def test_research_panel_attaches_bm_with_ff_timing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    panel = research_panel(store, "ind12")
    bm_rows = store.as_of(["FF.IND12BM.X0"], datetime(2026, 10, 2, tzinfo=UTC))
    fy2011 = bm_rows.filter(pl.col("event_time") == date(2011, 1, 1))["value"].item()
    x0 = panel.filter(pl.col("asset") == "X0")
    assert x0.filter(pl.col("date") == date(2012, 6, 1))["bm"].item() == pytest.approx(fy2011)
    assert x0.filter(pl.col("date") == date(2012, 5, 1))["bm"].item() != pytest.approx(fy2011)


def test_value_and_combo_signals(tmp_path: Path) -> None:
    panel = research_panel(_store(tmp_path), "ind12")
    v = value("log_bm").compute(panel)
    assert v.drop_nulls("signal").height > 0
    assert (v.drop_nulls("signal")["signal"] == v.drop_nulls("signal")["bm"].log()).all()
    combo = value_momentum(0.5).compute(panel).drop_nulls("signal")
    by_date = combo.group_by("date").agg(pl.col("signal").mean())
    assert np.abs(by_date["signal"].to_numpy()).max() < 1e-9  # z-score blend is centered
    rel = value("bm_vs_5y").compute(panel)
    assert rel["signal"].null_count() < rel.height
    assert {s.family for s in SIGNALS.values()} <= set(FAMILIES)
    assert all(s.reference for s in SIGNALS.values())


def test_vol_target_is_causal_and_capped() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for t in range(60):
        d = date(2000 + t // 12, t % 12 + 1, 1)
        sig = rng.normal(size=10)
        fwd = 0.01 * sig + rng.normal(0, 0.05, 10)
        rows += [
            {"date": d, "asset": f"A{i}", "signal": sig[i], "fwd_ret": fwd[i]} for i in range(10)
        ]
    base = backtest_signal(pl.DataFrame(rows), "s", n_quantiles=3, cost_bps=0)
    vt = vol_target(base, 0.10, window=6, max_leverage=2.0, cost_bps=0)
    assert vt.long_short.height == base.long_short.height - 6
    assert vt.long_short["scale"].to_numpy().max() <= 2.0
    # scale on a date depends only on earlier dates: changing a later return leaves it alone
    ls = base.long_short.with_columns(
        pl.when(pl.col("date") == pl.col("date").max())
        .then(5.0)
        .otherwise(pl.col("net"))
        .alias("net")
    )
    from dataclasses import replace

    vt2 = vol_target(replace(base, long_short=ls), 0.10, window=6, cost_bps=0)
    assert vt2.long_short["scale"][-1] == pytest.approx(vt.long_short["scale"][-1])
    ann = float(np.std(vt.long_short["net"].to_numpy(), ddof=1)) * math.sqrt(12)
    assert 0.03 < ann < 0.3


def test_market_regimes(tmp_path: Path) -> None:
    reg = market_regimes(_store(tmp_path), min_history=24)
    assert set(reg["regime"].unique().to_list()) <= set(REGIMES)
    assert reg["date"].to_list()[0] > date(2010, 1, 1)  # needs a full window first
    with pytest.raises(ValueError, match="MKTRF"):
        market_regimes(PITStore(tmp_path / "empty"))


def test_runner_gates_and_ensemble(tmp_path: Path) -> None:
    store = _store(tmp_path)
    log = TrialLog(tmp_path)
    ctx = ResearchContext(
        panel=research_panel(store, "ind12"),
        universe="ind12",
        log=log,
        regimes=market_regimes(store, min_history=24).select("date", "regime"),
        n_quantiles=3,
    )
    mom = run_signal(SIGNALS["mom_12_1"], ctx)
    vt = run_signal(vol_scaled_momentum(0.12), ctx)
    assert vt.n_trials == 2
    assert "scale" in vt.long_short.columns
    assert set(mom.regime_ic) <= set(REGIMES)

    reps = [run_signal(s, ctx) for s in FAMILIES["low_volatility"]]
    pbo, months = family_pbo(reps, n_splits=4)
    assert 0.0 <= pbo <= 1.0
    assert months > 0

    # An oracle signal (next month's return) must add to the ensemble; pure noise must not.
    base = signal_panel(SIGNALS["mom_12_1"], ctx)
    rng = np.random.default_rng(1)
    oracle = base.with_columns(pl.col("fwd_ret").alias("signal"))
    noise = base.with_columns(pl.Series("signal", rng.normal(size=base.height)))
    full, marginal = ensemble_marginal_icir({"oracle": oracle, "noise": noise, "mom": base})
    assert marginal["oracle"] > 0.5
    assert marginal["oracle"] > marginal["noise"]
    assert math.isfinite(full)

    g = evaluate_gates(SIGNALS["mom_12_1"], mom, pbo=0.05, marginal_icir=0.1)
    assert set(g.checks) == {"ic_tstat", "dsr", "pbo", "regimes", "marginal", "rationale"}
    assert g.checks["pbo"]
    assert g.checks["marginal"]
    assert g.checks["rationale"]
    bad = evaluate_gates(SIGNALS["mom_12_1"], mom, pbo=None, marginal_icir=float("nan"))
    assert not bad.passed
    assert {"pbo", "marginal"} <= set(bad.failed)


def test_cli_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    _store(data, n_months=144)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        out = tmp_path / "gates.json"
        res = CliRunner().invoke(
            app,
            [
                "research",
                "gates",
                "--universe",
                "ind12",
                "--quantiles",
                "3",
                "--splits",
                "4",
                "--out",
                str(out),
            ],
        )
        assert res.exit_code == 0, res.output
        assert "ensemble ICIR" in res.output
        assert out.exists()
        assert '"gates"' in out.read_text()
        res = CliRunner().invoke(
            app, ["research", "signal", "value_log_bm", "--universe", "ind12", "--quantiles", "3"]
        )
        assert res.exit_code == 0, res.output
        assert "ic_by_market_regime" in res.output
    finally:
        get_settings.cache_clear()


def test_factor_panel_and_residual_momentum(tmp_path: Path) -> None:
    panel = research_panel(_store(tmp_path), "ind12")
    assert {"f_mktrf", "f_smb", "f_hml", "f_rf"} <= set(panel.columns)
    assert panel["f_rf"].drop_nulls().unique().to_list() == [0.003]  # percent -> fraction

    # Same 1.5 market beta; HI gets +3%/month idiosyncratic return over the last year, LO -3%.
    # (A constant alpha would be absorbed by the intercept: residual momentum is about
    # *recent* idiosyncratic performance.)
    dates = panel.select("date").unique().sort("date")
    mkt = panel.select("date", "f_mktrf", "f_smb", "f_hml", "f_rf").unique("date").sort("date")
    rng = np.random.default_rng(3)
    n = dates.height
    recent = np.zeros(n)
    recent[n - 12 : n - 1] = 0.03
    base = pl.col("f_rf") + 1.5 * pl.col("f_mktrf") + pl.Series(rng.normal(0, 0.002, n))
    hi = mkt.with_columns(pl.lit("HI").alias("asset"), (base + pl.Series(recent)).alias("ret"))
    lo = mkt.with_columns(pl.lit("LO").alias("asset"), (base - pl.Series(recent)).alias("ret"))
    toy = pl.concat([hi, lo]).select("date", "asset", "ret", "f_mktrf", "f_smb", "f_hml", "f_rf")
    scores = residual_momentum_scores(toy, ("f_mktrf",), window=36)
    last = scores.drop_nulls("signal").sort("date").group_by("asset").agg(pl.col("signal").last())
    by = dict(last.iter_rows())
    assert by["HI"] > by["LO"]  # constant alpha gap shows up despite identical beta
    assert (
        scores.filter(pl.col("date") < dates["date"][35])["signal"].null_count()
        == scores.filter(pl.col("date") < dates["date"][35]).height
    )  # nothing before a full window

    missing = residual_momentum_scores(toy.drop("f_rf"), ("f_mktrf",))
    assert missing["signal"].null_count() == missing.height
    sig = residual_momentum(36, "ff3")
    assert sig.family == "residual_momentum"
    assert "signal" in sig.compute(panel).columns


def test_gate_suite_and_cards(tmp_path: Path) -> None:
    store = _store(tmp_path, n_months=144)
    ctx = ResearchContext(
        panel=research_panel(store, "ind12"),
        universe="ind12",
        log=TrialLog(tmp_path),
        regimes=market_regimes(store, min_history=24).select("date", "regime"),
        n_quantiles=3,
    )
    suite = run_gate_suite(ctx, n_splits=4)
    assert set(suite.gates) == set(SIGNALS)
    assert set(suite.pbo) == set(FAMILIES)
    distinct = {(s.name, str(sorted(s.params.items()))) for m in FAMILIES.values() for s in m}
    distinct |= {(s.name, str(sorted(s.params.items()))) for s in SIGNALS.values()}
    assert suite.n_trials == len(distinct)  # representative configs are not double-counted
    paths = write_cards(suite, tmp_path / "cards", generated=date(2026, 10, 14), cost_bps=10)
    assert len(paths) == len(SIGNALS) + 1
    card = (tmp_path / "cards" / "mom_12_1.md").read_text()
    for heading in (
        "## Why it should work",
        "## Promotion gates",
        "## Every configuration tried",
        "## Where it breaks",
    ):
        assert heading in card
    assert "Do not edit by hand" in card
    index = (tmp_path / "cards" / "README.md").read_text()
    assert "[`mom_12_1`](mom_12_1.md)" in index


def test_market_regimes_needs_history(tmp_path: Path) -> None:
    store = PITStore(tmp_path)
    store.write(make_obs_frame([("FF.FACTOR.MKTRF", date(2020, 1, 1), KNOWN, 1.0)]))
    with pytest.raises(ValueError, match=r"months of FF\.FACTOR\.MKTRF"):
        market_regimes(store)
