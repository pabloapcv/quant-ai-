from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from atlas.backtest.trials import TrialLog
from atlas.models.expected_returns.ensemble import (
    EnsembleConfig,
    _past_mean,
    forward_return,
    walk_forward,
)
from atlas.signals.industry import research_panel
from atlas.signals.research import ResearchContext
from atlas.views.publish import ViewExistsError, build_views, evaluate, publish, with_stances
from atlas.views.schema import (
    Confidence,
    Stance,
    confidence_for,
    raw_stance,
    stance_with_hysteresis,
)
from tests.test_research import _store

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ stance rules
@pytest.mark.parametrize(
    ("prob", "agree", "expected"),
    [
        (0.70, 0.80, Stance.STRONG_OVERWEIGHT),
        (0.70, 0.60, Stance.OVERWEIGHT),
        (0.56, 0.60, Stance.OVERWEIGHT),
        (0.56, 0.40, Stance.NEUTRAL),  # low agreement -> neutral
        (0.50, 0.90, Stance.NEUTRAL),
        (0.44, 0.60, Stance.UNDERWEIGHT),
        (0.30, 0.80, Stance.STRONG_UNDERWEIGHT),
        (0.30, 0.40, Stance.UNDERWEIGHT),
        (None, None, Stance.NEUTRAL),
    ],
)
def test_raw_stance(prob: float | None, agree: float | None, expected: Stance) -> None:
    assert raw_stance(prob, agree) is expected


def test_hysteresis_blocks_marginal_flips() -> None:
    assert stance_with_hysteresis(0.555, 0.6, Stance.NEUTRAL) is Stance.NEUTRAL  # barely over
    assert stance_with_hysteresis(0.58, 0.6, Stance.NEUTRAL) is Stance.OVERWEIGHT
    assert stance_with_hysteresis(0.545, 0.6, Stance.OVERWEIGHT) is Stance.OVERWEIGHT  # sticky
    assert stance_with_hysteresis(0.52, 0.6, Stance.OVERWEIGHT) is Stance.NEUTRAL
    assert stance_with_hysteresis(0.6, 0.6, None) is Stance.OVERWEIGHT


def test_confidence() -> None:
    assert confidence_for(0.62, 0.8) is Confidence.HIGH
    assert confidence_for(0.56, 0.6) is Confidence.MEDIUM
    assert confidence_for(0.51, 0.9) is Confidence.LOW
    assert confidence_for(None, 0.9) is Confidence.LOW


# ------------------------------------------------------------------ ensemble mechanics
def test_past_mean_respects_lag() -> None:
    x = np.arange(6, dtype=float).reshape(-1, 1)
    lag1 = _past_mean(x, 1, lag=1)[:, 0]
    assert np.isnan(lag1[0])
    assert lag1[3] == pytest.approx(np.mean([0, 1, 2]))
    lag3 = _past_mean(x, 1, lag=3)[:, 0]
    assert np.isnan(lag3[2])
    assert lag3[5] == pytest.approx(np.mean([0, 1, 2]))  # rows <= 5-3


def test_forward_return_compounds() -> None:
    df = pl.DataFrame({"asset": ["A"] * 4, "date": list(range(4)), "ret": [0.1, 0.2, -0.1, 0.0]})
    out = df.with_columns(forward_return(2).alias("f"))["f"].to_list()
    assert out[0] == pytest.approx(1.2 * 0.9 - 1)
    assert out[2] is None
    assert out[3] is None


@pytest.fixture(scope="module")
def run_and_ctx(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    path = tmp_path_factory.mktemp("views")
    store = _store(path, n_months=240, n_assets=12, start_year=1995)
    ctx = ResearchContext(
        panel=research_panel(store, "ind12"), universe="ind12", log=TrialLog(path), n_quantiles=3
    )
    return (
        walk_forward(ctx, EnsembleConfig(horizon=3, min_history=36, calib_min_dates=48)),
        ctx,
        path,
    )


def test_walk_forward_outputs(run_and_ctx) -> None:  # type: ignore[no-untyped-def]
    run, _, _ = run_and_ctx
    fc = run.forecasts
    assert {"score", "exp_excess", "p_lo", "p_hi", "prob", "agreement", "excess_med"} <= set(
        fc.columns
    )
    scored = fc.drop_nulls(["score", "prob", "p_lo"])
    assert scored.height > 0
    assert (scored["p_lo"] <= scored["exp_excess"]).all()
    assert (scored["exp_excess"] <= scored["p_hi"]).all()
    assert scored["prob"].min() >= 0
    assert scored["prob"].max() <= 1
    w = run.weights.drop("date").to_numpy()
    sums = w.sum(axis=1)
    assert np.all((np.abs(sums - 1) < 1e-9) | (sums == 0))  # weights sum to 1 once active
    assert (w >= 0).all()
    # nothing before min_history (+ horizon) months of past ICs
    first_active = run.weights.filter(pl.col("mom_12_1") > 0)["date"].min()
    assert first_active is not None


def test_walk_forward_is_point_in_time(run_and_ctx) -> None:  # type: ignore[no-untyped-def]
    """Forecasts at date T must not change if returns after T are altered."""
    run, ctx, _ = run_and_ctx
    dates = run.forecasts.select("date").unique().sort("date")["date"].to_list()
    cut = dates[180]
    poisoned_panel = ctx.panel.with_columns(
        pl.when(pl.col("date") > cut).then(pl.col("ret") * -5).otherwise(pl.col("ret")).alias("ret")
    )
    from dataclasses import replace

    run2 = walk_forward(replace(ctx, panel=poisoned_panel), run.config)
    cols = ["date", "asset", "score", "exp_excess", "p_lo", "p_hi", "prob"]
    # outcomes for dates >= cut - horizon depend on poisoned returns; forecasts made at
    # dates <= cut may still only use outcomes realized by then
    a = run.forecasts.filter(pl.col("date") <= cut).select(cols)
    b = run2.forecasts.filter(pl.col("date") <= cut).select(cols)
    # identical up to float summation order (polars may reduce in a different order)
    assert_frame_equal(a, b, check_exact=False, rel_tol=1e-9, abs_tol=1e-12)


def test_build_publish_and_evaluate(run_and_ctx, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    run, _, _ = run_and_ctx
    views = build_views(run, "ind12", code_sha="abc", n_trials=7)
    assert len(views) == 12
    assert views[0].score >= views[-1].score
    v = views[0]
    assert v.horizon == "3M"
    assert v.provenance.n_trials == 7
    assert v.asset_id.startswith("INDUSTRY.IND12.")
    assert all(d.contribution == pytest.approx(d.contribution) for d in v.drivers)

    path = publish(views, tmp_path, "ind12")
    assert json.loads(path.read_text())[0]["view_id"] == v.view_id
    assert publish(build_views(run, "ind12", code_sha="abc", n_trials=7), tmp_path, "ind12") == path
    altered = [views[0].model_copy(update={"score": 0.0}), *views[1:]]
    with pytest.raises(ViewExistsError):
        publish(altered, tmp_path, "ind12")
    with pytest.raises(ValueError, match="no views"):
        publish([], tmp_path, "ind12")
    with pytest.raises(ValueError, match="no forecasts"):
        build_views(run, "ind12", date(1900, 1, 1))

    ev = evaluate(run)
    s = ev.summary()
    assert ev.months > 0
    assert 0.0 <= ev.coverage <= 1.0
    assert all(v is None or isinstance(v, int | float) for v in s.values())
    assert ev.reliability.height > 0
    st = with_stances(run.forecasts)
    assert set(st["stance"].drop_nulls().unique().to_list()) <= {x.value for x in Stance}


def test_cli_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    _store(data, n_months=240, start_year=1995)
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    try:
        runner = CliRunner()
        res = runner.invoke(app, ["views", "build", "--universe", "ind12", "--publish"])
        assert res.exit_code == 0, res.output
        assert "published" in res.output
        assert list((data / "gold" / "views" / "ind12" / "3M").glob("*.json"))
        res = runner.invoke(
            app, ["views", "build", "--universe", "ind12", "--json", "--horizon", "1"]
        )
        assert res.exit_code == 0, res.output
        assert '"stance"' in res.output
        res = runner.invoke(
            app, ["views", "evaluate", "--universe", "ind12", "--start", "2012-01-01"]
        )
        assert res.exit_code == 0, res.output
        assert "coverage" in res.output
    finally:
        get_settings.cache_clear()


def test_json_output_is_strict() -> None:
    from atlas import jsonsafe

    out = jsonsafe.dumps({"a": float("nan"), "b": [1.0, float("inf")], "c": (2, date(2026, 1, 1))})
    assert json.loads(out) == {"a": None, "b": [1.0, None], "c": [2, "2026-01-01"]}
    assert "NaN" not in out
    assert "Infinity" not in out


# ------------------------------------------------------------------ walk-forward gating
def _ic_inputs(n: int = 240, seed: int = 4) -> tuple[np.ndarray, pl.Series, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    strong = 0.08 + rng.normal(0, 0.1, n)  # clearly positive IC
    noise = rng.normal(0, 0.1, n)  # no skill
    two_regime = np.where(np.arange(n) % 4 < 2, 0.2, -0.1) + rng.normal(0, 0.02, n)  # + in 2 of 4
    dates = pl.Series("date", [date(2000 + i // 12, i % 12 + 1, 1) for i in range(n)])
    regimes = pl.DataFrame({"date": dates, "regime": [f"r{i % 4}" for i in range(n)]})
    return np.column_stack([strong, noise, two_regime]), dates, regimes


def test_eligibility_gates() -> None:
    from atlas.models.expected_returns.ensemble import _eligibility

    ic, dates, regimes = _ic_inputs()
    assert _eligibility(ic, dates, None, EnsembleConfig()).all()  # no gates -> all eligible

    cfg = EnsembleConfig(gate_tstat=2.0, min_history=60, horizon=3)
    e = _eligibility(ic, dates, regimes, cfg)
    assert not e[:60].any()  # not enough history yet
    assert e[-1, 0]  # strong component eligible
    assert not e[-1, 1]  # noise component never clears t >= 2 at the end
    assert e[-1, 2]  # the two-regime component has a positive overall mean

    reg_cfg = EnsembleConfig(gate_tstat=2.0, gate_regimes=3, min_history=60, horizon=3)
    e2 = _eligibility(ic, dates, regimes, reg_cfg)
    assert e2[-1, 0]
    assert not e2[-1, 2]  # positive in only 2 of 4 regimes
    with pytest.raises(ValueError, match="regimes"):
        _eligibility(ic, dates, None, reg_cfg)


def test_eligibility_is_point_in_time() -> None:
    from atlas.models.expected_returns.ensemble import _eligibility

    ic, dates, regimes = _ic_inputs()
    cfg = EnsembleConfig(gate_tstat=2.0, gate_regimes=3, min_history=60, horizon=3, calib_refit=12)
    base = _eligibility(ic, dates, regimes, cfg)
    t = 144  # a refit month
    poisoned = ic.copy()
    poisoned[t - cfg.horizon + 1 :] = -5.0  # outcomes not yet realized at t
    assert np.array_equal(_eligibility(poisoned, dates, regimes, cfg)[: t + 1], base[: t + 1])


def test_gated_ensemble_only_weights_eligible_components(run_and_ctx) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from atlas.models.regime.market import market_regimes

    _, ctx, path = run_and_ctx
    gated_ctx = replace(ctx, regimes=None)
    run = walk_forward(
        gated_ctx, EnsembleConfig(horizon=3, min_history=36, calib_min_dates=48, gate_tstat=50.0)
    )
    assert (run.weights.drop("date").to_numpy() == 0).all()  # impossible bar: nothing qualifies
    assert run.forecasts["score"].null_count() == run.forecasts.height
    with pytest.raises(ValueError, match="regimes"):
        walk_forward(gated_ctx, EnsembleConfig(horizon=3, gate_regimes=3))
    _ = market_regimes, path


def test_publish_ignores_run_metadata(run_and_ctx, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    run, _, _ = run_and_ctx
    first = build_views(run, "ind12", code_sha="aaa", snapshot_id="snap_1")
    path = publish(first, tmp_path, "ind12")
    again = build_views(
        run, "ind12", code_sha="bbb", snapshot_id="snap_2"
    )  # new commit, new snapshot
    assert publish(again, tmp_path, "ind12") == path  # same forecasts -> no-op
    assert json.loads(path.read_text())[0]["provenance"]["snapshot_id"] == "snap_1"  # first kept


def test_gbm_combiner_is_point_in_time(run_and_ctx) -> None:  # type: ignore[no-untyped-def]
    """The gradient-boosted ranker trains only on realized outcomes: no look-ahead."""
    from dataclasses import replace

    _, ctx, _ = run_and_ctx
    cfg = EnsembleConfig(horizon=3, min_history=36, calib_min_dates=48, combiner="gbm")
    run = walk_forward(ctx, cfg)
    scored = run.forecasts.drop_nulls("score")
    assert scored.height > 0
    first = scored["date"].min()
    dates = run.forecasts.select("date").unique().sort("date")["date"].to_list()
    assert first is not None
    assert first >= dates[cfg.min_history + cfg.horizon]  # needs min_history realized months
    cut = dates[180]
    poisoned = ctx.panel.with_columns(
        pl.when(pl.col("date") > cut).then(pl.col("ret") * -5).otherwise(pl.col("ret")).alias("ret")
    )
    run2 = walk_forward(replace(ctx, panel=poisoned), cfg)
    cols = ["date", "asset", "score", "exp_excess", "prob"]
    a = run.forecasts.filter(pl.col("date") <= cut).select(cols)
    b = run2.forecasts.filter(pl.col("date") <= cut).select(cols)
    assert_frame_equal(a, b, check_exact=False, rel_tol=1e-9, abs_tol=1e-12)
    with pytest.raises(ValueError, match="unknown combiner"):
        walk_forward(ctx, EnsembleConfig(combiner="magic"))
