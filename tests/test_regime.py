from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from atlas.features.macro import RECESSION
from atlas.features.registry import FeatureSpec, Pillar
from atlas.models.regime.composites import composite, contributions
from atlas.models.regime.engine import RegimeEngine, data_month_for
from atlas.models.regime.evaluate import month_ends, walk_forward
from atlas.models.regime.hmm import fit_regime_hmm
from atlas.models.regime.quadrant import (
    Quadrant,
    classify,
    classify_expr,
    noise_scale,
    probabilities,
)
from atlas.models.regime.recession import fit_recession_probit, forward_label
from atlas.pit.cube import VintageCube
from atlas.pit.store import PITStore
from tests.conftest import make_obs_frame
from tests.synthetic import build_observations, months, true_regime

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_obs() -> pl.DataFrame:
    return build_observations()


@pytest.fixture(scope="module")
def engine(synthetic_obs: pl.DataFrame) -> RegimeEngine:
    return RegimeEngine(VintageCube(synthetic_obs))


# ------------------------------------------------------------------ quadrant & composites
@pytest.mark.parametrize(
    ("g", "i", "expected"),
    [
        (1.0, -1.0, Quadrant.GOLDILOCKS),
        (1.0, 1.0, Quadrant.OVERHEATING),
        (-1.0, 1.0, Quadrant.STAGFLATION),
        (-1.0, -1.0, Quadrant.DEFLATIONARY_SLOWDOWN),
    ],
)
def test_classify_scalar_and_expr_agree(g: float, i: float, expected: Quadrant) -> None:
    assert classify(g, i) is expected
    df = pl.DataFrame({"g": [g], "i": [i]})
    assert df.select(classify_expr(pl.col("g"), pl.col("i")))[0, 0] == expected.value


def test_probabilities_are_a_distribution() -> None:
    p = probabilities(0.0, 0.0, 1.0, 1.0)
    assert all(v == pytest.approx(0.25) for v in p.values())
    p = probabilities(2.0, -2.0, 0.5, 0.5)
    assert sum(p.values()) == pytest.approx(1.0)
    assert max(p, key=lambda q: p[q]) is Quadrant.GOLDILOCKS


def test_noise_scale() -> None:
    assert noise_scale(pl.Series([1.0, 2.0])) == 1.0  # too short -> neutral
    assert noise_scale(pl.Series([0.0] * 40)) == 0.1  # floor
    s = pl.Series(np.sin(np.arange(200) / 5.0))
    assert 0.1 < noise_scale(s) < 2.0


def test_composite_min_features_clip_and_contributions() -> None:
    specs = [
        FeatureSpec("a", "US.A.A", Pillar.GROWTH, ()),
        FeatureSpec("b", "US.B.B", Pillar.GROWTH, (), sign=-1),
        FeatureSpec("c", "US.C.C", Pillar.GROWTH, ()),
    ]
    m = pl.DataFrame(
        {
            "event_time": [date(2020, 1, 1), date(2020, 2, 1)],
            "a": [10.0, 1.0],
            "b": [-1.0, None],
            "c": [0.0, 1.0],
        }
    )
    out = composite(m, specs, "growth", min_features=3)
    assert out["growth"][0] == pytest.approx((3.0 + 1.0 + 0.0) / 3)  # a clipped to 3
    assert out["growth"][1] is None
    assert out["growth_n"][1] == 2
    contrib = contributions(m.row(0, named=True), specs)
    assert sum(contrib.values()) == pytest.approx(out["growth"][0])
    assert (
        composite(m, [FeatureSpec("zz", "US.Z.Z", Pillar.GROWTH, ())], "g")["g"].null_count() == 2
    )


def test_data_month_for() -> None:
    assert data_month_for(date(2026, 10, 14)) == date(2026, 9, 1)
    assert data_month_for(date(2026, 1, 31)) == date(2025, 12, 1)
    assert data_month_for(datetime(2026, 3, 1, tzinfo=UTC)) == date(2026, 2, 1)


# ------------------------------------------------------------------ recession & hmm
def test_forward_label() -> None:
    df = pl.DataFrame({"r": [0.0, 0.0, 1.0, 0.0, 0.0]})
    assert df.select(forward_label(pl.col("r"), 2))["r"].to_list() == [1.0, 1.0, 0.0, None, None]


def test_recession_probit_respects_label_lag() -> None:
    n = 400
    ev = [date(1980 + i // 12, i % 12 + 1, 1) for i in range(n)]
    rng = np.random.default_rng(0)
    curve = 1.5 * np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.3, n)  # persistent cycle
    rec = np.zeros(n)
    rec[6:] = (curve[:-6] < -1.0).astype(float)  # an inverted curve -> recession 6m later
    feats = pl.DataFrame({"event_time": ev, "curve": curve})
    recession = pl.DataFrame({"event_time": ev, "value": rec})
    res = fit_recession_probit(feats, recession, ["curve"], current_month=ev[-1])
    assert res is not None
    assert res.train_end == ev[-1 - 24]
    assert res.coefficients["curve"] < 0
    assert 0.0 <= res.prob_12m <= 1.0

    none = fit_recession_probit(
        feats, recession.with_columns(pl.lit(0.0).alias("value")), ["curve"], current_month=ev[-1]
    )
    assert none is None


def _collinear_case() -> tuple[pl.DataFrame, pl.DataFrame, list[date]]:
    """A true driver ``a`` plus a near-copy ``b`` that the data load with the wrong sign."""
    n = 480
    ev = [date(1980 + i // 12, i % 12 + 1, 1) for i in range(n)]
    rng = np.random.default_rng(3)
    a = np.zeros(n)
    for t in range(1, n):  # persistent, like a labour-market gauge
        a[t] = 0.95 * a[t - 1] + rng.normal(0, 0.3)
    b = a + rng.normal(0, 0.3, n)
    latent = 2.5 * a - 1.5 * b + rng.normal(0, 0.3, n)
    rec = np.zeros(n)
    rec[1:] = (latent[:-1] > 0.8).astype(float)
    feats = pl.DataFrame({"event_time": ev, "a": a, "b": b})
    return feats, pl.DataFrame({"event_time": ev, "value": rec}), ev


def test_recession_probit_drops_wrong_signed_input() -> None:
    feats, recession, ev = _collinear_case()
    free = fit_recession_probit(feats, recession, ["a", "b"], current_month=ev[-1])
    assert free is not None
    assert free.coefficients["b"] < 0 < free.coefficients["a"]  # the collinear sign flip
    assert free.dropped == []

    signed = fit_recession_probit(
        feats, recession, ["a", "b"], current_month=ev[-1], expected_signs={"a": 1, "b": 1}
    )
    assert signed is not None
    assert signed.dropped == ["b"]
    assert set(signed.coefficients) == {"const", "a"}
    assert signed.coefficients["a"] > 0
    assert set(signed.inputs) == {"a", "b"}  # still reported, just not used
    assert signed.train_end == free.train_end  # label-lag discipline unchanged
    assert 0.0 <= signed.prob_12m <= 1.0

    # Columns without an expected sign are left unrestricted.
    partial = fit_recession_probit(
        feats, recession, ["a", "b"], current_month=ev[-1], expected_signs={"a": 1}
    )
    assert partial is not None
    assert partial.coefficients == free.coefficients


def test_recession_probit_none_when_every_input_is_wrong_signed() -> None:
    feats, recession, ev = _collinear_case()
    res = fit_recession_probit(
        feats, recession, ["a"], current_month=ev[-1], expected_signs={"a": -1}
    )
    assert res is None


def test_recession_feature_expected_signs() -> None:
    signs = {s.name: s.sign for s in RECESSION}
    assert signs == {"curve_10y3m": -1, "sahm": 1, "nfci_level": 1, "unrate_12m_chg_pp": 1}


def test_hmm_finds_separated_states() -> None:
    rng = np.random.default_rng(1)
    calm = rng.normal([1.0, -0.5, -0.5], 0.2, size=(150, 3))
    stress = rng.normal([-1.5, 0.5, 1.5], 0.2, size=(60, 3))
    x = np.vstack([calm, stress])
    res = fit_regime_hmm(x, ("growth", "inflation", "financial_conditions"))
    assert res is not None
    assert res.n_states >= 2
    top = res.most_likely
    assert top.prob > 0.9
    assert top.label == "stagflation+stress"
    assert fit_regime_hmm(x[:50], ("growth", "inflation", "financial_conditions")) is None


# ------------------------------------------------------------------ engine end to end
@pytest.mark.parametrize(
    "at", [date(2005, 6, 30), date(2011, 3, 31), date(2015, 6, 30), date(2018, 9, 30)]
)
def test_engine_recovers_latent_regime(engine: RegimeEngine, at: date) -> None:
    rep = engine.run(at, with_hmm=False)
    g, f = true_regime(months().index(rep.data_month))
    assert rep.regime is classify(g, f)
    assert sum(rep.quadrant_probs.values()) == pytest.approx(1.0)
    assert rep.growth.n_features == 7
    assert rep.inflation.n_features == 5
    assert rep.growth.score is not None
    assert sum(rep.growth.contributions.values()) == pytest.approx(rep.growth.score)
    assert rep.missing_features == ["hy_oas"]  # not in the synthetic economy


def test_engine_full_report(engine: RegimeEngine) -> None:
    rep = engine.run(date(2015, 6, 30))
    assert rep.hmm is not None
    assert rep.hmm.n_states in (2, 3, 4)
    assert sum(s.prob for s in rep.hmm.states) == pytest.approx(1.0)
    assert rep.recession is not None
    assert rep.recession.train_end <= date(2013, 5, 1)
    assert rep.recession.dropped == []  # unconstrained by default
    assert rep.model_dump_json()  # serializable for the API / LLM layer


def test_engine_sign_constraint_is_opt_in(synthetic_obs: pl.DataFrame) -> None:
    constrained = RegimeEngine(VintageCube(synthetic_obs), recession_sign_constraint=True)
    rep = constrained.run(date(2015, 6, 30), with_hmm=False)
    assert rep.recession is not None
    signs = {s.name: s.sign for s in RECESSION}
    for name, coef in rep.recession.coefficients.items():
        assert name == "const" or signs[name] * coef > 0


def test_engine_is_point_in_time(synthetic_obs: pl.DataFrame) -> None:
    at = date(2012, 12, 31)
    before = RegimeEngine(VintageCube(synthetic_obs)).run(at)

    # Rewrite every value that became public after `at` with garbage.
    cutoff = datetime(2013, 1, 1, 5, 0, tzinfo=UTC)
    poisoned = synthetic_obs.with_columns(
        pl.when(pl.col("knowledge_time") > cutoff)
        .then(pl.lit(-1e6))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    after = RegimeEngine(VintageCube(poisoned)).run(at)
    assert after == before


def test_engine_with_no_data() -> None:
    rep = RegimeEngine(VintageCube(make_obs_frame([]))).run(date(2020, 1, 31))
    assert rep.regime is None
    assert rep.quadrant_probs == {}
    assert rep.hmm is None
    assert rep.recession is None


def test_walk_forward_scores(engine: RegimeEngine) -> None:
    bt = walk_forward(engine, date(2008, 1, 1), date(2010, 12, 31))
    assert bt.monthly.height == 36
    assert bt.balanced_accuracy is not None
    assert bt.balanced_accuracy > 0.8
    assert bt.recession_brier is not None
    assert bt.recession_brier_baseline is not None
    assert bt.recession_brier < bt.recession_brier_baseline
    assert set(bt.summary()) >= {"balanced_accuracy", "recession_auc"}


def test_month_ends() -> None:
    assert month_ends(date(2020, 1, 15), date(2020, 3, 31)) == [
        date(2020, 1, 31),
        date(2020, 2, 29),
        date(2020, 3, 31),
    ]


def test_cli_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_obs: pl.DataFrame
) -> None:
    from atlas.cli import app
    from atlas.settings import get_settings

    data = tmp_path / "data"
    PITStore(data).write(synthetic_obs.filter(pl.col("event_time") >= date(1990, 1, 1)))
    monkeypatch.setenv("ATLAS_DATA_DIR", str(data))
    monkeypatch.setenv("ATLAS_CONFIG_DIR", str(REPO / "config"))
    get_settings.cache_clear()
    runner = CliRunner()
    try:
        res = runner.invoke(app, ["regime", "now", "--at", "2015-06-30"])
        assert res.exit_code == 0, res.output
        assert "stagflation" in res.output
        assert "recession within 12m" in res.output

        res = runner.invoke(app, ["regime", "now", "--at", "2015-06-30", "--json", "--no-hmm"])
        assert res.exit_code == 0, res.output
        assert '"regime": "stagflation"' in res.output

        out = tmp_path / "bt.csv"
        res = runner.invoke(
            app,
            [
                "regime",
                "backtest",
                "--start",
                "2014-01-01",
                "--end",
                "2014-06-30",
                "--out",
                str(out),
            ],
        )
        assert res.exit_code == 0, res.output
        assert "balanced_accuracy" in res.output
        assert pl.read_csv(out).height == 6
    finally:
        get_settings.cache_clear()


def test_cube_latest_knowledge(synthetic_obs: pl.DataFrame) -> None:
    cube = VintageCube(synthetic_obs)
    latest = cube.latest_knowledge_date
    assert latest is not None
    assert latest > date(2025, 12, 31)
    assert VintageCube(make_obs_frame([])).latest_knowledge_date is None
    assert "US.IP" in cube
    assert "US.NOPE" not in cube
    assert cube.as_of("US.NOPE", date(2020, 1, 1)).is_empty()
    _ = timedelta


def test_hmm_duplicate_labels_are_disambiguated() -> None:
    from atlas.models.regime.hmm import HMMState, _disambiguate

    def st(label: str, g: float, i: float, f: float) -> HMMState:
        return HMMState(
            label=label,
            prob=0.25,
            means={"growth": g, "inflation": i, "financial_conditions": f},
            persistence=0.9,
        )

    two = _disambiguate(
        [
            st("stagflation+stress", -0.6, 1.4, 1.6),
            st("stagflation+stress", -1.0, 0.2, 0.6),
            st("goldilocks", 0.5, -0.5, -0.3),
        ]
    )
    assert [s.label for s in two] == [
        "stagflation+stress, higher inflation",
        "stagflation+stress, lower inflation",
        "goldilocks",
    ]
    three = _disambiguate([st("x", 0, 1, 0), st("x", 0, 3, 0), st("x", 0, 2, 0)])
    assert [s.label for s in three] == [
        "x (3 of 3 by inflation)",
        "x (1 of 3 by inflation)",
        "x (2 of 3 by inflation)",
    ]
    assert len({s.label for s in three}) == 3
