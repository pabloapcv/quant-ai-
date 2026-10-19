from datetime import date

import numpy as np
import polars as pl
import pytest

from atlas.models.scenario import AVERAGE, _block_rows, _full_move, fit_scenario_model


def _data(n: int = 240, seed: int = 1) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    months = pl.date_range(
        date(2000, 1, 1), date(2000 + (n - 1) // 12, (n - 1) % 12 + 1, 1), "1mo", eager=True
    )
    a = rng.normal(0, 1, n)
    b = 0.8 * a + 0.6 * rng.normal(0, 1, n)  # correlated with a
    changes = pl.DataFrame({"date": months, "fa": a, "fb": b})
    noise = rng.normal(0, 0.001, (n, 2))
    # X loads on fa only, Y on fb only; both get a common market term.
    mkt = 0.01 * rng.normal(0, 1, n)
    rx = mkt + 0.02 * a + noise[:, 0]
    ry = mkt + 0.02 * b + noise[:, 1]
    panel = pl.concat(
        [
            pl.DataFrame({"date": months, "asset": "X", "ret": rx}),
            pl.DataFrame({"date": months, "asset": "Y", "ret": ry}),
        ]
    )
    return changes, panel


def test_betas_and_conditional_moves() -> None:
    changes, panel = _data()
    model = fit_scenario_model(changes, panel)
    assert model.assets == ("X", "Y", AVERAGE)
    res = model.run({"fa": 1.0}, n_boot=50)
    implied = res.implied_factor_moves["fb"]
    assert implied == pytest.approx(0.8, abs=0.1)  # Sigma_ba / Sigma_aa
    by = {r.asset: r for r in res.assets}
    # X excess = (0.02 - avg)/..., computed from both slopes: X gets 0.02, Y gets 0.02*implied.
    assert by["X"].expected == pytest.approx((0.02 - 0.02 * implied) / 2, abs=1e-3)
    assert by["X"].p10 <= by["X"].expected <= by["X"].p90
    assert res.average.expected == pytest.approx(0.01 * (1 + implied), abs=1e-3)


def test_both_shocked_has_no_implied_moves() -> None:
    changes, panel = _data()
    res = fit_scenario_model(changes, panel).run({"fa": 1.0, "fb": -1.0}, n_boot=20)
    assert res.implied_factor_moves == {}
    by = {r.asset: r.expected for r in res.assets}
    assert by["X"] == pytest.approx(0.02, abs=1e-3)  # (0.02 - (-0.02)) / 2


def test_validation() -> None:
    changes, panel = _data()
    model = fit_scenario_model(changes, panel)
    with pytest.raises(ValueError, match="unknown factor"):
        model.run({"gold": 1.0})
    with pytest.raises(ValueError, match="at least one"):
        model.run({})
    with pytest.raises(ValueError, match="need 60"):
        fit_scenario_model(changes, panel, months=changes["date"].to_list()[:30])


def test_full_move_and_block_rows() -> None:
    x = np.random.default_rng(0).normal(size=(100, 3))
    move = _full_move(x, [1], np.array([2.0]))
    assert move[1] == 2.0
    rows = _block_rows(100, 12, np.random.default_rng(0))
    assert len(rows) == 100
    assert rows.min() >= 0
    assert rows.max() <= 99
