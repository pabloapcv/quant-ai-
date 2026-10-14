from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl
import pytest

from atlas.features.registry import FeatureSpec, Pillar, feature_matrix
from atlas.features.transforms import Annualized, Diff, Monthly, PctChange, RollingZ, apply_all
from atlas.pit.cube import VintageCube
from tests.conftest import make_obs_frame, utc


def frame(dates: Sequence[date], values: Sequence[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"event_time": list(dates), "value": list(values)},
        schema={"event_time": pl.Date, "value": pl.Float64},
    )


def monthly(values: Sequence[float | None], start: int = 2000) -> pl.DataFrame:
    return frame([date(start + i // 12, i % 12 + 1, 1) for i in range(len(values))], values)


def test_monthly_averages_and_fills_gaps() -> None:
    daily = frame([date(2020, 1, 2), date(2020, 1, 31), date(2020, 3, 5)], [1.0, 3.0, 10.0])
    out = Monthly().apply(daily)
    assert out.rows() == [
        (date(2020, 1, 1), 2.0),
        (date(2020, 2, 1), None),
        (date(2020, 3, 1), 10.0),
    ]
    assert Monthly("last").apply(daily)["value"][0] == 3.0
    assert Monthly().apply(daily.clear()).is_empty()


def test_change_transforms() -> None:
    df = monthly([100.0] * 12 + [110.0])
    assert PctChange(12).apply(df)["value"][-1] == pytest.approx(10.0)
    assert Diff(12).apply(df)["value"][-1] == pytest.approx(10.0)
    six = monthly([100.0] * 6 + [105.0])
    assert Annualized(6).apply(six)["value"][-1] == pytest.approx(100 * (1.05**2 - 1))


def test_rolling_z_is_causal() -> None:
    base = [float(i % 7) for i in range(80)]
    z1 = RollingZ(window=24, min_samples=12).apply(monthly(base))["value"]
    changed = base[:60] + [1000.0] * 20
    z2 = RollingZ(window=24, min_samples=12).apply(monthly(changed))["value"]
    assert z1[:60].to_list() == z2[:60].to_list()
    assert z1[:11].null_count() == 11
    flat = RollingZ(window=24, min_samples=12).apply(monthly([5.0] * 30))["value"]
    assert flat.null_count() == 30  # zero variance -> undefined, not inf


def test_apply_all_scrubs_non_finite() -> None:
    out = apply_all(monthly([0.0] * 12 + [5.0]), (Monthly(), PctChange(12)))
    assert out["value"][-1] is None


def test_feature_matrix_ragged_edge() -> None:
    a = "US.A.X"
    b = "US.B.X"
    rows = [(a, date(2020, m, 1), utc(2020, m, 20), float(m)) for m in range(1, 7)]
    rows += [(b, date(2020, m, 1), utc(2020, m, 20), float(-m)) for m in range(1, 3)]
    cube = VintageCube(make_obs_frame(rows))
    specs = [FeatureSpec("fa", a, Pillar.GROWTH, ()), FeatureSpec("fb", b, Pillar.GROWTH, ())]
    m = feature_matrix(cube, specs, date(2020, 7, 31), end_month=date(2020, 6, 1), ffill_limit=2)
    assert m["event_time"].to_list()[-1] == date(2020, 6, 1)
    assert m["fa"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert m["fb"].to_list() == [-1.0, -2.0, -2.0, -2.0, None, None]  # filled 2 months only

    # As of mid-March only Jan/Feb are public.
    early = feature_matrix(cube, specs, date(2020, 3, 10), end_month=date(2020, 2, 1))
    assert early["fa"].to_list() == [1.0, 2.0]

    empty = feature_matrix(
        VintageCube(make_obs_frame([])), specs, date(2020, 1, 1), end_month=date(2020, 1, 1)
    )
    assert empty.is_empty()
