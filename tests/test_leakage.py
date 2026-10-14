"""Look-ahead leakage suite (PRD §8.6). Blocking in CI.

Any feature or query computed as of ``t`` must be byte-identical whether or not data
published after ``t`` exists in the store.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from atlas.pit.clock import to_cutoff
from atlas.pit.store import PITStore
from tests.conftest import make_obs_frame, utc

S = "US.TEST.SERIES"
T = date(2022, 6, 30)


def _base_rows() -> list[tuple[str, date, datetime, float | None]]:
    rows: list[tuple[str, date, datetime, float | None]] = []
    for m in range(1, 13):
        ev = date(2021, m, 1)
        rows.append((S, ev, utc(2021, m, 1) + timedelta(days=40), float(m)))
        rows.append((S, ev, utc(2021, m, 1) + timedelta(days=400), float(m) + 0.25))  # revision
    return rows


def test_future_vintage_injection_does_not_change_past(tmp_path: Path) -> None:
    store = PITStore(tmp_path / "d")
    store.write(make_obs_frame(_base_rows()))
    before = store.as_of([S], T)
    panel_before = store.realtime_panel([S], [date(2022, 1, 31), T])

    # Inject revisions, new periods, and withdrawals that all become known AFTER T.
    after = to_cutoff(T) + timedelta(seconds=1)
    future = [
        (S, date(2021, 3, 1), after, -999.0),
        (S, date(2021, 4, 1), after + timedelta(days=30), None),
        (S, date(2022, 6, 1), after, 42.0),
    ]
    store.write(make_obs_frame(future, run_id="run_future"))

    assert store.as_of([S], T).equals(before)
    assert store.realtime_panel([S], [date(2022, 1, 31), T]).equals(panel_before)


_vintage = st.tuples(
    st.integers(min_value=0, max_value=11),  # event month offset
    st.integers(min_value=0, max_value=900),  # days after event start until known
    st.one_of(st.none(), st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)),
)


@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    vintages=st.lists(_vintage, min_size=1, max_size=40),
    cutoff_days=st.integers(min_value=0, max_value=1000),
)
def test_as_of_matches_brute_force(
    tmp_path_factory: pytest.TempPathFactory,
    vintages: list[tuple[int, int, float | None]],
    cutoff_days: int,
) -> None:
    store = PITStore(tmp_path_factory.mktemp("prop"))
    seen: dict[tuple[date, datetime], float | None] = {}
    for month, lag, value in vintages:
        ev = date(2020, month + 1, 1)
        seen[(ev, utc(2020, month + 1, 1) + timedelta(days=lag))] = value  # last write per key wins
    store.write(make_obs_frame([(S, ev, kt, v) for (ev, kt), v in seen.items()]))

    cutoff = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=cutoff_days)
    expected: dict[date, float] = {}
    latest_kt: dict[date, datetime] = {}
    for (ev, kt), v in seen.items():
        if kt <= cutoff and (ev not in latest_kt or kt > latest_kt[ev]):
            latest_kt[ev] = kt
            expected.pop(ev, None)
            if v is not None:
                expected[ev] = v

    got = store.as_of([S], cutoff)
    assert dict(zip(got["event_time"].to_list(), got["value"].to_list(), strict=True)) == expected
    assert got.filter(pl.col("knowledge_time") > cutoff).is_empty()
