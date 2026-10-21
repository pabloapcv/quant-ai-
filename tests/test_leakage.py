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


# ------------------------------------------------------------------ shuffle test (PRD §8.6)
def test_shuffled_target_has_no_ic(tmp_path: Path) -> None:
    """Shuffling forward returns within each date must destroy every signal's IC.

    If the backtester (or a signal) peeked at the target, a shuffled target would still
    show skill. 40 assets x 480 months puts the noise s.d. of the mean IC near 0.007.
    """
    import numpy as np

    from atlas.backtest.crosssection import backtest_signal
    from atlas.backtest.trials import TrialLog
    from atlas.signals.industry import SIGNALS, research_panel
    from atlas.signals.research import ResearchContext, signal_panel
    from tests.test_research import _store

    store = _store(tmp_path, n_months=480, n_assets=40, start_year=1980)
    ctx = ResearchContext(
        panel=research_panel(store, "ind12"), universe="ind12", log=TrialLog(tmp_path)
    )
    checked = 0
    for sig in SIGNALS.values():
        if sig.ranking_of is not None:
            continue
        panel = signal_panel(sig, ctx).drop_nulls(["signal", "fwd_ret"]).sort(["date", "asset"])
        if panel.is_empty():
            continue
        # deterministic shuffle within each date (numpy, one seed per date)
        dates = panel["date"].to_numpy()
        fwd = panel["fwd_ret"].to_numpy().copy()
        for i, d in enumerate(np.unique(dates)):
            idx = np.flatnonzero(dates == d)
            fwd[idx] = np.random.default_rng(i).permutation(fwd[idx])
        shuffled = panel.with_columns(pl.Series("fwd_ret", fwd))
        rep = backtest_signal(shuffled, sig.name, n_quantiles=3)
        assert abs(rep.ic_mean) < 0.03, (sig.name, rep.ic_mean)
        assert abs(rep.ic_tstat) < 3.5, (sig.name, rep.ic_tstat)  # no skill beyond chance
        checked += 1
    assert checked >= 5


# ------------------------------------------------------------------ timestamp audit (PRD §8.6)
def test_timestamp_audit_nothing_known_after_as_of(tmp_path: Path) -> None:
    """Every observation a point-in-time read returns was public by the as-of cutoff."""
    from atlas.features.macro import GROWTH
    from atlas.features.registry import feature_matrix

    rows = []
    for m in range(36):
        ev = date(2019 + m // 12, m % 12 + 1, 1)
        first = datetime(ev.year, ev.month, 1, tzinfo=UTC) + timedelta(days=45)
        rows += [(spec.series_id, ev, first, 100.0 + m) for spec in GROWTH] + [
            (spec.series_id, ev, first + timedelta(days=300), 101.0 + m) for spec in GROWTH
        ]
    store = PITStore(tmp_path / "d")
    store.write(make_obs_frame(rows))
    cube = store.load_cube()
    for at in (date(2019, 9, 30), date(2020, 6, 15), date(2021, 3, 31)):
        cutoff = to_cutoff(at)
        for spec in GROWTH:
            got = cube.as_of(spec.series_id, at)
            assert got.is_empty() or got["knowledge_time"].max() <= cutoff  # type: ignore[operator]
        # the feature matrix equals one built from a store that never saw later data
        visible = make_obs_frame([r for r in rows if r[2] <= cutoff])
        clean = PITStore(tmp_path / f"clean_{at}")
        clean.write(visible)
        a = feature_matrix(cube, GROWTH, at, end_month=date(at.year, at.month, 1))
        b = feature_matrix(clean.load_cube(), GROWTH, at, end_month=date(at.year, at.month, 1))
        assert a.equals(b)


def test_documents_are_point_in_time(tmp_path: Path) -> None:
    from atlas.text.documents import DocStore, Document

    store = DocStore(tmp_path)
    for day in (date(2020, 1, 29), date(2020, 3, 15), date(2020, 4, 29)):
        store.save(
            Document(
                doc_id=f"fomc-statement-{day}",
                source="fomc",
                kind="statement",
                title="t",
                url="u",
                published_at=datetime(day.year, day.month, day.day, 18, tzinfo=UTC),
                meeting_date=day,
                paragraphs=["a", "b"],
            )
        )
    at = datetime(2020, 3, 15, 17, 59, tzinfo=UTC)  # a minute before the March release
    seen = list(store.iter(as_of=at))
    assert [d.meeting_date for d in seen] == [date(2020, 1, 29)]
    assert all(d.published_at <= at for d in seen)
