from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from atlas.pit.clock import to_cutoff
from atlas.pit.store import PITStore
from atlas.quality.contracts import DataQualityError
from tests.conftest import MakeObs, make_obs_frame, utc

CPI = "US.CPI.CORE"

# CPI for Jan 2020: first print 100.0 on 2020-02-13, revised to 100.5 on 2021-02-19.
REVISED = [
    (CPI, date(2020, 1, 1), utc(2020, 2, 13, 13, 30), 100.0),
    (CPI, date(2020, 1, 1), utc(2021, 2, 19, 13, 30), 100.5),
    (CPI, date(2020, 2, 1), utc(2020, 3, 11, 12, 30), 101.0),
]


def test_as_of_returns_vintage_known_at_the_time(store: PITStore, make_obs: MakeObs) -> None:
    store.write(make_obs(REVISED))

    before_first = store.as_of([CPI], utc(2020, 2, 13, 13, 29))
    assert before_first.is_empty()

    first = store.as_of([CPI], date(2020, 2, 13))
    assert first["value"].to_list() == [100.0]

    mid = store.as_of([CPI], date(2020, 12, 31))
    assert mid.select("event_time", "value").rows() == [
        (date(2020, 1, 1), 100.0),
        (date(2020, 2, 1), 101.0),
    ]

    latest = store.as_of([CPI], date(2026, 1, 1))
    assert latest.filter(pl.col("event_time") == date(2020, 1, 1))["value"].item() == 100.5


def test_date_cutoff_is_end_of_day_new_york(store: PITStore, make_obs: MakeObs) -> None:
    # 23:30 ET on 2020-03-02 (EST, UTC-5) is 04:30 UTC on 2020-03-03.
    store.write(make_obs([(CPI, date(2020, 1, 1), datetime(2020, 3, 3, 4, 30, tzinfo=UTC), 1.0)]))
    assert store.as_of([CPI], date(2020, 3, 2)).height == 1
    assert store.as_of([CPI], date(2020, 3, 1)).height == 0


def test_naive_datetime_is_rejected(store: PITStore) -> None:
    with pytest.raises(ValueError, match="naive"):
        store.as_of([CPI], datetime(2020, 1, 1))


def test_withdrawn_value_is_not_resurrected(store: PITStore, make_obs: MakeObs) -> None:
    store.write(
        make_obs(
            [
                (CPI, date(2020, 1, 1), utc(2020, 2, 13), 100.0),
                (CPI, date(2020, 1, 1), utc(2020, 6, 1), None),
            ]
        )
    )
    assert store.as_of([CPI], date(2020, 3, 1)).height == 1
    assert store.as_of([CPI], date(2020, 7, 1)).is_empty()


def test_rewrite_is_idempotent(store: PITStore, make_obs: MakeObs) -> None:
    first = store.write(make_obs(REVISED))
    assert first[0].rows_new == 3
    assert first[0].part is not None

    again = store.write(make_obs(REVISED, run_id="run_2"))
    assert again[0].rows_new == 0
    assert again[0].part is None
    assert len(store.part_files([CPI])) == 1


def test_only_new_rows_are_appended(store: PITStore, make_obs: MakeObs) -> None:
    store.write(make_obs(REVISED[:2]))
    res = store.write(make_obs(REVISED))
    assert (res[0].rows_new, res[0].rows_conflicting) == (1, 0)
    assert len(store.part_files([CPI])) == 2
    assert store.scan([CPI]).collect().height == 3


def test_restated_vintage_latest_ingest_wins(store: PITStore, make_obs: MakeObs) -> None:
    row = (CPI, date(2020, 1, 1), utc(2020, 2, 13), 100.0)
    store.write(make_obs([row], ingested_at=datetime(2026, 1, 1, tzinfo=UTC)))
    res = store.write(make_obs([(*row[:3], 99.0)], ingested_at=datetime(2026, 2, 1, tzinfo=UTC)))
    assert res[0].rows_conflicting == 1
    assert store.as_of([CPI], date(2026, 3, 1))["value"].item() == 99.0


def test_history_returns_revision_path(store: PITStore, make_obs: MakeObs) -> None:
    store.write(make_obs(REVISED))
    hist = store.history(CPI, date(2020, 1, 1))
    assert hist["value"].to_list() == [100.0, 100.5]


def test_include_approx_toggle(store: PITStore, make_obs: MakeObs) -> None:
    store.write(make_obs([(CPI, date(1960, 1, 1), utc(1960, 2, 20), 30.0)], approx=True))
    assert store.as_of([CPI], date(1970, 1, 1)).height == 1
    assert store.as_of([CPI], date(1970, 1, 1), include_approx=False).is_empty()


def test_event_window_filters(store: PITStore, make_obs: MakeObs) -> None:
    store.write(make_obs(REVISED))
    out = store.as_of([CPI], date(2026, 1, 1), event_start=date(2020, 2, 1))
    assert out["event_time"].to_list() == [date(2020, 2, 1)]


def test_realtime_panel_matches_what_a_model_would_have_seen(
    store: PITStore, make_obs: MakeObs
) -> None:
    store.write(make_obs(REVISED))
    panel = store.realtime_panel(
        [CPI], [date(2020, 2, 1), date(2020, 2, 28), date(2020, 3, 31), date(2021, 3, 1)]
    )
    # 2020-02-01: nothing published yet -> no row
    assert panel.select("event_time", "value").rows() == [
        (date(2020, 1, 1), 100.0),  # Feb 28: latest period is Jan, first print
        (date(2020, 2, 1), 101.0),  # Mar 31: Feb print is out
        (date(2020, 2, 1), 101.0),  # 2021: still Feb is the latest period
    ]


def test_contract_blocks_bad_writes(store: PITStore, make_obs: MakeObs) -> None:
    dup = make_obs([REVISED[0], REVISED[0]])
    with pytest.raises(DataQualityError):
        store.write(dup)

    premature = make_obs([(CPI, date(2020, 5, 1), utc(2020, 4, 1), 1.0)])
    with pytest.raises(DataQualityError, match="known before"):
        store.write(premature)

    bad_freq = make_obs([REVISED[0]]).with_columns(pl.lit("X").alias("frequency"))
    with pytest.raises(DataQualityError):
        store.write(bad_freq)
    assert store.part_files() == []


def test_multiple_series_and_listing(store: PITStore) -> None:
    other = "US.LABOR.UNRATE"
    store.write(make_obs_frame([*REVISED, (other, date(2020, 1, 1), utc(2020, 2, 7), 3.5)]))
    assert store.series_ids() == [CPI, other]
    out = store.as_of([CPI, other], date(2020, 3, 1))
    assert set(out["series_id"].to_list()) == {CPI, other}
    last = store.last_knowledge_times()
    assert last[other] == utc(2020, 2, 7)


def test_empty_store_reads(store: PITStore) -> None:
    assert store.as_of([CPI], date(2020, 1, 1)).is_empty()
    assert store.series_ids() == []
    assert store.last_knowledge_times() == {}


def test_knowledge_time_never_exceeds_cutoff(store: PITStore, make_obs: MakeObs) -> None:
    rows = [
        (
            CPI,
            date(2020, 1, 1) + timedelta(days=31 * i),
            utc(2020, 2, 13) + timedelta(days=31 * i),
            float(i),
        )
        for i in range(24)
    ]
    store.write(make_obs(rows))
    for cutoff in [date(2020, 6, 30), date(2021, 1, 15), date(2021, 12, 31)]:
        out = store.as_of([CPI], cutoff)
        latest = out["knowledge_time"].max()
        assert isinstance(latest, datetime)
        assert latest <= to_cutoff(cutoff)
