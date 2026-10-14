"""Leakage-safe cross-validation for overlapping financial labels (López de Prado, AFML ch.7).

Observations are time periods ``0..T-1``; the label of period ``i`` spans ``[i, i+horizon]``.
A training period is *purged* if its label window overlaps any test label window, and
*embargoed* for ``embargo`` periods after the test block (serial correlation spill-over).
For panels, split on dates and map the date indices back to rows.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]


def _train_mask(n: int, test_idx: IntArray, horizon: int, embargo: int) -> NDArray[np.bool_]:
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    # contiguous runs of test periods
    runs = np.split(test_idx, np.where(np.diff(test_idx) != 1)[0] + 1)
    for run in runs:
        a, b = int(run[0]), int(run[-1])
        lo = max(0, a - horizon)  # train label [i, i+h] reaches into the test block
        hi = min(n, b + horizon + embargo + 1)  # test labels reach b+h, then embargo
        mask[lo:hi] = False
    return mask


@dataclass(frozen=True)
class PurgedKFold:
    n_splits: int = 5
    horizon: int = 1
    embargo: int = 0

    def split(self, n: int) -> Iterator[tuple[IntArray, IntArray]]:
        if self.n_splits < 2 or n < self.n_splits:
            raise ValueError("need n >= n_splits >= 2")
        for test in np.array_split(np.arange(n, dtype=np.int64), self.n_splits):
            train = np.flatnonzero(_train_mask(n, test, self.horizon, self.embargo))
            yield train.astype(np.int64), test


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """CPCV: every choice of ``n_test_groups`` out of ``n_groups`` blocks is a test set.

    Produces ``C(N, k)`` splits and ``C(N, k) * k / N`` complete backtest paths — a
    distribution of out-of-sample outcomes instead of a single walk-forward path.
    """

    n_groups: int = 10
    n_test_groups: int = 2
    horizon: int = 1
    embargo: int = 0

    @property
    def n_splits(self) -> int:
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        return self.n_splits * self.n_test_groups // self.n_groups

    def split(self, n: int) -> Iterator[tuple[IntArray, IntArray]]:
        if not 0 < self.n_test_groups < self.n_groups or n < self.n_groups:
            raise ValueError("need 0 < n_test_groups < n_groups <= n")
        groups = np.array_split(np.arange(n, dtype=np.int64), self.n_groups)
        for combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test = np.concatenate([groups[g] for g in combo])
            train = np.flatnonzero(_train_mask(n, test, self.horizon, self.embargo))
            yield train.astype(np.int64), test
