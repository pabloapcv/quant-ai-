"""Series catalog: canonical series ids and their source mapping (``config/series.yaml``)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import time
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from atlas.pit.schema import Frequency

CONSERVATIVE_RELEASE_TIME = time(23, 59)


class SeriesSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[A-Z0-9]+(\.[A-Z0-9]+)+$")
    source: str
    source_id: str
    frequency: Frequency
    publication_lag_days: int = Field(ge=0)
    release_time_et: time = CONSERVATIVE_RELEASE_TIME
    description: str
    tags: tuple[str, ...] = ()


class Catalog:
    def __init__(self, specs: list[SeriesSpec]) -> None:
        self._specs: dict[str, SeriesSpec] = {}
        for spec in specs:
            if spec.id in self._specs:
                raise ValueError(f"duplicate series id in catalog: {spec.id}")
            self._specs[spec.id] = spec

    @classmethod
    def load(cls, path: Path) -> Catalog:
        raw = yaml.safe_load(path.read_text())
        return cls([SeriesSpec.model_validate(item) for item in raw["series"]])

    def get(self, series_id: str) -> SeriesSpec:
        try:
            return self._specs[series_id]
        except KeyError:
            raise KeyError(f"unknown series id: {series_id}") from None

    def by_source(self, source: str) -> list[SeriesSpec]:
        return [s for s in self._specs.values() if s.source == source]

    def __iter__(self) -> Iterator[SeriesSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
