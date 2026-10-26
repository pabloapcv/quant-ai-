"""Day-scoped disk cache for slow, deterministic results (briefing, business lenses).

A result is keyed by the UTC date and the latest pipeline run, so it is rebuilt after new
data and reused otherwise: by API restarts and by the assistant's tools alike.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas import jsonsafe


def latest_run_name(data_dir: Path) -> str:
    runs = sorted((data_dir / "runs").glob("2*.json"))
    return runs[-1].name if runs else ""


def disk_cached(
    data_dir: Path, name: str, build: Callable[[], dict[str, Any]], variant: str = ""
) -> dict[str, Any]:
    folder = data_dir / "cache" / "web"
    stamp = f"{datetime.now(UTC).date()}_{latest_run_name(data_dir).removesuffix('.json')}"
    path = folder / f"{stamp}{variant}_{name}.json"
    if path.exists():
        try:
            cached: dict[str, Any] = json.loads(path.read_text())
            return cached
        except ValueError:
            pass  # half-written file: rebuild
    value = build()
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"*_{name}.json"):  # earlier days or runs
        old.unlink(missing_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(jsonsafe.dumps(value))
    tmp.replace(path)
    return value
