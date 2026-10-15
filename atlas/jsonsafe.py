"""Strict JSON output: NaN and infinities become null.

Python's ``json`` writes ``NaN``/``Infinity`` by default, which is not JSON: browsers and
most parsers reject the whole document. Every file or stdout payload Atlas emits for other
tools goes through ``dumps``.
"""

from __future__ import annotations

import json
import math
from typing import Any


def sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [sanitize(v) for v in obj]
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("default", str)
    return json.dumps(sanitize(obj), allow_nan=False, **kwargs)
