from __future__ import annotations

from pathlib import Path

import pytest
import typer

from atlas.cli import _load_signal_file

REPO = Path(__file__).resolve().parents[1]


def test_example_signal_file_loads() -> None:
    out = _load_signal_file(REPO / "docs" / "examples" / "my_signal.py")
    assert out["signal"].name == "my_mom_9"
    assert [s.name for s in out["family"]] == ["my_mom_6", "my_mom_9", "my_mom_12"]


def test_signal_file_needs_signal(tmp_path: Path) -> None:
    f = tmp_path / "bad.py"
    f.write_text("X = 1\n")
    with pytest.raises(typer.BadParameter, match="SIGNAL"):
        _load_signal_file(f)
    with pytest.raises(typer.BadParameter, match="no such file"):
        _load_signal_file(tmp_path / "missing.py")
