"""Immutable snapshots (PRD §9): pin exactly which data, config and code produced an output.

A snapshot is a JSON manifest listing every silver part file with its SHA-256, plus config
hashes and the git commit. Because store parts are append-only, a snapshot's file list
stays readable forever; ``PITStore.from_snapshot`` reads only those parts.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from atlas import __version__
from atlas.pit.store import PITStore

_REPO_ROOT = Path(__file__).resolve().parents[2]


class FileEntry(BaseModel):
    path: str  # relative to data_dir
    sha256: str
    bytes: int


class Snapshot(BaseModel):
    snapshot_id: str
    created_at: datetime
    package_version: str
    code_sha: str | None
    code_dirty: bool | None
    config_sha256: dict[str, str]
    files: list[FileEntry]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(_REPO_ROOT), *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def snapshots_dir(data_dir: Path) -> Path:
    return data_dir / "snapshots"


def create_snapshot(data_dir: Path, config_dir: Path, *, now: datetime | None = None) -> Snapshot:
    created = now or datetime.now(UTC)
    store = PITStore(data_dir)
    files = [
        FileEntry(path=str(p.relative_to(data_dir)), sha256=_sha256(p), bytes=p.stat().st_size)
        for p in store.part_files()
    ]
    configs = {
        str(p.relative_to(config_dir)): _sha256(p) for p in sorted(config_dir.rglob("*.yaml"))
    }
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")

    content = hashlib.sha256()
    for f in files:
        content.update(f"{f.path}:{f.sha256}\n".encode())
    for name, digest in configs.items():
        content.update(f"cfg:{name}:{digest}\n".encode())
    snapshot_id = f"snap_{created:%Y-%m-%dT%H%MZ}_{content.hexdigest()[:4]}"

    snap = Snapshot(
        snapshot_id=snapshot_id,
        created_at=created,
        package_version=__version__,
        code_sha=sha,
        code_dirty=None if status is None else bool(status),
        config_sha256=configs,
        files=files,
    )
    out = snapshots_dir(data_dir) / f"{snapshot_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(snap.model_dump_json(indent=2))
    return snap


def load_snapshot(data_dir: Path, snapshot_id: str) -> Snapshot:
    path = snapshots_dir(data_dir) / f"{snapshot_id}.json"
    return Snapshot.model_validate_json(path.read_text())


def verify_snapshot(data_dir: Path, snap: Snapshot) -> list[str]:
    """Return a list of problems; empty means every pinned file is present and unchanged."""
    problems = []
    for f in snap.files:
        path = data_dir / f.path
        if not path.exists():
            problems.append(f"missing: {f.path}")
        elif _sha256(path) != f.sha256:
            problems.append(f"modified: {f.path}")
    return problems


def store_from_snapshot(data_dir: Path, snap: Snapshot) -> PITStore:
    problems = verify_snapshot(data_dir, snap)
    if problems:
        raise ValueError(f"snapshot {snap.snapshot_id} is not reproducible: {problems[:5]}")
    return PITStore(data_dir, pinned_parts=[data_dir / f.path for f in snap.files])
