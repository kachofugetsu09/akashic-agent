"""Shared immutable records and path helpers for rehearsal snapshots."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CopyRecord:
    path: str
    kind: str
    size: int
    sha256: str | None = None
    link_target: str | None = None


@dataclass(frozen=True)
class SourceEntry:
    relative: Path
    kind: str
    size: int
    mtime_ns: int
    mode: int
    link_target: str | None = None
    sha256: str | None = None


class SnapshotDriftError(RuntimeError):
    """表示在线复制期间 Workspace 的纳入集合发生变化。"""


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        _ = path.relative_to(root)
    except ValueError:
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
