from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError(f"JSON 根必须是 object: {path}")
    return document


def atomic_write(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Fsync and atomically replace one operator-owned state file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, document: Mapping[str, object]) -> None:
    atomic_write(
        path,
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


@contextmanager
def release_lock(path: Path) -> Iterator[None]:
    """Reject a second release transaction instead of silently queueing it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"已有 Akashic release transaction: {path}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def activation_receipt(
    *,
    status: str,
    target_commit: str,
    previous_commit: str | None,
    detail: str | None = None,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schemaVersion": 1,
        "status": status,
        "targetCommit": target_commit,
        "previousCommit": previous_commit,
        "recordedAt": datetime.now(timezone.utc).isoformat(),
    }
    if detail:
        receipt["detail"] = detail
    return receipt
