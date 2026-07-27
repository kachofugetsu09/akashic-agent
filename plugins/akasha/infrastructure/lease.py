"""Hold one fail-loud process lease for the writable memory sidecar."""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import IO


class WriterLease:
    """Own the exclusive process lease for one Akasha database."""

    def __init__(self, memory_path: Path) -> None:
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = memory_path.with_suffix(memory_path.suffix + ".lock")
        self._handle: IO[str] = self.path.open("a+", encoding="utf-8")
        self._closed = False
        try:
            fcntl.flock(
                self._handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            self._handle.close()
            raise RuntimeError(
                f"Akasha memory already has a writer: {memory_path}"
            ) from error

    def close(self) -> None:
        if self._closed:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._closed = True
