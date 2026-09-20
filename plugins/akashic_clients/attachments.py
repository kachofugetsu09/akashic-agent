"""Filesystem staging owned by the Akashic clients plugin."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class AttachmentStore:
    """Keep transient upload files inside one plugin-owned workspace root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create_path(self, prefix: str, suffix: str) -> Path:
        return self._new_path(prefix=prefix, suffix=suffix)

    def create_persistent_path(self, prefix: str, suffix: str) -> Path:
        return self._new_path(prefix=prefix, suffix=suffix)

    def create_staging_path(
        self, *, prefix: str = ".upload-", suffix: str = ".part"
    ) -> Path:
        return self._new_path(prefix=prefix, suffix=suffix)

    def publish_staging(self, staging: Path, *, prefix: str, suffix: str) -> Path:
        target = self._new_path(prefix=prefix, suffix=suffix)
        staging.replace(target)
        return target

    def write_bytes(self, data: bytes, *, prefix: str, suffix: str) -> Path:
        if not isinstance(data, bytes):
            raise TypeError("attachment data 必须是 bytes")
        staging = self.create_staging_path(prefix=f".{prefix}", suffix=".part")
        try:
            with staging.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            return self.publish_staging(staging, prefix=prefix, suffix=suffix)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise

    def _new_path(self, *, prefix: str, suffix: str) -> Path:
        fd, raw = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=self.root)
        os.close(fd)
        path = Path(raw)
        path.unlink()
        return path


__all__ = ["AttachmentStore"]
