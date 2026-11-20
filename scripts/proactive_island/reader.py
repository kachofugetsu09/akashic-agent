"""Open legacy proactive SQLite files through one strict offline reader."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def open_legacy_sqlite(path: Path) -> Generator[sqlite3.Connection]:
    """Yield an immutable verified connection or expose an unquiet legacy store."""

    # 1. Immutable readers cannot see WAL, so require the offline checkpoint first.
    wal_path = path.with_name(path.name + "-wal")
    if wal_path.is_file() and wal_path.stat().st_size > 0:
        raise RuntimeError(f"legacy proactive SQLite has uncheckpointed WAL: {path}")

    # 2. Read without creating the database, journal, shared memory, or schema.
    database_uri = path.resolve(strict=False).as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(database_uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        _ = connection.execute("PRAGMA query_only = ON")
        check = [tuple(row) for row in connection.execute("PRAGMA quick_check")]
        if check != [("ok",)]:
            raise RuntimeError(f"legacy proactive SQLite quick_check failed: {path}")
        yield connection
    finally:
        connection.close()


__all__ = ["open_legacy_sqlite"]
