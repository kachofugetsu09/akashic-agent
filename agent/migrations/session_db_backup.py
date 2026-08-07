from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4


def _integrity_check(path: Path) -> None:
    """Verify a SQLite file before publishing it as recovery evidence."""

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        connection.close()
    if rows != [("ok",)]:
        raise RuntimeError(f"SQLite integrity_check 失败: {path}: {rows[:3]}")


def backup_sqlite_database(
    source: Path,
    backup_root: Path,
    *,
    migration: str,
) -> Path:
    """Create an online SQLite backup and manifest before a migration DDL write."""

    # 1. Allocate a private, recoverable directory before touching the source DB.
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(backup_root, 0o700)
    backup = backup_root / source.name
    candidate = backup.with_name(f".{backup.name}.{uuid4().hex}.tmp")
    try:
        source_connection = sqlite3.connect(source)
        try:
            target_connection = sqlite3.connect(candidate)
            try:
                source_connection.backup(target_connection)
                target_connection.commit()
            finally:
                target_connection.close()
        finally:
            source_connection.close()
        _integrity_check(candidate)
        candidate.replace(backup)

        # 2. Persist machine-readable location, digest, and integrity evidence.
        payload = {
            "schema_version": 1,
            "migration": migration,
            "source": str(source),
            "backup": backup.name,
            "sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
            "sqlite_integrity": "ok",
        }
        manifest = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        manifest_path = backup_root / "manifest.json"
        manifest_candidate = manifest_path.with_name(
            f".{manifest_path.name}.{uuid4().hex}.tmp"
        )
        manifest_candidate.write_bytes(manifest)
        manifest_candidate.chmod(0o600)
        with manifest_candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest_candidate.replace(manifest_path)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    return backup
