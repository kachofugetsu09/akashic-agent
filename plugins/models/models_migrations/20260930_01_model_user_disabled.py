"""Keep explicit model opt-outs separate from provider catalog availability."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = set()
__transactional__ = False

_ID = "20260930_01_model_user_disabled"
_TABLES = ("model_definitions", "embedding_models")
_COLUMN = "user_disabled INTEGER NOT NULL DEFAULT 0 CHECK (user_disabled IN (0, 1))"


def _backup_registry(path: Path, root: Path) -> None:
    """Keep this immutable plugin migration independent of private Core helpers."""

    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(root, 0o700)
    backup = root / path.name
    descriptor = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    with closing(sqlite3.connect(path)) as source:
        with closing(sqlite3.connect(backup)) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("model registry backup failed integrity check")
    with backup.open("rb") as stream:
        os.fsync(stream.fileno())
    manifest = {
        "schema_version": 1,
        "migration": _ID,
        "source": str(path),
        "backup": backup.name,
        "sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
        "sqlite_integrity": "ok",
    }
    descriptor = os.open(
        root / "manifest.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for directory in (root, root.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def add_user_disabled(_connection: object) -> None:
    """Back up the existing registry, then atomically add both opt-out columns."""

    workspace = current_migration_context().workspace
    path = workspace / "model-registry.sqlite3"
    if path.is_symlink():
        raise ValueError("model registry migration refuses a symbolic link")
    if not path.exists():
        return  # A fresh Models owner creates the current schema.
    if not path.is_file():
        raise ValueError("model registry is not a regular file")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            required = {
                "model_registry_meta",
                "model_connections",
                "model_role_bindings",
                *_TABLES,
            }
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not required.issubset(tables):
                raise RuntimeError("model registry schema lineage is incomplete")
            present = []
            for table in _TABLES:
                columns = {
                    row[1]: row
                    for row in connection.execute(f"PRAGMA table_info({table})")
                }
                if not {"id", "connection_id", "model", "enabled"}.issubset(columns):
                    raise RuntimeError(f"{table} schema lineage is incompatible")
                column = columns.get("user_disabled")
                present.append(column is not None)
                if column is not None:
                    sql = connection.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()[0]
                    if (str(column[2]).upper(), column[3], column[4], column[5]) != (
                        "INTEGER",
                        1,
                        "0",
                        0,
                    ) or "CHECK(USER_DISABLEDIN(0,1))" not in "".join(
                        sql.upper().split()
                    ):
                        raise RuntimeError(
                            f"{table}.user_disabled has an incompatible definition"
                        )
            if all(present):
                connection.rollback()
                return
            if any(present):
                raise RuntimeError("model user-disabled schema is partially installed")
            # A separate backup reader sees the committed pre-DDL state while
            # BEGIN IMMEDIATE prevents another writer from changing it.
            _backup_registry(
                path,
                workspace / "runtime" / "model-backups" / f"{_ID}-{uuid4().hex}",
            )
            for table in _TABLES:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {_COLUMN}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


steps = [step(add_user_disabled)]
