"""Keep explicit model opt-outs separate from provider catalog availability."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = set()
__transactional__ = False

_ID = "20260930_01_model_user_disabled"
_TABLES = ("model_definitions", "embedding_models")
_COLUMN = "user_disabled INTEGER NOT NULL DEFAULT 0 CHECK (user_disabled IN (0, 1))"


# Frozen owner schema from b7d29fa5, independent of future ModelsStore changes.
# Only its already-approved additive expansions may be absent or appended.
_REGISTRY_SCHEMA = """
CREATE TABLE model_registry_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    default_embedding_model_id TEXT DEFAULT NULL,
    host_epoch INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE model_connections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    catalog_provider_id TEXT NOT NULL DEFAULT '',
    auth_id TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL DEFAULT '',
    auth_kind TEXT NOT NULL DEFAULT '',
    auth_payload TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    driver_config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE model_definitions (
    id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL REFERENCES model_connections(id) ON DELETE RESTRICT,
    model TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    reasoning_effort TEXT NOT NULL DEFAULT '',
    supported_reasoning_efforts TEXT NOT NULL DEFAULT '[]',
    context_window INTEGER NOT NULL DEFAULT 0 CHECK (context_window >= 0),
    max_output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (max_output_tokens >= 0),
    input_modalities TEXT NOT NULL DEFAULT '["text"]',
    capability_source TEXT NOT NULL DEFAULT 'unknown',
    context_window_source TEXT NOT NULL DEFAULT 'unknown',
    max_output_tokens_source TEXT NOT NULL DEFAULT 'unknown',
    input_modalities_source TEXT NOT NULL DEFAULT 'unknown',
    effective_context_percent REAL NOT NULL DEFAULT 0.9,
    compaction_trigger_percent REAL NOT NULL DEFAULT 0.74,
    use_responses_lite INTEGER NOT NULL DEFAULT 0 CHECK (use_responses_lite IN (0, 1)),
    supports_parallel_tool_calls INTEGER NOT NULL DEFAULT 1 CHECK (supports_parallel_tool_calls IN (0, 1)),
    reasoning_summary TEXT NOT NULL DEFAULT 'none',
    capabilities_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(connection_id, model)
);

CREATE TABLE embedding_models (
    id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL REFERENCES model_connections(id) ON DELETE RESTRICT,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    capabilities_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(connection_id, model)
);

CREATE TABLE model_role_bindings (
    role TEXT PRIMARY KEY CHECK (role IN ('default', 'fast', 'agent', 'vision')),
    model_id TEXT NOT NULL REFERENCES model_definitions(id) ON DELETE RESTRICT,
    reasoning_effort TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
_ADDITIVE_COLUMNS = {
    "model_registry_meta": {"default_embedding_model_id", "host_epoch"},
    "model_connections": {"driver_config_json"},
    "model_definitions": {"capabilities_json"},
    "embedding_models": {"capabilities_json"},
    "model_role_bindings": set(),
}


def _sql_tokens(sql: str) -> tuple[str, ...]:
    tokens = re.findall(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`|\[[^\]]*\]|[\w]+|[^\s]", sql
    )
    return tuple(
        token if token.startswith("'") else token.strip('"`[]').lower()
        for token in tokens
    )


def _checks(sql: str) -> set[tuple[str, ...]]:
    tokens = _sql_tokens(sql)
    checks = set()
    for index, token in enumerate(tokens):
        if token != "check" or tokens[index + 1] != "(":
            continue
        depth = 1
        end = index + 2
        while depth:
            depth += (tokens[end] == "(") - (tokens[end] == ")")
            end += 1
        checks.add(tokens[index + 2 : end - 1])
    return checks


def _columns(connection: sqlite3.Connection, table: str) -> dict[str, tuple]:
    # PRAGMA captures types, nullability, defaults, PK position and hidden state;
    # it is insensitive to declaration order, quoting and whitespace in the DDL.
    return {
        row[1].lower(): (
            row[2].upper(),
            row[3],
            _sql_tokens(row[4]) if row[4] is not None else None,
            row[5],
            row[6],
        )
        for row in connection.execute(f"PRAGMA table_xinfo({table})")
    }


def _constraints(connection: sqlite3.Connection, table: str) -> tuple:
    foreign_keys = sorted(
        tuple(row[2:])
        for row in connection.execute(f"PRAGMA foreign_key_list({table})")
    )
    indexes = sorted(
        (
            index[2],
            index[3],
            index[4],
            tuple(
                (row[2], row[3], row[4])
                for row in connection.execute(f"PRAGMA index_xinfo('{index[1]}')")
                if row[5]
            ),
        )
        for index in connection.execute(f"PRAGMA index_list({table})")
    )
    options = next(
        tuple(row[4:])
        for row in connection.execute("PRAGMA table_list")
        if row[1] == table
    )
    return foreign_keys, indexes, options


def _require_registry_lineage(connection: sqlite3.Connection) -> list[bool]:
    with closing(sqlite3.connect(":memory:")) as expected:
        expected.executescript(_REGISTRY_SCHEMA)
        present = []
        for table, optional in _ADDITIVE_COLUMNS.items():
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if row is None:
                raise RuntimeError("model registry schema lineage is incomplete")
            # PRAGMA does not expose every conflict, deferral or column-collation
            # policy. None of these clauses exists in the approved owner schema;
            # reject them instead of silently treating altered policies as equal.
            if {"conflict", "deferrable", "collate", "autoincrement"}.intersection(
                _sql_tokens(row[0])
            ):
                raise RuntimeError(f"{table} schema lineage has unsupported policies")
            actual_columns = _columns(connection, table)
            expected_columns = _columns(expected, table)
            expected_sql = expected.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]
            expected_checks = _checks(expected_sql)
            for name in optional:
                if name not in actual_columns:
                    expected_columns.pop(name)
            # New registries start at epoch 1; the historical additive upgrade
            # used default 0. Both are explicit supported owner lineages.
            if actual_columns.get("host_epoch") == ("INTEGER", 1, ("0",), 0, 0):
                expected_columns["host_epoch"] = actual_columns["host_epoch"]
            if table in _TABLES:
                present.append("user_disabled" in actual_columns)
                if "user_disabled" in actual_columns:
                    expected_columns["user_disabled"] = ("INTEGER", 1, ("0",), 0, 0)
                    expected_checks.update(_checks(_COLUMN))
            if (
                actual_columns != expected_columns
                or _constraints(connection, table) != _constraints(expected, table)
                or _checks(row[0]) != expected_checks
                or connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE tbl_name=? AND type='trigger'",
                    (table,),
                ).fetchone()
            ):
                raise RuntimeError(f"{table} schema lineage is incompatible")
        return present


def _backup_registry(path: Path, root: Path) -> None:
    """Keep this immutable plugin migration independent of private Core helpers."""

    workspace = path.parent
    # Refuse every pre-existing symlink before creating any backup directories,
    # including links that point back inside the workspace or are dangling.
    relative = root.relative_to(workspace)
    current = workspace
    for component in (None, *relative.parts):
        if component is not None:
            current = current / component
        if current.is_symlink():
            raise ValueError("model registry backup refuses symbolic-link ancestors")
        if current.exists() and not current.is_dir():
            raise ValueError("model registry backup ancestor is not a directory")
    if not root.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("model registry backup escapes the workspace")

    # No-follow directory creation complements the complete preflight check.
    # The deployment runner's offline workspace lock serializes approved writers;
    # this does not defend against a hostile same-user process renaming parents.
    with ExitStack() as stack:
        descriptor = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, descriptor)
        directories = [descriptor]
        for index, name in enumerate(relative.parts):
            try:
                os.mkdir(name, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                if index == len(relative.parts) - 1:
                    raise
            descriptor = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            stack.callback(os.close, descriptor)
            directories.append(descriptor)
        root_fd = descriptor
        backup = root / path.name
        descriptor = os.open(
            path.name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        os.close(descriptor)
        with closing(sqlite3.connect(path)) as source:
            with closing(sqlite3.connect(backup)) as target:
                source.backup(target)
                if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise RuntimeError("model registry backup failed integrity check")
        with backup.open("rb") as stream:
            os.fsync(stream.fileno())
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {
            "schema_version": 1,
            "migration": _ID,
            "source": str(path),
            "backup": path.name,
            "sha256": digest,
            "sqlite_integrity": "ok",
        }
        descriptor = os.open(
            "manifest.json",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for descriptor in reversed(directories):
            os.fsync(descriptor)


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
            present = _require_registry_lineage(connection)
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
