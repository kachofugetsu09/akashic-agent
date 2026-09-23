"""Add the immutable runtime input reference to the existing update journal."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = {}

_OLD_TABLE = """CREATE TABLE plugin_updates (
        update_id TEXT PRIMARY KEY,
        plugin_id TEXT NOT NULL,
        plugin_base TEXT NOT NULL,
        previous_pointers_json TEXT,
        candidate_pointer TEXT NOT NULL,
        previous_enabled INTEGER CHECK (previous_enabled IN (0, 1)),
        phase TEXT NOT NULL CHECK (phase IN ('armed', 'committed', 'rolled_back')),
        reload_tx_id TEXT UNIQUE REFERENCES reload_transactions(tx_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT NOT NULL
    )"""
_NEW_TABLE = """CREATE TABLE plugin_updates (
        update_id TEXT PRIMARY KEY,
        plugin_id TEXT NOT NULL,
        plugin_base TEXT NOT NULL,
        previous_pointers_json TEXT,
        candidate_pointer TEXT NOT NULL,
        previous_enabled INTEGER CHECK (previous_enabled IN (0, 1)),
        phase TEXT NOT NULL CHECK (phase IN ('armed', 'committed', 'rolled_back')),
        reload_tx_id TEXT UNIQUE REFERENCES reload_transactions(tx_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT NOT NULL,
        input_ref TEXT
    )"""
# SQLite's ALTER TABLE path preserves a different comma/closing-paren shape
# from the fresh CREATE TABLE text.  Keep this fixed identity explicit.
_ALTERED_TABLE = """CREATE TABLE plugin_updates (
        update_id TEXT PRIMARY KEY,
        plugin_id TEXT NOT NULL,
        plugin_base TEXT NOT NULL,
        previous_pointers_json TEXT,
        candidate_pointer TEXT NOT NULL,
        previous_enabled INTEGER CHECK (previous_enabled IN (0, 1)),
        phase TEXT NOT NULL CHECK (phase IN ('armed', 'committed', 'rolled_back')),
        reload_tx_id TEXT UNIQUE REFERENCES reload_transactions(tx_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT NOT NULL ,
        input_ref TEXT)"""
_INDEX = """CREATE UNIQUE INDEX plugin_update_active
        ON plugin_updates(plugin_id) WHERE phase='armed'"""


def _normalise(sql: str) -> str:
    """Accept only insignificant DDL whitespace differences."""
    return " ".join(sql.strip().split())


def _sql(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE (type='table' OR type='index')"
        " AND name=?", (name,),
    ).fetchone()
    return None if row is None else str(row[0])


def _check_shape(conn: sqlite3.Connection) -> str:
    """Classify the exact old/new table and partial schemas."""
    table_sql = _sql(conn, "plugin_updates")
    index_sql = _sql(conn, "plugin_update_active")
    if table_sql is None and index_sql is None:
        return "missing"
    if table_sql is not None and index_sql is not None:
        if _normalise(table_sql) == _normalise(_OLD_TABLE) and _normalise(index_sql) == _normalise(_INDEX):
            return "old"
        if _normalise(table_sql) in {_normalise(_NEW_TABLE), _normalise(_ALTERED_TABLE)} and _normalise(index_sql) == _normalise(_INDEX):
            return "new"
    raise RuntimeError("plugin_updates schema 不完整或未知，拒绝自动迁移")


def _database_path() -> Path:
    return current_migration_context().workspace / "runtime" / "plugin-reloads.sqlite3"


def _backup_path(path: Path) -> Path:
    return path.with_name(
        f"{path.name}.before-input-ref.{uuid.uuid4().hex}.bak"
    )


def _copy_database(source_path: Path, target_path: Path) -> None:
    """Create a new immutable SQLite recovery point and verify it."""
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
        result = target.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise RuntimeError(f"SQLite backup integrity check failed: {result}")
    finally:
        target.close()
        source.close()


def _upgrade(connection: sqlite3.Connection) -> None:
    """Apply one additive column using the Yoyo callback connection contract."""
    _ = connection  # The ledger DB is not the plugin-reloads DB being migrated.
    path = _database_path()
    if not path.exists():
        # Fresh workspaces receive the current schema from ReloadJournal.
        return
    probe = sqlite3.connect(path)
    try:
        shape = _check_shape(probe)
    finally:
        probe.close()
    if shape == "new":
        return
    if shape == "missing":
        raise RuntimeError(
            "已有 plugin-reloads.sqlite3 缺少 plugin_updates，拒绝伪造历史表"
        )

    backup = _backup_path(path)
    _copy_database(path, backup)
    conn = sqlite3.connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ALTER TABLE plugin_updates ADD COLUMN input_ref TEXT")
        if _check_shape(conn) != "new":
            raise RuntimeError("ALTER 后 plugin_updates schema 不是已知新形状")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"SQLite migration integrity check failed: {integrity}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


# This migration is intentionally irreversible. A failed apply retains its
# named backup; restore is a separately authorized operator action, not Yoyo
# downgrade behavior.
steps = [step(_upgrade, None)]
