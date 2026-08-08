from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import cast
from uuid import uuid4


def _read_table_sql(connection: sqlite3.Connection, table: str) -> str:
    """读取表定义，并拒绝缺失的 schema owner。"""

    # 1. 从 SQLite 元数据读取规范 CREATE TABLE 定义。
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if table_row is None or not table_row[0]:
        raise RuntimeError(f"{table} schema lineage 不兼容，表定义缺失")

    # 2. 保留后续 schema 校验使用的原始文本。
    return str(table_row[0])


def _validate_table_columns(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[tuple[str, str, int, int], ...],
) -> None:
    """校验 SQLite 列顺序、类型、非空属性和主键序号。"""

    # 1. 读取 SQLite 按声明顺序返回的列元数据。
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )

    # 2. 将本迁移持有的 manifest 规范化为相同的比较形状。
    expected_columns = tuple(
        (name, column_type.upper(), not_null, pk_ordinal)
        for name, column_type, not_null, pk_ordinal in columns
    )
    if actual_columns != expected_columns:
        raise RuntimeError(f"{table} schema lineage 不兼容，列定义不匹配")


def _validate_sql_fragments(
    table: str,
    table_sql: str,
    fragments: tuple[str, ...],
) -> None:
    """校验表定义包含本迁移持有的全部 SQL 约束片段。"""

    # 1. 规范化 SQL 空白和大小写，不改变片段语义。
    normalized_sql = "".join(table_sql.upper().split())

    # 2. 在首个缺失约束处失败，保留迁移诊断信息。
    for fragment in fragments:
        if "".join(fragment.upper().split()) not in normalized_sql:
            raise RuntimeError(f"{table} schema lineage 不兼容，约束定义缺失")


def _read_table_indexes(
    connection: sqlite3.Connection,
    table: str,
) -> list[tuple[object, ...]]:
    """读取一张表的 SQLite 索引元数据。"""

    # 1. 让 SQLite 返回包含自动索引在内的完整列表。
    return connection.execute(f"PRAGMA index_list({table})").fetchall()


def _validate_named_indexes(
    connection: sqlite3.Connection,
    table: str,
    index_rows: list[tuple[object, ...]],
    named_indexes: dict[str, tuple[tuple[str, ...], int]],
) -> None:
    """校验显式命名索引的集合、唯一性和列顺序。"""

    # 1. 从命名索引合同中排除 SQLite 自动生成的索引。
    named_rows = {
        str(row[1]): (int(cast(int, row[2])), str(row[3]))
        for row in index_rows
        if not str(row[1]).startswith("sqlite_autoindex_")
    }
    if set(named_rows) != set(named_indexes):
        raise RuntimeError(f"{table} schema lineage 不兼容，索引集合不匹配")

    # 2. 校验每个本迁移索引的唯一性、来源和列顺序。
    for name, (expected_columns, expected_unique) in named_indexes.items():
        unique, origin = named_rows[name]
        if unique != expected_unique or origin != "c":
            raise RuntimeError(f"{table} schema lineage 不兼容，索引定义不匹配: {name}")
        actual_columns = tuple(
            str(row[2])
            for row in connection.execute(f"PRAGMA index_info({name!r})")
        )
        if actual_columns != expected_columns:
            raise RuntimeError(f"{table} schema lineage 不兼容，索引列不匹配: {name}")


def _validate_auto_indexes(
    connection: sqlite3.Connection,
    table: str,
    index_rows: list[tuple[object, ...]],
    auto_indexes: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    """校验 SQLite 为主键和唯一约束生成的索引。"""

    # 1. 收集 SQLite 自动索引的来源和列顺序。
    actual_auto_indexes = []
    for row in index_rows:
        name = str(row[1])
        if not name.startswith("sqlite_autoindex_"):
            continue
        actual_auto_indexes.append(
            (
                str(row[3]),
                tuple(
                    str(index_row[2])
                    for index_row in connection.execute(f"PRAGMA index_info({name!r})")
                ),
            )
        )

    # 2. 比较生成的 schema identity，不依赖 SQLite 返回顺序。
    if sorted(actual_auto_indexes) != sorted(auto_indexes):
        raise RuntimeError(f"{table} schema lineage 不兼容，约束索引不匹配")


def validate_table_schema(
    connection: sqlite3.Connection,
    *,
    table: str,
    columns: tuple[tuple[str, str, int, int], ...],
    named_indexes: dict[str, tuple[tuple[str, ...], int]],
    auto_indexes: tuple[tuple[str, tuple[str, ...]], ...],
    sql_fragments: tuple[str, ...] = (),
    validate_named_indexes: bool = True,
) -> None:
    """除非 SQLite 表符合本迁移的 schema identity，否则明确失败。"""

    # 1. 校验表定义、列顺序和内联约束。
    table_sql = _read_table_sql(connection, table)
    _validate_table_columns(connection, table, columns)
    _validate_sql_fragments(table, table_sql, sql_fragments)

    # 2. 在本阶段负责创建显式索引时校验它们。
    index_rows = _read_table_indexes(connection, table)
    if validate_named_indexes:
        _validate_named_indexes(connection, table, index_rows, named_indexes)

    # 3. 始终校验 SQLite 为约束生成的自动索引。
    _validate_auto_indexes(connection, table, index_rows, auto_indexes)


def _integrity_check(path: Path) -> None:
    """Verify a SQLite file before publishing it as recovery evidence."""

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        connection.close()
    if rows != [("ok",)]:
        raise RuntimeError(f"SQLite integrity_check 失败: {path}: {rows[:3]}")


def _fsync_directory(path: Path) -> None:
    """Durably publish one renamed backup and its manifest."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    manifest_candidate: Path | None = None
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
        candidate.chmod(0o600)
        with candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        _integrity_check(candidate)
        candidate.replace(backup)
        backup.chmod(0o600)

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
        _fsync_directory(backup_root)
    except BaseException:
        candidate.unlink(missing_ok=True)
        if manifest_candidate is not None:
            manifest_candidate.unlink(missing_ok=True)
        raise
    return backup
