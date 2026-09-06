from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


_SCHEMA = {
    "channel_identities": """CREATE TABLE IF NOT EXISTS channel_identities (
        channel TEXT NOT NULL, identity TEXT NOT NULL, chat_id TEXT NOT NULL,
        updated_at TEXT NOT NULL, PRIMARY KEY(channel, identity)
    )""",
    "channel_identity_migrations": """CREATE TABLE IF NOT EXISTS channel_identity_migrations (
        channel TEXT PRIMARY KEY, migrated_at TEXT NOT NULL
    )""",
}


def init_channel_identities(connection: sqlite3.Connection) -> None:
    """核对完整的已知 schema 后初始化空表，不改写已有身份。"""
    def sql(value: str) -> str:
        return re.sub(r"\s+", "", value.lower().replace("if not exists", "")).rstrip(";")

    # 1. 部分 schema 或未知结构必须先报告，不能自动修复。
    existing: set[str] = set()
    for name, statement in _SCHEMA.items():
        row = connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
        if row is not None:
            if sql(row[0]) != sql(statement):
                raise RuntimeError(f"{name} schema 不匹配")
            existing.add(name)
    if existing and existing != set(_SCHEMA):
        raise RuntimeError("channel identities schema 不完整")
    # 2. 由调用方的同一事务提交 DDL。
    for statement in _SCHEMA.values():
        _ = connection.execute(statement)


def delete_session_identities(connection: sqlite3.Connection, keys: Sequence[str]) -> None:
    """由旧 Session 删除 owner 在已备份的审计事务内减少对应路由。"""
    placeholders = ",".join("?" for _ in keys)
    _ = connection.execute(
        f"DELETE FROM channel_identities WHERE (channel || ':' || chat_id) IN ({placeholders})",
        keys,
    )


def seed_channel_identities(
    connection: sqlite3.Connection, channel: str, mapping: Mapping[str, tuple[str, str]],
) -> None:
    """在 owner 或显式迁移事务内一次发布路由与永久标记。"""
    if connection.execute(
        "SELECT 1 FROM channel_identity_migrations WHERE channel=?", (channel,),
    ).fetchone() is not None:
        return
    if connection.execute(
        "SELECT 1 FROM channel_identities WHERE channel=? LIMIT 1", (channel,),
    ).fetchone() is not None:
        raise ValueError(f"身份映射缺少迁移标记: {channel}")
    _ = connection.executemany(
        "INSERT INTO channel_identities(channel, identity, chat_id, updated_at) VALUES (?, ?, ?, ?)",
        ((channel, identity, chat_id, updated) for identity, (chat_id, updated) in mapping.items()),
    )
    _ = connection.execute(
        "INSERT INTO channel_identity_migrations(channel, migrated_at) VALUES (?, ?)",
        (channel, datetime.now(UTC).isoformat()),
    )


@dataclass(frozen=True)
class ChannelIdentityWriteReceipt:
    """保存精确提交版本与之前的路由，供失败接纳撤销。"""

    channel: str
    identity: str
    chat_id: str
    updated_at: str
    previous: tuple[str, str] | None


class ChannelIdentities:
    """独占渠道身份路由与迁移标记，不创建或改写 Session。"""

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            with self._conn:
                _ = self._conn.execute("BEGIN IMMEDIATE")
                init_channel_identities(self._conn)
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def load(self, channel: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT identity, chat_id FROM channel_identities WHERE channel=? ORDER BY identity",
                (channel,),
            ).fetchall()
        return {row["identity"]: row["chat_id"] for row in rows}

    def resolve(self, channel: str, identity: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT chat_id FROM channel_identities WHERE channel=? AND identity=?",
                (channel, identity),
            ).fetchone()
        return None if row is None else row["chat_id"]

    def migration_completed(self, channel: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM channel_identity_migrations WHERE channel=?", (channel,),
            ).fetchone()
        return row is not None

    def seed(self, channel: str, mapping: Mapping[str, tuple[str, str]]) -> None:
        """一次性接收来源已裁决的历史映射，保留永久迁移标记。"""
        with self._lock, self._conn:
            _ = self._conn.execute("BEGIN IMMEDIATE")
            seed_channel_identities(self._conn, channel, mapping)

    def remember(self, channel: str, identity: str, chat_id: str) -> ChannelIdentityWriteReceipt:
        """提交唯一 recipient；版本用于失败接纳的条件回滚。"""
        with self._lock, self._conn:
            _ = self._conn.execute("BEGIN IMMEDIATE")
            # 1. 读取原路由并选取不同的提交版本。
            row = self._conn.execute(
                "SELECT chat_id, updated_at FROM channel_identities WHERE channel=? AND identity=?",
                (channel, identity),
            ).fetchone()
            previous = None if row is None else (str(row["chat_id"]), str(row["updated_at"]))
            updated = datetime.now(UTC).isoformat()
            if previous is not None and updated == previous[1]:
                updated = (datetime.fromisoformat(updated) + timedelta(microseconds=1)).isoformat()
            # 2. 路由与标记一起发布，旧 metadata 从此不再裁决。
            _ = self._conn.execute(
                """INSERT INTO channel_identities(channel, identity, chat_id, updated_at)
                VALUES (?, ?, ?, ?) ON CONFLICT(channel, identity) DO UPDATE
                SET chat_id=excluded.chat_id, updated_at=excluded.updated_at""",
                (channel, identity, chat_id, updated),
            )
            _ = self._conn.execute(
                "INSERT INTO channel_identity_migrations(channel, migrated_at) VALUES (?, ?) ON CONFLICT DO NOTHING",
                (channel, updated),
            )
        return ChannelIdentityWriteReceipt(channel, identity, chat_id, updated, previous)

    def rollback(self, receipt: ChannelIdentityWriteReceipt) -> bool:
        """只撤销尚未被其他接纳替换的路由，不撤销迁移标记。"""
        with self._lock, self._conn:
            _ = self._conn.execute("BEGIN IMMEDIATE")
            match = (receipt.channel, receipt.identity, receipt.chat_id, receipt.updated_at)
            if receipt.previous is None:
                result = self._conn.execute(
                    "DELETE FROM channel_identities WHERE channel=? AND identity=? AND chat_id=? AND updated_at=?",
                    match,
                )
            else:
                result = self._conn.execute(
                    """UPDATE channel_identities SET chat_id=?, updated_at=?
                    WHERE channel=? AND identity=? AND chat_id=? AND updated_at=?""",
                    (*receipt.previous, *match),
                )
        return result.rowcount == 1
