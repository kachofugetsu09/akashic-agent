"""只迁移已知旧渠道的 metadata 路由，运行时不再读取这些字段。"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tomllib
from typing import cast

from agent.migrations.session_db_backup import backup_sqlite_database
from session.identities import init_channel_identities, seed_channel_identities
from session.log import _session_schemas, _sql


def _rules(config_path: Path) -> dict[str, tuple[str, bool]]:
    """固定历史内置规则，并读取当时允许配置的 Telegram 渠道名。"""
    document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    channels = document.get("channels", {})
    if not isinstance(channels, dict):
        raise ValueError("channels 必须是 table")
    telegram = cast(dict[str, object], channels).get("telegram", {})
    if not isinstance(telegram, dict):
        raise ValueError("channels.telegram 必须是 table")
    name = cast(dict[str, object], telegram).get("channel_name", "telegram")
    if not isinstance(name, str) or not name or name != name.strip() or ":" in name:
        raise ValueError("Telegram 历史渠道名无效")
    rules = {"feishu": ("feishu_open_id", False), "qq": ("user_id", False), "telegram": ("username", True)}
    if name in {"qq", "feishu", "akashic"}:
        raise ValueError("Telegram 历史渠道名与其他渠道冲突")
    rules[name] = ("username", True)
    return rules


def _mappings(connection: sqlite3.Connection, rules: dict[str, tuple[str, bool]]) -> dict[str, dict[str, tuple[str, str]]]:
    """按旧排序固定同名 winner；已迁移渠道完全不再解释 metadata。"""
    plans: dict[str, dict[str, tuple[str, str]]] = {}
    for channel, (field, lower) in rules.items():
        if connection.execute(
            "SELECT 1 FROM channel_identity_migrations WHERE channel=?", (channel,),
        ).fetchone() is not None:
            continue
        if connection.execute("SELECT 1 FROM channel_identities WHERE channel=? LIMIT 1", (channel,)).fetchone() is not None:
            raise ValueError(f"身份映射缺少迁移标记: {channel}")
        mapping: dict[str, tuple[str, str]] = {}
        # 使用精确前缀；配置中的下划线不是 SQL LIKE 通配符。
        prefix = channel + ":"
        rows = connection.execute(
            "SELECT key, metadata, updated_at FROM sessions WHERE substr(key,1,?)=? ORDER BY updated_at ASC,key ASC",
            (len(prefix), prefix),
        )
        for key, raw, updated in rows:
            metadata = json.loads(raw if raw is not None else "{}")
            if not isinstance(metadata, dict):
                raise ValueError(f"session metadata 必须是 JSON object: {key}")
            value = cast(dict[str, object], metadata).get(field)
            if not isinstance(value, str):
                continue
            identity = value.strip()
            if lower:
                identity = identity.lower()
            if identity:
                mapping[identity] = (key[len(prefix):], updated)
        plans[channel] = mapping
    return plans


def migrate(path: Path, config_path: Path, backup_root: Path) -> None:
    """整库备份后在单个事务内迁移路由，原 Session 和消息逐字保留。"""
    if not path.exists():
        return
    rules = _rules(config_path)
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("SELECT sql FROM sqlite_master WHERE name='sessions'").fetchone()
        if row is None or _sql(row[0]) not in _session_schemas():
            raise ValueError("渠道身份迁移遇到未知 Session schema")
        # 1. 预检在回滚事务中完成，不留下部分 schema 或 marker。
        _ = connection.execute("BEGIN IMMEDIATE")
        try:
            init_channel_identities(connection)
            pending = _mappings(connection, rules)
        finally:
            connection.rollback()
        if not pending:
            return
    _ = backup_sqlite_database(path, backup_root, migration="channel-identities")
    with closing(sqlite3.connect(path)) as connection, connection:
        _ = connection.execute("PRAGMA foreign_keys=ON")
        _ = connection.execute("BEGIN IMMEDIATE")
        init_channel_identities(connection)
        sessions = connection.execute("SELECT * FROM sessions ORDER BY key").fetchall()
        messages = connection.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall()
        # 2. 所有渠道一起提交；已有路由和未知渠道不受影响。
        plans = _mappings(connection, rules)
        for channel, mapping in plans.items():
            seed_channel_identities(connection, channel, mapping)
        if connection.execute("SELECT * FROM sessions ORDER BY key").fetchall() != sessions:
            raise ValueError("渠道身份迁移改变了 Session")
        if connection.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall() != messages:
            raise ValueError("渠道身份迁移改变了消息")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)] or connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("渠道身份迁移完整性检查失败")
