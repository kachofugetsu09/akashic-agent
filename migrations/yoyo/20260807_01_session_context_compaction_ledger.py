from __future__ import annotations

import os
import shutil
import sqlite3
import tomllib
from pathlib import Path
from typing import Any
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context


__depends__ = {"20260805_01_akasha_sparse_index_v9"}
__transactional__ = False

_MIGRATION_NAME = "session-context-compaction-ledger"


def _integrity_check(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        connection.close()
    if rows != [("ok",)]:
        raise RuntimeError(f"SQLite integrity_check 失败: {path}: {rows[:3]}")


def _backup_sqlite(source: Path, target: Path) -> None:
    """Create and verify a SQLite online backup."""

    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        source_connection = sqlite3.connect(source)
        target_connection = sqlite3.connect(candidate)
        try:
            source_connection.backup(target_connection)
            target_connection.commit()
        finally:
            target_connection.close()
            source_connection.close()
        _integrity_check(candidate)
        candidate.replace(target)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise


def _restore_file(path: Path, content: bytes | None, mode: int | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.restore")
    candidate.write_bytes(content)
    if mode is not None:
        candidate.chmod(mode)
    candidate.replace(path)


def _write_config(path: Path, document: Any, mode: int) -> None:
    rendered = tomlkit.dumps(document).encode("utf-8")
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    candidate.write_bytes(rendered)
    candidate.chmod(mode)
    with candidate.open("rb") as stream:
        os.fsync(stream.fileno())
    candidate.replace(path)


def _migrate_config(path: Path) -> None:
    """Move legacy window keys to agent.context.compaction."""

    if not path.is_file():
        return
    raw = path.read_text(encoding="utf-8")
    # Parse with stdlib first so malformed config remains a hard migration failure.
    tomllib.loads(raw)
    document = tomlkit.parse(raw)
    agent = document.setdefault("agent", tomlkit.table())
    if not isinstance(agent, dict):
        raise ValueError("agent 配置必须是 TOML table")
    context = agent.setdefault("context", tomlkit.table())
    if not isinstance(context, dict):
        raise ValueError("agent.context 配置必须是 TOML table")
    compaction = context.setdefault("compaction", tomlkit.table())
    if not isinstance(compaction, dict):
        raise ValueError("agent.context.compaction 配置必须是 TOML table")
    if "trigger_percent" not in compaction:
        compaction["trigger_percent"] = 0.74
    if "keep_recent_tokens" not in compaction:
        compaction["keep_recent_tokens"] = 20_000

    # Explicitly migrate removed keys; the runtime loader rejects them if they reappear.
    context.pop("memory_window", None)
    document.pop("memory_window", None)
    llm = document.get("llm")
    if isinstance(llm, dict):
        runtimes = llm.get("runtimes")
        if isinstance(runtimes, dict):
            for runtime in runtimes.values():
                if isinstance(runtime, dict):
                    runtime.pop("effective_context_percent", None)
                    runtime.pop("compaction_trigger_percent", None)
    _write_config(path, document, path.stat().st_mode & 0o777)


def _ensure_ledger_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS session_compactions (
            session_key TEXT NOT NULL,
            generation INTEGER NOT NULL,
            parent_generation INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            trigger TEXT NOT NULL,
            summary_format_version INTEGER NOT NULL,
            summary TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            source_from_seq INTEGER NOT NULL,
            consolidated_through_seq INTEGER NOT NULL,
            source_message_ids_json TEXT NOT NULL,
            retained_tail_json TEXT NOT NULL,
            model_runtime_id TEXT NOT NULL,
            model TEXT NOT NULL,
            context_window INTEGER NOT NULL,
            threshold_tokens INTEGER NOT NULL,
            hard_input_tokens INTEGER NOT NULL,
            keep_recent_tokens INTEGER NOT NULL,
            tokens_before INTEGER NOT NULL,
            tokens_after INTEGER NOT NULL,
            summary_usage_json TEXT NOT NULL,
            invalidated_at TEXT,
            invalidated_reason TEXT,
            PRIMARY KEY (session_key, generation),
            UNIQUE (session_key, source_ref)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_session_compactions_active
        ON session_compactions(session_key, invalidated_at, generation)
        """
    )
    connection.execute(
        "UPDATE sessions SET last_consolidated = 0 "
        "WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions')"
    )


def migrate_session_context_compaction_ledger(_connection: object) -> None:
    """Back up installation state and publish the session compaction schema."""

    current = current_migration_context()
    config = current.config_path
    sessions = current.workspace / "sessions.db"
    recent = current.workspace / "memory" / "RECENT_CONTEXT.md"
    backup_root = (
        current.workspace
        / "backups"
        / _MIGRATION_NAME
        / uuid4().hex
    )
    config_bytes = config.read_bytes() if config.exists() else None
    config_mode = config.stat().st_mode & 0o777 if config.exists() else None
    recent_bytes = recent.read_bytes() if recent.exists() else None
    recent_mode = recent.stat().st_mode & 0o777 if recent.exists() else None
    sessions_backup = backup_root / "sessions.db"
    if sessions.exists():
        _integrity_check(sessions)
        _backup_sqlite(sessions, sessions_backup)
    if config_bytes is not None:
        backup_root.mkdir(parents=True, exist_ok=True)
        (backup_root / config.name).write_bytes(config_bytes)
        if config_mode is not None:
            (backup_root / config.name).chmod(config_mode)
    if recent_bytes is not None:
        recent_backup = backup_root / "memory" / recent.name
        recent_backup.parent.mkdir(parents=True, exist_ok=True)
        recent_backup.write_bytes(recent_bytes)
        if recent_mode is not None:
            recent_backup.chmod(recent_mode)

    try:
        # 1. Migrate config and SessionDB before touching the retired projection.
        _migrate_config(config)
        if sessions.exists():
            connection = sqlite3.connect(sessions)
            try:
                _ensure_ledger_schema(connection)
                connection.commit()
            finally:
                connection.close()
            _integrity_check(sessions)

        # 2. Keep a verified archive but remove the retired workspace projection.
        if recent.exists():
            recent.unlink()
        if recent.exists() or recent.is_symlink():
            raise RuntimeError(f"RECENT_CONTEXT.md 删除失败: {recent}")
    except BaseException:
        # 3. Restore every touched persistent object; Yoyo will not record success.
        _restore_file(config, config_bytes, config_mode)
        if sessions_backup.exists():
            shutil.copy2(sessions_backup, sessions)
            _integrity_check(sessions)
        _restore_file(recent, recent_bytes, recent_mode)
        raise


steps = [step(migrate_session_context_compaction_ledger)]
