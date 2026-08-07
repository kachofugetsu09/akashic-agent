from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tomllib
from pathlib import Path
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context


__depends__ = {"20260805_01_akasha_sparse_index_v9"}
__transactional__ = False

_MIGRATION_NAME = "session-context-compaction-ledger"
_LEDGER_COLUMNS = {
    "session_key",
    "generation",
    "parent_generation",
    "created_at",
    "trigger",
    "summary_format_version",
    "summary",
    "source_ref",
    "source_from_seq",
    "consolidated_through_seq",
    "source_message_ids_json",
    "retained_tail_json",
    "model_runtime_id",
    "model",
    "context_window",
    "threshold_tokens",
    "hard_input_tokens",
    "keep_recent_tokens",
    "tokens_before",
    "tokens_after",
    "summary_usage_json",
    "invalidated_at",
    "invalidated_reason",
}


class _PathSnapshot:
    """Capture one migration input without following the path during restore."""

    def __init__(
        self,
        path: Path,
        existed: bool,
        kind: str,
        mode: int | None,
        symlink_target: str | None,
        resolved_target: Path | None,
        target_mode: int | None,
        content: bytes | None,
    ) -> None:
        self.path = path
        self.existed = existed
        self.kind = kind
        self.mode = mode
        self.symlink_target = symlink_target
        self.resolved_target = resolved_target
        self.target_mode = target_mode
        self.content = content


class _BackupRecord:
    """Describe a readable backup artifact and the source path identity."""

    def __init__(
        self,
        snapshot: _PathSnapshot,
        backup_path: Path | None,
        raw_backup_path: Path | None = None,
        sqlite_backup_path: Path | None = None,
        content_sha256: str | None = None,
        sqlite_sha256: str | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.backup_path = backup_path
        self.raw_backup_path = raw_backup_path
        self.sqlite_backup_path = sqlite_backup_path
        self.content_sha256 = content_sha256
        self.sqlite_sha256 = sqlite_sha256

    def as_dict(self, root: Path) -> dict[str, object]:
        snapshot = self.snapshot
        payload: dict[str, object] = {
            "existed": snapshot.existed,
            "kind": snapshot.kind,
            "mode": snapshot.mode,
            "symlink_target": snapshot.symlink_target,
            "content_sha256": self.content_sha256,
        }
        if snapshot.resolved_target is not None:
            payload["resolved_target"] = str(snapshot.resolved_target)
        if snapshot.target_mode is not None:
            payload["target_mode"] = snapshot.target_mode
        if self.backup_path is not None:
            payload["backup"] = str(self.backup_path.relative_to(root))
        if self.raw_backup_path is not None:
            payload["raw_backup"] = str(self.raw_backup_path.relative_to(root))
        if self.sqlite_backup_path is not None:
            payload["sqlite_backup"] = str(
                self.sqlite_backup_path.relative_to(root)
            )
            payload["sqlite_integrity"] = "ok"
            payload["sqlite_sha256"] = self.sqlite_sha256
        return payload


def _path_exists(path: Path) -> bool:
    """Return whether a path exists, including a dangling symbolic link."""

    return path.exists() or path.is_symlink()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _integrity_check(path: Path) -> None:
    """Verify a SQLite file before it can become a migration source or backup."""

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        connection.close()
    if rows != [("ok",)]:
        raise RuntimeError(f"SQLite integrity_check 失败: {path}: {rows[:3]}")


def _write_bytes(path: Path, payload: bytes, mode: int | None = None) -> None:
    """Write a staged or restored regular file with fsync and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        candidate.write_bytes(payload)
        if mode is not None:
            candidate.chmod(mode)
        with candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        candidate.replace(path)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise


def _backup_sqlite(source: Path, target: Path) -> None:
    """Create and verify a SQLite online backup at a new target path."""

    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
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
        candidate.replace(target)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise


def _backup_sqlite_in_place(source: Path, target: Path) -> None:
    """Copy a verified SQLite snapshot into an existing database inode."""

    source_connection = sqlite3.connect(source)
    try:
        target_connection = sqlite3.connect(target)
        try:
            source_connection.backup(target_connection)
            target_connection.commit()
        finally:
            target_connection.close()
    finally:
        source_connection.close()
    _integrity_check(target)


def _snapshot_path(path: Path, *, capture_content: bool = True) -> _PathSnapshot:
    """Capture file kind, link identity, mode, and optionally readable content."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _PathSnapshot(
            path=path,
            existed=False,
            kind="missing",
            mode=None,
            symlink_target=None,
            resolved_target=None,
            target_mode=None,
            content=None,
        )
    except OSError as exc:
        raise RuntimeError(f"无法读取迁移源元数据: {path}") from exc

    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            raise RuntimeError(f"迁移源软链接无法解析: {path}") from exc
        if path.is_dir() or not path.is_file():
            raise RuntimeError(f"迁移源不是可读文件: {path}")
        target_mode: int | None = None
        if path.is_file():
            target_mode = stat.S_IMODE(resolved.stat().st_mode)
        content = path.read_bytes() if capture_content and path.is_file() else None
        return _PathSnapshot(
            path=path,
            existed=True,
            kind="symlink",
            mode=mode,
            symlink_target=target,
            resolved_target=resolved,
            target_mode=target_mode,
            content=content,
        )

    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"迁移源必须是普通文件或软链接: {path}")
    content = path.read_bytes() if capture_content else None
    return _PathSnapshot(
        path=path,
        existed=True,
        kind="file",
        mode=mode,
        symlink_target=None,
        resolved_target=path,
        target_mode=mode,
        content=content,
    )


def _archive_file(
    snapshot: _PathSnapshot,
    backup_path: Path,
    *,
    mode: int | None = None,
) -> str | None:
    """Archive source bytes as a readable regular file and verify its digest."""

    if not snapshot.existed:
        return None
    if snapshot.content is None:
        raise RuntimeError(f"迁移源内容不可读: {snapshot.path}")
    archive_mode = mode if mode is not None else snapshot.target_mode
    _write_bytes(backup_path, snapshot.content, mode=archive_mode)
    digest = _sha256_bytes(backup_path.read_bytes())
    expected = _sha256_bytes(snapshot.content)
    if digest != expected:
        raise RuntimeError(f"备份内容校验失败: {snapshot.path}")
    return digest


def _render_config(raw: bytes) -> bytes:
    """Move removed context keys into the new compaction policy table."""

    text = raw.decode("utf-8")
    # 1. Parse with stdlib first so malformed config remains a hard failure.
    tomllib.loads(text)
    document = tomlkit.parse(text)
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

    # 2. Remove every retired location checked by the runtime boundary.
    document.pop("memory_window", None)
    context.pop("memory_window", None)
    compaction.pop("memory_window", None)
    llm = document.get("llm")
    if isinstance(llm, dict):
        for key in ("effective_context_percent", "compaction_trigger_percent"):
            llm.pop(key, None)
        main = llm.get("main")
        if isinstance(main, dict):
            for key in ("effective_context_percent", "compaction_trigger_percent"):
                main.pop(key, None)
        runtimes = llm.get("runtimes")
        if isinstance(runtimes, dict):
            for runtime in runtimes.values():
                if isinstance(runtime, dict):
                    runtime.pop("effective_context_percent", None)
                    runtime.pop("compaction_trigger_percent", None)
    return tomlkit.dumps(document).encode("utf-8")


def _migrate_config(path: Path) -> bytes:
    """Render a migrated config without mutating the source path."""

    return _render_config(path.read_bytes())


def _ensure_ledger_schema(connection: sqlite3.Connection) -> None:
    """Create the immutable compaction ledger and reset legacy cursors."""

    existing = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_compactions'"
    ).fetchone()
    if existing is not None:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(session_compactions)")
        }
        missing = sorted(_LEDGER_COLUMNS - columns)
        if missing:
            raise RuntimeError(
                "session_compactions schema lineage 不兼容，缺少列: "
                + ", ".join(missing)
            )
    else:
        connection.execute(
            """
            CREATE TABLE session_compactions (
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
    sessions_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
    ).fetchone()
    if sessions_exists is None:
        return
    session_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)")
    }
    if "last_consolidated" not in session_columns:
        raise RuntimeError("sessions schema lineage 不兼容，缺少列: last_consolidated")
    connection.execute("UPDATE sessions SET last_consolidated = 0")


def _write_manifest(root: Path, records: dict[str, _BackupRecord]) -> None:
    payload = {
        "schema_version": 1,
        "migration": _MIGRATION_NAME,
        "sources": {
            name: record.as_dict(root) for name, record in records.items()
        },
    }
    rendered = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_bytes(root / "manifest.json", rendered, mode=0o600)
    if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"迁移备份 manifest 校验失败: {root / 'manifest.json'}")


def _remove_current_path(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_dir() and not path.is_symlink():
        raise RuntimeError(f"无法原子恢复目录路径: {path}")
    path.unlink()


def _restore_snapshot(
    record: _BackupRecord,
    *,
    content_path: Path | None,
) -> None:
    """Restore bytes and path identity, including symbolic-link metadata."""

    snapshot = record.snapshot
    path = snapshot.path
    if not snapshot.existed:
        _remove_current_path(path)
        return

    if content_path is None or not content_path.is_file():
        raise RuntimeError(f"恢复备份缺失: {path}")
    content = content_path.read_bytes()
    if record.content_sha256 != _sha256_bytes(content):
        raise RuntimeError(f"恢复备份摘要不匹配: {path}")

    if snapshot.kind == "symlink":
        if snapshot.resolved_target is None or snapshot.symlink_target is None:
            raise RuntimeError(f"软链接快照不完整: {path}")
        target = snapshot.resolved_target
        _write_bytes(target, content, mode=snapshot.target_mode)
        _remove_current_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(snapshot.symlink_target, path)
        if os.readlink(path) != snapshot.symlink_target or path.read_bytes() != content:
            raise RuntimeError(f"软链接恢复校验失败: {path}")
        return

    _remove_current_path(path)
    _write_bytes(path, content, mode=snapshot.mode)
    if path.read_bytes() != content:
        raise RuntimeError(f"文件恢复校验失败: {path}")


def _restore_sqlite_snapshot(record: _BackupRecord) -> None:
    """Restore a database from the verified online backup, including WAL state."""

    snapshot = record.snapshot
    path = snapshot.path
    if not snapshot.existed:
        _remove_current_path(path)
        return
    backup = record.sqlite_backup_path
    if backup is None or not backup.is_file():
        raise RuntimeError(f"SQLite 恢复备份缺失: {path}")
    _integrity_check(backup)
    if record.sqlite_sha256 != _sha256_bytes(backup.read_bytes()):
        raise RuntimeError(f"SQLite 恢复备份摘要不匹配: {path}")

    target = snapshot.resolved_target if snapshot.kind == "symlink" else path
    if target is None:
        raise RuntimeError(f"SQLite 恢复目标路径缺失: {path}")
    _backup_sqlite_in_place(backup, target)

    if snapshot.kind == "symlink":
        if snapshot.symlink_target is None:
            raise RuntimeError(f"SQLite 软链接快照不完整: {path}")
        if not path.is_symlink() or os.readlink(path) != snapshot.symlink_target:
            _remove_current_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(snapshot.symlink_target, path)
    elif path.is_symlink():
        raise RuntimeError(f"SQLite 文件类型恢复失败: {path}")
    _integrity_check(path)


def _publish_staged(snapshot: _PathSnapshot, staged: Path) -> None:
    """Publish one staged file while preserving an original symbolic link."""

    if not snapshot.existed:
        return
    target = snapshot.resolved_target if snapshot.kind == "symlink" else snapshot.path
    if target is None:
        raise RuntimeError(f"迁移目标路径缺失: {snapshot.path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, target)


def _publish_staged_sqlite(snapshot: _PathSnapshot, staged: Path) -> None:
    """Publish a staged SQLite database without replacing its live inode."""

    if not snapshot.existed:
        return
    target = snapshot.resolved_target if snapshot.kind == "symlink" else snapshot.path
    if target is None:
        raise RuntimeError(f"SQLite 迁移目标路径缺失: {snapshot.path}")
    _backup_sqlite_in_place(staged, target)
    staged.unlink()


def migrate_session_context_compaction_ledger(_connection: object) -> None:
    """Back up installation state, stage the new schema, and publish it atomically."""

    current = current_migration_context()
    config = current.config_path
    sessions = current.workspace / "sessions.db"
    recent = current.workspace / "memory" / "RECENT_CONTEXT.md"
    backup_root = (
        current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex
    )
    staging_root = backup_root / "staging"
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(backup_root, 0o700)
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)

    # 1. Snapshot and verify every source before any live path changes.
    config_snapshot = _snapshot_path(config)
    sessions_snapshot = _snapshot_path(sessions, capture_content=False)
    recent_snapshot = _snapshot_path(recent)
    records: dict[str, _BackupRecord] = {}
    config_backup = backup_root / config.name
    config_digest = _archive_file(config_snapshot, config_backup)
    records["config"] = _BackupRecord(
        config_snapshot,
        config_backup if config_snapshot.existed else None,
        content_sha256=config_digest,
    )
    recent_backup = backup_root / "memory" / recent.name
    recent_digest = _archive_file(recent_snapshot, recent_backup)
    records["recent_context"] = _BackupRecord(
        recent_snapshot,
        recent_backup if recent_snapshot.existed else None,
        content_sha256=recent_digest,
    )

    sessions_backup = backup_root / "sessions.db"
    sessions_raw_backup = backup_root / "sessions.db.raw"
    sessions_sqlite_path: Path | None = None
    sessions_raw_path: Path | None = None
    sessions_digest: str | None = None
    sessions_sqlite_digest: str | None = None
    if sessions_snapshot.existed:
        _integrity_check(sessions)
        _backup_sqlite(sessions, sessions_backup)
        raw = sessions.read_bytes()
        _write_bytes(sessions_raw_backup, raw, mode=sessions_snapshot.target_mode)
        if sessions_raw_backup.read_bytes() != raw:
            raise RuntimeError(f"sessions.db 原始备份校验失败: {sessions}")
        sessions_sqlite_path = sessions_backup
        sessions_raw_path = sessions_raw_backup
        sessions_digest = _sha256_bytes(raw)
        sessions_sqlite_digest = _sha256_bytes(sessions_backup.read_bytes())
    records["sessions"] = _BackupRecord(
        sessions_snapshot,
        sessions_backup if sessions_snapshot.existed else None,
        raw_backup_path=sessions_raw_path,
        sqlite_backup_path=sessions_sqlite_path,
        content_sha256=sessions_digest,
        sqlite_sha256=sessions_sqlite_digest,
    )

    # 2. Prepare all replacement bytes under one unique staging directory.
    staged_config: Path | None = None
    if config_snapshot.existed:
        staged_config = staging_root / "config.toml"
        _write_bytes(
            staged_config,
            _migrate_config(config),
            mode=config_snapshot.target_mode,
        )
    staged_sessions: Path | None = None
    if sessions_snapshot.existed:
        staged_sessions = staging_root / "sessions.db"
        _backup_sqlite(sessions, staged_sessions)
        connection = sqlite3.connect(staged_sessions)
        try:
            _ensure_ledger_schema(connection)
            connection.commit()
        finally:
            connection.close()
        _integrity_check(staged_sessions)
    _write_manifest(backup_root, records)

    try:
        # 3. Publish staged config and SessionDB, then remove the retired projection.
        if staged_config is not None:
            _publish_staged(config_snapshot, staged_config)
        if staged_sessions is not None:
            _publish_staged_sqlite(sessions_snapshot, staged_sessions)
        if recent_snapshot.existed:
            recent.unlink()
            if _path_exists(recent):
                raise RuntimeError(f"RECENT_CONTEXT.md 删除失败: {recent}")
        staging_root.rmdir()
    except BaseException as migration_error:
        # 4. Restore all three path identities and bytes; never report a false success.
        try:
            _restore_snapshot(
                records["config"],
                content_path=config_backup if config_snapshot.existed else None,
            )
            _restore_sqlite_snapshot(records["sessions"])
            _restore_snapshot(
                records["recent_context"],
                content_path=recent_backup if recent_snapshot.existed else None,
            )
        except BaseException as restore_error:
            raise RuntimeError(
                f"session compaction migration failed and restore failed: {migration_error}"
            ) from restore_error
        raise
steps = [step(migrate_session_context_compaction_ledger)]
