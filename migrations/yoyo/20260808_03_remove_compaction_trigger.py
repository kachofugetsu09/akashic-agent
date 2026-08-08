from __future__ import annotations

import hashlib
import json
import os
import stat
import tomllib
from pathlib import Path
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context


__depends__ = {"20260808_05_activate_session_compaction_cursor"}
__transactional__ = False

_MIGRATION_NAME = "remove-compaction-trigger"


class _ConfigSnapshot:
    """Capture config bytes and path identity before the external rewrite."""

    def __init__(
        self,
        path: Path,
        content: bytes,
        mode: int,
        kind: str,
        resolved_target: Path,
        symlink_target: str | None,
    ) -> None:
        self.path = path
        self.content = content
        self.mode = mode
        self.kind = kind
        self.resolved_target = resolved_target
        self.symlink_target = symlink_target


def _snapshot_config(path: Path) -> _ConfigSnapshot | None:
    """Read one regular config file while preserving symbolic-link identity."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"无法读取配置迁移源元数据: {path}") from exc

    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        resolved = path.resolve(strict=False)
        if not path.is_file():
            raise RuntimeError(f"配置迁移源不是可读文件: {path}")
        return _ConfigSnapshot(
            path,
            path.read_bytes(),
            stat.S_IMODE(resolved.stat().st_mode),
            "symlink",
            resolved,
            target,
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"配置迁移源必须是普通文件或软链接: {path}")
    return _ConfigSnapshot(
        path,
        path.read_bytes(),
        mode,
        "file",
        path,
        None,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably publish one atomic file replacement."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, mode: int) -> None:
    """Write bytes through a same-directory fsynced replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        candidate.write_bytes(payload)
        candidate.chmod(mode)
        with candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        candidate.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise


def _render_config(raw: bytes) -> bytes | None:
    """Remove the retired trigger while retaining the fixed token watermark."""

    text = raw.decode("utf-8")
    # 1. Parse both representations so malformed configuration remains a hard failure.
    tomllib.loads(text)
    document = tomlkit.parse(text)
    agent = document.get("agent")
    if not isinstance(agent, dict):
        return None
    context = agent.get("context")
    if not isinstance(context, dict):
        return None
    compaction = context.get("compaction")
    if not isinstance(compaction, dict) or "trigger_percent" not in compaction:
        return None

    # 2. Keep the new fixed watermark and remove only the retired policy key.
    compaction.pop("trigger_percent")
    compaction.setdefault("keep_recent_tokens", 20_000)
    rendered = tomlkit.dumps(document).encode("utf-8")
    parsed = tomllib.loads(rendered.decode("utf-8"))
    final_compaction = parsed.get("agent", {}).get("context", {}).get("compaction", {})
    if "trigger_percent" in final_compaction:
        raise RuntimeError("配置迁移后仍存在 agent.context.compaction.trigger_percent")
    if "keep_recent_tokens" not in final_compaction:
        raise RuntimeError("配置迁移后缺少 agent.context.compaction.keep_recent_tokens")
    return rendered


def _write_backup(snapshot: _ConfigSnapshot, root: Path) -> Path:
    """Write a recoverable config snapshot and a machine-readable manifest."""

    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(root, 0o700)
    backup = root / snapshot.path.name
    _write_atomic(backup, snapshot.content, 0o600)
    if backup.read_bytes() != snapshot.content:
        raise RuntimeError(f"配置迁移备份校验失败: {snapshot.path}")
    payload = {
        "schema_version": 1,
        "migration": _MIGRATION_NAME,
        "source": {
            "path": str(snapshot.path),
            "kind": snapshot.kind,
            "symlink_target": snapshot.symlink_target,
            "resolved_target": str(snapshot.resolved_target),
            "backup": snapshot.path.name,
            "sha256": _sha256(snapshot.content),
        },
    }
    manifest = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_atomic(root / "manifest.json", manifest, 0o600)
    return backup


def _publish_config(snapshot: _ConfigSnapshot, rendered: bytes) -> None:
    """Publish config bytes without replacing a pre-existing symbolic link."""

    _write_atomic(snapshot.resolved_target, rendered, snapshot.mode)
    if snapshot.kind == "symlink":
        if not snapshot.path.is_symlink() or os.readlink(snapshot.path) != snapshot.symlink_target:
            raise RuntimeError(f"配置软链接身份发生变化: {snapshot.path}")
    if snapshot.resolved_target.read_bytes() != rendered:
        raise RuntimeError(f"配置迁移发布校验失败: {snapshot.path}")


def remove_compaction_trigger(_connection: object) -> None:
    """Back up and remove the retired percentage trigger from installation config."""

    _ = _connection
    current = current_migration_context()
    snapshot = _snapshot_config(current.config_path)
    if snapshot is None:
        return
    rendered = _render_config(snapshot.content)
    if rendered is None:
        return

    # 3. Persist a recoverable source snapshot before the external config write.
    backup_root = (
        current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex
    )
    _write_backup(snapshot, backup_root)

    # 4. Publish and verify the fixed policy; Yoyo records the ID only after return.
    _publish_config(snapshot, rendered)


steps = [step(remove_compaction_trigger)]
