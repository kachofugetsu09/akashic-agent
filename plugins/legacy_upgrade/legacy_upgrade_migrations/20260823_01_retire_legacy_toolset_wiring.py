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

__depends__ = {"20260817_01_akasha_sparse_index_v10"}
__transactional__ = False

_MIGRATION_NAME = "retire-legacy-toolset-wiring"
_LEGACY_TOOLSETS = ("meta_common", "spawn", "schedule")
_CURRENT_TOOLSETS = ("meta_common",)


class _ConfigSnapshot:
    """Capture bytes, permissions, and link identity before rewriting config."""

    def __init__(
        self,
        *,
        path: Path,
        content: bytes,
        kind: str,
        mode: int,
        resolved_target: Path,
        symlink_target: str | None,
    ) -> None:
        self.path = path
        self.content = content
        self.kind = kind
        self.mode = mode
        self.resolved_target = resolved_target
        self.symlink_target = symlink_target


def _snapshot_config(path: Path) -> _ConfigSnapshot | None:
    """Read one regular config file without losing a symbolic-link boundary."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"无法读取配置迁移源元数据: {path}") from exc

    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        resolved = path.resolve(strict=False)
        if not path.is_file() or not resolved.is_file():
            raise RuntimeError(f"配置迁移源软链接不是可读文件: {path}")
        return _ConfigSnapshot(
            path=path,
            content=path.read_bytes(),
            kind="symlink",
            mode=stat.S_IMODE(resolved.stat().st_mode),
            resolved_target=resolved,
            symlink_target=target,
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"配置迁移源必须是普通文件或软链接: {path}")
    return _ConfigSnapshot(
        path=path,
        content=path.read_bytes(),
        kind="file",
        mode=stat.S_IMODE(metadata.st_mode),
        resolved_target=path,
        symlink_target=None,
    )


def _sha256(payload: bytes) -> str:
    """Return the digest used to prove a recoverable config snapshot."""

    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist an already-replaced file entry in its parent directory."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, mode: int) -> None:
    """Write one fsynced file through a same-directory atomic replacement."""

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
    """Replace only the exact retired default toolset sequence."""

    text = raw.decode("utf-8")
    # 1. Parse before mutation so malformed config remains a startup error.
    tomllib.loads(text)
    document = tomlkit.parse(text)
    agent = document.get("agent")
    if not isinstance(agent, dict):
        return None
    wiring = agent.get("wiring")
    if not isinstance(wiring, dict):
        return None
    toolsets = wiring.get("toolsets")
    if not isinstance(toolsets, list) or tuple(toolsets) != _LEGACY_TOOLSETS:
        return None

    # 2. Preserve every other user-owned field and remove only retired entries.
    wiring["toolsets"] = list(_CURRENT_TOOLSETS)
    rendered = tomlkit.dumps(document).encode("utf-8")
    parsed = tomllib.loads(rendered.decode("utf-8"))
    final_agent = parsed.get("agent")
    final_wiring = final_agent.get("wiring") if isinstance(final_agent, dict) else None
    if (
        not isinstance(final_wiring, dict)
        or tuple(final_wiring.get("toolsets", ())) != _CURRENT_TOOLSETS
    ):
        raise RuntimeError("配置迁移后 agent.wiring.toolsets 不等于当前默认值")
    return rendered


def _write_backup(snapshot: _ConfigSnapshot, root: Path) -> Path:
    """Store a verified 0600 source copy and manifest before publication."""

    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(root, 0o700)
    backup = root / "config.toml.before"
    _write_atomic(backup, snapshot.content, 0o600)
    digest = _sha256(snapshot.content)
    if _sha256(backup.read_bytes()) != digest:
        raise RuntimeError(f"配置迁移备份校验失败: {snapshot.path}")
    manifest = {
        "schema_version": 1,
        "migration": _MIGRATION_NAME,
        "source": {
            "path": str(snapshot.path),
            "kind": snapshot.kind,
            "resolved_target": str(snapshot.resolved_target),
            "symlink_target": snapshot.symlink_target,
            "mode": snapshot.mode,
            "backup": backup.name,
            "sha256": digest,
        },
    }
    rendered = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_atomic(root / "manifest.json", rendered, 0o600)
    if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"配置迁移备份 manifest 校验失败: {root}")
    return backup


def _publish_config(snapshot: _ConfigSnapshot, rendered: bytes) -> None:
    """Publish config bytes while preserving an existing symbolic link."""

    _write_atomic(snapshot.resolved_target, rendered, snapshot.mode)
    if snapshot.kind == "symlink" and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"配置软链接身份发生变化: {snapshot.path}")
    if snapshot.path.read_bytes() != rendered:
        raise RuntimeError(f"配置迁移发布校验失败: {snapshot.path}")


def _restore_config(snapshot: _ConfigSnapshot, backup: Path) -> None:
    """Restore the exact pre-migration bytes after a failed publication."""

    payload = backup.read_bytes()
    if _sha256(payload) != _sha256(snapshot.content):
        raise RuntimeError(f"配置迁移恢复备份摘要不匹配: {snapshot.path}")
    _write_atomic(snapshot.resolved_target, payload, snapshot.mode)
    if snapshot.kind == "symlink" and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"配置软链接恢复身份不匹配: {snapshot.path}")
    if snapshot.path.read_bytes() != snapshot.content:
        raise RuntimeError(f"配置迁移恢复校验失败: {snapshot.path}")


def retire_legacy_toolset_wiring(_connection: object) -> None:
    """Migrate the exact pre-plugin toolset default without touching custom config."""

    _ = _connection
    current = current_migration_context()
    snapshot = _snapshot_config(current.config_path)
    if snapshot is None:
        return
    rendered = _render_config(snapshot.content)
    if rendered is None:
        return

    # 1. Persist a recoverable source snapshot before the only external write.
    backup_root = current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex
    backup = _write_backup(snapshot, backup_root)

    # 2. Publish the narrow replacement and restore it if verification fails.
    try:
        _publish_config(snapshot, rendered)
    except BaseException as migration_error:
        try:
            _restore_config(snapshot, backup)
        except BaseException as restore_error:
            raise RuntimeError(
                f"toolset wiring migration failed and restore failed: {migration_error}"
            ) from restore_error
        raise


steps = [step(retire_legacy_toolset_wiring)]
