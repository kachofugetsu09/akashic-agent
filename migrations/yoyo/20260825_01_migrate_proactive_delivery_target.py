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

__depends__ = {"20260823_01_retire_legacy_toolset_wiring"}
__transactional__ = False

_MIGRATION_NAME = "migrate-proactive-delivery-target"


class _FileSnapshot:
    """Capture one file's bytes, permissions, and symbolic-link identity."""

    def __init__(
        self,
        *,
        path: Path,
        content: bytes | None,
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


class _MigrationPlan:
    """Describe the two file publications required by one legacy config."""

    def __init__(self, *, config: bytes, wake: bytes | None) -> None:
        self.config = config
        self.wake = wake


def _snapshot_file(path: Path, *, absent_mode: int = 0o600) -> _FileSnapshot:
    """Read a regular file while preserving an existing symbolic-link boundary."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _FileSnapshot(
            path=path,
            content=None,
            kind="absent",
            mode=absent_mode,
            resolved_target=path,
            symlink_target=None,
        )
    except OSError as exc:
        raise RuntimeError(f"无法读取迁移源元数据: {path}") from exc

    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        resolved = path.resolve(strict=False)
        if not path.is_file() or not resolved.is_file():
            raise RuntimeError(f"迁移源软链接不是可读文件: {path}")
        return _FileSnapshot(
            path=path,
            content=path.read_bytes(),
            kind="symlink",
            mode=stat.S_IMODE(resolved.stat().st_mode),
            resolved_target=resolved,
            symlink_target=target,
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"迁移源必须是普通文件或软链接: {path}")
    return _FileSnapshot(
        path=path,
        content=path.read_bytes(),
        kind="file",
        mode=stat.S_IMODE(metadata.st_mode),
        resolved_target=path,
        symlink_target=None,
    )


def _sha256(payload: bytes) -> str:
    """Return the digest used to verify a recoverable source snapshot."""

    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist directory entries after an atomic replacement or removal."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, mode: int) -> None:
    """Publish one fsynced regular file through a same-directory replacement."""

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


def _validated_text(value: object, *, field: str) -> str:
    """Validate one legacy target identity at the configuration boundary."""

    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} 必须是非空且无首尾空白的字符串")
    return value


def _legacy_delivery(proactive: object) -> dict[str, str] | None:
    """Map one enabled legacy target to Wake's delivery schema."""

    if not isinstance(proactive, dict):
        raise ValueError("proactive 必须是 table")
    enabled = proactive.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("proactive.enabled 必须是布尔值")
    if not enabled:
        return None
    target = proactive.get("target")
    if not isinstance(target, dict):
        raise ValueError("已启用 proactive 时必须配置 proactive.target")
    channel = _validated_text(target.get("channel"), field="proactive.target.channel")
    recipient = _validated_text(target.get("chat_id"), field="proactive.target.chat_id")
    return {
        "channel": channel,
        "recipient": recipient,
        "session_id": f"{channel}:{recipient}",
    }


def _render_plan(config_raw: bytes, wake_raw: bytes | None) -> _MigrationPlan | None:
    """Remove retired proactive config and preserve its enabled delivery target."""

    # 1. Parse both trust-boundary files before producing any mutation.
    config_text = config_raw.decode("utf-8")
    config_data = tomllib.loads(config_text)
    if "proactive" not in config_data:
        return None
    delivery = _legacy_delivery(config_data["proactive"])
    config_document = tomlkit.parse(config_text)
    del config_document["proactive"]
    rendered_config = tomlkit.dumps(config_document).encode("utf-8")
    if "proactive" in tomllib.loads(rendered_config.decode("utf-8")):
        raise RuntimeError("迁移后主配置仍含 proactive")

    # 2. Add only the missing Wake delivery, preserving matching plugin config.
    if wake_raw is None:
        wake_document = tomlkit.document()
    else:
        wake_text = wake_raw.decode("utf-8")
        tomllib.loads(wake_text)
        wake_document = tomlkit.parse(wake_text)
    existing = wake_document.get("delivery")
    if delivery is None:
        rendered_wake = wake_raw
    elif existing is None:
        wake_document["delivery"] = delivery
        rendered_wake = tomlkit.dumps(wake_document).encode("utf-8")
    elif dict(existing) == delivery:
        rendered_wake = wake_raw
    else:
        raise RuntimeError("Wake delivery 已存在且与 proactive.target 冲突")

    if rendered_wake is not None:
        parsed_wake = tomllib.loads(rendered_wake.decode("utf-8"))
        if delivery is not None and parsed_wake.get("delivery") != delivery:
            raise RuntimeError("迁移后 Wake delivery 与旧 proactive.target 不一致")
    return _MigrationPlan(config=rendered_config, wake=rendered_wake)


def _write_backup(
    snapshots: tuple[_FileSnapshot, ...], root: Path
) -> dict[Path, Path | None]:
    """Store verified 0600 source copies and an integrity manifest."""

    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(root, 0o700)
    backups: dict[Path, Path | None] = {}
    sources: dict[str, object] = {}
    for index, snapshot in enumerate(snapshots):
        backup: Path | None = None
        digest: str | None = None
        if snapshot.content is not None:
            backup = root / f"source-{index}.before"
            _write_atomic(backup, snapshot.content, 0o600)
            digest = _sha256(snapshot.content)
            if _sha256(backup.read_bytes()) != digest:
                raise RuntimeError(f"迁移备份校验失败: {snapshot.path}")
        backups[snapshot.path] = backup
        sources[str(snapshot.path)] = {
            "kind": snapshot.kind,
            "resolved_target": str(snapshot.resolved_target),
            "symlink_target": snapshot.symlink_target,
            "mode": snapshot.mode,
            "backup": None if backup is None else backup.name,
            "sha256": digest,
        }
    manifest = {
        "schema_version": 1,
        "migration": _MIGRATION_NAME,
        "sources": sources,
    }
    payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_atomic(root / "manifest.json", payload, 0o600)
    if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"迁移备份 manifest 校验失败: {root}")
    return backups


def _publish(snapshot: _FileSnapshot, payload: bytes | None) -> None:
    """Publish or remove one file while preserving its original link boundary."""

    if payload is None:
        if snapshot.kind != "absent":
            snapshot.path.unlink()
            _fsync_directory(snapshot.path.parent)
        return
    target = snapshot.resolved_target if snapshot.kind == "symlink" else snapshot.path
    _write_atomic(target, payload, snapshot.mode)
    if snapshot.kind == "symlink" and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"迁移发布改变了软链接身份: {snapshot.path}")
    if snapshot.path.read_bytes() != payload:
        raise RuntimeError(f"迁移发布校验失败: {snapshot.path}")


def _restore(snapshot: _FileSnapshot, backup: Path | None) -> None:
    """Restore one file to its exact pre-migration presence and bytes."""

    if snapshot.content is None:
        if snapshot.path.exists() or snapshot.path.is_symlink():
            snapshot.path.unlink()
            _fsync_directory(snapshot.path.parent)
        return
    if backup is None:
        raise RuntimeError(f"迁移恢复缺少备份: {snapshot.path}")
    payload = backup.read_bytes()
    if _sha256(payload) != _sha256(snapshot.content):
        raise RuntimeError(f"迁移恢复备份摘要不匹配: {snapshot.path}")
    target = snapshot.resolved_target if snapshot.kind == "symlink" else snapshot.path
    _write_atomic(target, payload, snapshot.mode)
    if snapshot.kind == "symlink" and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"迁移恢复改变了软链接身份: {snapshot.path}")
    if snapshot.path.read_bytes() != snapshot.content:
        raise RuntimeError(f"迁移恢复校验失败: {snapshot.path}")


def migrate_proactive_delivery_target(_connection: object) -> None:
    """Move the enabled legacy proactive target into Wake plugin configuration."""

    _ = _connection
    current = current_migration_context()
    config = _snapshot_file(current.config_path)
    if config.content is None:
        return
    wake_path = current.workspace / "plugin-data" / "wake-builtin" / "config.local.toml"
    wake = _snapshot_file(wake_path)
    plan = _render_plan(config.content, wake.content)
    if plan is None:
        return

    # 1. Persist both source states before publishing either destination.
    backup_root = current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex
    backups = _write_backup((config, wake), backup_root)

    # 2. Publish Wake first, then retire legacy config; restore both on failure.
    try:
        if plan.wake != wake.content:
            _publish(wake, plan.wake)
        _publish(config, plan.config)
    except BaseException as migration_error:
        restore_errors: list[BaseException] = []
        for snapshot in (wake, config):
            try:
                _restore(snapshot, backups[snapshot.path])
            except BaseException as exc:
                restore_errors.append(exc)
        if restore_errors:
            raise RuntimeError(
                f"proactive target migration failed and restore failed: {migration_error}"
            ) from restore_errors[0]
        raise


steps = [step(migrate_proactive_delivery_target)]
