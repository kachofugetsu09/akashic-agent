"""Retire the old Mobile transport setting without deleting its stored data."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context


__depends__ = {"20260913_01_akashic_clients_config"}
__transactional__ = False
_BACKUP_SUFFIX = ".before-retire-mobile-20260928.bak"


def retire_mobile_config(connection: object) -> None:
    """Remove the retired key after saving the exact prior config."""

    _ = connection
    context = current_migration_context()
    root = context.bundle_data_roots.get("akashic_clients_config")
    if root is None:
        raise RuntimeError("迁移缺少 akashic_clients data root")
    workspace = context.workspace.resolve(strict=False)
    target = root / "config.local.toml"
    if not root.resolve(strict=False).is_relative_to(workspace / "plugin-data"):
        raise ValueError(f"插件配置越过 workspace: {root}")
    if target.is_symlink():
        raise ValueError(f"插件配置不能是符号链接: {target}")
    if not target.exists():
        return
    if not target.is_file():
        raise ValueError(f"插件配置不是普通文件: {target}")

    source = target.read_bytes()
    document = tomlkit.parse(source.decode("utf-8"))
    mobile = document.get("mobile_realtime")
    if mobile is None:
        return
    if not isinstance(mobile, Mapping):
        raise ValueError("mobile_realtime 必须是 TOML table")
    web = document.get("web")
    if document.get("enabled", True) and isinstance(web, Mapping) and web.get("enabled") is False:
        raise ValueError("旧配置仅启用 Mobile；需先明确 Web 启用策略")

    backup = target.with_name(target.name + _BACKUP_SUFFIX)
    if backup.exists() or backup.is_symlink():
        if backup.is_symlink() or not backup.is_file() or backup.read_bytes() != source:
            raise FileExistsError(f"配置恢复点已存在且内容不同: {backup}")
    else:
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
    _sync_directory(target.parent)

    document.pop("mobile_realtime")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(tomlkit.dumps(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, target.stat().st_mode & 0o777)
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    """Persist the backup and replacement names before migration success."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


steps = [step(retire_mobile_config)]
