"""Retire the old Mobile transport setting without deleting its stored data."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

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
    target = root / "config.input.json"
    if not root.resolve(strict=False).is_relative_to(workspace / "plugin-data"):
        raise ValueError(f"插件配置越过 workspace: {root}")
    legacy = root / "config.local.toml"
    if legacy.exists() or legacy.is_symlink():
        raise ValueError("发现旧 TOML 配置；须先离线显式升级为固定配置输入")
    if target.is_symlink():
        raise ValueError(f"插件配置不能是符号链接: {target}")
    if not target.exists():
        return
    if not target.is_file():
        raise ValueError(f"插件配置不是普通文件: {target}")

    source = target.read_bytes()
    # 1. 修改固定输入的映射；其余编码值和凭据引用原样保留。
    document = json.loads(source)
    if (
        not isinstance(document, dict)
        or set(document) != {"version", "config"}
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        raise ValueError("固定配置输入格式错误")
    config = _config_map(document["config"])
    if "mobile_realtime" not in config:
        return
    _config_map(config["mobile_realtime"])
    enabled = config.get("enabled", True)
    web_enabled = _config_map(config["web"]).get("enabled", True) if "web" in config else True
    if type(enabled) is not bool or type(web_enabled) is not bool:
        raise ValueError("客户端 enabled 必须是布尔值")
    if enabled and not web_enabled:
        raise ValueError("旧配置仅启用 Mobile；需先明确 Web 启用策略")

    # 2. 先保存原始字节，再原子替换输入；重试不覆盖恢复点。
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

    config.pop("mobile_realtime")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, target.stat().st_mode & 0o777)
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _config_map(value: object) -> dict[str, object]:
    """读取固定配置格式的 map，不解析无关字段或凭据。"""
    if (
        not isinstance(value, list)
        or len(value) != 2
        or value[0] != "map"
        or not isinstance(value[1], dict)
    ):
        raise ValueError("固定配置输入必须是 map")
    return value[1]


def _sync_directory(path: Path) -> None:
    """Persist the backup and replacement names before migration success."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


steps = [step(retire_mobile_config)]
