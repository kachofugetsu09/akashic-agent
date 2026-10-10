#!/usr/bin/env python3
"""Migrate legacy ``[channels.telegram]`` once.

The running application never reads these tables as channel owners.  This
command copies their validated values into ordinary plugin data, writes a
recoverable config backup, and removes the old tables from the source config.
"""

from __future__ import annotations

import argparse
import asyncio
import tomllib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import tomlkit

from agent.plugins.manifest import ensure_workspace_plugin_data_dir, workspace_plugin_data_dir
from agent.plugin_composition.credentials import CredentialRef
from agent.plugin_composition.config_input import load_config, save_config, save_credential
from agent.plugins.channel_credentials import CoreProviderClientFactory


class MigrationConflict(RuntimeError):
    """Refuse to overwrite an existing plugin config or recovery point."""


def migrate_legacy_channels(config_path: Path, workspace: Path, *, marketplace: str) -> tuple[str, ...]:
    """Move legacy channel tables into plugin data and return migrated channels."""

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    source = config_path.read_text(encoding="utf-8")
    document = tomlkit.parse(source)
    channels = document.get("channels")
    if not isinstance(channels, Mapping):
        return ()
    telegram = channels.get("telegram")
    if telegram is None:
        return ()

    if not isinstance(telegram, Mapping):
        raise ValueError("channels.telegram 必须是 TOML table")
    targets = (
        ("telegram_channel", _render_telegram(telegram, workspace)),
        ("telegram_sender", _render_telegram_sender(telegram, workspace)),
    )
    outputs: list[tuple[Path, dict[str, Any]]] = []
    for plugin_name, content in targets:
        directory = workspace_plugin_data_dir(workspace, plugin_name, marketplace)
        values = tomllib.loads(content)
        current, revision = load_config(directory)
        if current:
            if not asyncio.run(_matches_config(directory, current, revision, values)):
                raise MigrationConflict(f"插件配置已存在，拒绝覆盖: {directory}")
            continue
        outputs.append((directory, values))
    backup = config_path.with_name(config_path.name + ".before-channel-plugin-migration.bak")
    if backup.exists() and backup.read_text(encoding="utf-8") != source:
        raise MigrationConflict(f"配置恢复点与本次输入不同，拒绝覆盖: {backup}")

    # 1. 原输入先完整备份；部分目标已发布时保留它们供同输入重试。
    if not backup.exists():
        shutil.copy2(config_path, backup)
        os.chmod(backup, 0o600)
        with backup.open("rb") as stream:
            os.fsync(stream.fileno())
    # 2. 此命令拥有旧渠道字段，Core writer 只接收无明文的固定映射。
    for directory, values in outputs:
        ensure_workspace_plugin_data_dir(directory, workspace)
        token = values.pop("token", None)
        if token:
            values["token"] = save_credential(directory, token)
        save_config(directory, values)
    # 3. 所有新 owner 可读后才移除旧主配置入口；恢复点始终保留。
    if config_path.read_text(encoding="utf-8") != source:
        raise MigrationConflict("迁移期间主配置变化；已发布目标和恢复点保留")
    channels.pop("telegram", None)
    _atomic_write(config_path, tomlkit.dumps(document), mode=config_path.stat().st_mode & 0o777)
    return ("telegram_channel",)


async def _matches_config(directory: Path, current: dict[str, object], revision: str,
                          expected: dict[str, Any]) -> bool:
    """通过同一个授权 factory 核对本命令已发布的目标，允许失败后重试。"""
    values = dict(current)
    ref = values.get("token")
    factory = CoreProviderClientFactory(directory, current, revision)
    try:
        if isinstance(ref, CredentialRef):
            client = await factory.create({"token": ref})
            values["token"] = client.credential(ref)
        return values == expected
    finally:
        await factory.aclose()


def _render_telegram(table: Mapping[str, Any], workspace: Path) -> str:
    channel_name = str(table.get("channel_name", "telegram")).strip()
    if channel_name != "telegram":
        raise ValueError(
            "自定义 channels.telegram.channel_name 无法直接迁移到静态 telegram channel；"
            "请为该身份发布独立 channel 插件后再迁移"
        )
    enabled = _as_bool(table.get("enabled", True), "channels.telegram.enabled")
    raw_token = str(table.get("token", ""))
    token = _resolve(raw_token, workspace).strip()
    allow_from = table.get("allow_from", table.get("allowFrom", []))
    values: dict[str, Any] = {
        "enabled": enabled and bool(token),
        "allow_from": [
            str(item) for item in _string_list(allow_from, "channels.telegram.allow_from")
        ],
    }
    if token:
        values["token"] = token
    return tomlkit.dumps(values)


def _render_telegram_sender(table: Mapping[str, Any], workspace: Path) -> str:
    """Create the matching delivery sender from the legacy Telegram token."""

    channel_name = str(table.get("channel_name", "telegram")).strip()
    if channel_name != "telegram":
        raise ValueError(
            "自定义 channels.telegram.channel_name 无法直接迁移到静态 telegram sender；"
            "请为该身份发布独立 sender 插件后再迁移"
        )
    enabled = _as_bool(table.get("enabled", True), "channels.telegram.enabled")
    token = _resolve(str(table.get("token", "")), workspace).strip()
    values: dict[str, Any] = {
        "enabled": enabled and bool(token),
        "channel": "telegram",
    }
    if token:
        values["token"] = token
    return tomlkit.dumps(values)


def _atomic_write(path: Path, text: str, *, mode: int) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode or 0o600)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _resolve(value: str, workspace: Path) -> str:
    resolved = re.sub(r"\$\{(\w+)\}", lambda match: os.environ.get(match.group(1), match.group(0)), value)
    match = re.fullmatch(r"\$\{(\w+)\}", resolved)
    if match:
        secret_file = workspace / "memory" / match.group(1)
        if secret_file.exists():
            return secret_file.read_text(encoding="utf-8").strip()
    return resolved


def _as_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} 必须是布尔值")
    return value


def _string_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field} 必须是字符串数组")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} 必须只包含字符串")
    return list(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--marketplace", required=True, help="目标插件的安装 marketplace")
    args = parser.parse_args()
    migrated = migrate_legacy_channels(args.config, args.workspace, marketplace=args.marketplace)
    if migrated:
        print("已迁移: " + ", ".join(migrated))
    else:
        print("没有需要迁移的 legacy channel 配置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
