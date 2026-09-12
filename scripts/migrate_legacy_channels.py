#!/usr/bin/env python3
"""Migrate legacy ``[channels.telegram]`` and ``[channels.qq]`` once.

The running application never reads these tables as channel owners.  This
command copies their validated values into ordinary plugin data, writes a
recoverable config backup, and removes the old tables from the source config.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import tomlkit

from agent.plugins.manifest import ensure_workspace_plugin_data_dir, workspace_plugin_data_dir


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
    qq = channels.get("qq")
    if telegram is None and qq is None:
        return ()

    for name, value in (("telegram", telegram), ("qq", qq)):
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"channels.{name} 必须是 TOML table")
    targets: list[tuple[str, str]] = []
    migrated_channels: list[str] = []
    if isinstance(telegram, Mapping):
        migrated_channels.append("telegram_channel")
        targets.extend(
            (
                ("telegram_channel", _render_telegram(telegram, workspace)),
                ("telegram_sender", _render_telegram_sender(telegram, workspace)),
            )
        )
    if isinstance(qq, Mapping):
        migrated_channels.append("qq_channel")
        targets.extend(
            (
                ("qq_channel", _render_qq(qq)),
                ("qq_sender", _render_qq_sender(qq, workspace)),
            )
        )
    if not targets:
        raise ValueError("legacy channels.telegram/qq 必须是 TOML table")
    for plugin_name, content in targets:
        target = workspace_plugin_data_dir(workspace, plugin_name, marketplace) / "config.local.toml"
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise MigrationConflict(f"插件配置已存在，拒绝覆盖: {target}")
    backup = config_path.with_name(config_path.name + ".before-channel-plugin-migration.bak")
    if backup.exists() and backup.read_text(encoding="utf-8") != source:
        raise MigrationConflict(f"配置恢复点与本次输入不同，拒绝覆盖: {backup}")

    # 1. Validate and stage all plugin outputs before changing the source config.
    staged: list[tuple[Path, Path]] = []
    try:
        for plugin_name, content in targets:
            directory = workspace_plugin_data_dir(workspace, plugin_name, marketplace)
            ensure_workspace_plugin_data_dir(directory, workspace)
            fd, temporary = tempfile.mkstemp(prefix=".channel-migration.", dir=directory)
            temp_path = Path(temporary)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            staged.append((temp_path, directory / "config.local.toml"))

        # 2. Keep a named source recovery point before removing legacy owner data.
        if not backup.exists():
            shutil.copy2(config_path, backup)
        # 先发布全部目标，最后移除旧入口；中断后同内容目标允许安全续做。
        for temporary, target in staged:
            os.replace(temporary, target)
        channels.pop("telegram", None)
        channels.pop("qq", None)
        _atomic_write(config_path, tomlkit.dumps(document), mode=config_path.stat().st_mode & 0o777)
    except BaseException:
        for temporary, _target in staged:
            temporary.unlink(missing_ok=True)
        raise
    return tuple(migrated_channels)


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


def _render_qq(table: Mapping[str, Any]) -> str:
    enabled = _as_bool(table.get("enabled", True), "channels.qq.enabled")
    bot_uin = str(table.get("bot_uin", "")).strip()
    groups_raw = table.get("groups", [])
    if not isinstance(groups_raw, list | tuple):
        raise ValueError("channels.qq.groups 必须是数组")
    groups: list[dict[str, Any]] = []
    for index, raw in enumerate(groups_raw):
        if not isinstance(raw, Mapping):
            raise ValueError(f"channels.qq.groups[{index}] 必须是 table")
        group_id = str(raw.get("group_id", raw.get("groupId", ""))).strip()
        if not group_id:
            raise ValueError(f"channels.qq.groups[{index}].group_id 不能为空")
        groups.append(
            {
                "group_id": group_id,
                "allow_from": [
                    str(item)
                    for item in _string_list(
                        raw.get("allow_from", raw.get("allowFrom", [])),
                        f"channels.qq.groups[{index}].allow_from",
                    )
                ],
                "require_at": _as_bool(
                    raw.get("require_at", raw.get("requireAt", True)),
                    f"channels.qq.groups[{index}].require_at",
                ),
            }
        )
    timeout = float(table.get("websocket_open_timeout_seconds", 5.0))
    if timeout <= 0:
        raise ValueError("channels.qq.websocket_open_timeout_seconds 必须大于 0")
    return tomlkit.dumps(
        {
            "enabled": enabled and bool(bot_uin),
            "bot_uin": bot_uin,
            "allow_from": [
                str(item)
                for item in _string_list(
                    table.get("allow_from", table.get("allowFrom", [])),
                    "channels.qq.allow_from",
                )
            ],
            "groups": groups,
            "websocket_open_timeout_seconds": timeout,
        }
    )


def _render_qq_sender(table: Mapping[str, Any], workspace: Path) -> str:
    """Create a sender only when the old config names a real OneBot endpoint."""

    enabled = _as_bool(table.get("enabled", True), "channels.qq.enabled")
    bot_uin = str(table.get("bot_uin", "")).strip()
    active = enabled and bool(bot_uin)
    raw_endpoint = table.get("sender_endpoint", table.get("endpoint", ""))
    endpoint = str(raw_endpoint).strip()
    token = _resolve(
        str(table.get("sender_token", table.get("token", ""))), workspace
    ).strip()
    if active and not endpoint:
        raise ValueError(
            "启用 QQ channel 迁移需要 channels.qq.sender_endpoint（OneBot WS API）；"
            "不能从 bot_uin 猜测 QQ sender 地址"
        )
    if endpoint:
        _validate_qq_sender_endpoint(endpoint)
    values: dict[str, Any] = {"enabled": active, "channel": "qq"}
    if endpoint:
        values["endpoint"] = endpoint
    if token:
        values["token"] = token
    return tomlkit.dumps(values)


def _validate_qq_sender_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"ws", "wss"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") == "/event"
    ):
        raise ValueError(
            "channels.qq.sender_endpoint 必须是无凭据、无 query/fragment 的 OneBot WS API URL"
        )


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
