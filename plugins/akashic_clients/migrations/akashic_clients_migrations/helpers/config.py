"""Move the legacy client configuration into this plugin's data root."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomlkit

from agent.migrations.context import current_migration_context


BUNDLE_ID = "akashic_clients_config"
_BACKUP_SUFFIX = ".before-akashic-clients-plugin-migration.bak"
_LEGACY_CHANNEL_KEYS = frozenset(
    {"enabled", "channel_name", "socket_path", "host", "port"}
)
_MOBILE_KEYS = frozenset(
    {
        "enabled",
        "host",
        "port",
        "database",
        "lan_hostname",
        "public_url",
        "max_attachment_mb",
        "inbox_retention_days",
        "key_encryption",
    }
)
_KEY_ENCRYPTION_KEYS = frozenset(
    {"provider", "master_key_namespace", "master_key_file", "keyset_manifest"}
)


class MigrationConflict(RuntimeError):
    """Refuse to overwrite an unrelated plugin config or recovery point."""


def migrate(connection: object) -> None:
    """Move legacy client tables after the migration runner has bound its owner."""

    _ = connection
    context = current_migration_context()
    data_root = context.bundle_data_roots.get(BUNDLE_ID)
    if data_root is None:
        raise RuntimeError(f"迁移缺少 bundle data root: {BUNDLE_ID}")
    migrate_config(context.config_path, context.workspace, data_root)


def migrate_config(config_path: Path, workspace: Path, data_root: Path) -> bool:
    """Copy client settings, retain a source backup, and remove old tables."""

    if not config_path.is_file() or config_path.is_symlink():
        raise FileNotFoundError(f"主配置必须是普通文件: {config_path}")
    source_bytes = config_path.read_bytes()
    try:
        source_text = source_bytes.decode("utf-8")
        document = tomlkit.parse(source_text)
    except (UnicodeDecodeError, tomlkit.exceptions.ParseError) as error:
        raise ValueError(f"主配置无法解析: {config_path}") from error

    channels = document.get("channels")
    if channels is not None and not isinstance(channels, Mapping):
        raise ValueError("channels 必须是 TOML table")
    chat = channels.get("chat") if isinstance(channels, Mapping) else None
    if chat is not None and not isinstance(chat, Mapping):
        raise ValueError("channels.chat 必须是 TOML table")
    mobile = document.get("mobile_realtime")
    if mobile is not None and not isinstance(mobile, Mapping):
        raise ValueError("mobile_realtime 必须是 TOML table")
    if chat is None and mobile is None:
        return False

    client_config = _build_client_config(chat, mobile)
    target_text = tomlkit.dumps(client_config)
    target = _safe_target(data_root, workspace)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise MigrationConflict(f"插件配置不是普通文件，拒绝覆盖: {target}")
        try:
            existing = tomllib_loads(target.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise MigrationConflict(f"插件配置无法解析，拒绝覆盖: {target}") from error
        if existing != tomllib_loads(target_text):
            raise MigrationConflict(f"插件配置已存在且内容不同，拒绝覆盖: {target}")

    backup = config_path.with_name(config_path.name + _BACKUP_SUFFIX)
    if backup.exists():
        if backup.is_symlink() or not backup.is_file() or backup.read_bytes() != source_bytes:
            raise MigrationConflict(f"配置恢复点与本次输入不同，拒绝覆盖: {backup}")

    _ensure_data_root(data_root, workspace)
    if not backup.exists():
        shutil.copy2(config_path, backup)
    if not target.exists():
        _atomic_write(target, target_text, mode=0o600)

    if isinstance(channels, Mapping):
        channels.pop("chat", None)
        if not channels:
            document.pop("channels", None)
    document.pop("mobile_realtime", None)
    _atomic_write(config_path, tomlkit.dumps(document), mode=config_path.stat().st_mode & 0o777)
    return True


def _build_client_config(
    chat: Mapping[str, Any] | None,
    mobile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    chat_values = dict(chat or {})
    unknown_chat = set(chat_values) - _LEGACY_CHANNEL_KEYS
    if unknown_chat:
        raise ValueError(
            "channels.chat 含无法迁移的字段: " + ", ".join(sorted(unknown_chat))
        )
    if "enabled" in chat_values and not isinstance(chat_values["enabled"], bool):
        raise ValueError("channels.chat.enabled 必须是布尔值")
    if "channel_name" in chat_values:
        channel_name = chat_values["channel_name"]
        if channel_name not in {"", "web", "akashic"}:
            raise ValueError(
                "channels.chat.channel_name 不是 Akashic clients 的可迁移身份: "
                f"{channel_name!r}"
            )
    if "socket_path" in chat_values and not isinstance(chat_values["socket_path"], str):
        raise ValueError("channels.chat.socket_path 必须是字符串")
    if "host" in chat_values and not isinstance(chat_values["host"], str):
        raise ValueError("channels.chat.host 必须是字符串")
    if "port" in chat_values and (
        isinstance(chat_values["port"], bool) or not isinstance(chat_values["port"], int)
    ):
        raise ValueError("channels.chat.port 必须是整数")

    mobile_values = dict(mobile or {})
    unknown_mobile = set(mobile_values) - _MOBILE_KEYS
    if unknown_mobile:
        raise ValueError(
            "mobile_realtime 含无法迁移的字段: " + ", ".join(sorted(unknown_mobile))
        )
    if "key_encryption" in mobile_values:
        key_values = mobile_values["key_encryption"]
        if not isinstance(key_values, Mapping):
            raise ValueError("mobile_realtime.key_encryption 必须是 TOML table")
        unknown_key = set(key_values) - _KEY_ENCRYPTION_KEYS
        if unknown_key:
            raise ValueError(
                "mobile_realtime.key_encryption 含无法迁移的字段: "
                + ", ".join(sorted(unknown_key))
            )
        mobile_values["key_encryption"] = dict(key_values)
    if "enabled" in mobile_values and not isinstance(mobile_values["enabled"], bool):
        raise ValueError("mobile_realtime.enabled 必须是布尔值")

    web_enabled = bool(chat_values.get("enabled", True))
    mobile_enabled = bool(mobile_values.get("enabled", False))
    result: dict[str, Any] = {
        "enabled": web_enabled or mobile_enabled,
        "web": {"enabled": web_enabled},
    }
    if mobile is not None:
        result["mobile_realtime"] = mobile_values
    if "socket_path" in chat_values:
        result["web"]["socket_path"] = chat_values["socket_path"]
    return result


def _safe_target(data_root: Path, workspace: Path) -> Path:
    _ensure_data_root(data_root, workspace)
    return data_root / "config.local.toml"


def _ensure_data_root(data_root: Path, workspace: Path) -> None:
    root = workspace.resolve(strict=False)
    candidate = data_root.expanduser().resolve(strict=False)
    plugin_root = root / "plugin-data"
    if not candidate.is_relative_to(plugin_root):
        raise ValueError(f"插件迁移数据目录越过 workspace: {data_root}")
    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"插件迁移数据目录不能穿过符号链接: {current}")
    candidate.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, text: str, *, mode: int) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode or 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tomllib_loads(text: str) -> dict[str, Any]:
    """Parse TOML for semantic equality without preserving formatting."""

    import tomllib

    value = tomllib.loads(text)
    if not isinstance(value, dict):
        raise ValueError("TOML 顶层必须是 table")
    return value
