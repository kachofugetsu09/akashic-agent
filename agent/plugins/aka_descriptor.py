from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_AKA_PLUGIN_FILE = ".aka-plugin/plugin.json"


@dataclass(frozen=True)
class PluginDescriptor:
    name: str
    version: str
    description: str
    root: Path
    raw_manifest: dict[str, object]
    lifecycle_entry: Path | None = None
    lifecycle_class: str = ""
    skill_roots: tuple[Path, ...] = ()
    drift_skill_roots: tuple[Path, ...] = ()
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_plugin_descriptor(plugin_root: Path) -> PluginDescriptor | None:
    manifest_path = plugin_root / _AKA_PLUGIN_FILE
    if not manifest_path.exists():
        return None
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("aka plugin manifest 读取失败 (%s): %s", manifest_path, e)
        return None
    if not isinstance(loaded, dict):
        logger.warning("aka plugin manifest 格式错误，期望 dict (%s)", manifest_path)
        return None

    raw = cast(dict[str, object], loaded)
    name = str(raw.get("name") or plugin_root.name).strip()
    if not name:
        logger.warning("aka plugin manifest 缺少 name (%s)", manifest_path)
        return None

    version = str(raw.get("version") or "").strip()
    description = str(raw.get("description") or "").strip()
    paths = _as_dict(raw.get("paths"))
    akashic = _as_dict(raw.get("akashic"))
    lifecycle = _as_dict(akashic.get("lifecycle"))

    lifecycle_entry = _resolve_optional_path(
        plugin_root,
        str(lifecycle.get("entry") or "").strip(),
    )
    lifecycle_class = str(lifecycle.get("class") or "").strip()
    skill_roots = _resolve_root_dirs(plugin_root, paths.get("skills"))
    drift_skill_roots = _resolve_root_dirs(plugin_root, paths.get("drift_skills"))
    mcp_servers = _load_mcp_servers(
        plugin_root,
        paths.get("mcp_servers"),
    )

    return PluginDescriptor(
        name=name,
        version=version,
        description=description,
        root=plugin_root,
        raw_manifest=raw,
        lifecycle_entry=lifecycle_entry,
        lifecycle_class=lifecycle_class,
        skill_roots=skill_roots,
        drift_skill_roots=drift_skill_roots,
        mcp_servers=mcp_servers,
    )


def _resolve_root_dirs(plugin_root: Path, raw_value: object) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for path_text in _as_str_list(raw_value):
        path = (plugin_root / path_text).resolve(strict=False)
        if path.is_dir():
            resolved.append(path)
    return tuple(resolved)


def _load_mcp_servers(
    plugin_root: Path,
    raw_value: object,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path_text in _as_str_list(raw_value):
        config_path = (plugin_root / path_text).resolve(strict=False)
        if not config_path.exists():
            continue
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("aka plugin mcp 配置读取失败 (%s): %s", config_path, e)
            continue
        if not isinstance(loaded, dict):
            logger.warning("aka plugin mcp 配置格式错误，期望 dict (%s)", config_path)
            continue
        servers = _as_dict(cast(dict[str, object], loaded).get("servers"))
        for server_name, server_value in servers.items():
            server = _normalize_server(plugin_root, server_name, server_value)
            if server is None:
                continue
            merged[server_name] = server
    return merged


def _normalize_server(
    plugin_root: Path,
    server_name: str,
    raw_value: object,
) -> dict[str, Any] | None:
    server = _as_dict(raw_value)
    command = _as_str_list(server.get("command"))
    if not command:
        logger.warning("aka plugin mcp server 缺少 command (%s)", server_name)
        return None
    normalized_command = [
        _normalize_command_item(plugin_root, item)
        for item in command
    ]
    env = {
        str(key): str(value)
        for key, value in _as_dict(server.get("env")).items()
    }
    cwd_raw = str(server.get("cwd") or "").strip()
    cwd = (
        str((plugin_root / cwd_raw).resolve(strict=False))
        if cwd_raw and not Path(cwd_raw).is_absolute()
        else cwd_raw or str(plugin_root.resolve(strict=False))
    )
    return {
        "command": normalized_command,
        "env": env,
        "cwd": cwd,
    }


def _resolve_optional_path(plugin_root: Path, value: str) -> Path | None:
    if not value:
        return None
    path = (plugin_root / value).resolve(strict=False)
    if not path.exists():
        logger.warning("aka plugin lifecycle 入口不存在 (%s)", path)
        return None
    return path


def _normalize_command_item(plugin_root: Path, value: str) -> str:
    if not value:
        return value
    if Path(value).is_absolute():
        return value
    if "/" not in value and "\\" not in value and not value.startswith("."):
        return value
    return str((plugin_root / value).resolve(strict=False))


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return result
