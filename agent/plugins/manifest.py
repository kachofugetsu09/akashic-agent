from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Mapping, cast

from infra.persistence.json_store import atomic_write_text


def plugins_root(plugins_home: Path | None = None) -> Path:
    if plugins_home is not None:
        return plugins_home
    configured = os.environ.get("AKASHIC_PLUGIN_HOME")
    if configured is not None:
        configured = configured.strip()
        if not configured:
            raise ValueError("AKASHIC_PLUGIN_HOME 不能为空")
        return Path(configured).expanduser().resolve(strict=False)
    return Path.home() / ".akashic-plugin"


def manifest_path(plugins_home: Path | None = None) -> Path:
    return plugins_root(plugins_home) / "manifest.toml"


def builtin_plugin_data_dir(
    plugin_name: str,
    workspace: Path,
) -> Path:
    """返回当前 workspace 内的内置插件数据目录。"""

    return workspace_plugin_data_dir(workspace, plugin_name, "builtin")


def workspace_plugin_data_dir(
    workspace: Path,
    plugin_name: str,
    marketplace: str,
) -> Path:
    """解析 workspace 内的插件数据目录，不创建或迁移数据。"""

    # 1. 插件身份只能形成单一路径段
    for label, value in (("name", plugin_name), ("marketplace", marketplace)):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
            raise ValueError(f"插件 {label} 不是安全路径段: {value!r}")

    # 2. 数据根始终归属于显式 workspace
    return workspace.resolve(strict=False) / "plugin-data" / f"{plugin_name}-{marketplace}"


def ensure_workspace_plugin_data_dir(path: Path, workspace: Path) -> None:
    """安全创建 workspace 内的数据目录，并拒绝中间符号链接。"""

    validate_workspace_plugin_data_path(path, workspace)
    path.mkdir(parents=True, exist_ok=True)
    validate_workspace_plugin_data_path(path, workspace)


def validate_workspace_plugin_data_path(path: Path, workspace: Path) -> None:
    """校验插件数据路径归属 workspace，且现有路径不穿过符号链接。"""

    # 1. 校验目标仍归属于 workspace
    root = workspace.resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"插件数据目录越界: {path}") from error

    # 2. 逐级拒绝现有符号链接
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"插件数据目录不能穿过符号链接: {current}")


def load_plugin_manifest(
    plugins_home: Path | None = None,
) -> dict[str, bool]:
    path = manifest_path(plugins_home)
    if not path.exists():
        return {}
    loaded = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_plugins = loaded.get("plugins")
    if not isinstance(raw_plugins, dict):
        raise ValueError("manifest.toml 缺少 [plugins] 配置")
    result: dict[str, bool] = {}
    for plugin_id, raw_entry in cast(dict[object, object], raw_plugins).items():
        if not isinstance(plugin_id, str) or not isinstance(raw_entry, dict):
            raise ValueError("manifest.toml 插件条目格式错误")
        enabled = cast(dict[object, object], raw_entry).get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(f"manifest.toml 插件缺少 enabled: {plugin_id}")
        result[plugin_id] = enabled
    return result


def load_package_manifest(
    plugins_home: Path | None = None,
) -> dict[str, bool]:
    path = manifest_path(plugins_home)
    if not path.exists():
        return {}
    loaded = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_packages = loaded.get("packages", {})
    if not isinstance(raw_packages, dict):
        raise ValueError("manifest.toml [packages] 配置格式错误")
    result: dict[str, bool] = {}
    for package_id, raw_entry in cast(dict[object, object], raw_packages).items():
        if not isinstance(package_id, str) or not isinstance(raw_entry, dict):
            raise ValueError("manifest.toml 插件包条目格式错误")
        enabled = cast(dict[object, object], raw_entry).get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(f"manifest.toml 插件包缺少 enabled: {package_id}")
        result[package_id] = enabled
    return result


def upsert_plugin_manifest(
    plugin_id: str,
    *,
    enabled: bool,
    plugins_home: Path | None = None,
) -> Path:
    entries = load_plugin_manifest(plugins_home)
    entries[plugin_id] = enabled
    return write_plugin_manifest(entries, plugins_home=plugins_home)


def set_plugin_enabled(
    plugin_id: str,
    *,
    enabled: bool,
    plugins_home: Path | None = None,
) -> Path:
    entries = load_plugin_manifest(plugins_home)
    if plugin_id not in entries:
        raise ValueError(f"插件未安装: {plugin_id}")
    entries[plugin_id] = enabled
    return write_plugin_manifest(entries, plugins_home=plugins_home)


def remove_plugin_manifest_entry(
    plugin_id: str,
    *,
    plugins_home: Path | None = None,
) -> Path:
    entries = load_plugin_manifest(plugins_home)
    if plugin_id not in entries:
        raise ValueError(f"插件未安装: {plugin_id}")
    del entries[plugin_id]
    return write_plugin_manifest(entries, plugins_home=plugins_home)


def write_package_manifest(
    packages: Mapping[str, bool],
    *,
    plugins_home: Path | None = None,
) -> Path:
    plugins = load_plugin_manifest(plugins_home)
    path = manifest_path(plugins_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[plugins]", ""]
    for plugin_id, enabled in sorted(plugins.items()):
        escaped = plugin_id.replace("\\", "\\\\").replace('"', '\\"')
        lines.extend([
            f'[plugins."{escaped}"]',
            f"enabled = {'true' if enabled else 'false'}",
            "",
        ])
    lines.extend(["[packages]", ""])
    for package_id, enabled in sorted(packages.items()):
        escaped = package_id.replace("\\", "\\\\").replace('"', '\\"')
        lines.extend([
            f'[packages."{escaped}"]',
            f"enabled = {'true' if enabled else 'false'}",
            "",
        ])
    return _atomic_write(path, "\n".join(lines))


def write_plugin_manifest(
    entries: Mapping[str, bool],
    *,
    plugins_home: Path | None = None,
) -> Path:
    path = manifest_path(plugins_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    packages = load_package_manifest(plugins_home)
    lines = ["[plugins]", ""]
    for plugin_id, enabled in sorted(entries.items()):
        escaped = plugin_id.replace("\\", "\\\\").replace('"', '\\"')
        lines.extend(
            [
                f'[plugins."{escaped}"]',
                f"enabled = {'true' if enabled else 'false'}",
                "",
            ]
        )
    if packages:
        lines.extend(["[packages]", ""])
        for package_id, enabled in sorted(packages.items()):
            escaped = package_id.replace("\\", "\\\\").replace('"', '\\"')
            lines.extend(
                [
                    f'[packages."{escaped}"]',
                    f"enabled = {'true' if enabled else 'false'}",
                    "",
                ]
            )
    content = "\n".join(lines)
    return _atomic_write(path, content)


def _atomic_write(path: Path, content: str) -> Path:
    atomic_write_text(path, content, domain="plugin_manifest")
    return path
