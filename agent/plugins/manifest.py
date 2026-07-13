from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Mapping, cast

from infra.persistence.json_store import atomic_write_text


def plugins_root(plugins_home: Path | None = None) -> Path:
    return plugins_home or Path.home() / ".akashic-plugin"


def manifest_path(plugins_home: Path | None = None) -> Path:
    return plugins_root(plugins_home) / "manifest.toml"


def builtin_plugin_data_dir(
    plugin_name: str,
    plugins_home: Path | None = None,
) -> Path:
    """返回内置插件的用户数据目录。"""

    return plugins_root(plugins_home) / "data" / f"{plugin_name}-builtin"


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


def set_package_enabled(
    package_id: str,
    *,
    enabled: bool,
    plugins_home: Path | None = None,
) -> Path:
    packages = load_package_manifest(plugins_home)
    if package_id not in packages:
        raise ValueError(f"插件包未安装: {package_id}")
    packages[package_id] = enabled
    return write_package_manifest(packages, plugins_home=plugins_home)


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
