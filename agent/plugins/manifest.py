from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Mapping, cast


def plugins_root(plugins_home: Path | None = None) -> Path:
    return plugins_home or Path.home() / ".akashic-plugin"


def manifest_path(plugins_home: Path | None = None) -> Path:
    return plugins_root(plugins_home) / "manifest.toml"


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


def upsert_plugin_manifest(
    plugin_id: str,
    *,
    enabled: bool,
    plugins_home: Path | None = None,
) -> Path:
    entries = load_plugin_manifest(plugins_home)
    entries[plugin_id] = enabled
    return write_plugin_manifest(entries, plugins_home=plugins_home)


def write_plugin_manifest(
    entries: Mapping[str, bool],
    *,
    plugins_home: Path | None = None,
) -> Path:
    path = manifest_path(plugins_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for plugin_id, enabled in sorted(entries.items()):
        escaped = plugin_id.replace("\\", "\\\\").replace('"', '\\"')
        lines.extend(
            [
                f'[plugins."{escaped}"]',
                f"enabled = {'true' if enabled else 'false'}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
