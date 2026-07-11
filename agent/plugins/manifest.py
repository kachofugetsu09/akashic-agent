from __future__ import annotations

import os
import tempfile
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


def write_plugin_manifest(
    entries: Mapping[str, bool],
    *,
    plugins_home: Path | None = None,
) -> Path:
    path = manifest_path(plugins_home)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    content = "\n".join(lines)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix="manifest-",
        suffix=".toml",
        delete=False,
    ) as stream:
        _ = stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)
    return path
