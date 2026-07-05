from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast


def aka_plugins_root(plugins_home: Path | None = None) -> Path:
    return plugins_home or (Path.home() / ".akashic-plugin")


def global_registry_path(plugins_home: Path | None = None) -> Path:
    return aka_plugins_root(plugins_home) / "registry.json"


def load_plugin_registry(
    plugins_home: Path | None = None,
) -> dict[str, dict[str, object]]:
    path = global_registry_path(plugins_home)
    return _load_plugins(_load_registry(path))


def upsert_plugin_registry_entry(
    plugin_id: str,
    entry: Mapping[str, object],
    *,
    plugins_home: Path | None = None,
) -> Path:
    path = global_registry_path(plugins_home)
    current = _load_registry(path)
    plugins = _load_plugins(current)
    merged = dict(plugins.get(plugin_id, {}))
    merged.update(dict(entry))
    plugins[plugin_id] = merged
    return _write_registry(path, plugins)


def replace_plugin_registry(
    entries: Mapping[str, Mapping[str, object]],
    *,
    plugins_home: Path | None = None,
) -> Path:
    path = global_registry_path(plugins_home)
    current = _load_registry(path)
    previous = _load_plugins(current)
    plugins: dict[str, dict[str, object]] = {}
    for plugin_id, entry in sorted(entries.items()):
        merged = dict(previous.get(plugin_id, {}))
        merged.update(dict(entry))
        plugins[plugin_id] = merged
    return _write_registry(path, plugins)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded_obj = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded_obj, dict):
        return {}
    loaded = cast(dict[object, object], loaded_obj)
    return {
        str(key): value
        for key, value in loaded.items()
    }


def _load_plugins(data: Mapping[str, Any]) -> dict[str, dict[str, object]]:
    raw_plugins = data.get("plugins")
    if not isinstance(raw_plugins, dict):
        return {}
    result: dict[str, dict[str, object]] = {}
    for raw_plugin_id, raw_entry in cast(dict[object, object], raw_plugins).items():
        if not isinstance(raw_plugin_id, str) or not isinstance(raw_entry, dict):
            continue
        entry = cast(dict[object, object], raw_entry)
        result[raw_plugin_id] = {
            str(key): value
            for key, value in entry.items()
        }
    return result


def _write_registry(
    path: Path,
    plugins: Mapping[str, Mapping[str, object]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "plugins_home": str(path.parent),
        "plugins": {
            plugin_id: dict(entry)
            for plugin_id, entry in sorted(plugins.items())
        },
    }
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
