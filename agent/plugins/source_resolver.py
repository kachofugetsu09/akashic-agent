from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_AKA_PLUGIN_FILE = ".aka-plugin/plugin.json"


@dataclass(frozen=True)
class ResolvedPluginSource:
    plugin_root: Path
    source_type: Literal["builtin", "installed"]
    marketplace: str = ""


def resolve_plugin_sources(
    plugin_dirs: list[Path],
    *,
    installed_cache_root: Path | None = None,
) -> list[ResolvedPluginSource]:
    discovered: list[ResolvedPluginSource] = []
    seen: set[Path] = set()
    if installed_cache_root is not None:
        for source in _iter_installed_plugin_roots(installed_cache_root):
            normalized = source.plugin_root.resolve(strict=False)
            if normalized in seen:
                continue
            seen.add(normalized)
            discovered.append(source)
    for root in plugin_dirs:
        for plugin_root in _iter_declared_plugin_roots(root):
            normalized = plugin_root.resolve(strict=False)
            if normalized in seen:
                continue
            seen.add(normalized)
            discovered.append(
                ResolvedPluginSource(
                    plugin_root=normalized,
                    source_type="builtin",
                )
            )
    return discovered


def _iter_declared_plugin_roots(root: Path) -> list[Path]:
    if _is_plugin_root(root):
        return [root]
    if not root.is_dir():
        return []
    result: list[Path] = []
    for child in sorted(root.iterdir()):
        if _is_plugin_root(child):
            result.append(child)
    return result


def _iter_installed_plugin_roots(installed_cache_root: Path) -> list[ResolvedPluginSource]:
    if not installed_cache_root.is_dir():
        return []
    result: list[ResolvedPluginSource] = []
    for marketplace_dir in sorted(installed_cache_root.iterdir()):
        if not marketplace_dir.is_dir():
            continue
        for plugin_dir in sorted(marketplace_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue
            version_dirs = [
                child for child in sorted(plugin_dir.iterdir())
                if child.is_dir() and _is_plugin_root(child)
            ]
            if not version_dirs:
                continue
            result.append(
                ResolvedPluginSource(
                    plugin_root=version_dirs[-1],
                    source_type="installed",
                    marketplace=marketplace_dir.name,
                )
            )
    return result


def _is_plugin_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "plugin.py").exists() or (path / _AKA_PLUGIN_FILE).exists()
