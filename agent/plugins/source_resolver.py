from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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
    if not installed_cache_root.exists() and not installed_cache_root.is_symlink():
        return []
    if installed_cache_root.is_symlink():
        raise ValueError(f"installed cache root 不能是符号链接: {installed_cache_root}")
    if not installed_cache_root.is_dir():
        raise ValueError(f"installed cache root 不是目录: {installed_cache_root}")
    result: list[ResolvedPluginSource] = []
    for marketplace_dir in sorted(installed_cache_root.iterdir()):
        if marketplace_dir.name.startswith("."):
            continue
        _require_cache_directory(marketplace_dir, "marketplace")
        _require_safe_cache_segment(marketplace_dir, "marketplace")
        for plugin_dir in sorted(marketplace_dir.iterdir()):
            if plugin_dir.name.startswith("."):
                continue
            _require_cache_directory(plugin_dir, "plugin")
            _require_safe_cache_segment(plugin_dir, "plugin")
            version_dirs: list[Path] = []
            for child in sorted(plugin_dir.iterdir()):
                if child.name.startswith("."):
                    continue
                _require_safe_cache_segment(child, "version")
                _require_cache_directory(child, "version")
                version_dirs.append(child)
            if len(version_dirs) > 1:
                paths = ", ".join(str(path) for path in version_dirs)
                raise ValueError(f"installed cache 可见版本冲突: {paths}")
            if len(version_dirs) != 1:
                continue
            _require_plugin_root(version_dirs[0])
            result.append(
                ResolvedPluginSource(
                    plugin_root=version_dirs[0],
                    source_type="installed",
                    marketplace=marketplace_dir.name,
                )
            )
    return result


def _require_cache_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"installed cache {label} 不能是符号链接: {path}")
    if not path.is_dir():
        raise ValueError(f"installed cache {label} 不是目录: {path}")


def _require_safe_cache_segment(path: Path, label: str) -> None:
    if not _is_safe_cache_segment(path.name):
        raise ValueError(f"installed cache {label} 路径段无效: {path}")


def _require_plugin_root(path: Path) -> None:
    plugin_file = path / "plugin.py"
    if plugin_file.is_symlink():
        raise ValueError(f"installed cache plugin.py 不能是符号链接: {plugin_file}")
    if not plugin_file.is_file():
        raise ValueError(f"installed cache 缺少 plugin.py: {plugin_file}")


def _is_plugin_root(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    plugin_file = path / "plugin.py"
    return not plugin_file.is_symlink() and plugin_file.is_file()


def _is_safe_cache_segment(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is not None
