from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from agent.plugins.artifacts import ArtifactSelector, read_pointers, resolve_pointer
from agent.plugins.static_manifest import (
    StaticPluginManifest,
    load_static_plugin_manifest,
)


@dataclass(frozen=True)
class ResolvedPluginSource:
    plugin_root: Path
    source_type: Literal["builtin", "installed"]
    marketplace: str = ""
    plugin_name: str = ""
    entrypoint: str = "plugin.py"
    static_manifest: StaticPluginManifest | None = None


def resolve_plugin_sources(
    plugin_dirs: Sequence[Path] = (),
    *,
    installed_cache_root: Path | None = None,
    installed_selector: ArtifactSelector = "stable",
) -> list[ResolvedPluginSource]:
    discovered: list[ResolvedPluginSource] = []
    seen: set[Path] = set()
    if installed_cache_root is not None:
        for source in _iter_installed_plugin_roots(
            installed_cache_root,
            selector=installed_selector,
        ):
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
            static_manifest = _load_optional_static_manifest(normalized)
            discovered.append(
                ResolvedPluginSource(
                    plugin_root=normalized,
                    source_type="builtin",
                    plugin_name=(
                        static_manifest.name if static_manifest is not None else ""
                    ),
                    entrypoint=(
                        static_manifest.entrypoint
                        if static_manifest is not None
                        else "plugin.py"
                    ),
                    static_manifest=static_manifest,
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


def _iter_installed_plugin_roots(
    installed_cache_root: Path,
    *,
    selector: ArtifactSelector,
) -> list[ResolvedPluginSource]:
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
            has_pointers, selected = _resolve_installed_pointer(plugin_dir, selector)
            if has_pointers:
                if selected is not None:
                    static_manifest = _require_installed_plugin_root(selected)
                    _validate_installed_identity(
                        plugin_dir.name,
                        static_manifest,
                    )
                    result.append(
                        ResolvedPluginSource(
                            plugin_root=selected,
                            source_type="installed",
                            marketplace=marketplace_dir.name,
                            plugin_name=plugin_dir.name,
                            entrypoint=static_manifest.entrypoint,
                            static_manifest=static_manifest,
                        )
                    )
                continue
            visible = tuple(
                child
                for child in sorted(plugin_dir.iterdir())
                if not child.name.startswith(".")
            )
            if visible:
                paths = ", ".join(str(path) for path in visible)
                raise ValueError(
                    f"installed cache 含不受支持的旧版可见目录: {paths}"
                )
    return result


def _resolve_installed_pointer(
    plugin_dir: Path,
    selector: ArtifactSelector,
) -> tuple[bool, Path | None]:
    pointers = read_pointers(plugin_dir)
    if pointers is None:
        return False, None
    pointer = pointers.stable if selector == "stable" else pointers.latest
    return True, resolve_pointer(plugin_dir, pointer)


def _require_cache_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"installed cache {label} 不能是符号链接: {path}")
    if not path.is_dir():
        if not path.exists():
            raise FileNotFoundError(f"installed cache {label} 扫描期间已变化: {path}")
        raise ValueError(f"installed cache {label} 不是目录: {path}")


def _require_safe_cache_segment(path: Path, label: str) -> None:
    if not _is_safe_cache_segment(path.name):
        raise ValueError(f"installed cache {label} 路径段无效: {path}")


def _require_installed_plugin_root(path: Path) -> StaticPluginManifest:
    manifest_path = path / "akashic.plugin.toml"
    if manifest_path.exists() or manifest_path.is_symlink():
        return load_static_plugin_manifest(path)
    if not path.exists():
        raise FileNotFoundError(f"installed cache 版本扫描期间已变化: {path}")
    raise ValueError(f"installed cache 缺少静态 v3 manifest: {manifest_path}")


def _load_optional_static_manifest(path: Path) -> StaticPluginManifest | None:
    manifest_path = path / "akashic.plugin.toml"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None
    return load_static_plugin_manifest(path)


def _validate_installed_identity(
    cache_name: str,
    manifest: StaticPluginManifest,
) -> None:
    if manifest.name != cache_name:
        raise ValueError(
            "installed cache 插件目录与静态 manifest name 不一致: "
            f"directory={cache_name} manifest={manifest.name}"
        )


def _is_plugin_root(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    manifest_path = path / "akashic.plugin.toml"
    if manifest_path.exists() or manifest_path.is_symlink():
        _ = load_static_plugin_manifest(path)
        return True
    # Built-ins may keep the conventional plugin.py entrypoint without an install manifest.
    plugin_file = path / "plugin.py"
    return not plugin_file.is_symlink() and plugin_file.is_file()


def _is_safe_cache_segment(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is not None
