from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.static_manifest import (
    PluginSourceContentError,
    StaticPluginManifest,
    load_static_plugin_manifest,
    source_error_details,
)


@dataclass(frozen=True)
class ResolvedPluginSource:
    plugin_root: Path
    source_type: Literal["builtin", "installed"]
    marketplace: str = ""
    plugin_name: str = ""
    static_manifest: StaticPluginManifest | None = None


@dataclass(frozen=True)
class PluginSourceFailure:
    """A source-local diagnostic without a runtime or durable owner."""

    source_root: Path
    source_type: Literal["builtin", "installed"]
    phase: str
    error_type: str
    error_text: str
    plugin_id: str | None = None


@dataclass(frozen=True)
class PluginSourceScan:
    """Pure source scan output; Manager owns any retained diagnostics."""

    sources: tuple[ResolvedPluginSource, ...]
    failures: tuple[PluginSourceFailure, ...]


def resolve_plugin_sources(
    plugin_dirs: Sequence[Path] = (),
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
            static_manifest = load_static_plugin_manifest(normalized)
            discovered.append(
                ResolvedPluginSource(
                    plugin_root=normalized,
                    source_type="builtin",
                    plugin_name=static_manifest.name,
                    static_manifest=static_manifest,
                )
            )
    return discovered


def scan_plugin_sources(
    plugin_dirs: Sequence[Path] = (),
    *,
    installed_cache_root: Path | None = None,
) -> PluginSourceScan:
    """Scan source content without weakening shared boundary failures."""

    raw: list[ResolvedPluginSource] = []
    if installed_cache_root is not None:
        raw.extend(
            _iter_installed_plugin_roots(
                installed_cache_root,
                load_manifests=False,
            )
        )
    for root in plugin_dirs:
        raw.extend(
            ResolvedPluginSource(
                plugin_root=plugin_root.resolve(strict=False),
                source_type="builtin",
            )
            for plugin_root in _iter_declared_plugin_roots(root)
        )

    sources: list[ResolvedPluginSource] = []
    failures: list[PluginSourceFailure] = []
    seen: set[Path] = set()
    for source in raw:
        normalized = source.plugin_root.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            static_manifest = load_static_plugin_manifest(normalized)
            if source.source_type == "installed":
                _validate_installed_identity(source.plugin_name, static_manifest)
        except PluginSourceContentError as error:
            error_type, error_text = source_error_details(error)
            failures.append(
                PluginSourceFailure(
                    source_root=normalized,
                    source_type=source.source_type,
                    phase="identity",
                    error_type=error_type,
                    error_text=error_text,
                )
            )
            continue
        except FileNotFoundError as error:
            if source.source_type == "installed":
                raise
            if normalized.exists() and (normalized / "plugin.py").exists():
                raise
            failures.append(
                PluginSourceFailure(
                    source_root=normalized,
                    source_type=source.source_type,
                    phase="source",
                    error_type="SourceUnavailable",
                    error_text=str(error) or "插件源码目录暂时不可用",
                )
            )
            continue
        sources.append(
            ResolvedPluginSource(
                plugin_root=normalized,
                source_type=source.source_type,
                marketplace=source.marketplace,
                plugin_name=source.plugin_name or static_manifest.name,
                static_manifest=static_manifest,
            )
        )
    return PluginSourceScan(tuple(sources), tuple(failures))


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
    load_manifests: bool = True,
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
            has_pointers, selected = _resolve_installed_pointer(
                plugin_dir,
                validate_content=load_manifests,
            )
            if has_pointers:
                if selected is not None:
                    static_manifest = None
                    if load_manifests:
                        static_manifest = load_static_plugin_manifest(selected)
                        _validate_installed_identity(plugin_dir.name, static_manifest)
                    result.append(
                        ResolvedPluginSource(
                            plugin_root=selected,
                            source_type="installed",
                            marketplace=marketplace_dir.name,
                            plugin_name=plugin_dir.name,
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
    *,
    validate_content: bool = True,
) -> tuple[bool, Path | None]:
    pointers = read_pointers(plugin_dir, validate_content=validate_content)
    if pointers is None:
        return False, None
    if pointers.stable != pointers.latest:
        raise RuntimeError(
            f"插件仍有历史候选指针对，须先处理未决更新: {plugin_dir}"
        )
    pointer = pointers.stable
    return True, resolve_pointer(
        plugin_dir, pointer, validate_content=validate_content,
    )


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
    plugin_file = path / "plugin.py"
    if plugin_file.is_symlink():
        raise ValueError(f"插件 plugin.py 不能是符号链接: {plugin_file}")
    return plugin_file.is_file()


def _is_safe_cache_segment(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is not None
