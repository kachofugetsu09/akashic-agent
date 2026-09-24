"""Build one fixed plugin input before a runtime generation exists."""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from agent.plugin_composition.config_input import load_config
from agent.plugins.archive import PluginArchive, encode_config
from agent.plugins.manifest import (
    validate_workspace_plugin_data_path,
    workspace_plugin_data_dir,
)
from agent.plugins.python_environment import ENVIRONMENT_FILE, read_environment_refs
from agent.plugins.static_manifest import (
    PluginSourceCompileError,
    PluginSourceContentError,
    StaticPluginManifest,
    load_static_plugin_manifest,
)

PLUGIN_ARCHIVE_BINDING_API = 3


@dataclass(frozen=True, slots=True)
class PreparedPluginInput:
    """The archived input and source facts used by one runtime generation."""

    plugin_id: str
    archive_ref: str
    source_revision: str
    config_revision: str
    config: dict[str, object]
    static_manifest: StaticPluginManifest
    plugin_dir: Path
    code_dir: Path
    data_dir: Path
    source_type: Literal["builtin", "installed"]


def prepare_plugin_input(
    mod: Mapping[str, str], *, workspace: Path, archive: PluginArchive,
) -> PreparedPluginInput:
    """Check, compile, and archive one source without loading its module."""

    # 1. Check the discovered source identity and read its current config.
    plugin_id = _resolve_plugin_id(mod)
    plugin_dir = Path(mod["plugin_root"])
    entry = plugin_dir / "plugin.py"
    if Path(mod["module_path"]).absolute() != entry.absolute():
        raise RuntimeError("source discovery module path 与制品 plugin.py 不一致")
    if entry.is_symlink() or not entry.is_file():
        raise ValueError(f"插件 plugin.py 必须是普通文件: {entry}")
    identity = load_static_plugin_manifest(plugin_dir)
    if mod.get("manifest_digest", "") != identity.identity_digest:
        raise RuntimeError("source discovery identity 已漂移")
    revision = _source_revision(plugin_dir)
    data_dir = _resolve_plugin_data_dir(mod["name"], mod, workspace)
    validate_workspace_plugin_data_path(data_dir, workspace)
    config, config_revision = load_config(data_dir)

    # 2. Fix the code tree, check its identity, and compile every Python file.
    code_ref = archive.save(
        plugin_dir, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE}),
    )
    code_dir = archive.open(code_ref)
    try:
        archived_identity = load_static_plugin_manifest(code_dir)
    except PluginSourceContentError as error:
        raise RuntimeError("插件归档静态身份损坏") from error
    if _source_revision(code_dir) != revision or archived_identity != identity:
        raise RuntimeError("插件源码或安装身份在归档前发生变化")
    for source_path in sorted(code_dir.rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "node_modules"} for part in source_path.parts):
            continue
        try:
            source_text = source_path.read_text(encoding="utf-8")
            compile(source_text, str(source_path), "exec")
        except (SyntaxError, UnicodeError) as error:
            raise PluginSourceCompileError(
                f"插件源码无法编译: {source_path}"
            ) from error

    # 3. Save the same v4 descriptor after all source checks succeed.
    environments: dict[str, str] = {}
    if identity.python:
        if (plugin_dir / ENVIRONMENT_FILE).exists():
            environments = read_environment_refs(plugin_dir, identity)
        elif mod["source_type"] == "installed":
            raise RuntimeError("插件尚未准备固定 Python 环境；请通过安装流程重建")
    ref = archive.save_descriptor({
        "version": 4, "code": code_ref, "python_environments": environments,
        "plugin_id": plugin_id, "source_revision": revision,
        "config_revision": config_revision, "config": encode_config(config),
        "source_type": mod["source_type"],
        "data_dir": data_dir.resolve().relative_to(workspace.resolve()).as_posix(),
        "runtime": {"python_tag": sys.implementation.cache_tag, "binding_api": PLUGIN_ARCHIVE_BINDING_API},
    })
    return PreparedPluginInput(
        plugin_id=plugin_id, archive_ref=ref,
        source_revision=revision, config_revision=config_revision, config=config,
        static_manifest=identity, plugin_dir=plugin_dir, code_dir=code_dir,
        data_dir=data_dir,
        source_type=cast(Literal["builtin", "installed"], mod["source_type"]),
    )


def _resolve_plugin_id(mod: Mapping[str, str]) -> str:
    name = mod["name"]
    marketplace = mod.get("marketplace", "").strip()
    if not marketplace:
        return name
    return f"{name}@{marketplace}"


def _resolve_plugin_data_dir(
    name: str, mod: Mapping[str, str], workspace: Path,
) -> Path:
    """Use the plugin's existing private data path in this workspace."""
    marketplace = mod.get("marketplace", "").strip()
    suffix = marketplace or "builtin"
    return workspace_plugin_data_dir(workspace, name, suffix)


def _require_plugin_path(plugin_dir: Path, path: Path, label: str) -> None:
    try:
        _ = path.relative_to(plugin_dir)
    except ValueError as error:
        raise RuntimeError(f"插件 {label} 越界: {path}") from error


def _source_revision(plugin_dir: Path) -> str:
    """Hash the plugin source tree with the existing exclusion and path rules."""
    digest = hashlib.sha256()
    root = plugin_dir.resolve(strict=False)
    excluded = {
        ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".venv", "__pycache__", "node_modules", ENVIRONMENT_FILE,
    }
    for current, directories, filenames in os.walk(plugin_dir, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in excluded)
        current_path = Path(current)
        for name in [*directories, *sorted(filenames)]:
            if name in excluded:
                continue
            path = current_path / name
            relative = path.relative_to(plugin_dir)
            if path.is_symlink():
                resolved = path.resolve(strict=False)
                _require_plugin_path(root, resolved, "源码符号链接")
                digest.update(str(relative).encode())
                digest.update(os.readlink(path).encode())
                if resolved.is_file():
                    digest.update(resolved.read_bytes())
                continue
            if not path.is_file():
                continue
            resolved = path.resolve(strict=False)
            _require_plugin_path(root, resolved, "源码文件")
            digest.update(str(relative).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
