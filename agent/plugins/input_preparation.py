"""准备当前安装输入，不复制插件运行代码。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from agent.plugin_composition.config_input import load_config, config_bytes, CONFIG_INPUT
from agent.plugins.selection import PluginSelection
from agent.plugins.manifest import (
    validate_workspace_plugin_data_path,
    workspace_plugin_data_dir,
)
from agent.plugins.python_environment import ENVIRONMENT_FILE, PythonEnvironments, read_environment_refs
from agent.plugins.static_manifest import (
    PluginSourceCompileError,
    StaticPluginManifest,
    load_static_plugin_manifest,
)

PLUGIN_INPUT_API = 3
# watcher 与运行变化比较共用排除规则；安装来源摘要仍包含来源标签。
SOURCE_EXCLUDED_NAMES = frozenset({
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "node_modules", ENVIRONMENT_FILE, ".akashic-source.json",
})
_SOURCE_LABEL = ".akashic-source.json"


@dataclass(frozen=True, slots=True)
class PreparedPluginInput:
    """一次加载的安装路径、配置与来源事实。"""

    plugin_id: str
    input_ref: str
    source_revision: str
    runtime_revision: str
    config_revision: str
    config: dict[str, object]
    static_manifest: StaticPluginManifest
    plugin_dir: Path
    code_dir: Path
    data_dir: Path
    source_type: Literal["builtin", "installed"]


def prepare_plugin_input(
    mod: Mapping[str, str], *, workspace: Path, selection: PluginSelection, initial: bool = False,
) -> PreparedPluginInput:
    """校验并编译已安装源码，不提前导入模块。"""

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
    revision, runtime_revision = source_revisions(plugin_dir)
    data_dir = _resolve_plugin_data_dir(mod["name"], mod, workspace)
    validate_workspace_plugin_data_path(data_dir, workspace)
    config, config_revision = load_config(data_dir)
    defaults = plugin_dir / "initial_config.json"
    if initial and not (data_dir / CONFIG_INPUT).exists() and defaults.exists():
        if defaults.is_symlink():
            raise ValueError("初始配置不能是符号链接")
        config = json.loads(defaults.read_bytes())
        if not isinstance(config, dict):
            raise ValueError("初始配置必须是映射")
        config_revision = hashlib.sha256(config_bytes(config)).hexdigest()

    # 2. 直接编译安装目录；外部环境变化不会被另一个代码快照隐藏。
    code_dir = plugin_dir.resolve(strict=True)
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

    # 3. 准备当前选择的元数据；代码字段只表示实际安装路径。
    environments: dict[str, str] = {}
    if identity.python:
        if (plugin_dir / ENVIRONMENT_FILE).exists():
            environments = read_environment_refs(plugin_dir, identity)
        elif mod["source_type"] == "installed":
            raise RuntimeError("插件尚未准备固定 Python 环境；请通过安装流程重建")
        elif "distribution_source" in mod:
            owner = PythonEnvironments(workspace)
            environments = {
                runtime.runtime_root: owner.prepared(
                    plugin_dir, runtime,
                    wheel_digest=mod.get("wheel_tree_sha256", "")
                    if (plugin_dir / runtime.requirements).read_text().strip() else "",
                ) for runtime in identity.python
            }
    ref = selection.prepare({
        "version": 6, "code": str(code_dir), "python_environments": environments,
        "entrypoints": dict(identity.entrypoints),
        "plugin_id": plugin_id, "source_revision": revision,
        "config_revision": config_revision,
        "source_type": mod["source_type"],
        **({"distribution_source": mod["distribution_source"]} if "distribution_source" in mod else {}),
        "data_dir": data_dir.resolve().relative_to(workspace.resolve()).as_posix(),
        "runtime": {"python_tag": sys.implementation.cache_tag, "binding_api": PLUGIN_INPUT_API},
    })
    return PreparedPluginInput(
        plugin_id=plugin_id, input_ref=ref,
        source_revision=revision, runtime_revision=runtime_revision,
        config_revision=config_revision, config=config,
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


def _source_revision(plugin_dir: Path, *, runtime_only: bool = False) -> str:
    """校验完整源码；仅在比较运行变化时忽略来源标签。"""
    (digest,) = _digest_tree(plugin_dir, (not runtime_only,))
    return digest


# 一次遍历同时得到完整摘要与忽略来源标签的运行摘要。
def source_revisions(plugin_dir: Path) -> tuple[str, str]:
    source, runtime = _digest_tree(plugin_dir, (True, False))
    return source, runtime


# 每个摘要按是否包含来源标签取条目；喂入的字节与逐个单独遍历完全相同。
def _digest_tree(plugin_dir: Path, include_label: tuple[bool, ...]) -> tuple[str, ...]:
    digests = tuple(hashlib.sha256() for _ in include_label)
    labelled = tuple(digest for digest, include in zip(digests, include_label) if include)
    root = plugin_dir.resolve(strict=False)
    excluded = SOURCE_EXCLUDED_NAMES - {_SOURCE_LABEL} if labelled else SOURCE_EXCLUDED_NAMES
    for current, directories, filenames in os.walk(plugin_dir, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in excluded)
        current_path = Path(current)
        for name in [*directories, *sorted(filenames)]:
            if name in excluded:
                continue
            path = current_path / name
            relative = path.relative_to(plugin_dir)
            targets = labelled if _SOURCE_LABEL in relative.parts else digests
            for chunk in _source_entry_chunks(root, path, relative):
                for digest in targets:
                    digest.update(chunk)
    return tuple(digest.hexdigest() for digest in digests)


# 单个源码条目参与摘要的字节；越界路径在读取前失败。
def _source_entry_chunks(root: Path, path: Path, relative: Path) -> tuple[bytes, ...]:
    if path.is_symlink():
        resolved = path.resolve(strict=False)
        _require_plugin_path(root, resolved, "源码符号链接")
        link = (str(relative).encode(), os.readlink(path).encode())
        return (*link, resolved.read_bytes()) if resolved.is_file() else link
    if not path.is_file():
        return ()
    resolved = path.resolve(strict=False)
    _require_plugin_path(root, resolved, "源码文件")
    return str(relative).encode(), path.read_bytes()
