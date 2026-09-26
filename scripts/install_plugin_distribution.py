#!/usr/bin/env python3
"""验证 Core/插件发布身份，并按显式 profile 走正式 Git 插件安装链。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from typing import Any


_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from agent.plugins.install import install_git_plugin
from agent.migrations.release_backup import backup_release_state
from agent.migrations.runner import MigrationRunner
from agent.plugins.source_resolver import ResolvedPluginSource
from agent.plugin_composition.archive import encode_tree, sync_directory, tree_entries
from agent.plugin_composition.config_input import CONFIG_INPUT, load_config, save_config
from agent.migrations.runner import initialize_empty_workspace
from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.manifest import load_plugin_manifest, workspace_plugin_data_dir
from agent.plugins.static_manifest import load_static_plugin_manifest
from agent.plugins.python_environment import ENVIRONMENT_FILE
from agent.plugins.python_environment import OfflineWheels, preflight_offline_runtime, wheel_tree_sha256
from agent.plugins.input_preparation import prepare_plugin_input, _source_revision
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.selection import PluginSelection, SelectionConflictError
from agent.migrations.bundles import validate_migration_artifact
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceMaintenanceLock

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_FORMAL_INSTALLER = "agent.plugins.install.install_git_plugin"


def _external_path(root: Path, relative: object, *, directory: bool) -> Path:
    """Resolve a plain POSIX input path without following a link."""

    if (not isinstance(relative, str) or not relative or "\\" in relative
        or "\x00" in relative or relative.startswith("/")):
        raise ValueError("external input 必须是安全的 POSIX 相对路径")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("external input 包含不安全的路径段")
    current = root
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("external input root 必须是普通目录")
    for part in parts:
        current /= part
        mode = current.lstat().st_mode
        if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
            raise ValueError(f"external input 不能包含链接或特殊文件: {current}")
    mode = current.lstat().st_mode
    if directory != stat.S_ISDIR(mode):
        raise ValueError(f"external input 类型不符: {current}")
    return current


def load_deployment_plan(path: Path) -> tuple[str, str, list[dict[str, Any]], tuple[str, ...]]:
    """校验部署者指定的精确目标与迁移清单。"""
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"部署清单 JSON key 重复: {key}")
            result[key] = value
        return result

    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("部署清单必须是普通文件")
    raw = path.read_bytes()
    document = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(document, dict) or set(document) != {"schema_version", "expected_root_ref", "targets", "migrations"}:
        raise ValueError("部署清单需要 schema_version、expected_root_ref、targets、migrations")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("部署清单 schema_version 错误")
    root_ref = document["expected_root_ref"]
    if not isinstance(root_ref, str) or _SHA256.fullmatch(root_ref) is None:
        raise ValueError("expected_root_ref 必须是完整 Root SHA-256")
    migrations = document["migrations"]
    if (not isinstance(migrations, list)
        or any(not isinstance(item, str) or not item or item.strip() != item for item in migrations)
        or len(set(migrations)) != len(migrations)):
        raise ValueError("migrations 必须是无重复的 migration ID 数组")
    targets = document["targets"]
    if not isinstance(targets, list):
        raise ValueError("targets 必须是数组；空数组保留全部插件")
    seen: set[str] = set()
    for item in targets:
        if not isinstance(item, dict):
            raise ValueError("target 必须是对象")
        plugin_id = item.get("plugin_id")
        if (not isinstance(plugin_id, str) or plugin_id.count("@") != 1
            or any(_PATH_SEGMENT.fullmatch(part) is None for part in plugin_id.split("@"))
            or plugin_id in seen):
            raise ValueError(f"plugin_id 重复或无效: {plugin_id}")
        seen.add(plugin_id)
        fields = set(item) - {"offline_wheels"}
        if fields == {"plugin_id", "bundled"}:
            if item["bundled"] is not True:
                raise ValueError("bundled 必须为 true")
        elif fields == {"plugin_id", "bundle_relative_path", "bundle_sha256", "target_commit"}:
            if not isinstance(item["bundle_sha256"], str) or _SHA256.fullmatch(item["bundle_sha256"]) is None:
                raise ValueError(f"bundle_sha256 无效: {plugin_id}")
            if not isinstance(item["target_commit"], str) or _REVISION.fullmatch(item["target_commit"]) is None:
                raise ValueError(f"target_commit 无效: {plugin_id}")
            _external_path_syntax(item["bundle_relative_path"])
        else:
            raise ValueError(f"target 字段错误: {plugin_id}")
        if "offline_wheels" in item:
            wheels = item["offline_wheels"]
            if (not isinstance(wheels, dict) or set(wheels) != {"relative_path", "tree_sha256"}
                or not isinstance(wheels["tree_sha256"], str)
                or _SHA256.fullmatch(wheels["tree_sha256"]) is None):
                raise ValueError(f"offline_wheels 无效: {plugin_id}")
            _external_path_syntax(wheels["relative_path"])
    return hashlib.sha256(raw).hexdigest(), root_ref, targets, tuple(migrations)


def _external_path_syntax(relative: object) -> str:
    if (not isinstance(relative, str) or not relative or relative.startswith("/")
        or "\\" in relative or "\x00" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("external input 路径格式错误")
    return relative


def _stage_deployment_targets(
    *, targets: list[dict[str, Any]], distribution: Path, inputs: Path, staged: Path, expected_root_ref: str,
    selection: PluginSelection, workspace: Path, plugins_home: Path,
) -> list[dict[str, Any]]:
    """在临时目录固定 bundle/wheels，并核对每个显式目标。"""

    # 1. Require each target to own a coherent selected installed input.
    report = verify_distribution(distribution)
    bundled = {row["name"]: row for row in report["plugins"]}
    _, selected = _selected_components(selection, expected_root_ref)
    manifest = load_plugin_manifest(plugins_home)
    result: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        plugin_id = target["plugin_id"]
        if plugin_id not in selected or manifest.get(plugin_id) is not True:
            raise SelectionConflictError(f"external target 未选择或未启用: {plugin_id}")
        old_ref, descriptor, old_code = selected[plugin_id]
        name, marketplace = plugin_id.split("@")
        if (descriptor["source_type"] != "installed"
            or descriptor.get("data_dir") != workspace_plugin_data_dir(workspace, name, marketplace).relative_to(workspace).as_posix()):
            raise SelectionConflictError(f"external target 非已安装输入: {plugin_id}")
        artifact, current_code, current_source = _current_artifact(
            workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id,
        )
        # 2. Copy fixed bundle bytes into tmpfs and resolve its exact commit.
        if target.get("bundled"):
            row = bundled.get(name)
            if row is None:
                raise ValueError(f"目标 distribution 不包含插件: {name}")
            bundle = _distribution_file(distribution, row["file"], plugin_id)
            commit, bundle_digest = row["source_revision"], row["sha256"]
        else:
            bundle = _external_path(inputs, target["bundle_relative_path"], directory=False)
            commit, bundle_digest = target["target_commit"], target["bundle_sha256"]
        _check_sha256(bundle, bundle_digest, f"{plugin_id} bundle")
        local = staged / f"external-{index}.bundle"
        shutil.copyfile(bundle, local)
        _check_sha256(local, bundle_digest, f"staged {plugin_id} bundle")
        code = staged / f"external-{index}-code"
        _ = _git("clone", "--no-local", "--no-checkout", str(local), str(code))
        _ = _git("-C", str(code), "checkout", "--detach", commit)
        if _git("-C", str(code), "rev-parse", "HEAD") != commit:
            raise ValueError(f"external target commit 不一致: {plugin_id}")
        if target.get("bundled") and _provenance(code) != {
            "commit": report["source_commit"], "path": row["source_path"],
        }:
            raise ValueError(f"bundle provenance 与 distribution 不符: {plugin_id}")
        # 已安装目标版本可继续准备输入；第三种 cache 状态必须显式修复。
        target_code = _code_identity(code)
        same_current = descriptor["code"] == current_code and _provenance(old_code) == current_source
        installed_target = current_code == target_code and _git("-C", str(artifact), "rev-parse", "HEAD") == commit
        if not same_current and not installed_target:
            raise SelectionConflictError(f"selection/cache 既不是原输入也不是本次目标: {plugin_id}")
        identity = load_static_plugin_manifest(code)
        if identity.name != name:
            raise ValueError(f"external target 静态身份不一致: {plugin_id}")
        _ = validate_migration_artifact(code, static_manifest=identity)
        required = any((code / runtime.requirements).read_text(encoding="utf-8").strip() for runtime in identity.python)
        wheels_spec = target.get("offline_wheels")
        if required != (wheels_spec is not None):
            raise ValueError(f"external target wheel 输入与 requirements 不匹配: {plugin_id}")
        # 3. Check the target interpreter's transitive wheel closure before migration.
        wheels: OfflineWheels | None = None
        if wheels_spec is not None:
            source_wheels = _external_path(inputs, wheels_spec["relative_path"], directory=True)
            if wheel_tree_sha256(source_wheels) != wheels_spec["tree_sha256"]:
                raise ValueError(f"external target wheel digest 不一致: {plugin_id}")
            copied = staged / f"external-{index}-wheels"
            copied.mkdir()
            for file in source_wheels.iterdir():
                shutil.copyfile(file, copied / file.name)
            wheels = OfflineWheels(copied, wheels_spec["tree_sha256"])
            if wheel_tree_sha256(copied) != wheels.tree_sha256:
                raise ValueError(f"staged external wheel digest 不一致: {plugin_id}")
            for runtime_index, runtime in enumerate(identity.python):
                if not (code / runtime.requirements).read_text(encoding="utf-8").strip():
                    continue
                destination = staged / f"external-{index}-download-{runtime_index}"
                destination.mkdir()
                preflight_offline_runtime(code, runtime, wheels, destination)
        result.append({"plugin_id": plugin_id, "old_ref": old_ref, "bundle": local,
                       "commit": commit, "code": code, "installed_target": installed_target,
                       "wheels": wheels, "bundle_sha256": bundle_digest})
    return result


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON 根必须是 object: {path}")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _distribution_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} 必须是相对路径")
    path = (root / relative).resolve(strict=True)
    if not _under(path, root):
        raise ValueError(f"{label} 越过 distribution 根: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path


def _check_sha256(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError(f"{label} 缺少合法 sha256")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 不一致: expected={expected} actual={actual}")


def _git(*arguments: str) -> str:
    """运行只读 Git 校验并返回标准输出。"""

    result = subprocess.run(
        ["git", "--no-optional-locks", *arguments], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def verify_distribution(distribution: Path) -> dict[str, Any]:
    """核对报告、Core tar 和每个独立 bundle 的不可变身份。"""

    root = distribution.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"distribution 不是目录: {root}")
    report_path = _distribution_file(root, "distribution.json", "distribution report")
    report = _read_json(report_path)
    commit = report.get("source_commit")
    tree = report.get("source_tree")
    if not isinstance(commit, str) or _REVISION.fullmatch(commit) is None:
        raise ValueError("distribution source_commit 必须是完整 40 位 SHA")
    if not isinstance(tree, str) or _REVISION.fullmatch(tree) is None:
        raise ValueError("distribution source_tree 必须是完整 40 位 SHA")

    core = report.get("core")
    if not isinstance(core, dict):
        raise ValueError("distribution 缺少 core report")
    core_path = _distribution_file(root, core.get("file"), "Core artifact")
    _check_sha256(core_path, core.get("sha256"), "Core artifact")

    plugins = report.get("plugins")
    if not isinstance(plugins, list) or not plugins:
        raise ValueError("distribution 必须包含至少一个插件 bundle")
    names: set[str] = set()
    for index, raw in enumerate(plugins):
        if not isinstance(raw, dict):
            raise ValueError(f"plugins[{index}] 必须是 object")
        name = raw.get("name")
        source_revision = raw.get("source_revision")
        if (
            not isinstance(name, str)
            or _PATH_SEGMENT.fullmatch(name) is None
            or name in names
        ):
            raise ValueError(f"plugins[{index}] name 重复或非法")
        if not isinstance(source_revision, str) or _REVISION.fullmatch(source_revision) is None:
            raise ValueError(f"plugins[{index}] source_revision 非法")
        if raw.get("source_commit") != commit:
            raise ValueError(f"plugins[{index}] source_commit 与 distribution 不一致")
        bundle = _distribution_file(root, raw.get("file"), f"plugins[{index}] bundle")
        _check_sha256(bundle, raw.get("sha256"), f"plugins[{index}] bundle")
        names.add(name)

    profiles = report.get("profiles", [])
    if not isinstance(profiles, list):
        raise ValueError("distribution profiles 必须是 array")
    for index, raw in enumerate(profiles):
        if not isinstance(raw, dict):
            raise ValueError(f"profiles[{index}] 必须是 object")
        profile_path = _distribution_file(
            root, raw.get("path"), f"profiles[{index}]"
        )
        _check_sha256(profile_path, raw.get("sha256"), f"profiles[{index}]")
    wiring = report.get("runtime_wiring", [])
    if not isinstance(wiring, list):
        raise ValueError("distribution runtime_wiring 必须是 array")
    for index, raw in enumerate(wiring):
        if not isinstance(raw, dict):
            raise ValueError(f"runtime_wiring[{index}] 必须是 object")
        wiring_path = _distribution_file(
            root, raw.get("path"), f"runtime_wiring[{index}]"
        )
        _check_sha256(wiring_path, raw.get("sha256"), f"runtime_wiring[{index}]")
    return report


def _tar_names(core: Path) -> tuple[str, ...]:
    with tarfile.open(core, mode="r:") as archive:
        names = tuple(member.name for member in archive.getmembers())
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Core tar 含越界路径: {name}")
        if name == "plugins" or name.startswith("plugins/"):
            raise ValueError(f"Core tar 不能含业务 plugins 源码: {name}")
    return names


def extract_core(
    distribution: Path,
    destination: Path,
    *,
    report: dict[str, Any] | None = None,
) -> Path:
    """解包验证过的 Core；目标必须是一次性空目录。"""

    root = distribution.expanduser().resolve(strict=True)
    verified = verify_distribution(root) if report is None else report
    core_data = verified["core"]
    assert isinstance(core_data, dict)
    core = _distribution_file(root, core_data["file"], "Core artifact")
    _check_sha256(core, core_data["sha256"], "Core artifact")
    names = _tar_names(core)
    raw_target = destination.expanduser()
    if raw_target.is_symlink():
        raise FileExistsError(f"Core 目标不能是符号链接: {raw_target}")
    target = raw_target.resolve(strict=False)
    if target.exists():
        if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(f"Core 目标必须是一次性空目录: {target}")
    else:
        target.mkdir(parents=True)
    with tarfile.open(core, mode="r:") as archive:
        archive.extractall(target, filter="data")

    if bool(core_data.get("config_example")) and not (target / "config.example.toml").is_file():
        raise ValueError("Core artifact 缺少 config.example.toml")
    web = core_data.get("web")
    if isinstance(web, dict) and web.get("enabled") is True:
        for relative in ("static/chat/index.html", "static/dashboard/index.html"):
            if not (target / relative).is_file():
                raise ValueError(f"Core artifact 缺少 Web 静态产物: {relative}")
    return target


def _preflight_bundle(
    bundle: Path,
    *,
    row: dict[str, Any],
    source_commit: str,
    code_identity: bool = False,
) -> str | None:
    """在正式安装前验证 bundle revision 与不可变来源记录。"""

    with tempfile.TemporaryDirectory(prefix="akashic-plugin-preflight-") as directory:
        # `git bundle verify` needs a repository for prerequisite checks.  The
        # Core artifact has no checkout, so use a fresh empty bare repository
        # instead of inheriting whatever directory launched the installer.
        verify_repository = Path(directory) / "verify.git"
        _ = _git("init", "--bare", str(verify_repository))
        _ = _git(
            "-C", str(verify_repository), "bundle", "verify", str(bundle)
        )
        clone = Path(directory) / "source"
        _ = _git("clone", "--no-local", "--no-checkout", str(bundle), str(clone))
        revision = str(row["source_revision"])
        actual = _git(
            "-C", str(clone), "rev-parse", "--verify", f"{revision}^{{commit}}"
        )
        if actual != revision:
            raise ValueError(
                f"插件 {row['name']} bundle source_revision 不可解析: {revision}"
            )
        provenance = json.loads(
            _git("-C", str(clone), "show", f"{revision}:.akashic-source.json")
        )
        expected = {"commit": source_commit, "path": row["source_path"]}
        if provenance != expected:
            raise ValueError(f"插件 {row['name']} bundle provenance 不一致")
        if not code_identity:
            return None
        _ = _git("-C", str(clone), "checkout", "--detach", revision)
        if _provenance(clone) != expected:
            raise ValueError(f"插件 {row['name']} bundle provenance 路径无效")
        identity = load_static_plugin_manifest(clone)
        if identity.name != row["name"]:
            raise ValueError(f"插件 {row['name']} bundle 静态身份不一致")
        return _code_identity(clone)


def _code_identity(root: Path) -> str:
    """Use the archive owner's tree identity without writing an archive."""
    entries = tree_entries(root, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE}))
    return hashlib.sha256(encode_tree(entries)).hexdigest()


def _provenance(root: Path) -> dict[str, str] | None:
    """Read an optional, well-formed source claim from a fixed code tree."""
    path = root / ".akashic-source.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"插件 provenance 路径无效: {path}")
    raw = _read_json(path)
    commit, source_path = raw.get("commit"), raw.get("path")
    if (
        set(raw) != {"commit", "path"}
        or not isinstance(commit, str) or _REVISION.fullmatch(commit) is None
        or not isinstance(source_path, str) or not source_path
        or Path(source_path).is_absolute()
        or Path(source_path).as_posix() != source_path
        or not Path(source_path).parts
        or any(_PATH_SEGMENT.fullmatch(part) is None for part in Path(source_path).parts)
    ):
        raise ValueError(f"插件 provenance 内容无效: {path}")
    return {"commit": commit, "path": source_path}


def _validate_toml_value(value: object, *, location: str, depth: int = 0) -> None:
    """Validate the JSON value subset that the generic plugin config can write."""

    if depth > 32:
        raise ValueError(f"{location} 嵌套过深")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or _CONFIG_KEY.fullmatch(key) is None:
                raise ValueError(f"{location} 含非法 TOML key: {key!r}")
            _validate_toml_value(child, location=f"{location}.{key}", depth=depth + 1)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_toml_value(child, location=f"{location}[{index}]", depth=depth + 1)
        return
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{location} 含不支持的 TOML 值: {type(value).__name__}")


def _load_profile(path: Path) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    document = _read_json(path.expanduser().resolve(strict=True))
    if document.get("schema_version") != 1:
        raise ValueError("profile schema_version 必须为 1")
    profile_name = document.get("name")
    marketplace = document.get("marketplace")
    entries = document.get("plugins")
    initialization = document.get("initialization", {})
    if (
        not isinstance(profile_name, str)
        or not profile_name
        or not isinstance(marketplace, str)
        or _PATH_SEGMENT.fullmatch(marketplace) is None
        or not isinstance(entries, list)
        or not isinstance(initialization, dict)
    ):
        raise ValueError("profile 缺少合法 name/marketplace/plugins/initialization")

    normalized: list[dict[str, Any]] = []
    selected: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"profile.plugins[{index}] 必须是 object")
        name = raw.get("name")
        depends_on = raw.get("depends_on", [])
        reason = raw.get("reason")
        if (
            not isinstance(name, str)
            or _PATH_SEGMENT.fullmatch(name) is None
            or name in selected
            or not isinstance(depends_on, list)
            or any(not isinstance(item, str) for item in depends_on)
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError(f"profile.plugins[{index}] 字段无效或重复")
        if any(item not in selected for item in depends_on):
            raise ValueError(
                f"profile.plugins[{index}] 依赖必须先在 profile 中声明: {name}"
            )
        normalized.append({"name": name, "depends_on": list(depends_on), "reason": reason})
        selected.add(name)

    unknown_initialization = set(initialization) - {"plugin_configs"}
    if unknown_initialization:
        raise ValueError(
            "profile initialization 只接受通用 plugin_configs: "
            f"{sorted(unknown_initialization)}"
        )
    raw_configs = initialization.get("plugin_configs", [])
    if not isinstance(raw_configs, list):
        raise ValueError("profile initialization.plugin_configs 必须是 array")
    plugin_configs: list[dict[str, Any]] = []
    config_owners: set[str] = set()
    for index, raw in enumerate(raw_configs):
        if not isinstance(raw, dict):
            raise ValueError(f"profile initialization.plugin_configs[{index}] 必须是 object")
        owner = raw.get("owner")
        config = raw.get("config")
        if (
            not isinstance(owner, str)
            or _PATH_SEGMENT.fullmatch(owner) is None
            or owner not in selected
            or owner in config_owners
            or not isinstance(config, dict)
        ):
            raise ValueError(
                f"profile initialization.plugin_configs[{index}] owner/config 无效或重复"
            )
        _validate_toml_value(config, location=f"plugin_configs[{index}].config")
        plugin_configs.append({"owner": owner, "config": config})
        config_owners.add(owner)
    initialization = {"plugin_configs": plugin_configs}
    return profile_name, marketplace, normalized, initialization


def _write_plugin_configs(
    workspace: Path,
    *,
    marketplace: str,
    declarations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """创建无秘密的分发默认输入；凭据由插件配置命令单独授予。"""

    results: list[dict[str, str]] = []
    for declaration in declarations:
        owner = declaration["owner"]
        config = declaration["config"]
        data_dir = workspace_plugin_data_dir(workspace, owner, marketplace)
        config_path = data_dir / CONFIG_INPUT
        _ = load_config(data_dir)
        if config_path.exists():
            results.append({"owner": owner, "path": str(config_path), "status": "existing"})
            continue
        save_config(data_dir, config)
        results.append({"owner": owner, "path": str(config_path), "status": "created"})
    return results


def install_profile(
    distribution: Path,
    profile: Path,
    *,
    workspace: Path,
    plugins_home: Path,
    config_path: Path,
) -> dict[str, Any]:
    """按 profile 顺序调用正式 install_git_plugin，不扫描 checkout。"""

    distribution_root = distribution.expanduser().resolve(strict=True)
    report = verify_distribution(distribution_root)
    profile_path = profile.expanduser().resolve(strict=True)
    profile_rows = report.get("profiles", [])
    if not any(
        isinstance(item, dict)
        and _distribution_file(distribution_root, item.get("path"), "profile") == profile_path
        for item in profile_rows
    ):
        raise ValueError("profile 不属于已验证的 distribution artifact")
    profile_name, marketplace, entries, initialization = _load_profile(profile)
    rows = {
        item["name"]: item
        for item in report["plugins"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    selected_rows: list[tuple[dict[str, Any], Path]] = []
    for entry in entries:
        name = str(entry["name"])
        row = rows.get(name)
        if row is None:
            raise ValueError(f"profile 需要的插件 bundle 不存在: {name}")
        bundle = _distribution_file(
            distribution_root, row["file"], f"插件 {name} bundle"
        )
        _check_sha256(bundle, row["sha256"], f"插件 {name} bundle")
        _preflight_bundle(
            bundle,
            row=row,
            source_commit=str(report["source_commit"]),
        )
        selected_rows.append((row, bundle))
    workspace = workspace.expanduser().resolve(strict=False)
    plugins_home = plugins_home.expanduser().resolve(strict=False)
    config_path = config_path.expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise ValueError(f"runtime config 必须是普通文件: {config_path}")
    workspace.mkdir(parents=True, exist_ok=True)
    initialize_empty_workspace(
        repo_root=_SOURCE_ROOT,
        workspace=workspace,
        config_path=config_path,
    )
    plugins_home.mkdir(parents=True, exist_ok=True)

    installed: list[dict[str, Any]] = []
    for entry, (row, bundle) in zip(entries, selected_rows, strict=True):
        name = str(entry["name"])
        try:
            result = install_git_plugin(
                workspace=workspace,
                source=str(bundle),
                marketplace=marketplace,
                ref_name=str(row["source_revision"]),
                plugins_home=plugins_home,
            )
        except Exception as error:
            error.add_note(f"正式安装插件失败: {name}")
            raise
        provenance_path = result.installed_path / ".akashic-source.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_provenance = {
            "commit": report["source_commit"],
            "path": row["source_path"],
        }
        if provenance != expected_provenance:
            raise RuntimeError(f"插件 {name} provenance 不一致")
        installed.append(
            {
                "name": result.plugin_name,
                "marketplace": result.marketplace,
                "source_revision": result.source_revision,
                "installed_path": str(result.installed_path),
                "data_path": str(result.data_path),
            }
        )
    plugin_configs = initialization.get("plugin_configs", [])
    assert isinstance(plugin_configs, list)
    config_results = _write_plugin_configs(
        workspace,
        marketplace=marketplace,
        declarations=plugin_configs,
    )
    return {
        "schema_version": 1,
        "distribution_source_commit": report["source_commit"],
        "distribution_source_tree": report["source_tree"],
        "profile": profile_name,
        "marketplace": marketplace,
        "workspace": str(workspace),
        "plugins_home": str(plugins_home),
        "initialization": initialization,
        "plugin_configs": config_results,
        "formal_installer": _FORMAL_INSTALLER,
        "installed": installed,
    }


def _validate_receipt_state(
    receipt: dict[str, Any],
    *,
    workspace: Path,
    plugins_home: Path,
) -> None:
    """Validate one historical receipt without treating it as current composition."""

    if receipt.get("schema_version") != 1:
        raise ValueError("distribution receipt schema_version 必须为 1")
    for key in ("distribution_source_commit", "distribution_source_tree"):
        value = receipt.get(key)
        if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
            raise ValueError(f"distribution receipt {key} 无效")
    profile_name = receipt.get("profile")
    marketplace = receipt.get("marketplace")
    if (
        not isinstance(profile_name, str)
        or not profile_name.strip()
        or not isinstance(marketplace, str)
        or _PATH_SEGMENT.fullmatch(marketplace) is None
    ):
        raise ValueError("distribution receipt profile/marketplace 无效")
    expected_workspace = str(workspace.expanduser().resolve(strict=False))
    expected_plugins_home = str(plugins_home.expanduser().resolve(strict=False))
    if receipt.get("workspace") != expected_workspace:
        raise ValueError("distribution receipt workspace 与当前运行不一致")
    if receipt.get("plugins_home") != expected_plugins_home:
        raise ValueError("distribution receipt plugins_home 与当前运行不一致")
    if receipt.get("formal_installer") != _FORMAL_INSTALLER:
        raise ValueError("distribution receipt 缺少正式安装器身份")

    installed = receipt.get("installed")
    if not isinstance(installed, list):
        raise ValueError("distribution receipt installed 必须是 array")
    historical_ids: set[str] = set()
    for item in installed:
        if not isinstance(item, dict):
            raise ValueError("distribution receipt installed 条目必须是 object")
        name = item.get("name")
        item_marketplace = item.get("marketplace")
        source_revision = item.get("source_revision")
        installed_path = item.get("installed_path")
        data_path = item.get("data_path")
        if (
            not isinstance(name, str)
            or _PATH_SEGMENT.fullmatch(name) is None
            or not isinstance(item_marketplace, str)
            or _PATH_SEGMENT.fullmatch(item_marketplace) is None
            or item_marketplace != marketplace
            or not isinstance(source_revision, str)
            or _REVISION.fullmatch(source_revision) is None
            or not isinstance(installed_path, str)
            or not Path(installed_path).is_absolute()
            or not isinstance(data_path, str)
            or not Path(data_path).is_absolute()
            or not _under(
                Path(installed_path),
                plugins_home / "cache" / item_marketplace / name,
            )
            or not _under(
                Path(data_path),
                workspace_plugin_data_dir(workspace, name, item_marketplace),
            )
        ):
            raise ValueError("distribution receipt installed 条目身份无效")
        historical_id = f"{name}@{item_marketplace}"
        if historical_id in historical_ids:
            raise ValueError(f"distribution receipt installed 条目重复: {historical_id}")
        historical_ids.add(historical_id)

    initialization = receipt.get("initialization")
    if (
        not isinstance(initialization, dict)
        or set(initialization) != {"plugin_configs"}
        or not isinstance(initialization.get("plugin_configs"), list)
    ):
        raise ValueError("distribution receipt initialization 无效")
    historical_names = {item.split("@", 1)[0] for item in historical_ids}
    config_owners: set[str] = set()
    for index, item in enumerate(initialization["plugin_configs"]):
        if not isinstance(item, dict):
            raise ValueError(f"distribution receipt initialization.plugin_configs[{index}] 无效")
        owner = item.get("owner")
        config = item.get("config")
        if (
            not isinstance(owner, str)
            or _PATH_SEGMENT.fullmatch(owner) is None
            or owner not in historical_names
            or owner in config_owners
            or not isinstance(config, dict)
        ):
            raise ValueError(
                f"distribution receipt initialization.plugin_configs[{index}] 无效"
            )
        _validate_toml_value(
            config,
            location=f"distribution receipt plugin_configs[{index}].config",
        )
        config_owners.add(owner)

    config_results = receipt.get("plugin_configs")
    if not isinstance(config_results, list) or len(config_results) != len(config_owners):
        raise ValueError("distribution receipt plugin_configs 无效")
    result_owners: set[str] = set()
    for index, item in enumerate(config_results):
        if not isinstance(item, dict):
            raise ValueError(f"distribution receipt plugin_configs[{index}] 无效")
        owner = item.get("owner")
        path = item.get("path")
        status = item.get("status")
        if (
            not isinstance(owner, str)
            or owner not in config_owners
            or owner in result_owners
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            or not _under(Path(path), workspace)
            or status not in {"created", "existing"}
        ):
            raise ValueError(f"distribution receipt plugin_configs[{index}] 无效")
        result_owners.add(owner)


def _validate_current_plugins(
    *,
    workspace: Path,
    plugins_home: Path,
) -> None:
    """Validate current manifest/artifacts independently of historical receipt rows."""

    manifest = load_plugin_manifest(plugins_home)
    for plugin_id, enabled in manifest.items():
        name, separator, item_marketplace = plugin_id.rpartition("@")
        if (
            not separator
            or _PATH_SEGMENT.fullmatch(name) is None
            or _PATH_SEGMENT.fullmatch(item_marketplace) is None
            or not isinstance(enabled, bool)
        ):
            raise ValueError(f"当前 plugin manifest 身份无效: {plugin_id}")
        plugin_base = plugins_home / "cache" / item_marketplace / name
        pointers = read_pointers(plugin_base)
        if pointers is None or pointers.stable.path is None:
            raise ValueError(f"当前插件缺少 stable artifact: {plugin_id}")
        artifact = resolve_pointer(plugin_base, pointers.stable)
        if artifact is None:
            raise ValueError(f"当前插件 stable artifact 为空: {plugin_id}")
        static_manifest = load_static_plugin_manifest(artifact)
        if static_manifest.name != name:
            raise ValueError(
                f"当前 artifact 身份不一致: {plugin_id} -> {static_manifest.name}"
            )
        data_path = workspace_plugin_data_dir(workspace, name, item_marketplace)
        if data_path.is_symlink() or not data_path.is_dir():
            raise ValueError(f"当前插件数据目录缺失: {data_path}")


def _selected_components(selection: PluginSelection, root_ref: str) -> tuple[tuple[str, ...], dict[str, tuple[str, Mapping[str, object], Path]]]:
    """Check every selected code closure before changing an install pointer."""
    root = selection.archive.read_descriptor(root_ref)
    components = root["components"]
    if root["version"] != 1 or not isinstance(components, tuple):
        raise ValueError("stable 完整记录格式无效")
    found: dict[str, tuple[str, Mapping[str, object], Path]] = {}
    checked_refs: list[str] = []
    for ref in components:
        if not isinstance(ref, str):
            raise ValueError("stable component ref 无效")
        record = selection.archive.read_descriptor(ref)
        plugin_id, code_ref = record["plugin_id"], record["code"]
        if record["version"] != 4 or not isinstance(plugin_id, str) or not isinstance(code_ref, str):
            raise ValueError(f"selected descriptor 格式无效: {ref}")
        if plugin_id in found:
            raise ValueError(f"stable 重复插件身份: {plugin_id}")
        code = selection.archive.open(code_ref)
        identity = load_static_plugin_manifest(code)
        if identity.name != plugin_id.split("@", 1)[0]:
            raise ValueError(f"selected 静态身份不一致: {plugin_id}")
        if record.get("source_revision") != _source_revision(code):
            raise ValueError(f"selected 源码身份不一致: {plugin_id}")
        config_revision = record.get("config_revision")
        if not isinstance(config_revision, str) or _SHA256.fullmatch(config_revision) is None:
            raise ValueError(f"selected 配置身份无效: {plugin_id}")
        if record["source_type"] not in {"builtin", "installed"}:
            raise ValueError(f"selected 来源类型无效: {plugin_id}")
        found[plugin_id] = (ref, record, code)
        checked_refs.append(ref)
    return tuple(checked_refs), found


def _current_artifact(
    *, workspace: Path, plugins_home: Path, plugin_id: str,
) -> tuple[Path, str, dict[str, str] | None]:
    """Read one coherent installed pointer and its fixed code identity."""
    name, marketplace = plugin_id.rsplit("@", 1)
    base = plugins_home / "cache" / marketplace / name
    for directory in (plugins_home / "cache", plugins_home / "cache" / marketplace, base):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"incomplete_or_drift: cache 目录无效: {directory}")
    pointers = read_pointers(base)
    if pointers is None or pointers.stable.path is None or pointers.latest != pointers.stable:
        raise ValueError(f"incomplete_or_drift: current pointer 无效: {plugin_id}")
    artifact = resolve_pointer(base, pointers.stable)
    if artifact is None:
        raise ValueError(f"incomplete_or_drift: stable artifact 缺失: {plugin_id}")
    identity = load_static_plugin_manifest(artifact)
    if identity.name != name:
        raise ValueError(f"incomplete_or_drift: artifact 身份不一致: {plugin_id}")
    data = workspace_plugin_data_dir(workspace, name, marketplace)
    if data.is_symlink() or not data.is_dir():
        raise ValueError(f"incomplete_or_drift: plugin-data 身份无效: {data}")
    return artifact, _code_identity(artifact), _provenance(artifact)


def publish_distribution(
    *, distribution: Path, workspace: Path, plugins_home: Path,
    config_path: Path, plan: Path, inputs: Path,
    backup_dir: Path | None = None, preflight_only: bool = False,
) -> dict[str, Any]:
    """只发布清单内的输入；备份可选，迁移由清单授权。"""
    digest, expected_root, requested, migrations = load_deployment_plan(plan)
    workspace, plugins_home = workspace.resolve(strict=True), plugins_home.resolve(strict=True)
    selection = PluginSelection(workspace)
    if selection.read() != expected_root:
        raise SelectionConflictError("当前 Root 与部署清单基线不同")
    maintenance = WorkspaceMaintenanceLock(workspace)
    publication = PluginPublicationLock(plugins_home)
    # 在线预检只读；停止期执行仍重新取得锁并核对同一基线。
    if not preflight_only:
        maintenance.acquire()
    try:
        if not preflight_only:
            publication.acquire()
        try:
            if selection.read() != expected_root:
                raise SelectionConflictError("取得发布锁期间 Root 改变")
            with tempfile.TemporaryDirectory(prefix="akashic-deploy-") as temporary:
                staged = Path(temporary)
                targets = _stage_deployment_targets(
                    targets=requested, distribution=distribution, inputs=inputs, staged=staged,
                    expected_root_ref=expected_root, selection=selection,
                    workspace=workspace, plugins_home=plugins_home,
                )
                components, selected = _selected_components(selection, expected_root)
                replacements = {item["plugin_id"]: item["code"] for item in targets}
                sources: list[ResolvedPluginSource] = []
                for plugin_id, (_, descriptor, old_code) in selected.items():
                    name, marketplace = plugin_id.split("@")
                    if plugin_id not in replacements and descriptor["source_type"] == "installed":
                        _, code_hash, provenance = _current_artifact(
                            workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id,
                        )
                        if code_hash != descriptor["code"] or provenance != _provenance(old_code):
                            raise SelectionConflictError(f"未触及插件 selection/cache 漂移: {plugin_id}")
                    code = replacements.get(plugin_id, old_code)
                    sources.append(ResolvedPluginSource(
                        plugin_root=code, source_type=descriptor["source_type"],
                        plugin_name=name, marketplace=marketplace,
                        static_manifest=load_static_plugin_manifest(code),
                    ))
                runner = MigrationRunner(repo_root=_SOURCE_ROOT, config_path=config_path,
                                         workspace=workspace, fixed_sources=sources)
                pending = runner.check(migrations)
                result: dict[str, Any] = {
                    "status": "preflight_ok", "plan_sha256": digest,
                    "old_root_ref": expected_root, "migration_ids": list(pending),
                    "target_ids": list(replacements), "backup_dir": None,
                }
                if preflight_only:
                    return result
                if backup_dir is not None:
                    state = workspace.parent
                    if plugins_home.parent != state or config_path.resolve().parent != state:
                        raise ValueError("全状态备份要求 config、workspace、plugin-home 位于同一 state")
                    backup_release_state(state, backup_dir)
                    result["backup_dir"] = str(backup_dir)
                # Core 先更新自己的账本结构，业务 step 只由相应插件执行。
                runner.run_under_maintenance(maintenance, core_only=True, approved_migrations=migrations)
                with ReloadJournal.inspect_existing(workspace) as journal:
                    if journal.pending_recovery or journal.armed_updates:
                        raise RuntimeError("reload/install owner 尚未结算；不得覆盖未决事实")
                runner.run_under_maintenance(maintenance, approved_migrations=migrations)
                changed: dict[str, str] = {}
                for target in targets:
                    plugin_id = target["plugin_id"]
                    name, marketplace = plugin_id.split("@")
                    if target["installed_target"]:
                        artifact, _, _ = _current_artifact(workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id)
                        update_id = None
                    else:
                        installed = install_git_plugin(
                            workspace=workspace, source=str(target["bundle"]), marketplace=marketplace,
                            ref_name=target["commit"], plugins_home=plugins_home,
                            refresh_existing_artifact=False, offline_wheels=target["wheels"],
                        )
                        artifact, update_id = installed.installed_path, installed.update_id
                    identity = load_static_plugin_manifest(artifact)
                    prepared = prepare_plugin_input(
                        {"name": name, "plugin_root": str(artifact), "module_path": str(artifact / "plugin.py"),
                         "manifest_digest": identity.identity_digest, "marketplace": marketplace, "source_type": "installed"},
                        workspace=workspace, archive=selection.archive,
                    )
                    if prepared.plugin_id != plugin_id:
                        raise RuntimeError(f"安装输入身份不符: {plugin_id}")
                    if update_id is not None:
                        ReloadJournal(workspace).set_input_ref(update_id, prepared.archive_ref)
                    changed[selected[plugin_id][0]] = prepared.archive_ref
                new_components = tuple(changed.get(ref, ref) for ref in components)
                new_root = (selection.commit(new_components, expected_ref=expected_root)
                            if new_components != components else expected_root)
                return {**result, "status": "selected_not_started", "new_root_ref": new_root,
                        "ordered_components": list(new_components)}
        finally:
            if not preflight_only:
                publication.release()
    finally:
        if not preflight_only:
            maintenance.release()


def ensure_profile(
    distribution: Path,
    profile: Path,
    *,
    workspace: Path,
    plugins_home: Path,
    config_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Install once, then validate the durable receipt without changing composition."""

    receipt_path = receipt_path.expanduser()
    if receipt_path.is_symlink():
        raise ValueError(f"distribution receipt 不能是符号链接: {receipt_path}")
    config_path = config_path.expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise ValueError(f"runtime config 必须是普通文件: {config_path}")
    if receipt_path.exists():
        if not receipt_path.is_file():
            raise ValueError(f"distribution receipt 不是普通文件: {receipt_path}")
        distribution_root = distribution.expanduser().resolve(strict=True)
        verify_distribution(distribution_root)
        # A profile is only the first-install recipe.  Receipt-present
        # startup must follow the current manifest, even after operator
        # replacement or removal of an originally selected provider.
        receipt = _read_json(receipt_path)
        _validate_receipt_state(
            receipt,
            workspace=workspace.expanduser().resolve(strict=False),
            plugins_home=plugins_home.expanduser().resolve(strict=False),
        )
        _validate_current_plugins(
            workspace=workspace.expanduser().resolve(strict=False),
            plugins_home=plugins_home.expanduser().resolve(strict=False),
        )
        return {**receipt, "status": "existing"}

    result = install_profile(
        distribution,
        profile,
        workspace=workspace,
        plugins_home=plugins_home,
        config_path=config_path,
    )
    return {**result, "status": "installed"}


def _write_receipt(path: Path, result: dict[str, Any]) -> None:
    """Atomically publish one durable installer receipt."""

    path = path.expanduser()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"distribution receipt 不是普通文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plugins-home", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--ensure-profile", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if sum((args.verify_only, args.ensure_profile, args.publish)) > 1:
        parser.error("--verify-only、--ensure-profile、--publish 互斥")
    if args.publish:
        if args.config is None or args.plan is None or args.inputs is None:
            parser.error("--publish 需要 --config、--plan、--inputs")
        result = publish_distribution(
            distribution=args.distribution, workspace=args.workspace,
            plugins_home=args.plugins_home, config_path=args.config,
            plan=args.plan, inputs=args.inputs, backup_dir=args.backup_dir,
            preflight_only=args.preflight_only,
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    report = verify_distribution(args.distribution)
    if args.verify_only:
        result: dict[str, Any] = {
            "status": "verified",
            "distribution_source_commit": report["source_commit"],
            "plugin_count": len(report["plugins"]),
        }
    else:
        if args.config is None:
            parser.error("安装 profile 必须提供 --config")
        if args.core_root is not None:
            extract_core(args.distribution, args.core_root, report=report)
        if args.ensure_profile:
            if args.receipt is None:
                parser.error("--ensure-profile 必须提供 --receipt")
            result = ensure_profile(
                args.distribution,
                args.profile,
                workspace=args.workspace,
                plugins_home=args.plugins_home,
                config_path=args.config,
                receipt_path=args.receipt,
            )
        else:
            result = install_profile(
                args.distribution,
                args.profile,
                workspace=args.workspace,
                plugins_home=args.plugins_home,
                config_path=args.config,
            )
    if args.receipt is not None:
        if not (args.ensure_profile and result.get("status") == "existing"):
            _write_receipt(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
