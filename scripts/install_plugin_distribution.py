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
from uuid import uuid4


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
from agent.plugins.reload_journal import ReloadJournal, JournalPreflight
from agent.plugins.selection import PluginSelection, SelectionConflictError, SelectionWriteError
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


def _external_plan(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    """Validate the sole operator authority for explicit external targets."""

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"external plan JSON key 重复: {key}")
            result[key] = value
        return result

    # 1. Parse exact JSON once; duplicated object keys cannot hide an input.
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("external plan 必须是普通文件")
    raw = path.read_bytes()
    document = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(document, dict) or set(document) != {"schema_version", "expected_root_ref", "targets"}:
        raise ValueError("external plan 根字段错误")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("external plan schema_version 错误")
    root_ref = document["expected_root_ref"]
    if not isinstance(root_ref, str) or _SHA256.fullmatch(root_ref) is None:
        raise ValueError("external plan expected_root_ref 错误")
    targets = document["targets"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("external plan targets 必须非空")
    # 2. Validate every declared target before resolving input bytes.
    seen: set[str] = set()
    for item in targets:
        if not isinstance(item, dict) or set(item) not in ({"plugin_id", "bundle_relative_path", "bundle_sha256", "target_commit"}, {"plugin_id", "bundle_relative_path", "bundle_sha256", "target_commit", "offline_wheels"}):
            raise ValueError("external target 字段错误")
        plugin_id = item["plugin_id"]
        if (not isinstance(plugin_id, str) or plugin_id.count("@") != 1
            or any(_PATH_SEGMENT.fullmatch(part) is None for part in plugin_id.split("@"))
            or plugin_id in seen):
            raise ValueError(f"external target plugin_id 重复或无效: {plugin_id}")
        seen.add(plugin_id)
        if not isinstance(item["bundle_sha256"], str) or _SHA256.fullmatch(item["bundle_sha256"]) is None:
            raise ValueError(f"external bundle_sha256 无效: {plugin_id}")
        if not isinstance(item["target_commit"], str) or _REVISION.fullmatch(item["target_commit"]) is None:
            raise ValueError(f"external target_commit 无效: {plugin_id}")
        _ = _external_path_syntax(item["bundle_relative_path"])
        wheels = item.get("offline_wheels")
        if "offline_wheels" in item:
            if (not isinstance(wheels, dict) or set(wheels) != {"relative_path", "tree_sha256"}
                or not isinstance(wheels["tree_sha256"], str)
                or _SHA256.fullmatch(wheels["tree_sha256"]) is None):
                raise ValueError(f"external offline_wheels 无效: {plugin_id}")
            _ = _external_path_syntax(wheels["relative_path"])
    return hashlib.sha256(raw).hexdigest(), root_ref, targets


def _external_path_syntax(relative: object) -> str:
    if (not isinstance(relative, str) or not relative or relative.startswith("/")
        or "\\" in relative or "\x00" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("external input 路径格式错误")
    return relative


def _stage_external_targets(
    *, plan: Path, inputs: Path, staged: Path, expected_root_ref: str,
    selection: PluginSelection, workspace: Path, plugins_home: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Fix bundle and wheel bytes in tmpfs and classify every explicit target."""

    # 1. Require each target to own a coherent selected installed input.
    plan_digest, requested_root, targets = _external_plan(plan)
    if requested_root != expected_root_ref:
        raise SelectionConflictError("external plan expected_root_ref 与当前 Root 不符")
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
        _, current_code, current_source = _current_artifact(
            workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id,
        )
        if descriptor["code"] != current_code or _provenance(old_code) != current_source:
            raise SelectionConflictError(f"external target selection/cache 漂移: {plugin_id}")
        # 2. Copy fixed bundle bytes into tmpfs and resolve its exact commit.
        bundle = _external_path(inputs, target["bundle_relative_path"], directory=False)
        _check_sha256(bundle, target["bundle_sha256"], f"external {plugin_id} bundle")
        local = staged / f"external-{index}.bundle"
        shutil.copyfile(bundle, local)
        _check_sha256(local, target["bundle_sha256"], f"staged {plugin_id} bundle")
        code = staged / f"external-{index}-code"
        _ = _git("clone", "--no-local", "--no-checkout", str(local), str(code))
        _ = _git("-C", str(code), "checkout", "--detach", target["target_commit"])
        if _git("-C", str(code), "rev-parse", "HEAD") != target["target_commit"]:
            raise ValueError(f"external target commit 不一致: {plugin_id}")
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
                       "commit": target["target_commit"], "code": code,
                       "wheels": wheels, "bundle_sha256": target["bundle_sha256"]})
    return plan_digest, result


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
        ["git", *arguments], check=True, capture_output=True, text=True
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
    if _REVISION.fullmatch(_git("-C", str(artifact), "rev-parse", "HEAD")) is None or _git(
        "-C", str(artifact), "diff", "--name-only", "HEAD", "--"
    ):
        raise ValueError(f"incomplete_or_drift: artifact Git 工作树与 HEAD 不一致: {plugin_id}")
    data = workspace_plugin_data_dir(workspace, name, marketplace)
    if data.is_symlink() or not data.is_dir():
        raise ValueError(f"incomplete_or_drift: plugin-data 身份无效: {data}")
    return artifact, _code_identity(artifact), _provenance(artifact)


def _plain(path: Path, *, directory: bool = False) -> None:
    if path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"元数据路径必须是普通{'目录' if directory else '文件'}: {path}")


def _save(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        path.chmod(0o600)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != content:
        raise RuntimeError(f"备份校验失败: {path}")


def _backup_adoption(
    *, workspace: Path, plugins_home: Path, backup_dir: Path,
    root_ref: str, plugin_ids: tuple[str, ...], bundle_sha256: dict[str, str],
    journal: JournalPreflight,
) -> Path:
    """Save only publication metadata before the first install changes it."""
    workspace, plugins_home = workspace.resolve(), plugins_home.resolve()
    backup_dir = backup_dir.absolute()
    if backup_dir.is_relative_to(workspace) or backup_dir.is_relative_to(plugins_home):
        raise ValueError("恢复点必须在 workspace 和 plugin-home 外")
    _plain(backup_dir.parent, directory=True)
    backup_dir = backup_dir.parent.resolve() / backup_dir.name
    if backup_dir.is_relative_to(workspace) or backup_dir.is_relative_to(plugins_home):
        raise ValueError("恢复点父目录不能链接到运行数据内")
    sources = [
        (workspace / "runtime/plugin-stable.json", "workspace/runtime/plugin-stable.json"),
        (plugins_home / "manifest.toml", "plugins/manifest.toml"),
    ]
    for plugin_id in plugin_ids:
        name, marketplace = plugin_id.rsplit("@", 1)
        relative = f"plugins/cache/{marketplace}/{name}/.pointers.json"
        sources.append((plugins_home / "cache" / marketplace / name / ".pointers.json", relative))
    for source, _ in sources:
        _plain(source)
    backup_dir.mkdir(mode=0o700)
    sync_directory(backup_dir.parent)
    entries: list[dict[str, str]] = []
    for source, relative in sources:
        content = source.read_bytes()
        target = backup_dir / relative
        _save(target, content)
        if source.read_bytes() != content:
            raise RuntimeError(f"备份期间元数据变化: {source}")
        entries.append({"source": str(source), "backup": relative, "sha256": _sha256(target)})
    journal_target = backup_dir / "workspace/runtime/plugin-reloads.sqlite3"
    journal_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    journal.backup_to(journal_target)
    with journal_target.open("rb") as stream:
        os.fsync(stream.fileno())
    entries.append({"source": str(workspace / "runtime/plugin-reloads.sqlite3"),
                    "backup": "workspace/runtime/plugin-reloads.sqlite3", "sha256": _sha256(journal_target)})
    _save(backup_dir / "recovery.json", json.dumps({
        "purpose": "stopped-bundled-publication; explicit recovery only",
        "workspace": str(workspace), "plugins_home": str(plugins_home),
        "previous_selection": root_ref, "plugin_ids": plugin_ids,
        "bundle_sha256": bundle_sha256, "files": entries,
    }, ensure_ascii=False, indent=2).encode())
    for current, _, _ in os.walk(backup_dir, topdown=False):
        sync_directory(Path(current))
    return backup_dir


def adopt_bundled_distribution(
    *, distribution: Path, profile: Path, workspace: Path, plugins_home: Path,
    config_path: Path, receipt_path: Path, backup_dir: Path, expected_root_ref: str,
    held_locks: tuple[WorkspaceMaintenanceLock, PluginPublicationLock] | None = None,
    previous_source_commit: str | None = None,
    external_targets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select exact bundled inputs while the runtime is stopped."""
    if _SHA256.fullmatch(expected_root_ref) is None:
        raise ValueError("expected_root_ref 必须是完整 SHA-256")
    if previous_source_commit is not None and _REVISION.fullmatch(previous_source_commit) is None:
        raise ValueError("previous_source_commit 必须是完整 Git commit")
    _plain(workspace, directory=True)
    _plain(plugins_home, directory=True)
    workspace, plugins_home = workspace.resolve(), plugins_home.resolve()
    _plain(config_path)
    _plain(receipt_path)
    maintenance = WorkspaceMaintenanceLock(workspace) if held_locks is None else held_locks[0]
    if held_locks is None:
        maintenance.acquire()
    elif maintenance.paths != (workspace / ".supervisor.lock", workspace / ".instance.lock") or len(maintenance._streams) != 2:
        raise RuntimeError("bundle adoption 缺少 workspace maintenance owner")
    try:
        publication = PluginPublicationLock(plugins_home) if held_locks is None else held_locks[1]
        if held_locks is None:
            publication.acquire()
        elif publication.path != plugins_home / ".publication.lock" or publication._stream is None:
            raise RuntimeError("bundle adoption 缺少 plugin publication owner")
        try:
            # 1. Fix the supplied bundle and inspect all formal current facts.
            report = verify_distribution(distribution)
            root = distribution.expanduser().resolve(strict=True)
            profile_path = profile.expanduser().resolve(strict=True)
            if not any(
                isinstance(item, dict) and _distribution_file(root, item.get("path"), "profile") == profile_path
                for item in report["profiles"]
            ):
                raise ValueError("profile 不属于已验证的 distribution")
            profile_name, marketplace, entries, _ = _load_profile(profile_path)
            rows = {str(item["name"]): item for item in report["plugins"]}
            bundles: dict[str, tuple[dict[str, Any], Path, str]] = {}
            for entry in entries:
                name = str(entry["name"])
                row = rows.get(name)
                if row is None:
                    raise ValueError(f"profile 缺少 bundle: {name}")
                bundle = _distribution_file(root, row["file"], f"插件 {name} bundle")
                bundled_code = _preflight_bundle(
                    bundle, row=row, source_commit=str(report["source_commit"]), code_identity=True,
                )
                assert bundled_code is not None
                bundles[name] = (row, bundle, bundled_code)
            receipt = _read_json(receipt_path)
            _validate_receipt_state(receipt, workspace=workspace, plugins_home=plugins_home)
            if receipt["marketplace"] != marketplace:
                raise ValueError("历史 receipt marketplace 与目标 profile 不一致")
            historical = {f"{item['name']}@{item['marketplace']}" for item in receipt["installed"]}
            selection = PluginSelection(workspace)
            old_root = selection.read()
            if old_root != expected_root_ref:
                raise SelectionConflictError("stable 基线与 expected_root_ref 不一致")
            assert old_root is not None
            components, selected = _selected_components(selection, old_root)
            _plain(plugins_home / "manifest.toml")
            manifest = load_plugin_manifest(plugins_home)
            for plugin_id, enabled in manifest.items():
                name, separator, item_marketplace = plugin_id.rpartition("@")
                if (not separator or _PATH_SEGMENT.fullmatch(name) is None
                    or _PATH_SEGMENT.fullmatch(item_marketplace) is None
                    or not isinstance(enabled, bool)):
                    raise ValueError(f"manifest 插件身份无效: {plugin_id}")
            changed: list[dict[str, str]] = []
            already: list[str] = []
            external: list[str] = []
            excluded: list[dict[str, str]] = []
            target_rows: list[tuple[str, dict[str, Any], Path]] = []
            classified_ids: set[str] = set()
            with ReloadJournal.inspect_existing(workspace) as journal:
                if journal.pending_recovery or journal.armed_updates:
                    raise RuntimeError("pending reload 或 armed install 必须由原 owner 结算")
                for entry in entries:
                    name = str(entry["name"])
                    plugin_id = f"{name}@{marketplace}"
                    if plugin_id not in historical or plugin_id not in selected:
                        excluded.append({"plugin_id": plugin_id, "reason": "new_or_uninstalled"})
                        continue
                    classified_ids.add(plugin_id)
                    if manifest.get(plugin_id) is not True:
                        raise ValueError(f"incomplete_or_drift: 已选插件未在 manifest enabled: {plugin_id}")
                    old_ref, descriptor, old_code = selected[plugin_id]
                    if descriptor["source_type"] != "installed":
                        raise ValueError(f"incomplete_or_drift: 目标不是 installed source: {plugin_id}")
                    expected_data = workspace_plugin_data_dir(workspace, name, marketplace).relative_to(workspace).as_posix()
                    if descriptor.get("data_dir") != expected_data:
                        raise ValueError(f"incomplete_or_drift: selected plugin-data 身份不一致: {plugin_id}")
                    selected_source = _provenance(old_code)
                    release_owned = (
                        selected_source is not None
                        and selected_source["path"] == bundles[name][0]["source_path"]
                        and (previous_source_commit is None
                             or selected_source["commit"] == previous_source_commit)
                    )
                    artifact, current_code, current_source = _current_artifact(
                        workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id,
                    )
                    row, bundle, bundled_code = bundles[name]
                    expected_source = {"commit": str(report["source_commit"]), "path": str(row["source_path"])}
                    if (selected_source == expected_source and descriptor["code"] != bundled_code) or (
                        current_source == expected_source and current_code != bundled_code
                    ):
                        raise ValueError(f"incomplete_or_drift: provenance 与代码不一致: {plugin_id}")
                    selected_is_b = descriptor["code"] == bundled_code and selected_source == expected_source
                    current_is_b = current_code == bundled_code and current_source == expected_source and _git("-C", str(artifact), "rev-parse", "HEAD") == row["source_revision"]
                    if selected_is_b and current_is_b:
                        already.append(plugin_id)
                        continue
                    same_current = descriptor["code"] == current_code and selected_source == current_source
                    if selected_is_b or (not same_current and not current_is_b):
                        raise ValueError(f"incomplete_or_drift: selection/cache 不一致: {plugin_id}")
                    if current_is_b:
                        if not release_owned:
                            raise ValueError(f"incomplete_or_drift: B pointer 不能覆盖外部选择: {plugin_id}")
                        target_rows.append((plugin_id, row, bundle))
                        changed.append({"plugin_id": plugin_id, "old_ref": old_ref, "mode": "resume_install_B"})
                        continue
                    if not release_owned:
                        external.append(plugin_id)
                        continue
                    target_rows.append((plugin_id, row, bundle))
                    changed.append({"plugin_id": plugin_id, "old_ref": old_ref, "mode": "install_B"})
                for plugin_id, (_, descriptor, code) in selected.items():
                    if plugin_id in classified_ids or descriptor["source_type"] != "installed":
                        continue
                    if manifest.get(plugin_id) is not True:
                        raise ValueError(f"incomplete_or_drift: 已选插件未在 manifest enabled: {plugin_id}")
                    _, current_code, current_source = _current_artifact(
                        workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id,
                    )
                    if descriptor["code"] != current_code or _provenance(code) != current_source:
                        raise ValueError(f"incomplete_or_drift: 未触及插件的 selection/cache 不一致: {plugin_id}")
                external_rows = external_targets or []
                for target in external_rows:
                    plugin_id = target["plugin_id"]
                    if plugin_id not in selected or selected[plugin_id][0] != target["old_ref"]:
                        raise SelectionConflictError(f"external target selection 已改变: {plugin_id}")
                    if any(item[0] == plugin_id for item in target_rows):
                        raise SelectionConflictError(f"external target 与 bundled 目标重叠: {plugin_id}")
                    name, marketplace_name = plugin_id.split("@")
                    bundled = bundles.get(name) if marketplace_name == marketplace else None
                    if bundled is not None and previous_source_commit is not None:
                        prior = _provenance(selected[plugin_id][2])
                        if prior == {"commit": previous_source_commit,
                                     "path": bundled[0]["source_path"]}:
                            raise SelectionConflictError(f"external target 属于 release bundled owner: {plugin_id}")
                    if manifest.get(plugin_id) is not True:
                        raise SelectionConflictError(f"external target 已禁用: {plugin_id}")
                    _, current_code, current_source = _current_artifact(
                        workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id,
                    )
                    if (selected[plugin_id][1]["code"] != current_code
                        or _provenance(selected[plugin_id][2]) != current_source):
                        raise SelectionConflictError(f"external target cache 漂移: {plugin_id}")
                    changed.append({"plugin_id": plugin_id, "old_ref": target["old_ref"], "mode": "external"})
                if target_rows or external_rows:
                    recovery = _backup_adoption(
                        workspace=workspace, plugins_home=plugins_home, backup_dir=backup_dir,
                        root_ref=old_root,
                        plugin_ids=tuple([item[0] for item in target_rows]
                                         + [item["plugin_id"] for item in external_rows]),
                        bundle_sha256={**{item[0]: str(item[1]["sha256"]) for item in target_rows},
                                       **{item["plugin_id"]: item["bundle_sha256"] for item in external_rows}},
                        journal=journal,
                    )
                else:
                    recovery = None
            # 2. The installer commits cache/history; prepare each complete input.
            replacements: dict[str, str] = {}
            for plugin_id, row, bundle in target_rows:
                phase = "install"
                try:
                    installed = install_git_plugin(
                        workspace=workspace, source=str(bundle), marketplace=marketplace,
                        ref_name=str(row["source_revision"]), plugins_home=plugins_home,
                        refresh_existing_artifact=False,
                        update_id=uuid4().hex,
                    )
                    if f"{installed.plugin_name}@{installed.marketplace}" != plugin_id or installed.source_revision != row["source_revision"]:
                        raise RuntimeError(f"install 结果身份不一致: {plugin_id}")
                    phase = "validate_install"
                    if _provenance(installed.installed_path) != {"commit": report["source_commit"], "path": row["source_path"]}:
                        raise RuntimeError(f"install provenance 不一致: {plugin_id}")
                    identity = load_static_plugin_manifest(installed.installed_path)
                    mod = {"name": installed.plugin_name, "plugin_root": str(installed.installed_path),
                           "module_path": str(installed.installed_path / "plugin.py"),
                           "manifest_digest": identity.identity_digest,
                           "marketplace": marketplace, "source_type": "installed"}
                    phase = "prepare_input"
                    prepared = prepare_plugin_input(mod, workspace=workspace, archive=selection.archive)
                    if prepared.plugin_id != plugin_id:
                        raise RuntimeError(f"prepare 结果身份不一致: {plugin_id}")
                    phase = "set_input_ref"
                    ReloadJournal(workspace).set_input_ref(installed.update_id, prepared.archive_ref)
                    replacements[plugin_id] = prepared.archive_ref
                    changed[next(i for i, item in enumerate(changed) if item["plugin_id"] == plugin_id)]["new_ref"] = prepared.archive_ref
                except Exception as error:
                    error.add_note(f"B2 phase={phase} plugin={plugin_id} recovery={recovery} prepared={tuple(replacements)}")
                    raise
            for target in external_rows:
                plugin_id = target["plugin_id"]
                name, marketplace_name = plugin_id.split("@")
                phase = "external_install"
                try:
                    wheels = target["wheels"]
                    if wheels is not None and wheel_tree_sha256(wheels.directory) != wheels.tree_sha256:
                        raise ValueError(f"external wheel 输入变化: {plugin_id}")
                    _check_sha256(target["bundle"], target["bundle_sha256"], f"external {plugin_id} bundle")
                    installed = install_git_plugin(
                        workspace=workspace, source=str(target["bundle"]),
                        marketplace=marketplace_name, ref_name=target["commit"],
                        plugins_home=plugins_home, refresh_existing_artifact=False,
                        update_id=uuid4().hex, offline_wheels=wheels,
                    )
                    if (f"{installed.plugin_name}@{installed.marketplace}" != plugin_id
                        or installed.source_revision != target["commit"]):
                        raise RuntimeError(f"external install 身份不一致: {plugin_id}")
                    phase = "external_prepare"
                    identity = load_static_plugin_manifest(installed.installed_path)
                    mod = {"name": name, "plugin_root": str(installed.installed_path),
                           "module_path": str(installed.installed_path / "plugin.py"),
                           "manifest_digest": identity.identity_digest,
                           "marketplace": marketplace_name, "source_type": "installed"}
                    prepared = prepare_plugin_input(mod, workspace=workspace, archive=selection.archive)
                    if prepared.plugin_id != plugin_id:
                        raise RuntimeError(f"external prepare 身份不一致: {plugin_id}")
                    ReloadJournal(workspace).set_input_ref(installed.update_id, prepared.archive_ref)
                    replacements[plugin_id] = prepared.archive_ref
                    changed[next(i for i, item in enumerate(changed) if item["plugin_id"] == plugin_id)]["new_ref"] = prepared.archive_ref
                except Exception as error:
                    error.add_note(f"external phase={phase} plugin={plugin_id} recovery={recovery} prepared={tuple(replacements)}")
                    raise
            if replacements:
                replacement_refs = {selected[plugin_id][0]: new_ref for plugin_id, new_ref in replacements.items()}
                new_components = tuple(replacement_refs.get(ref, ref) for ref in components)
                try:
                    new_root = selection.commit(new_components, expected_ref=old_root)
                except (SelectionConflictError, SelectionWriteError) as error:
                    error.add_note(f"B2 phase=selection recovery={recovery} prepared={tuple(replacements)}")
                    raise
            else:
                new_root = old_root
            skipped_external = [plugin_id for plugin_id in external if plugin_id not in {item["plugin_id"] for item in external_rows}]
            status = (
                "partial_selected_not_started" if skipped_external else
                "selected_not_started" if replacements else
                "already_selected_not_started" if already else "no_eligible_targets"
            )
            return {"mode": "operator_trusted_offline_distribution",
                    "status": status, "old_root_ref": old_root, "new_root_ref": new_root,
                    "ordered_components": list(new_components if replacements else components),
                    "changed": changed, "already_selected": already,
                    "skipped_external": skipped_external, "excluded": excluded,
                    "backup_dir": None if recovery is None else str(recovery),
                    "profile": profile_name, "distribution_source_commit": report["source_commit"]}
        finally:
            if held_locks is None:
                publication.release()
    finally:
        if held_locks is None:
            maintenance.release()


def _fixed_release_sources(
    *, distribution: Path, profile: Path, workspace: Path, plugins_home: Path, receipt_path: Path,
    selection: PluginSelection, root_ref: str, staging: Path, previous_source_commit: str,
    external_targets: list[dict[str, Any]] | None = None,
) -> tuple[ResolvedPluginSource, ...]:
    """Bind migration code to target bundles and retained selected artifacts."""

    report = verify_distribution(distribution)
    root = distribution.resolve(strict=True)
    profile_path = profile.resolve(strict=True)
    if not any(
        isinstance(item, dict) and _distribution_file(root, item.get("path"), "profile") == profile_path
        for item in report["profiles"]
    ):
        raise ValueError("目标 profile 不属于固定 distribution")
    _, marketplace, entries, _ = _load_profile(profile_path)
    receipt = _read_json(receipt_path)
    _validate_receipt_state(receipt, workspace=workspace, plugins_home=plugins_home)
    historical = {f"{item['name']}@{item['marketplace']}" for item in receipt["installed"]}
    rows = {str(item["name"]): item for item in report["plugins"]}
    _, selected = _selected_components(selection, root_ref)
    external_code = {item["plugin_id"]: item["code"] for item in external_targets or []}
    sources: list[ResolvedPluginSource] = []
    for plugin_id, (_, descriptor, old_code) in selected.items():
        name, separator, owner_marketplace = plugin_id.rpartition("@")
        if not separator:
            name, owner_marketplace = plugin_id, ""
        code = external_code.get(plugin_id, old_code)
        row = rows.get(name)
        if (plugin_id in external_code and owner_marketplace == marketplace
            and row is not None and plugin_id in historical
            and _provenance(old_code) == {"commit": previous_source_commit,
                                          "path": row["source_path"]}):
            raise SelectionConflictError(f"external target 属于 release bundled owner: {plugin_id}")
        if (plugin_id not in external_code and owner_marketplace == marketplace and row is not None
            and plugin_id in historical and descriptor["source_type"] == "installed"
            and _provenance(old_code) is not None
            and _provenance(old_code) == {"commit": previous_source_commit,
                                           "path": row["source_path"]}):
            bundle = _distribution_file(root, row["file"], f"插件 {name} bundle")
            _check_sha256(bundle, row["sha256"], f"插件 {name} bundle")
            expected_code = _preflight_bundle(bundle, row=row, source_commit=str(report["source_commit"]), code_identity=True)
            code = staging / plugin_id
            _ = _git("clone", "--no-local", "--no-checkout", str(bundle), str(code))
            _ = _git("-C", str(code), "checkout", "--detach", str(row["source_revision"]))
            if (_git("-C", str(code), "rev-parse", "HEAD") != row["source_revision"]
                or _code_identity(code) != expected_code
                or _provenance(code) != {"commit": report["source_commit"], "path": row["source_path"]}):
                raise RuntimeError(f"目标 migration bundle 身份漂移: {plugin_id}")
        identity = load_static_plugin_manifest(code)
        if identity.name != name:
            raise RuntimeError(f"selected migration owner 身份不一致: {plugin_id}")
        sources.append(ResolvedPluginSource(
            plugin_root=code, source_type="installed" if separator else "builtin",
            plugin_name=name, marketplace=owner_marketplace,
            static_manifest=identity,
        ))
    return tuple(sources)


def upgrade_bundled_distribution(
    *, distribution: Path, profile: Path, workspace: Path, plugins_home: Path,
    config_path: Path, receipt_path: Path, backup_dir: Path, expected_root_ref: str,
    previous_source_commit: str,
    external_plan: Path | None = None, external_inputs: Path | None = None,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Migrate target code and data, then commit one stopped full selection."""

    _plain(workspace, directory=True)
    _plain(plugins_home, directory=True)
    _plain(config_path)
    workspace, plugins_home = workspace.resolve(), plugins_home.resolve()
    if not _SHA256.fullmatch(expected_root_ref):
        raise ValueError("release upgrade 需要完整 expected_root_ref")
    if _REVISION.fullmatch(previous_source_commit) is None:
        raise ValueError("release upgrade 需要完整 previous_source_commit")
    if (external_plan is None) != (external_inputs is None):
        raise ValueError("external plan 和 input root 必须同时提供")
    maintenance = WorkspaceMaintenanceLock(workspace)
    maintenance.acquire()
    try:
        publication = PluginPublicationLock(plugins_home)
        publication.acquire()
        try:
            selection = PluginSelection(workspace)
            if selection.read() != expected_root_ref:
                raise SelectionConflictError("release upgrade stable 基线改变")
            with tempfile.TemporaryDirectory(prefix="akashic-release-migrations-") as temporary:
                staged = Path(temporary)
                plan_digest: str | None = None
                external_targets: list[dict[str, Any]] = []
                if external_plan is not None and external_inputs is not None:
                    plan_digest, external_targets = _stage_external_targets(
                        plan=external_plan, inputs=external_inputs, staged=staged,
                        expected_root_ref=expected_root_ref, selection=selection,
                        workspace=workspace, plugins_home=plugins_home,
                    )
                sources = _fixed_release_sources(
                    distribution=distribution, profile=profile, workspace=workspace,
                    plugins_home=plugins_home,
                    receipt_path=receipt_path, selection=selection,
                    root_ref=expected_root_ref, staging=staged,
                    previous_source_commit=previous_source_commit,
                    external_targets=external_targets,
                )
                if preflight_only:
                    if plan_digest is None:
                        raise ValueError("external preflight 需要 plan")
                    return {"status": "preflight_ok", "old_root_ref": expected_root_ref,
                            "external_plan_sha256": plan_digest,
                            "target_count": len(external_targets),
                            "selected_source_count": len(sources)}
                state = workspace.parent
                if plugins_home.parent != state or config_path.resolve().parent != state:
                    raise ValueError("release upgrade 需要同一 state 下的配置与 plugin-home")
                saved = backup_release_state(state, backup_dir)
                runner = MigrationRunner(
                    repo_root=_SOURCE_ROOT, config_path=config_path, workspace=workspace,
                    fixed_sources=sources, installed_cache_root=staged / "empty-cache",
                )
                # Core owns the journal shape. Check its pending facts before plugin data steps.
                core = runner.run_under_maintenance(maintenance, core_only=True)
                with ReloadJournal.inspect_existing(workspace) as journal:
                    if journal.pending_recovery or journal.armed_updates:
                        raise RuntimeError("pending reload 或 armed install 必须先由原 owner 结算")
                plugin = runner.run_under_maintenance(maintenance)
                adopted = adopt_bundled_distribution(
                    distribution=distribution, profile=profile, workspace=workspace,
                    plugins_home=plugins_home, config_path=config_path,
                    receipt_path=receipt_path, backup_dir=backup_dir / "adoption",
                    expected_root_ref=expected_root_ref,
                    held_locks=(maintenance, publication),
                    previous_source_commit=previous_source_commit,
                    external_targets=external_targets,
                )
                files = saved["files"]
                if not isinstance(files, list):
                    raise RuntimeError("release backup manifest files 无效")
                return {"status": adopted["status"], "migration_ids": [
                    *core.migrations, *plugin.migrations], "backup_dir": str(backup_dir),
                    "backup_files": len(files), "adoption": adopted,
                    "old_root_ref": expected_root_ref, "new_root_ref": adopted["new_root_ref"],
                    "ordered_components": adopted["ordered_components"],
                    "external_plan_sha256": plan_digest}
        finally:
            publication.release()
    finally:
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
    parser.add_argument("--adopt-bundled", action="store_true")
    parser.add_argument("--upgrade-bundled", action="store_true")
    parser.add_argument("--expected-root-ref")
    parser.add_argument("--previous-source-commit")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--external-plan", type=Path)
    parser.add_argument("--external-inputs", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if sum((args.verify_only, args.ensure_profile, args.adopt_bundled, args.upgrade_bundled)) > 1:
        parser.error("--verify-only、--ensure-profile、--adopt-bundled 与 --upgrade-bundled 互斥")
    if args.upgrade_bundled:
        if (args.config is None or args.receipt is None or args.backup_dir is None
            or args.expected_root_ref is None or args.previous_source_commit is None):
            parser.error("--upgrade-bundled 必须提供 --config、--receipt、--backup-dir、--expected-root-ref、--previous-source-commit")
        if args.core_root is not None:
            parser.error("--upgrade-bundled 不解包 Core")
        try:
            upgraded = upgrade_bundled_distribution(
                distribution=args.distribution, profile=args.profile,
                workspace=args.workspace, plugins_home=args.plugins_home,
                config_path=args.config, receipt_path=args.receipt,
                backup_dir=args.backup_dir, expected_root_ref=args.expected_root_ref,
                previous_source_commit=args.previous_source_commit,
                external_plan=args.external_plan, external_inputs=args.external_inputs,
                preflight_only=args.preflight_only,
            )
        except Exception as error:
            print(json.dumps({"status": "failed", "phase": "upgrade",
                              "error_type": type(error).__name__, "error": str(error),
                              "details": list(getattr(error, "__notes__", ())),
                              "requested_backup_dir": str(args.backup_dir)}, ensure_ascii=False))
            parser.exit(1)
        print(json.dumps(upgraded, ensure_ascii=False))
        parser.exit(0)
    if args.adopt_bundled:
        if args.config is None or args.receipt is None or args.backup_dir is None or args.expected_root_ref is None:
            parser.error("--adopt-bundled 必须提供 --config、--receipt、--backup-dir、--expected-root-ref")
        if args.core_root is not None:
            parser.error("--adopt-bundled 不解包 Core")
        try:
            adopted = adopt_bundled_distribution(
                distribution=args.distribution, profile=args.profile,
                workspace=args.workspace, plugins_home=args.plugins_home,
                config_path=args.config, receipt_path=args.receipt,
                backup_dir=args.backup_dir, expected_root_ref=args.expected_root_ref,
            )
        except Exception as error:
            details = list(getattr(error, "__notes__", ()))
            reason = (
                "selection_conflict" if isinstance(error, SelectionConflictError) else
                "selection_uncertain" if isinstance(error, SelectionWriteError) and error.outcome == "uncertain" else
                "selection_write_failed" if isinstance(error, SelectionWriteError) else
                "incomplete_or_drift" if "incomplete_or_drift" in str(error) else "invalid_state"
            )
            failed: dict[str, Any] = {"status": "failed", "reason": reason,
                                      "error_type": type(error).__name__, "error": str(error),
                                      "details": details,
                                      "expected_root_ref": args.expected_root_ref,
                                      "requested_backup_dir": str(args.backup_dir),
                                      "recovery_point_complete": True if any("recovery=" in note for note in details) else None}
            if isinstance(error, SelectionWriteError):
                failed.update({"target_ref": error.target_ref, "observed_ref": error.observed_ref,
                               "outcome": error.outcome,
                               "observation_error": repr(error.observation_error)})
            print(json.dumps(failed, ensure_ascii=False))
            parser.exit(1)
        print(json.dumps(adopted, ensure_ascii=False))
        parser.exit(3 if adopted["status"] == "partial_selected_not_started" else
                    4 if adopted["status"] == "no_eligible_targets" else 0)
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
