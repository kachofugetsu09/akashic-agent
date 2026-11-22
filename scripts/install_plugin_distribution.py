#!/usr/bin/env python3
"""验证 Core/插件发布身份，并按显式 bundle 走正式 Git 插件安装链。"""

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
from typing import Any, Literal, cast


_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from agent.plugins.bundles import absorbed_plugin_ids, composition_rows, plugin_choices, preset_plugin_ids, set_plugin_choice
from agent.plugins.prerequisites import artifact_head, clear_disabled, describe, parse_needs, record_declared, record_disabled
from agent.plugins.install import install_git_plugin, set_artifact_origin
from agent.migrations.release_backup import backup_release_state
from agent.migrations.runner import MigrationRunner
from agent.plugins.source_resolver import ResolvedPluginSource
from agent.plugins.source_resolver import scan_plugin_sources
from agent.plugins.distribution_sources import adoption_without, distribution_sources, distribution_migration_sources, DistributionSources, is_distribution_input, is_distribution_owned, check_distribution_adoption_format
from agent.plugins.files import encode_tree, tree_entries
from agent.plugin_composition.config_input import CONFIG_INPUT, load_config, save_config
from agent.migrations.runner import initialize_empty_workspace
from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.manifest import workspace_plugin_data_dir, ensure_workspace_plugin_data_dir
from agent.plugins.static_manifest import PluginSourceCompileError, load_static_plugin_manifest
from agent.plugins.dependency_constraints import (
    DependencyBatch, build_constraints, core_lock, installed_pins, normalize, verify_selected_environments,
)
from agent.plugins.dependency_paths import site_packages
from agent.plugins.resources import check_archive_dir, ensure_resources, read_manifest, repair_installed_resources
from agent.plugins.python_environment import ENVIRONMENT_FILE, INPROCESS_KEY, PythonEnvironments, inprocess_runtime
from agent.plugins.python_environment import OfflineWheels, preflight_offline_runtime, wheel_tree_sha256
from agent.plugins.input_preparation import prepare_plugin_input, PLUGIN_INPUT_API, PluginInitialConfigError
from agent.plugins.input_preparation import check_plugin_source, read_initial_config
from agent.plugins.reload_journal import ReloadJournal, PendingPublicationError, check_pending_publication
from agent.plugins.selection import PluginSelection, SelectionConflictError, SelectionWriteError
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceMaintenanceLock
from scripts.akashic_release.manifest import write_json
from scripts.akashic_release.snapshot import capture_state, restore_state
from scripts.akashic_release.wheels import WheelResolutionFailure
from utils.timing import measure

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_FORMAL_INSTALLER = "agent.plugins.install.install_git_plugin"
_KINDS = frozenset({"upgrade", "install", "switch"})


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


def load_deployment_plan(path: Path) -> tuple[str, str, list[dict[str, Any]], dict[str, Any] | None]:
    """校验部署者指定的精确外置目标。"""
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
    if not isinstance(document, dict) or set(document) - {"distribution_adoption", "retire"} != {"schema_version", "expected_root_ref", "targets"}:
        raise ValueError("部署清单需要 schema_version、expected_root_ref、targets")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("部署清单 schema_version 错误")
    root_ref = document["expected_root_ref"]
    if not isinstance(root_ref, str) or _SHA256.fullmatch(root_ref) is None:
        raise ValueError("expected_root_ref 必须是完整 Root SHA-256")
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
        fields = set(item) - {"offline_wheels", "kind", "upstream", "images", "core_lock_sha256", "inprocess_pins", "offline_resources"}
        _check_target_kind(item, bundled="bundled" in item)
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
        if "offline_resources" in item:
            resources = item["offline_resources"]
            if (not isinstance(resources, dict) or set(resources) != {"relative_path", "sha256"}
                or not isinstance(resources["sha256"], list)
                or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in resources["sha256"])):
                raise ValueError(f"offline_resources 无效: {plugin_id}")
            _external_path_syntax(resources["relative_path"])
    digest = hashlib.sha256(raw).hexdigest()
    adoption = None
    if "distribution_adoption" in document:
        value = document["distribution_adoption"]
        if not isinstance(value, dict) or set(value) != {"distribution_source_commit", "entries"}:
            raise ValueError("distribution_adoption 需要目标发行版和已批准的精确条目")
        adoption = {"schema_version": 1, "base_root_ref": root_ref, "plan_sha256": digest, **value}
        check_distribution_adoption_format(adoption)
    return digest, root_ref, targets, adoption


# 清单条目的 kind、upstream、images：首装与换来源必须带真实上游地址；bundled 只能是 upgrade。
def _check_target_kind(item: dict[str, Any], *, bundled: bool) -> None:
    kind = item.get("kind", "upgrade")
    if kind not in _KINDS or (bundled and kind != "upgrade"):
        raise ValueError(f"target kind 无效: {item.get('plugin_id')}")
    upstream, images = item.get("upstream"), item.get("images", [])
    if ("upstream" in item and (not isinstance(upstream, str) or not upstream or upstream != upstream.strip())) \
            or (kind != "upgrade" and "upstream" not in item):
        raise ValueError(f"target upstream 无效: {item.get('plugin_id')}")
    if not isinstance(images, list) or not all(isinstance(value, str) for value in images):
        raise ValueError(f"target images 无效: {item.get('plugin_id')}")


# 清单里部署者确认要随新发行版消失的在用内置插件；缺省为空。
def plan_retire_ids(path: Path) -> frozenset[str]:
    value = json.loads(path.read_text(encoding="utf-8")).get("retire", [])
    if (not isinstance(value, list) or not all(
            isinstance(item, str) and item.count("@") == 1
            and all(_PATH_SEGMENT.fullmatch(part) for part in item.split("@")) for item in value)):
        raise ValueError("部署清单 retire 必须是插件身份数组")
    return frozenset(value)


def _external_path_syntax(relative: object) -> str:
    if (not isinstance(relative, str) or not relative or relative.startswith("/")
        or "\\" in relative or "\x00" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("external input 路径格式错误")
    return relative


def _stage_deployment_targets(
    *, targets: list[dict[str, Any]], distribution: Path, inputs: Path, staged: Path, expected_root_ref: str | None,
    selection: PluginSelection, workspace: Path, plugins_home: Path,
) -> list[dict[str, Any]]:
    """在临时目录固定 bundle/wheels，并核对每个显式目标。"""

    if not targets:
        return []

    # 1. Require each target to own a coherent selected installed input.
    report = verify_distribution(distribution)
    bundled = {row["name"]: row for row in report["plugins"]}
    # 同一份批次事实：本批每个包的进程内解析结果；宿主求 wheel、预检与停机窗口内的安装都基于它。
    batch = _dependency_batch(targets)
    selected = ({} if expected_root_ref is None
                else _selected_components(selection, expected_root_ref, replacing_distribution=True,
                                       absorbed=frozenset(absorbed_plugin_ids(distribution)))[1])
    provided = {f"{row['name']}@{report['marketplace']}" for row in report["plugins"]}
    result: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        plugin_id = target["plugin_id"]
        kind = target.get("kind", "upgrade")
        name, marketplace = plugin_id.split("@")
        old_ref = None
        if kind == "upgrade":
            if plugin_id not in selected:
                raise SelectionConflictError(f"external target 未选择: {plugin_id}")
            old_ref, descriptor, old_code = selected[plugin_id]
            if (descriptor["source_type"] != "installed"
                or descriptor.get("data_dir") != workspace_plugin_data_dir(workspace, name, marketplace).relative_to(workspace).as_posix()):
                raise SelectionConflictError(f"external target 非已安装输入: {plugin_id}")
            artifact, current_code, current_source = _current_artifact(
                workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id,
            )
        else:
            _check_new_source_target(kind, plugin_id, selected, provided)
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
        if kind == "upgrade":
            same_current = old_code == artifact and _provenance(old_code) == current_source
            installed_target = current_code == target_code and _git("-C", str(artifact), "rev-parse", "HEAD") == commit
            if not same_current and not installed_target:
                raise SelectionConflictError(f"selection/cache 既不是原输入也不是本次目标: {plugin_id}")
        identity = load_static_plugin_manifest(code)
        if identity.name != name:
            raise ValueError(f"external target 静态身份不一致: {plugin_id}")
        # 宿主声明已预拉的镜像必须与包静态声明一致；其余安装期会拒绝的问题也在预检阶段暴露。
        if ("images" in target or kind != "upgrade") and sorted(identity.images) != sorted(target.get("images", [])):
            raise ValueError(f"external target 镜像声明与宿主预拉清单不一致: {plugin_id}")
        check_plugin_source(code)
        _ = read_initial_config(code)
        inprocess = inprocess_runtime(code)
        required = inprocess is not None or any(
            (code / runtime.requirements).read_text(encoding="utf-8").strip() for runtime in identity.python)
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
            # 进程内依赖：Core 锁摘要与解析结果必须齐全，且就是这份镜像里的锁、这批 wheel（缺失即拒绝）。
            constraints = None
            if inprocess is not None:
                _check_inprocess_entry(target, wheels.directory, plugin_id)
                constraints = build_constraints(workspace, identity.name, PythonEnvironments(workspace),
                                                _scratch(staged, f"external-{index}-constraints"), batch, plugins_home)
            for runtime_index, runtime in enumerate((*identity.python, *(() if inprocess is None else (inprocess,)))):
                if not (code / runtime.requirements).read_text(encoding="utf-8").strip():
                    continue
                destination = staged / f"external-{index}-download-{runtime_index}"
                destination.mkdir()
                # 预检用与安装期相同的约束：预检通过就意味着停机窗口内的约束安装也能通过。
                preflight_offline_runtime(code, runtime, wheels, destination,
                                          constraints if runtime is inprocess else None)
        _publish_target_resources(code, target, inputs, staged, index, workspace, plugin_id)
        result.append({"plugin_id": plugin_id, "old_ref": old_ref, "bundle": local,
                       "commit": commit, "code": code, "kind": kind, "upstream": target.get("upstream"), "inprocess_pins": target.get("inprocess_pins"), "needs": parse_needs(target.get("needs", []), plugin_id), "unmet": (None if target.get("unmet") is None else parse_needs(target["unmet"], plugin_id)),
                       "wheels": wheels, "bundle_sha256": bundle_digest})
    return result


# 本批每个目标的新状态：带进程内依赖的是它的解析结果，不带的（升级后去掉依赖、首装无依赖）是空映射——
# 它们都替换盘上旧选择里的 pin，旧 pin 不能作为"不在本批的已装包"留在约束里。
def _dependency_batch(targets: list[dict[str, Any]]) -> DependencyBatch:
    return DependencyBatch({str(item["plugin_id"]).partition("@")[0]: item.get("inprocess_pins") or {} for item in targets})


def _scratch(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    return path


# 声明了资源的目标：清单必须带归档目录，每份哈希与声明一致；在预检里就发布到持久根（停机窗口内复用，不再需要网络）。
def _publish_target_resources(
    code: Path, target: dict[str, Any], inputs: Path, staged: Path, index: int, workspace: Path, plugin_id: str,
) -> None:
    spec = target.get("offline_resources")
    if bool(read_manifest(code)) != (spec is not None):
        raise ValueError(f"external target 资源输入与 resources.json 不匹配: {plugin_id}")
    if spec is None:
        return
    source = _external_path(inputs, spec["relative_path"], directory=True)
    copied = staged / f"external-{index}-resources"
    shutil.copytree(source, copied)
    _ = check_archive_dir(code, copied, spec["sha256"])
    _ = ensure_resources(code, workspace, archives=copied)


# 进程内依赖条目的完整性：有效的 Core 锁摘要（且等于这份镜像里的锁）、解析结果是 {包: 版本} 且与 wheel 目录一致。
def _check_inprocess_entry(target: dict[str, Any], wheels: Path, plugin_id: str) -> None:
    recorded, pins = target.get("core_lock_sha256"), target.get("inprocess_pins")
    lock, _ = core_lock()
    if not isinstance(recorded, str) or _SHA256.fullmatch(recorded) is None:
        raise ValueError(f"external target 带进程内依赖但缺少有效的 core_lock_sha256: {plugin_id}")
    if lock is None or hashlib.sha256(lock.read_bytes()).hexdigest() != recorded:
        raise ValueError(f"external target 的 wheel 是按另一份 Core 锁解析的: {plugin_id}")
    if not isinstance(pins, dict) or not pins or not all(isinstance(k, str) and isinstance(v, str) for k, v in pins.items()):
        raise ValueError(f"external target 缺少进程内依赖解析结果 inprocess_pins: {plugin_id}")
    # 同一个目录可能合法地含同名不同版本（进程内与子进程 runtime 各自解析）：按 (名字, 版本) 对核对，与目录顺序无关。
    present = {(normalize(parts[0]), parts[1]) for parts in (item.name.split("-") for item in wheels.glob("*.whl"))}
    if any((normalize(name), version) not in present for name, version in pins.items()):
        raise ValueError(f"external target 的 inprocess_pins 与 wheel 目录不一致: {plugin_id}")


# 首装与换来源的前提：目标发行版不再自带该身份；换来源要求旧选择是内置输入，首装要求选择里没有。
def _check_new_source_target(
    kind: str, plugin_id: str, selected: Mapping[str, tuple[str, Mapping[str, object], Path]], provided: set[str],
) -> None:
    old = selected.get(plugin_id)
    if plugin_id in provided:
        raise SelectionConflictError(f"目标发行版仍自带该插件，不能从外置来源接管: {plugin_id}")
    if kind == "install" and old is not None:
        raise SelectionConflictError(f"首装目标已在选择里: {plugin_id}")
    if kind == "switch" and (old is None or old[1]["source_type"] != "builtin"):
        raise SelectionConflictError(f"换来源目标不是内置输入: {plugin_id}")


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
    """Measure complete artifact verification at the publication boundary."""
    with measure("distribution.verify"):
        return _verify_distribution(distribution)


def _verify_distribution(distribution: Path) -> dict[str, Any]:
    """核对报告、Core tar 和每个独立 bundle 的不可变身份。"""

    root = distribution.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"distribution 不是目录: {root}")
    report_path = _distribution_file(root, "distribution.json", "distribution report")
    report = _read_json(report_path)
    if type(report.get("schema_version")) is not int or report["schema_version"] != 3:
        raise ValueError("需要包含 TOML bundle 的 distribution schema 3")
    marketplace = report.get("marketplace")
    if not isinstance(marketplace, str) or _PATH_SEGMENT.fullmatch(marketplace) is None:
        raise ValueError("distribution marketplace 无效")
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

    bundles = report.get("bundles", [])
    if not isinstance(bundles, list):
        raise ValueError("distribution bundles 必须是 array")
    for index, raw in enumerate(bundles):
        if not isinstance(raw, dict):
            raise ValueError(f"bundles[{index}] 必须是 object")
        profile_path = _distribution_file(
            root, raw.get("path"), f"bundles[{index}]"
        )
        _check_sha256(profile_path, raw.get("sha256"), f"bundles[{index}]")
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
    _tar_names(core)
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
        for relative in ("static/chat/chat-app.js", "static/dashboard/index.html"):
            if not (target / relative).is_file():
                raise ValueError(f"Core artifact 缺少 Web 静态产物: {relative}")
    return target


def _code_identity(root: Path) -> str:
    """Use the archive owner's tree identity without writing an archive."""
    entries = tree_entries(root, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE}))
    return hashlib.sha256(encode_tree(entries)).hexdigest()


def _distribution_code_identity(root: Path) -> str:
    """Compare executable input without the release's provenance stamp."""
    entries = tree_entries(root, exclude=frozenset({
        ".venv", "node_modules", ENVIRONMENT_FILE, ".akashic-source.json",
    }))
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


def install_bundle(
    distribution: Path,
    bundle_file: Path,
    *,
    workspace: Path,
    plugins_home: Path,
    config_path: Path,
    initialize_workspace: bool = True,
) -> dict[str, Any]:
    """记录发行组合的首次安装，不复制制品到全局 cache。"""

    distribution_root = distribution.expanduser().resolve(strict=True)
    report = verify_distribution(distribution_root)
    bundle_path = bundle_file.expanduser().resolve(strict=True)
    if bundle_file.stem != os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base"):
        raise ValueError("bundle 与 AKASHIC_PLUGIN_BUNDLE 不一致")
    bundle_rows = report.get("bundles", [])
    if not any(
        isinstance(item, dict)
        and _distribution_file(distribution_root, item.get("path"), "bundle") == bundle_path
        for item in bundle_rows
    ):
        raise ValueError("bundle 不属于已验证的 distribution artifact")
    bundle_name, marketplace = bundle_file.stem, report["marketplace"]
    declarations = composition_rows(workspace, distribution_root)
    entries = tuple(row for row in declarations if not row.disabled and row.plugin.rpartition("@")[2] == marketplace)
    initialization = {"plugin_configs": [
        {"owner": row.plugin.split("@")[0], "config": dict(row.config)}
        for row in entries if row.config
    ]}
    workspace = workspace.expanduser().resolve(strict=False)
    plugins_home = plugins_home.expanduser().resolve(strict=False)
    config_path = config_path.expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise ValueError(f"runtime config 必须是普通文件: {config_path}")
    workspace.mkdir(parents=True, exist_ok=True)
    if initialize_workspace:
        initialize_empty_workspace(
            repo_root=_SOURCE_ROOT,
            workspace=workspace,
            config_path=config_path,
        )
    plugins_home.mkdir(parents=True, exist_ok=True)

    # 宿主 journal 的初始化不依赖至少安装一个插件。
    ReloadJournal(workspace)
    plugin_configs = initialization.get("plugin_configs", [])
    assert isinstance(plugin_configs, list)
    config_results = _write_plugin_configs(
        workspace,
        marketplace=marketplace,
        declarations=plugin_configs,
    )
    return {
        "schema_version": 2,
        "distribution_source_commit": report["source_commit"],
        "distribution_source_tree": report["source_tree"],
        "bundle": bundle_name,
        "marketplace": marketplace,
        "workspace": str(workspace),
        "plugins_home": str(plugins_home),
        "initialization": initialization,
        "plugin_configs": config_results,
        "formal_installer": "scripts.install_plugin_distribution.install_bundle",
        "installed": [],
    }


def _validate_receipt_state(
    receipt: dict[str, Any],
    *,
    workspace: Path,
    plugins_home: Path,
) -> None:
    """Validate one historical receipt without treating it as current composition."""

    version = receipt.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("distribution receipt schema_version 无效")
    for key in ("distribution_source_commit", "distribution_source_tree"):
        value = receipt.get(key)
        if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
            raise ValueError(f"distribution receipt {key} 无效")
    profile_name = receipt.get("profile" if version == 1 else "bundle")
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
    installer = _FORMAL_INSTALLER if version == 1 else "scripts.install_plugin_distribution.install_bundle"
    if receipt.get("formal_installer") != installer:
        raise ValueError("distribution receipt 缺少正式安装器身份")

    installed = receipt.get("installed")
    if not isinstance(installed, list) or (version == 2 and installed):
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
            or (version == 1 and owner not in historical_names)
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


def _selected_components(
    selection: PluginSelection, root_ref: str, *, replacing_distribution: bool = False, absorbed: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], dict[str, tuple[str, Mapping[str, object], Path]]]:
    """外置输入检查实际目录；被替换的镜像输入只读原选择身份；已被 Core 吸收的外置身份在最早处剔除，不做任何校验。"""
    components = selection.components(root_ref)
    if not isinstance(components, tuple):
        raise ValueError("stable 完整记录格式无效")
    found: dict[str, tuple[str, Mapping[str, object], Path]] = {}
    checked_refs: list[str] = []
    for ref in components:
        if not isinstance(ref, str):
            raise ValueError("stable component ref 无效")
        record = selection.read_input(ref)
        plugin_id, code_ref = record["plugin_id"], record["code"]
        if record["version"] not in {5, 6} or not isinstance(plugin_id, str) or not isinstance(code_ref, str):
            raise ValueError(f"selected descriptor 格式无效: {ref}")
        # 已吸收的外置身份本来就要被移出：它的 runtime、artifact、静态身份都不再校验（旧 Python 或 binding_api 下也一样）。
        # ref 仍留在返回的 components 里，新旧选择比较时才会发现它被移出并提交；它不进入 found，后面的候选、迁移、环境引用都看不到它。
        if record.get("source_type") == "installed" and plugin_id in absorbed:
            checked_refs.append(ref)
            continue
        if plugin_id in found:
            raise ValueError(f"stable 重复插件身份: {plugin_id}")
        replacing = replacing_distribution and is_distribution_owned(record)
        code = Path(code_ref).resolve(strict=not replacing)
        if not replacing:
            identity = load_static_plugin_manifest(code)
            if identity.name != plugin_id.split("@", 1)[0]:
                raise ValueError(f"selected 静态身份不一致: {plugin_id}")
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


def _distribution_candidate(
    *, distribution: Path, workspace: Path, plugins_home: Path,
    selected: dict[str, tuple[str, Mapping[str, object], Path]],
    replacement_ids: frozenset[str] = frozenset(),
    adoption: Mapping[str, object] | None = None,
) -> tuple[DistributionSources, dict[str, ResolvedPluginSource]]:
    """Overlay fixed distribution sources while retaining exact external selections."""
    available = distribution_sources(workspace, plugins_home, distribution, adoption=adoption,
                                     replacing_distribution=True)
    scan = scan_plugin_sources(installed_cache_root=plugins_home / "cache",
                               ignored_installed_roots=available.ignored_installed_roots)
    if scan.failures:
        raise RuntimeError(f"installed source is unavailable: {scan.failures}")
    installed = {f"{source.plugin_name}@{source.marketplace}": source for source in scan.sources}
    external_names = {source.plugin_name for source in scan.sources}
    choices = plugin_choices(workspace, distribution)
    candidate: dict[str, ResolvedPluginSource] = {}
    for plugin_id, (_, descriptor, code) in selected.items():
        name, separator, marketplace = plugin_id.rpartition("@")
        if not separator:
            name, marketplace = plugin_id, ""
        if is_distribution_owned(descriptor) or plugin_id in available.legacy_ids:
            if plugin_id in installed:
                raise SelectionConflictError(f"distribution/installed selection drift: {plugin_id}")
            if name in external_names:
                raise SelectionConflictError(f"ambiguous selected distribution and installed name: {name}")
            if plugin_id in available.legacy_ids and descriptor["source_type"] == "installed":
                current_artifact, _, _ = _current_artifact(workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id)
                if current_artifact != Path(cast(str, descriptor["code"])):
                    raise SelectionConflictError(f"legacy distribution selection/cache drift: {plugin_id}")
            # No source in the current artifact means retirement, never data deletion.
            continue
        # Explicit targets get their own original-or-exact-target cache proof in
        # _stage_deployment_targets before any persistent mutation. They are not
        # preserved inputs: their old interpreter/code may be what is replaced.
        if descriptor["source_type"] == "installed" and plugin_id not in replacement_ids:
            current_artifact, _, _ = _current_artifact(workspace=workspace, plugins_home=plugins_home, plugin_id=plugin_id)
            if current_artifact != code:
                # The installer may have committed the new pointer before live activation failed.
                base = plugins_home / "cache" / marketplace / name
                with ReloadJournal.inspect_existing(workspace) as journal:
                    if not journal.owns_installed_artifact(plugin_id, base, current_artifact):
                        raise SelectionConflictError(f"external selection/cache drift: {plugin_id}")
                code = current_artifact
        if (plugin_id not in replacement_ids and
            descriptor["runtime"] != {"python_tag": sys.implementation.cache_tag, "binding_api": PLUGIN_INPUT_API}):
            raise RuntimeError(f"preserved plugin runtime is incompatible with this Core: {plugin_id}; explicit reinstall required")
        if choices.get(plugin_id, False):
            candidate[plugin_id] = ResolvedPluginSource(code, cast(Literal["builtin", "installed"], descriptor["source_type"]), marketplace, name,
                                                       load_static_plugin_manifest(code))
    for plugin_id, source in installed.items():
        if plugin_id not in selected and choices.get(plugin_id, False):
            candidate[plugin_id] = source
    for source in available.sources:
        plugin_id = f"{source.plugin_name}@{source.marketplace}"
        # Match existing discovery precedence, including a disabled installed override.
        if source.plugin_name in external_names or not choices.get(plugin_id, False):
            continue
        if plugin_id in candidate:
            raise SelectionConflictError(f"distribution identity already selected from another source: {plugin_id}")
        candidate[plugin_id] = source
    return available, candidate


# 在用的内置输入（选择里的发行版来源输入、仍被启用）若目标发行版不再提供、又没有被换来源，就会随这次发布消失。
# 必须在清单里显式确认（retire）；手工清单或 --no-extra 漏掉时在预检拦住，不依赖宿主侧。
def _dropped_inuse(
    *, selected: Mapping[str, tuple[str, Mapping[str, object], Path]], candidate: Mapping[str, ResolvedPluginSource],
    workspace: Path, distribution: Path, acknowledged: frozenset[str],
) -> list[str]:
    choices = plugin_choices(workspace, distribution)
    names = {source.plugin_name for source in candidate.values()}
    return sorted(
        plugin_id for plugin_id, (_, descriptor, _) in selected.items()
        if is_distribution_owned(descriptor) and choices.get(plugin_id, False) and plugin_id not in candidate
        and plugin_id.rpartition("@")[0] not in names and plugin_id not in acknowledged)


def _require_inuse_covered(
    *, selected: Mapping[str, tuple[str, Mapping[str, object], Path]], candidate: Mapping[str, ResolvedPluginSource],
    workspace: Path, distribution: Path, acknowledged: frozenset[str],
) -> None:
    dropped = _dropped_inuse(selected=selected, candidate=candidate, workspace=workspace,
                             distribution=distribution, acknowledged=acknowledged)
    if dropped:
        raise RuntimeError("在用的内置插件目标发行版不再提供，也没有换来源或被确认移除，发布后会消失: "
                           + ", ".join(dropped) + "。从 extra 换来源，或在清单 retire 里（akashic-release install --allow-retire）确认")


# preset 启用的行必须真正进入候选组合，首装和发布都不能静默少装；同名外置覆盖是用户的显式替换。
def _require_preset_rows(
    *, candidate: Mapping[str, ResolvedPluginSource], workspace: Path, distribution: Path,
) -> None:
    choices = plugin_choices(workspace, distribution)
    names = {source.plugin_name for source in candidate.values()}
    missing = sorted(
        plugin_id for plugin_id in preset_plugin_ids(distribution)
        if choices.get(plugin_id, False) and plugin_id not in candidate
        and plugin_id.rpartition("@")[0] not in names
    )
    if missing:
        raise RuntimeError("preset 启用的插件没有进入选择，停止发布: " + ", ".join(missing)
                           + "。发行版已不再自带的插件需先从 extra 仓库装上：用 scripts/start.py（自带补装），"
                           "或给 --ensure-bundle 传 --extra-repo / --extra-inputs，生产部署用 akashic-release install")


def _prepare_distribution_inputs(
    *, available: DistributionSources, candidate: dict[str, ResolvedPluginSource],
    selected: dict[str, tuple[str, Mapping[str, object], Path]],
    distribution: Path, workspace: Path, plugins_home: Path, selection: PluginSelection,
) -> tuple[str, ...]:
    """Prepare immutable inputs, then let the caller publish one complete selection."""
    for row in composition_rows(workspace, distribution):
        if row.config and not row.disabled and row.plugin in candidate:
            name, marketplace = row.plugin.rsplit("@", 1)
            _write_plugin_configs(workspace, marketplace=marketplace,
                                  declarations=[{"owner": name, "config": dict(row.config)}])
    environments = _prepare_distribution_environments(available, distribution, workspace, selected)
    prepared: dict[str, str] = {}
    for plugin_id, source in candidate.items():
        old = selected.get(plugin_id)
        if not source.distribution_source and old is not None and old[2] == source.plugin_root:
            prepared[plugin_id] = old[0]
            continue
        data_dir = workspace_plugin_data_dir(workspace, source.plugin_name, source.marketplace)
        ensure_workspace_plugin_data_dir(data_dir, workspace)
        identity = source.static_manifest
        assert identity is not None
        code, source_commit = source.plugin_root, source.distribution_source
        if old is not None and old[1]["data_dir"] != data_dir.relative_to(workspace).as_posix():
            raise SelectionConflictError(f"distribution data identity changed: {plugin_id}")
        # The old immutable input already passed compilation. Reuse only when
        # code, post-migration config, dependencies, and the Core contract match.
        with measure("plugin.input", plugin=plugin_id) as timing:
            if old is not None and (
                old[1]["source_type"] == "builtin"
                and old[1].get("distribution_source") == source_commit
                and is_distribution_input(old[1], old[2])
                and old[1]["runtime"] == {"python_tag": sys.implementation.cache_tag,
                                           "binding_api": PLUGIN_INPUT_API}
                and code == old[2]
                and load_config(data_dir)[1] == old[1]["config_revision"]
                and dict(cast(Mapping[str, str], old[1]["python_environments"])) == environments.get(plugin_id, {})
            ):
                prepared[plugin_id] = old[0]
                timing.update(reused=True, ref=old[0])
                continue
            try:
                result = prepare_plugin_input(
                    {"name": source.plugin_name, "marketplace": source.marketplace,
                     "plugin_root": str(code), "module_path": str(code / "plugin.py"),
                     "manifest_digest": identity.identity_digest, "source_type": source.source_type,
                     **({"distribution_source": source_commit, "wheel_tree_sha256": source.wheel_tree_sha256}
                        if source.distribution_source else {})},
                    workspace=workspace, selection=selection, initial=old is None,
                )
            except (PluginSourceCompileError, PluginInitialConfigError) as error:
                if source.source_type != "installed":
                    raise
                # Existing bad artifacts stay installed and enabled for repair through the live manager.
                print(f"外置插件无法准备输入: {plugin_id}: {error}", file=sys.stderr, flush=True)
                continue
            timing.update(reused=False, ref=result.input_ref)
        # 停止期已结算配置 owner；读取迁移后的持久输入，也支持迁移成功后的发布重试。
        prepared[plugin_id] = result.input_ref
    ordered = [prepared.pop(plugin_id) for plugin_id in selected if plugin_id in prepared]
    return tuple(ordered + [prepared[plugin_id] for plugin_id in sorted(prepared)])


def _can_boot_for_recovery(
    selected: dict[str, tuple[str, Mapping[str, object], Path]],
    candidate: dict[str, ResolvedPluginSource],
    choices: Mapping[str, bool],
) -> bool:
    """Let the existing runtime finish cleanup and honor an explicit disable."""
    # The manager settles old owners first, then drops disabled installed inputs.
    expected = {
        plugin_id for plugin_id, (_, descriptor, _) in selected.items()
        if descriptor["source_type"] != "installed" or choices.get(plugin_id) is not False
    }
    if expected != candidate.keys():
        return False
    for plugin_id, source in candidate.items():
        descriptor = selected[plugin_id][1]
        if (descriptor["source_type"] != source.source_type
            or (not source.distribution_source and Path(cast(str, descriptor["code"])) != source.plugin_root.resolve())
            or (source.distribution_source and (
                descriptor.get("distribution_source") != source.distribution_source
                or not is_distribution_input(descriptor, selected[plugin_id][2])
                or _distribution_code_identity(selected[plugin_id][2]) != _distribution_code_identity(source.plugin_root)))
            or descriptor["runtime"] != {"python_tag": sys.implementation.cache_tag,
                                         "binding_api": PLUGIN_INPUT_API}):
            return False
    return True


def _check_distribution_sources(distribution: Path, report: dict[str, Any]) -> None:
    """镜像构建时核对解包源码与 bundle；镜像只读，开机和发布不再重复（ADR 0077）。"""
    for row in report["plugins"]:
        with measure("distribution.source", plugin=row["name"]):
            source = distribution / "sources" / row["name"]
            if _provenance(source) != {"commit": report["source_commit"], "path": row["source_path"]}:
                raise ValueError(f"distribution source provenance mismatch: {row['name']}")
            with tempfile.TemporaryDirectory(prefix="akashic-source-check-") as temporary:
                root = Path(temporary) / "source"
                _git("clone", "--no-local", "--no-checkout", str(distribution / row["file"]), str(root))
                _git("-C", str(root), "checkout", "--detach", row["source_revision"])
                if _code_identity(root) != _code_identity(source):
                    raise ValueError(f"distribution source/bundle mismatch: {row['name']}")
            for entry in source.rglob("*.py"):
                compile(entry.read_bytes(), str(entry), "exec")

def _prepare_distribution_environments(
    available: DistributionSources, distribution: Path, workspace: Path,
    selected: dict[str, tuple[str, Mapping[str, object], Path]],
) -> dict[str, dict[str, str]]:
    """Warm immutable caches before downtime, including disabled plugin environments."""
    owner = PythonEnvironments(workspace)
    prepared: dict[str, dict[str, str]] = {}
    for source in available.sources:
        identity = source.static_manifest
        assert identity is not None
        plugin_id = f"{source.plugin_name}@{source.marketplace}"
        code = source.plugin_root
        refs: dict[str, str] = {}
        for runtime in identity.python:
            wheels = None
            if (source.plugin_root / runtime.requirements).read_text().strip():
                wheels = OfflineWheels(distribution / "wheels" / source.plugin_name, source.wheel_tree_sha256)
            with measure("distribution.environment", plugin=plugin_id, runtime=runtime.runtime_root):
                refs[runtime.runtime_root] = owner.prepare(code, runtime, offline_wheels=wheels)
        prepared[plugin_id] = refs
    return prepared


# Both publication locks are held and the host has verified that this exact boot exited.
def _settle_closed_boot(workspace: Path, boot_id: str) -> None:
    from bootstrap.runtime_readiness import read_closed_boot

    _, required = read_closed_boot(workspace, boot_id)
    journal = ReloadJournal(workspace)
    pending = journal.pending_recovery()
    receipt = f"release-closed:{boot_id}"
    by_id = {action.tx_id: action for action in pending}
    for tx_id in required:
        action = by_id.get(tx_id)
        if action is None:
            # A stopped publish can be retried after it has already settled this receipt.
            if any(event.details.get("cleanup_receipt") == receipt and
                   event.details.get("event") == "generation_cleanup_settled" for event in journal.events(tx_id)):
                continue
        elif journal.is_boot_generation_cleanup(action, boot_id):
            continue
        raise PendingPublicationError("关闭回执所列事务不是该 boot 的纯运行期代清理")
    for action in pending:
        _ = journal.settle_exited_generation(action, boot_id, receipt)


# 把已核对的清单目标装成 installed 输入，并写入候选组合与已选输入；发布与首装/启动补装共用。
def _install_targets(
    targets: list[dict[str, Any]], *, workspace: Path, plugins_home: Path, selection: PluginSelection,
    selected: dict[str, tuple[str, Mapping[str, object], Path]], candidate: dict[str, ResolvedPluginSource],
    distribution_root: Path | None = None,
) -> tuple[list[tuple[str, list[str], str]], list[str], list[tuple[str, list[str], str]]]:
    batch = _dependency_batch(targets)
    closed: list[tuple[str, list[str], str]] = []
    met: list[str] = []
    declared: list[tuple[str, list[str], str]] = []
    for target in targets:
        plugin_id = target["plugin_id"]
        name, marketplace = plugin_id.split("@")
        # Matching code does not prove an existing Python environment belongs to this interpreter.
        # 清单是部署者信任的显式替换：目标发行版是否自带已在预检核对，这里放行防接管。
        with measure("extra.install", plugin=name):
            installed = install_git_plugin(
                workspace=workspace, source=str(target["bundle"]), marketplace=marketplace,
                ref_name=target["commit"], plugins_home=plugins_home,
                refresh_existing_artifact=False, offline_wheels=target["wheels"],
                takeover=frozenset({plugin_id}), dependency_batch=batch,
            )
        artifact = installed.installed_path
        if target["upstream"]:
            set_artifact_origin(artifact, target["upstream"])
        # 首次安装的包若声明了外部前置条件而宿主不满足：照装（artifact 已落地），但写 disabled 行，不进选择；
        # 判定在宿主侧预取时做好并随目标带来（unmet），这里只读取，不自己探测；原因记录等选择提交后再写（见 ensure_bundle）。
        # 只针对 kind=install 的新增包，已在用的包不受影响。
        if target["kind"] == "install" and target.get("needs"):
            declared.append((plugin_id, list(target["needs"]), artifact_head(artifact)))
        missing = [str(item) for item in (target.get("unmet") or ())] if target["kind"] == "install" else []
        if target["kind"] == "install" and not missing and target.get("unmet") is not None:
            met.append(plugin_id)
        if missing:
            _ = set_plugin_choice(workspace, plugin_id, enabled=False, distribution=distribution_root)
            closed.append((plugin_id, missing, artifact_head(artifact)))
            candidate.pop(plugin_id, None)
            print(f"{plugin_id}: 已安装但未启用：{describe(missing)}", file=sys.stderr, flush=True)
            continue
        prepared = prepare_plugin_input(
            {"name": name, "plugin_root": str(artifact), "module_path": str(artifact / "plugin.py"),
             "manifest_digest": load_static_plugin_manifest(artifact).identity_digest,
             "marketplace": marketplace, "source_type": "installed"},
            workspace=workspace, selection=selection, initial=target["kind"] == "install",
        )
        if prepared.plugin_id != plugin_id:
            raise RuntimeError(f"安装输入身份不符: {plugin_id}")
        ReloadJournal(workspace).set_input_ref(installed.update_id, prepared.input_ref)
        selected[plugin_id] = (prepared.input_ref, selection.read_input(prepared.input_ref), prepared.code_dir)
        candidate[plugin_id] = ResolvedPluginSource(prepared.code_dir, "installed", marketplace, name,
                                                   load_static_plugin_manifest(prepared.code_dir))
    # 提交选择前整体核对一次：新选择里所有进程内环境，同一个包不能有两个版本。
    if any(batch.pins.values()):
        environments = PythonEnvironments(workspace)
        verify_selected_environments({
            pid: installed_pins(site_packages(environments.open(ref)))
            for pid, (_, record, _) in selected.items()
            if (ref := cast(Mapping[str, str], record["python_environments"]).get(INPROCESS_KEY)) is not None})
    return closed, met, declared


# 最终提交的归属凭证：部署清单自带的转换凭证原样提交；否则从已提交凭证里去掉本次首装或换来源的身份
# （历史归属对未在选择里的身份依然保留，首装不释放就会让指针校验失败）。
def _final_adoption(
    selection: PluginSelection, adoption: Mapping[str, object] | None, switched: set[str],
) -> tuple[Mapping[str, object] | None, bool]:
    if adoption is not None:
        return adoption, False
    current = selection.adoption()
    stripped = current
    for plugin_id in switched:
        stripped = adoption_without(stripped, plugin_id)
    return (stripped, True) if stripped != current else (None, False)


def publish_distribution(
    *, distribution: Path, workspace: Path, plugins_home: Path,
    config_path: Path, plan: Path, inputs: Path,
    backup_dir: Path | None = None, preflight_only: bool = False, closed_boot: str | None = None,
) -> dict[str, Any]:
    """更新内置 preset 和指定外置输入；先完成 Core 与内置迁移。"""
    digest, expected_root, requested, adoption = load_deployment_plan(plan)
    workspace, plugins_home = workspace.resolve(strict=True), plugins_home.resolve(strict=True)
    selection = PluginSelection(workspace)
    if selection.read() != expected_root:
        raise SelectionConflictError("当前 Root 与部署清单基线不同")
    maintenance = WorkspaceMaintenanceLock(workspace)
    publication = PluginPublicationLock(plugins_home)
    # Online preparation adds immutable caches only; state and selection stay fixed.
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
                with measure("publish.verify"):
                    report = verify_distribution(distribution)
                if adoption is not None and adoption["distribution_source_commit"] != report["source_commit"]:
                    raise SelectionConflictError("历史归属转换的目标发行版不符")
                components, selected = _selected_components(selection, expected_root, replacing_distribution=True,
                                                    absorbed=frozenset(absorbed_plugin_ids(distribution)))
                with measure("publish.candidate"):
                    available, candidate = _distribution_candidate(
                        distribution=distribution, workspace=workspace, plugins_home=plugins_home, selected=selected,
                        replacement_ids=frozenset(item["plugin_id"] for item in requested),
                        adoption=adoption,
                    )
                    _prepare_distribution_environments(available, distribution, workspace, selected)
                external_requests = [item for item in requested if not (
                    item.get("bundled") and item["plugin_id"] in candidate
                    and candidate[item["plugin_id"]].source_type == "builtin")]
                targets = _stage_deployment_targets(
                    targets=external_requests, distribution=distribution, inputs=inputs, staged=staged,
                    expected_root_ref=expected_root, selection=selection,
                    workspace=workspace, plugins_home=plugins_home,
                )
                replacements = {item["plugin_id"]: item["code"] for item in targets}
                with measure("publish.migration_sources"):
                    migration_sources = distribution_migration_sources(workspace, plugins_home, distribution,
                                                                        adoption=adoption, replacing_distribution=True)
                reserved_ids = {f"{item.plugin_name}@{item.marketplace}" for item in migration_sources}
                reserved_ids.update(available.legacy_ids)
                # 首装与换来源在预检里已核对目标发行版不再自带；只有已装升级仍按内置数据身份拒绝。
                taken_over = frozenset(item["plugin_id"] for item in targets if item["kind"] != "upgrade")
                if reserved_ids.difference(taken_over).intersection(replacements):
                    raise SelectionConflictError("外置部署目标不能接管内置数据身份；替代插件须使用独立身份")
                for plugin_id, code in replacements.items():
                    name, marketplace = plugin_id.rsplit("@", 1)
                    candidate[plugin_id] = ResolvedPluginSource(code, "installed", marketplace, name,
                                                               load_static_plugin_manifest(code))
                _require_preset_rows(candidate=candidate, workspace=workspace, distribution=distribution)
                _require_inuse_covered(selected=selected, candidate=candidate, workspace=workspace,
                                       distribution=distribution, acknowledged=plan_retire_ids(plan))
                runner = MigrationRunner(repo_root=_SOURCE_ROOT, config_path=config_path,
                                     plugins_home=plugins_home, bundle_directory=distribution / "bundles",
                                         workspace=workspace, fixed_sources=migration_sources,
                                         replacement_sources=tuple(candidate[item] for item in replacements),
                                         distribution_state=available)
                with measure("publish.migrations.check"):
                    pending = runner.check()
                result: dict[str, Any] = {
                    "status": "preflight_ok", "plan_sha256": digest,
                    "old_root_ref": expected_root, "migration_ids": list(pending),
                    "target_ids": list(replacements), "backup_dir": None,
                    "distribution_source_commit": report["source_commit"],
                    "distribution_ids": [key for key, source in candidate.items() if source.distribution_source],
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
                with measure("publish.migrations.run"):
                    runner.run_under_maintenance(maintenance, core_only=True)
                    if closed_boot is not None:
                        _settle_closed_boot(workspace, closed_boot)
                    check_pending_publication(workspace)
                    runner.run_under_maintenance(maintenance)
                # 迁移可能改变声明输入；只按迁移后的选择提交，显式外置目标重新核对启停。
                with measure("publish.candidate.migrated"):
                    available, candidate = _distribution_candidate(
                        distribution=distribution, workspace=workspace, plugins_home=plugins_home, selected=selected,
                        replacement_ids=frozenset(replacements), adoption=adoption,
                    )
                    choices = plugin_choices(workspace, distribution)
                for plugin_id, code in replacements.items():
                    if not choices.get(plugin_id, False):
                        raise SelectionConflictError(f"迁移后的外置目标未启用: {plugin_id}")
                    name, marketplace = plugin_id.rsplit("@", 1)
                    candidate[plugin_id] = ResolvedPluginSource(code, "installed", marketplace, name,
                                                               load_static_plugin_manifest(code))
                _require_preset_rows(candidate=candidate, workspace=workspace, distribution=distribution)
                prepared_selected = dict(selected)
                _install_targets(targets, workspace=workspace, plugins_home=plugins_home, selection=selection,
                                 selected=prepared_selected, candidate=candidate)
                with measure("publish.inputs"):
                    new_components = _prepare_distribution_inputs(
                        available=available, candidate=candidate, selected=prepared_selected,
                        distribution=distribution, workspace=workspace, plugins_home=plugins_home, selection=selection,
                    )
                # 换来源与最终选择提交合并成这一次 CAS：新输入与历史归属条目的移除同时发布。
                final_adoption, replace = _final_adoption(
                    selection, adoption, {item["plugin_id"] for item in targets if item["kind"] != "upgrade"})
                new_root = (selection.commit(new_components, expected_ref=expected_root,
                                             distribution_adoption=final_adoption, replace_adoption=replace)
                            if new_components != components or adoption is not None or replace else expected_root)
                return {**result, "status": "selected_not_started", "new_root_ref": new_root,
                        "ordered_components": list(new_components)}
        finally:
            if not preflight_only:
                publication.release()
    finally:
        if not preflight_only:
            maintenance.release()


# 启动补装的暂存目录放在状态盘：服务容器的 /tmp 是 256MB 内存盘，放不下依赖 wheel 的下载与暂存两份副本。
# 放在插件目录而不是 workspace：补装期间默认临时目录指向这里，而 journal 预检要求临时副本不在 workspace 内。
def _extra_staging_root(workspace: Path, plugins_home: Path) -> Path | None:
    if plugins_home.resolve().is_relative_to(workspace.resolve()):
        return None
    plugins_home.mkdir(parents=True, exist_ok=True)
    # 调用方持有插件目录的发布锁；上次被强杀留下的暂存目录在这里回收。
    for stale in plugins_home.glob(".extra-staging-*"):
        shutil.rmtree(stale, ignore_errors=True)
    return Path(tempfile.mkdtemp(prefix=".extra-staging-", dir=plugins_home))


def ensure_bundle(
    distribution: Path,
    bundle_file: Path,
    *,
    workspace: Path,
    plugins_home: Path,
    config_path: Path,
    receipt_path: Path,
    extra_repo: str | None = None,
    extra_inputs: Path | None = None,
    allow_retire: bool = False,
    install_extra: tuple[str, ...] = (),
    profile_patch: Path | None = None,
) -> dict[str, Any]:
    """First-install defaults, then compose the deployed distribution with external inputs.

    给了 extra_repo 或 extra_inputs 时，发行版已不再自带、而 preset 仍启用的插件先从 extra 装上；
    没给则保持原行为（缺失即停止发布）。
    """
    workspace = workspace.expanduser().resolve(strict=False)
    plugins_home = plugins_home.expanduser().resolve(strict=False)
    distribution = distribution.expanduser().resolve(strict=True)
    receipt_path = receipt_path.expanduser()
    if receipt_path.is_symlink():
        raise ValueError(f"distribution receipt 不能是符号链接: {receipt_path}")
    config_path = config_path.expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise ValueError(f"runtime config 必须是普通文件: {config_path}")
    verify_distribution(distribution)
    if not receipt_path.exists():
        selection = PluginSelection(workspace)
        if selection.path.exists() and selection.read() is not None:
            raise RuntimeError("既有 selection 缺少首次分发 receipt；不能猜测插件来源或重装默认组合")
        workspace.mkdir(parents=True, exist_ok=True)
        initialize_empty_workspace(repo_root=_SOURCE_ROOT, workspace=workspace, config_path=config_path)
        # 首次安装的选择（未选的预设行关掉、选中的非预设包启用）：必须在初始化之后、第一次候选计算之前写入；已有 patch 不覆盖。
        if profile_patch is not None and not (workspace / "bundle.patch.toml").exists():
            _ = (workspace / "bundle.patch.toml").write_bytes(profile_patch.read_bytes())
    maintenance = WorkspaceMaintenanceLock(workspace)
    publication = PluginPublicationLock(plugins_home)
    maintenance.acquire()
    previous_tmp = (os.environ.get("TMPDIR"), tempfile.tempdir)
    extra_root: Path | None = None
    try:
        publication.acquire()
        try:
            # 暂存目录在插件目录的发布锁内创建并回收遗留：多个 workspace 共用一个插件目录时，
            # 回收只会发生在没有别的进程持有发布锁、也就没有别的进程在用暂存目录的时候。
            # 本进程与它启动的 pip、git 子进程的临时文件都落在暂存目录里，结束时一并删除并复原。
            # 插件目录在 workspace 内时没有合适的状态盘位置，退回系统临时目录。
            staging = _extra_staging_root(workspace, plugins_home)
            extra_root = staging or Path(tempfile.mkdtemp(prefix="akashic-extra-"))
            if staging is not None:
                scratch = staging / "tmp"
                scratch.mkdir()
                os.environ["TMPDIR"] = tempfile.tempdir = str(scratch)
            _recover_extra_install(workspace)
            if not receipt_path.exists():
                current = PluginSelection(workspace)
                if current.path.exists() and current.read() is not None:
                    raise RuntimeError("取得维护锁后 selection 已变化；缺少首次分发 receipt，停止初始化")
                receipt = install_bundle(distribution, bundle_file, workspace=workspace,
                                          plugins_home=plugins_home, config_path=config_path,
                                          initialize_workspace=False)
                # Keep the first receipt unchanged, including retired bootstrap sources.
                _write_receipt(receipt_path, receipt)
            with measure("inputs.selection"):
                receipt = _read_json(receipt_path)
                _validate_receipt_state(receipt, workspace=workspace, plugins_home=plugins_home)
                selection = PluginSelection(workspace)
                expected = selection.read()
                components, selected = ((), {}) if expected is None else _selected_components(selection, expected, absorbed=frozenset(absorbed_plugin_ids(distribution)))
            runner = MigrationRunner(repo_root=_SOURCE_ROOT, config_path=config_path,
                                     plugins_home=plugins_home, bundle_directory=distribution / "bundles", workspace=workspace,
                                     fixed_sources=distribution_migration_sources(workspace, plugins_home, distribution))
            with measure("inputs.migrations.core"):
                pending = runner.check()
                runner.run_under_maintenance(maintenance, core_only=True)
            # Both installation locks are held. Finish interrupted pointer writes before comparing inputs.
            _ = ReloadJournal(workspace).settle_orphaned_installs(plugins_home)
            with measure("inputs.candidate"):
                available, candidate = _distribution_candidate(
                    distribution=distribution, workspace=workspace, plugins_home=plugins_home, selected=selected,
                )
                extra_targets: list[dict[str, Any]] = []
                if extra_repo is not None or extra_inputs is not None:
                    selected = dict(selected)
                    extra_targets = _stage_missing_from_extra(
                        distribution=distribution, workspace=workspace, plugins_home=plugins_home, selection=selection,
                        expected=expected, repo=extra_repo, inputs=extra_inputs, root=extra_root,
                        allow_retire=allow_retire, install_extra=install_extra)
                    for target in extra_targets:
                        name, marketplace = target["plugin_id"].split("@")
                        candidate[target["plugin_id"]] = ResolvedPluginSource(
                            target["code"], "installed", marketplace, name, load_static_plugin_manifest(target["code"]))
                    # 与发布路径相同的目标语义：迁移看到的是目标代码与已核对的发行版状态，不按运行态重读旧归属。
                    runner.replacement_sources = tuple(candidate[item["plugin_id"]] for item in extra_targets)
                    runner.distribution_state = available
                # 与部署同一实现：在用的内置插件若发行版不再提供，要么已换来源，要么必须用 --allow-retire 明确确认，不能静默移出选择。
                dropped = _dropped_inuse(selected=selected, candidate=candidate, workspace=workspace,
                                         distribution=distribution, acknowledged=frozenset())
                if dropped and not allow_retire:
                    raise RuntimeError("在用的内置插件发行版不再提供，也没有从 extra 换来源，启动后会消失: " + ", ".join(dropped)
                                       + "。请先放到 extra 仓库（--extra-repo / --extra-inputs），或用 --allow-retire 确认移除")
                if dropped:
                    print("将被移除（--allow-retire，在用的内置插件发行版不再提供）: " + ", ".join(dropped), file=sys.stderr, flush=True)
                _require_preset_rows(candidate=candidate, workspace=workspace, distribution=distribution)
            try:
                check_pending_publication(workspace)
            except PendingPublicationError:
                # The normal entrypoint must let the original runtime repair its
                # committed config projection. Do not prepare from stale files or
                # publish anything while that recovery owner is unsettled.
                if pending or expected is None or not _can_boot_for_recovery(
                    selected, candidate, plugin_choices(workspace, distribution)
                ):
                    raise
                return {**receipt, "status": "existing", "new_root_ref": expected,
                        "recovery_pending": True}
            with measure("inputs.migrations"):
                runner.run_under_maintenance(maintenance)
            # 补装的安装、输入准备与选择提交是一个单元：恢复点先落盘（进程死亡后下次启动先还原），
            # 失败时按选择的落盘事实决定是否还原，不能留下"指针已推进、选择未提交"或"选择已提交、制品被删"。
            restore_file = _save_extra_restore(workspace, plugins_home, extra_targets, expected)
            try:
                closed, met, declared = _install_targets(extra_targets, workspace=workspace, plugins_home=plugins_home, selection=selection,
                                               selected=selected, candidate=candidate, distribution_root=distribution)
                with measure("inputs.prepare"):
                    new_components = _prepare_distribution_inputs(
                        available=available, candidate=candidate, selected=selected,
                        distribution=distribution, workspace=workspace, plugins_home=plugins_home, selection=selection,
                    )
                # 空组合也必须提交，首次启动不能重新发现并启用全部制品。
                new_root = expected
                final_adoption, replace = _final_adoption(
                    selection, None, {item["plugin_id"] for item in extra_targets if item["kind"] != "upgrade"})
                # 前置条件记录（声明与原因）在选择提交之前写，并在恢复点里（capture_state 含这两份文件）：
                # 提交前失败或进程死亡，恢复会还原成事务前的内容；提交后不存在"记录还没写"的崩溃窗口。
                for plugin_id, needs, head in declared:
                    record_declared(workspace, plugin_id, needs, head)
                for plugin_id, missing, head in closed:
                    record_disabled(workspace, plugin_id, missing, head)
                for plugin_id in met:
                    clear_disabled(workspace, plugin_id)
                if expected is None or new_components != components or replace:
                    new_root = _commit_selection(selection, new_components, expected, final_adoption, replace)
            except BaseException:
                if extra_targets and PluginSelection(workspace).read() == expected:
                    restore_state(json.loads(cast(Path, restore_file).read_text(encoding="utf-8"))["state"])
                if restore_file is not None:
                    restore_file.unlink(missing_ok=True)
                raise
            if restore_file is not None:
                restore_file.unlink(missing_ok=True)
            return {**receipt, "status": "existing", "new_root_ref": new_root}
        finally:
            publication.release()
    finally:
        tempfile.tempdir = previous_tmp[1]
        if previous_tmp[0] is None:
            _ = os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = previous_tmp[0]
        if extra_root is not None:
            shutil.rmtree(extra_root, ignore_errors=True)
        maintenance.release()


# 选择里已装（extra 等外置来源）的 artifact：[(插件身份, artifact 目录)]；选择文件不存在返回空。
def _installed_artifacts(workspace: Path) -> list[tuple[str, Path]]:
    path = workspace / "runtime/plugin-stable.json"
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    return [(record["plugin_id"], Path(record["code"])) for record in document["inputs"].values()
            if record["source_type"] == "installed"]


# 每次启动、在补装事务之后执行的尽力而为步骤：已装包声明的资源缺失或损坏就对着它自己的 artifact 取回
# （不取代码、不求 wheel、不碰选择与安装记录）。任何失败只打警告，服务照常启动，该资源在取用时报 RESOURCE_UNAVAILABLE。
# 补装的早返回分支（没有缺口）也会走到这里，因为它在 ensure_bundle 返回之后。
def _repair_resources_best_effort(workspace: Path) -> None:
    try:
        failures = repair_installed_resources(_installed_artifacts(workspace), workspace)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"警告: 已装包的资源修复步骤失败，已跳过: {error}", file=sys.stderr, flush=True)
        return
    for name, reason in failures:
        print(f"警告: {name}: 资源修复失败，保留现有安装；服务照常启动，该资源在用到时报 RESOURCE_UNAVAILABLE，可执行 `akashic pkg upgrade {name.split('@', 1)[0]}` 修复（{reason}）",
              file=sys.stderr, flush=True)


_EXTRA_RESTORE = "runtime/extra-install-restore.json"


# 启动补装的耐久恢复点：第一笔可见安装写入之前落盘，附带选择基线；没有补装目标时不写。
def _save_extra_restore(
    workspace: Path, plugins_home: Path, targets: list[dict[str, Any]], base_root: str | None,
) -> Path | None:
    if not targets:
        return None
    path = workspace / _EXTRA_RESTORE
    write_json(path, {"base_root_ref": base_root,
                      "state": capture_state(workspace, plugins_home, [item["plugin_id"] for item in targets])})
    return path


# 上次补装进程死在安装之后、选择提交之前：选择仍是基线就先还原插件状态，选择已前进说明已提交，只清理恢复点。
def _recover_extra_install(workspace: Path) -> None:
    path = workspace / _EXTRA_RESTORE
    if not path.exists():
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    if PluginSelection(workspace).read() == document["base_root_ref"]:
        restore_state(document["state"])
    path.unlink()


# 选择提交；写入结果不确定时按落盘事实判断（与换来源一致）：基线已变说明新选择已发布，视为已提交。
def _commit_selection(
    selection: PluginSelection, components: tuple[str, ...], expected: str | None,
    adoption: Mapping[str, object] | None, replace: bool,
) -> str:
    try:
        return selection.commit(components, expected_ref=expected, distribution_adoption=adoption,
                                replace_adoption=replace)
    except SelectionWriteError as error:
        if error.outcome == "uncertain" and error.observed_ref not in (None, expected):
            return cast(str, error.observed_ref)
        raise


# 首装与启动时补装的第一步：preset 启用、发行版不再自带的插件从 extra 取代码并核对（只补缺口，不升级）。
# 只在 root 下固定 bundle 与代码、求依赖 wheel，不写 cache、选择或 patch；联网失败直接停止，并说明怎么用预先下载好的输入目录。
# 失败发生时，此前的初始化、内置安装、receipt 与 Core 迁移已经发生：它们都是幂等的，所以修好后重试成立；
# 本次 extra 包的安装（指针、环境、选择）不会半途生效。
def _stage_missing_from_extra(
    *, distribution: Path, workspace: Path, plugins_home: Path, selection: PluginSelection, expected: str | None,
    repo: str | None, inputs: Path | None, root: Path, allow_retire: bool = False, install_extra: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    from scripts.akashic_release.extra import ExtraOptions, prepare_extra_inputs

    options = ExtraOptions(repo=repo, inputs=inputs, pull_images=False, gaps_only=True, allow_retire=allow_retire,
                           install_extra=install_extra)
    try:
        found = prepare_extra_inputs(workspace=workspace, plugins_home=plugins_home, source=distribution,
                                     mode=os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base"), root_ref=expected,
                                     options=options, out=root / "inputs", run=subprocess.run)
    except WheelResolutionFailure:
        raise
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        if "--allow-retire" in str(error):
            raise RuntimeError(str(error)) from error
        raise RuntimeError(
            f"无法取得 preset 启用但发行版已不再自带、或需要重建依赖环境的插件（{str(error).splitlines()[0] if str(error) else type(error).__name__}）。请检查网络与 extra 仓库地址"
            "（AKASHIC_EXTRA_REPO），或用 --extra-inputs 指向预先下载好的输入目录后重试") from error
    for identity, note in found.skipped:
        print(f"{identity}: {note}", file=sys.stderr, flush=True)
    if not found.targets:
        return []
    (root / "staged").mkdir()
    return _stage_deployment_targets(
        targets=list(found.targets), distribution=distribution, inputs=root / "inputs", staged=root / "staged",
        expected_root_ref=expected, selection=selection, workspace=workspace, plugins_home=plugins_home)


# 首次安装的联网部分单独做完：取包、校验、求 wheel 到 out（与 --extra-inputs 同格式），不写 cache、选择或 patch。
# 失败直接抛错，状态不变；之后的 ensure 用 --extra-inputs 离线安装，不再联网。
def _prefetch_extra(distribution: Path, workspace: Path, plugins_home: Path, repo: str, out: Path,
                    install_extra: tuple[str, ...] = (), profile_patch: Path | None = None) -> dict[str, Any]:
    from scripts.akashic_release.extra import ExtraOptions, prepare_extra_inputs

    out.mkdir(parents=True, exist_ok=True)
    found = prepare_extra_inputs(
        workspace=workspace, plugins_home=plugins_home, source=distribution, mode=os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base"),
        root_ref=PluginSelection(workspace).read(), options=ExtraOptions(repo=repo, pull_images=False, gaps_only=True, record_needs=True, install_extra=install_extra,
                                                      patch_file=profile_patch),
        out=out, run=subprocess.run)
    return {"prefetched": [item["plugin_id"] for item in found.targets]}


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
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plugins-home", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--ensure-bundle", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--closed-boot", help=argparse.SUPPRESS)
    parser.add_argument("--extra-repo", help="--ensure-bundle：preset 启用而发行版已不再自带的插件从该 extra 仓库补装")
    parser.add_argument("--extra-inputs", type=Path, help="--ensure-bundle：预先下载好的 extra 输入目录（不联网）")
    parser.add_argument("--prefetch-extra", type=Path, help="只联网取好首次安装需要的 extra 包（含 wheel）到该目录，之后用 --extra-inputs 离线安装")
    parser.add_argument("--install-extra", default="", help="首次安装选定要装的 extra 包（逗号分隔的 manifest 名）")
    parser.add_argument("--profile-patch", type=Path, help="--ensure-bundle：首次安装的选择（bundle.patch.toml 内容），初始化后写入")
    parser.add_argument("--allow-retire", action="store_true", help="--ensure-bundle：确认让在用而发行版不再提供、上游也没有的内置插件消失")
    args = parser.parse_args()

    if args.prefetch_extra is not None:
        if args.extra_repo is None:
            parser.error("--prefetch-extra 需要 --extra-repo")
        print(json.dumps(_prefetch_extra(args.distribution, args.workspace, args.plugins_home, args.extra_repo, args.prefetch_extra,
                                                    tuple(item for item in args.install_extra.split(",") if item), args.profile_patch),
                         ensure_ascii=False))
        return
    if sum((args.verify_only, args.ensure_bundle, args.publish)) > 1:
        parser.error("--verify-only、--ensure-bundle、--publish 互斥")
    if args.closed_boot is not None and (not args.publish or args.preflight_only):
        parser.error("--closed-boot 仅适用于停止后的发布")
    if args.publish:
        if args.config is None or args.plan is None or args.inputs is None:
            parser.error("--publish 需要 --config、--plan、--inputs")
        result = publish_distribution(
            distribution=args.distribution, workspace=args.workspace,
            plugins_home=args.plugins_home, config_path=args.config,
            plan=args.plan, inputs=args.inputs, backup_dir=args.backup_dir,
            preflight_only=args.preflight_only, closed_boot=args.closed_boot,
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.verify_only:
        report = verify_distribution(args.distribution)
        _check_distribution_sources(args.distribution, report)
        result: dict[str, Any] = {
            "status": "verified",
            "distribution_source_commit": report["source_commit"],
            "plugin_count": len(report["plugins"]),
        }
    else:
        if args.config is None:
            parser.error("安装 bundle 必须提供 --config")
        if args.core_root is not None:
            extract_core(args.distribution, args.core_root)
        if args.ensure_bundle:
            if args.receipt is None:
                parser.error("--ensure-bundle 必须提供 --receipt")
            with measure("runtime.inputs"):
                result = ensure_bundle(
                    args.distribution,
                    args.bundle,
                    workspace=args.workspace,
                    plugins_home=args.plugins_home,
                    config_path=args.config,
                    receipt_path=args.receipt,
                    extra_repo=args.extra_repo,
                    extra_inputs=args.extra_inputs,
                    allow_retire=args.allow_retire,
                    install_extra=tuple(item for item in args.install_extra.split(",") if item),
                    profile_patch=args.profile_patch,
                )
            _repair_resources_best_effort(args.workspace.expanduser().resolve(strict=False))
        else:
            result = install_bundle(
                args.distribution,
                args.bundle,
                workspace=args.workspace,
                plugins_home=args.plugins_home,
                config_path=args.config,
            )
    if args.receipt is not None:
        if not (args.ensure_bundle and result.get("status") == "existing"):
            _write_receipt(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
