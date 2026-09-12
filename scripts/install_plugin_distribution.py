#!/usr/bin/env python3
"""验证 Core/插件发布身份，并按显式 profile 走正式 Git 插件安装链。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from agent.plugins.install import install_git_plugin

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
        _ = _distribution_file(root, raw.get("file"), f"plugins[{index}] bundle")
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
    target = destination.expanduser().resolve(strict=False)
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

    for key in ("authorization", "prompt"):
        owner = initialization.get(key)
        if not isinstance(owner, dict) or not isinstance(owner.get("owner"), str):
            raise ValueError(f"profile initialization.{key} 必须声明 owner")
        if owner["owner"] not in selected:
            raise ValueError(
                f"profile initialization.{key}.owner 不在已选插件中: {owner['owner']}"
            )
    materials = initialization.get("context_materials")
    if materials is not None:
        if not isinstance(materials, dict):
            raise ValueError("profile initialization.context_materials 必须是 object")
        if not isinstance(materials.get("owner"), str) or materials["owner"] not in selected:
            raise ValueError(
                "profile initialization.context_materials.owner 不在已选插件中"
            )
        prompt_sources = materials.get("prompt_sources")
        if (
            not isinstance(prompt_sources, dict)
            or not prompt_sources
            or any(
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(owner, str)
                or not owner.strip()
                or owner not in selected
                for name, owner in prompt_sources.items()
            )
        ):
            raise ValueError(
                "profile initialization.context_materials.prompt_sources 必须引用已选插件"
            )
        summary_source = materials.get("summary_source")
        if (
            not isinstance(summary_source, list)
            or len(summary_source) != 2
            or any(not isinstance(item, str) or not item.strip() for item in summary_source)
            or summary_source[1] not in selected
        ):
            raise ValueError(
                "profile initialization.context_materials.summary_source 必须引用已选插件"
            )
    return profile_name, marketplace, normalized, initialization


def _write_context_materials(
    workspace: Path,
    *,
    marketplace: str,
    declaration: dict[str, Any],
) -> dict[str, str]:
    """原子创建 profile 声明的 Context 材料授权，绝不覆盖既有工作区配置。"""

    config_dir = workspace / "plugin-data" / f"context-{marketplace}"
    config_path = config_dir / "config.local.toml"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or not config_path.is_file():
            raise ValueError(f"Context 材料配置不是普通文件: {config_path}")
        return {"path": str(config_path), "status": "existing"}

    prompt_sources = declaration["prompt_sources"]
    summary_source = declaration["summary_source"]
    prompt_entries = ", ".join(
        f'{name} = "{owner}@{marketplace}"'
        for name, owner in sorted(prompt_sources.items())
    )
    content = (
        f"prompt_sources = {{{prompt_entries}}}\n"
        f'summary_source = ["{summary_source[0]}", "{summary_source[1]}@{marketplace}"]\n'
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config.local.", suffix=".tmp", dir=config_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, config_path)
        except FileExistsError:
            if config_path.is_symlink() or not config_path.is_file():
                raise ValueError(f"Context 材料配置不是普通文件: {config_path}")
            return {"path": str(config_path), "status": "existing"}
        directory_fd = os.open(config_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(config_path), "status": "created"}


def install_profile(
    distribution: Path,
    profile: Path,
    *,
    workspace: Path,
    plugins_home: Path,
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
    workspace = workspace.expanduser().resolve(strict=False)
    plugins_home = plugins_home.expanduser().resolve(strict=False)
    workspace.mkdir(parents=True, exist_ok=True)
    plugins_home.mkdir(parents=True, exist_ok=True)

    installed: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry["name"])
        row = rows.get(name)
        if row is None:
            raise ValueError(f"profile 需要的插件 bundle 不存在: {name}")
        bundle = _distribution_file(
            distribution_root, row["file"], f"插件 {name} bundle"
        )
        _check_sha256(bundle, row["sha256"], f"插件 {name} bundle")
        try:
            result = install_git_plugin(
                workspace=workspace,
                source=str(bundle),
                marketplace=marketplace,
                ref_name=str(row["source_revision"]),
                plugins_home=plugins_home,
            )
        except Exception as error:
            raise RuntimeError(f"正式安装插件失败: {name}") from error
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
    context_materials = initialization.get("context_materials")
    context_config = None
    if context_materials is not None:
        assert isinstance(context_materials, dict)
        context_config = _write_context_materials(
            workspace,
            marketplace=marketplace,
            declaration=context_materials,
        )
    return {
        "schema_version": 1,
        "distribution_source_commit": report["source_commit"],
        "distribution_source_tree": report["source_tree"],
        "profile": profile_name,
        "marketplace": marketplace,
        "initialization_owners": initialization,
        "context_materials_config": context_config,
        "formal_installer": "agent.plugins.install.install_git_plugin",
        "installed": installed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plugins-home", type=Path, required=True)
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    report = verify_distribution(args.distribution)
    if args.verify_only:
        result: dict[str, Any] = {
            "status": "verified",
            "distribution_source_commit": report["source_commit"],
            "plugin_count": len(report["plugins"]),
        }
    else:
        if args.core_root is not None:
            extract_core(args.distribution, args.core_root, report=report)
        result = install_profile(
            args.distribution,
            args.profile,
            workspace=args.workspace,
            plugins_home=args.plugins_home,
        )
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
