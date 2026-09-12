#!/usr/bin/env python3
"""验证 Core 制品与第一方插件 bundle 的真实外置安装和 apply。

这个脚本只把 Core 解包到临时目录，并通过 ``install_git_plugin`` 安装
bundle。它不会把本仓库的 ``plugins/`` 加回导入路径，也不会把服务枚举
当成能力调用成功。全量模式把所有 bundle 安装到同一个临时组合中，因而
能暴露真实的跨插件实现导入和组合依赖欠账。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


_CORE_PACKAGES = frozenset(
    {
        "agent",
        "bootstrap",
        "bus",
        "core",
        "host_bridge",
        "infra",
        "mcp_servers",
        "memory2",
        "prompts",
        "session",
        "utils",
    }
)
_SAFE_CAPABILITY_ENTRYPOINTS = {
    "message.display:model.facts": "plugins.models.projection.display_facts",
}


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkout_root(path: Path) -> Path | None:
    """返回 path 所在 Git checkout，包含 worktree 的 .git 文件。"""

    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        if marker.is_dir() and (marker / "HEAD").is_file():
            return candidate.resolve(strict=False)
        if marker.is_file():
            try:
                header = marker.read_text(encoding="utf-8").strip()
            except OSError:
                header = ""
            if header.startswith("gitdir:"):
                return candidate.resolve(strict=False)
    return None


def _is_dependency_path(path: Path) -> bool:
    """允许 venv 的 site-packages；其内的 editable 源码仍由后续规则检查。"""

    return any(part in {"site-packages", "dist-packages"} for part in path.parts)


def _looks_like_editable_source(path: Path, repo_root: Path) -> bool:
    """识别没有 .git marker 的旧 editable SDK worktree 路径。"""

    if _under(path, repo_root) or _is_dependency_path(path):
        return False
    parts = path.parts
    if "worktrees" in parts:
        return True
    return len(parts) >= 3 and parts[-3:] == ("sdk", "python", "src")


def _source_checkout(path_or_url: str, repo_root: Path) -> Path | None:
    """校验外部 checkout 或 bundle；仓库内源码一律拒绝。"""

    if "://" in path_or_url or path_or_url.startswith("git@"):
        return None
    path = Path(path_or_url).expanduser().resolve(strict=True)
    if _under(path, repo_root):
        raise ValueError(
            "source checkout 在本仓库内；外置验收拒绝把 builtin checkout 当 external artifact"
        )
    if path.is_file():
        subprocess.run(
            ["git", "bundle", "list-heads", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return path
    if not path.is_dir():
        raise ValueError(f"external source 不是 Git 目录或 bundle: {path}")
    try:
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"external source 不是可复现 Git checkout: {path}") from error
    if _checkout_root(path) is None:
        raise ValueError(f"external source 缺少 Git checkout 身份: {path}")
    return path


def _purge_modules(roots: tuple[Path, ...]) -> list[str]:
    removed: list[str] = []
    for module_name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            if module_name == "plugins" or module_name.startswith("plugins."):
                sys.modules.pop(module_name, None)
                removed.append(module_name)
            continue
        path = Path(module_file).resolve(strict=False)
        if any(_under(path, root) for root in roots):
            sys.modules.pop(module_name, None)
            removed.append(module_name)
    return removed


def _hide_checkouts(
    repo_root: Path, source_checkout: Path | None
) -> dict[str, list[str]]:
    """清除源码和 editable checkout，防止 importer 以隐式路径兜底。"""

    roots = [repo_root.resolve(strict=False)]
    if source_checkout is not None and source_checkout.is_dir():
        roots.append(source_checkout.resolve(strict=False))
    removed_paths: list[str] = []
    filtered: list[str] = []
    for raw in sys.path:
        try:
            candidate = Path(raw or os.curdir).resolve(strict=False)
        except OSError:
            filtered.append(raw)
            continue
        checkout = _checkout_root(candidate)
        if any(_under(candidate, root) for root in roots) or (
            checkout is not None
            and not _is_dependency_path(candidate)
            and not any(_under(checkout, root) for root in roots)
        ) or _looks_like_editable_source(candidate, repo_root):
            removed_paths.append(str(candidate))
            continue
        filtered.append(raw)
    sys.path[:] = filtered
    for path in removed_paths:
        sys.path_importer_cache.pop(path, None)
    removed_modules = _purge_modules(tuple(roots))
    # An editable SDK can already have loaded modules from a sibling worktree.
    for module_name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        module_path = Path(module_file).resolve(strict=False)
        if _is_dependency_path(module_path):
            continue
        checkout = _checkout_root(module_path)
        if (
            checkout is not None
            and not any(_under(checkout, root) for root in roots)
        ) or _looks_like_editable_source(module_path, repo_root):
            sys.modules.pop(module_name, None)
            removed_modules.append(module_name)
    return {"paths": removed_paths, "modules": sorted(set(removed_modules))}


def _visible_checkout_modules(
    repo_root: Path, source_checkout: Path | None
) -> list[dict[str, str]]:
    roots = [repo_root / "plugins"]
    if source_checkout is not None and source_checkout.is_dir():
        roots.append(source_checkout)
    evidence: list[dict[str, str]] = []
    for module_name, module in sorted(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        path = Path(module_file).resolve(strict=False)
        for root in roots:
            if _under(path, root):
                evidence.append({"module": module_name, "file": str(path)})
                break
    return evidence


def _core_module_violations(core_root: Path) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for module_name, module in sorted(sys.modules.items()):
        top_level = module_name.split(".", 1)[0]
        if top_level not in _CORE_PACKAGES:
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        path = Path(module_file).resolve(strict=False)
        if not _under(path, core_root):
            violations.append({"module": module_name, "file": str(path)})
    return violations


def _ensure_empty_directory(path: Path, label: str) -> None:
    """只允许一次性空目录，避免把既有 workspace/cache 当验收输入。"""

    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"{label} 必须是目录而不能是文件或符号链接: {path}")
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} 必须为空的一次性目录: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_core_root(core_root: Path, repo_root: Path) -> Path:
    root = core_root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Core 制品目录无效: {root}")
    if _under(root, repo_root) or _checkout_root(root) is not None:
        raise ValueError("外置验收必须提供仓库之外、非 Git checkout 的 Core 制品目录")
    if (root / "plugins").exists():
        raise ValueError("Core 制品目录不能包含业务 plugins/ 源码")
    return root


def _extract_core_tar(core_tar: Path, destination: Path, repo_root: Path) -> Path:
    """解包不可变 Core tar，并立即检查它没有业务源码。"""

    source = core_tar.expanduser().resolve(strict=True)
    if not source.is_file() or _under(source, repo_root):
        raise ValueError("Core tar 必须是仓库外的正式制品文件")
    root = destination / "core"
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, mode="r:") as archive:
        archive.extractall(root, filter="data")
    return _validate_core_root(root, repo_root)


def _prepare_runtime(
    *, repo_root: Path, source_checkout: Path | None, core_root: Path, workspace: Path
) -> dict[str, Any]:
    removed = _hide_checkouts(repo_root, source_checkout)
    # A caller may have imported the Core from another archive in this process.
    _purge_modules((repo_root,))
    os.chdir(workspace)
    sys.path.insert(0, str(core_root))
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("plugins")
    if spec is not None:
        raise ValueError("运行环境仍能解析 checkout 的 plugins 命名空间")
    return {
        "checkout_paths_removed": removed["paths"],
        "checkout_modules_removed": removed["modules"],
        "core_root": str(core_root),
        "cwd": str(workspace),
    }


def _plugin_id_from_manifest(artifact: Path, marketplace: str) -> tuple[str, str]:
    from agent.plugins.static_manifest import load_static_plugin_manifest

    manifest = load_static_plugin_manifest(artifact)
    return f"{manifest.name}@{marketplace}", manifest.entrypoint


def _load_source_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source mapping 必须是 JSON object")
    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise ValueError("source mapping 的 key/value 必须是非空字符串")
        result[key] = value
    return result


def _load_capability_calls(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capability calls 必须是 JSON object")
    result: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("capability calls 的 key/value 类型无效")
        result[key] = dict(value)
    return result


def _load_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("packages"), list):
        raise ValueError("inventory JSON 缺少 packages 数组")
    return payload


def _source_for_entry(
    entry: dict[str, Any], sources: dict[str, str]
) -> str | None:
    package = str(entry.get("package", ""))
    manifest = entry.get("manifest")
    manifest_name = manifest.get("name") if isinstance(manifest, dict) else None
    for key in (package, str(manifest_name) if manifest_name else ""):
        if key in sources:
            return sources[key]
    return None


def _load_distribution(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("distribution JSON 必须是 object")
    commit = payload.get("source_commit")
    core = payload.get("core")
    rows = payload.get("plugins")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or not isinstance(core, dict)
        or not isinstance(core.get("file"), str)
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("distribution JSON 缺少固定 source commit、Core 或 plugins")
    parent = path.resolve().parent
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("distribution plugin row 无效")
        name = row["name"]
        row_commit = row.get("source_commit")
        if row_commit is not None and row_commit != commit:
            raise ValueError(f"distribution 插件 source commit 不一致: {name}")
        if name in names:
            raise ValueError(f"distribution 插件名称重复: {name}")
        names.add(name)
        bundle = parent / str(row.get("file", ""))
        if not bundle.is_file():
            raise ValueError(f"distribution bundle 缺失: {bundle}")
        digest = row.get("sha256")
        if isinstance(digest, str) and _sha256(bundle) != digest:
            raise ValueError(f"distribution bundle SHA256 不匹配: {bundle}")
    core_tar = parent / core["file"]
    if not core_tar.is_file():
        raise ValueError(f"distribution Core tar 缺失: {core_tar}")
    if isinstance(core.get("sha256"), str) and _sha256(core_tar) != core["sha256"]:
        raise ValueError(f"distribution Core SHA256 不匹配: {core_tar}")
    payload["_path"] = str(path.resolve())
    payload["_root"] = str(parent)
    return payload


def _capability_for(
    calls: dict[str, dict[str, Any]], *keys: str | None
) -> dict[str, Any] | None:
    for key in keys:
        if key and key in calls:
            return calls[key]
    return None


async def _invoke_capability(
    *, root: Any, plugin_id: str, spec: dict[str, Any]
) -> dict[str, Any]:
    """按登记的安全 oracle 调用一次真实 provider；绝不以枚举代替调用。"""

    service = spec.get("service")
    entrypoint = spec.get("entrypoint")
    input_value = spec.get("input")
    expected = spec.get("expect")
    if not isinstance(service, str) or not service.strip():
        raise ValueError("能力 oracle 缺少非空 service")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("能力 oracle 必须声明精确 entrypoint")
    if not isinstance(expected, dict) or "value" not in expected:
        raise ValueError("能力 oracle 必须声明 expect.value")
    from agent.plugin_composition import ServiceKey

    provided = root.provided_services(plugin_ids=frozenset({plugin_id}))
    evidence: dict[str, Any] = {
        "service": service,
        "entrypoint": entrypoint,
        "input": input_value,
        "expected": expected,
        "provided_service_names": sorted(key.name for key in provided),
        "call_executed": False,
    }
    key = ServiceKey(service)
    if key not in provided:
        raise ValueError(f"选定能力不是目标插件提供的服务: {service}")
    value = provided[key]
    if not callable(value):
        raise TypeError("选定能力不可调用；服务枚举和对象读取不能充当行为验收")
    safe_entrypoint = _SAFE_CAPABILITY_ENTRYPOINTS.get(service)
    if safe_entrypoint != entrypoint:
        raise ValueError(
            f"能力 oracle 未登记为无外部副作用的测试入口: {service} {entrypoint}"
        )
    if service == "message.display:model.facts":
        if (
            not isinstance(input_value, dict)
            or set(input_value) != {"kind", "value"}
            or input_value["kind"] != "model.facts"
            or not isinstance(input_value["value"], dict)
        ):
            raise ValueError("model.facts oracle input 必须是 {kind,value} ContentPart")
        from session.message import ContentPart

        actual = value(ContentPart(input_value["kind"], input_value["value"]))
    else:  # pragma: no cover - guarded by the registry above
        raise ValueError(f"未实现安全能力 oracle: {service}")
    if inspect.isawaitable(actual):
        actual = await actual
    evidence["actual"] = actual
    evidence["actual_type"] = type(actual).__name__
    if "type" in expected and type(actual).__name__ != expected["type"]:
        raise AssertionError(
            f"能力返回类型不符: expected={expected['type']} actual={type(actual).__name__}"
        )
    if actual != expected["value"]:
        raise AssertionError(f"能力返回值不符: expected={expected['value']!r} actual={actual!r}")
    evidence["call_executed"] = True
    evidence["status"] = "passed"
    return evidence


def _generation_evidence(
    *,
    generation: Any,
    artifact: Path,
    entrypoint: str,
    workspace: Path,
    repo_root: Path,
    source_checkout: Path | None,
) -> dict[str, Any]:
    module = sys.modules.get(generation.module_path)
    module_file = getattr(module, "__file__", None)
    checks: dict[str, bool] = {"apply": generation is not None and module is not None}
    evidence: dict[str, Any] = {
        "generation_id": generation.generation_id,
        "module_name": None if module is None else module.__name__,
        "module_file": module_file,
        "checks": checks,
    }
    if not isinstance(module_file, str):
        return evidence
    module_path = Path(module_file).resolve(strict=True)
    archive_root = (workspace / "runtime" / "plugin-archives").resolve(strict=False)
    installed_entrypoint = (artifact / entrypoint).resolve(strict=True)
    evidence["module_file_sha256"] = _sha256(module_path)
    evidence["installed_entrypoint_sha256"] = _sha256(installed_entrypoint)
    checks["module_file_is_core_archive"] = _under(module_path, archive_root)
    checks["module_file_not_checkout"] = not any(
        _under(module_path, root)
        for root in (repo_root / "plugins", source_checkout)
        if root is not None
    )
    checks["module_bytes_match_installed_artifact"] = (
        evidence["module_file_sha256"] == evidence["installed_entrypoint_sha256"]
    )
    return evidence


async def _exercise(
    *,
    source: str,
    repo_root: Path,
    marketplace: str,
    workspace: Path,
    plugins_home: Path,
    capability_service: str | None,
    core_root: Path | None = None,
    capability_spec: dict[str, Any] | None = None,
    require_capability: bool = False,
) -> dict[str, Any]:
    """在一次外部临时组合中正式安装并执行一个插件。"""

    source_checkout = _source_checkout(source, repo_root)
    evidence: dict[str, Any] = {
        "source": source,
        "source_checkout": None if source_checkout is None else str(source_checkout),
        "checks": {"source_checkout_is_external": True},
    }
    manager: Any = None
    event_bus: Any = None
    try:
        if core_root is None:
            raise ValueError("外置验收必须提供仓库外的 Core 制品目录")
        core_root = _validate_core_root(core_root, repo_root)
        _ensure_empty_directory(workspace, "workspace")
        _ensure_empty_directory(plugins_home, "plugins-home")
        evidence["runtime"] = _prepare_runtime(
            repo_root=repo_root,
            source_checkout=source_checkout,
            core_root=core_root,
            workspace=workspace,
        )
        evidence["checks"]["core_artifact_external"] = True
        from agent.plugins.install import install_git_plugin
        from agent.plugins.manager import PluginManager
        from bus.event_bus import EventBus

        result = install_git_plugin(
            workspace=workspace,
            source=source,
            marketplace=marketplace,
            plugins_home=plugins_home,
        )
        artifact = result.installed_path.resolve(strict=True)
        plugin_id, entrypoint = _plugin_id_from_manifest(artifact, marketplace)
        evidence.update(
            {
                "plugin_id": plugin_id,
                "source_revision": result.source_revision,
                "installed_artifact": str(artifact),
                "installed_entrypoint": str(artifact / entrypoint),
            }
        )
        evidence["checks"].update(
            {
                "formal_install_artifact": artifact.is_relative_to(
                    (plugins_home / "cache").resolve(strict=False)
                ),
                "installed_manifest_identity": True,
                "installed_source_has_no_sibling_plugins": not (artifact / "plugins").exists(),
            }
        )
        event_bus = EventBus()
        manager = PluginManager(
            [],
            event_bus=event_bus,
            workspace=workspace,
            installed_cache_root=plugins_home / "cache",
        )
        load_error: str | None = None
        try:
            await manager.load_all()
        except Exception as error:
            load_error = f"{type(error).__name__}: {error}"
        evidence["load_error"] = load_error
        snapshot = manager.current_snapshot
        generation = None if snapshot is None else snapshot.generations.get(plugin_id)
        if generation is not None:
            evidence.update(
                _generation_evidence(
                    generation=generation,
                    artifact=artifact,
                    entrypoint=entrypoint,
                    workspace=workspace,
                    repo_root=repo_root,
                    source_checkout=source_checkout,
                )
            )
        else:
            evidence["checks"]["apply"] = False
            gate = manager.latest_gate(plugin_id)
            if gate is not None:
                evidence["gate"] = {
                    "status": gate.status,
                    "failure_reason": gate.failure_reason,
                    "checks": [
                        {"id": item.check_id, "status": item.status, "evidence": item.evidence}
                        for item in gate.checks
                    ],
                }
            evidence["error"] = (
                (gate.failure_reason if gate is not None else None)
                or load_error
                or "formal install 后未形成 stable generation/module"
            )
        visible = _visible_checkout_modules(repo_root, source_checkout)
        evidence["checkout_modules_visible"] = visible
        evidence["checks"]["checkout_invisible"] = not visible
        evidence["core_module_violations"] = _core_module_violations(core_root)
        evidence["checks"]["core_modules_from_artifact"] = not evidence[
            "core_module_violations"
        ]
        if capability_service is not None and capability_spec is None:
            capability_spec = {
                "service": capability_service,
                "entrypoint": "",
                "input": None,
                "expect": {},
            }
        if capability_spec is not None:
            evidence["capability_call"] = {
                "status": "not_run",
                "call_executed": False,
            }
            if generation is None:
                evidence["capability_call"]["error"] = "apply 未成功，不能调用能力"
            else:
                from agent.plugins.snapshot import lease_runtime_snapshot

                try:
                    async with lease_runtime_snapshot(manager.snapshot_store) as leased:
                        if leased.composition_root is None:
                            raise RuntimeError("snapshot 缺少 composition root")
                        evidence["capability_call"] = await _invoke_capability(
                            root=leased.composition_root,
                            plugin_id=plugin_id,
                            spec=capability_spec,
                        )
                    evidence["checks"]["capability_call"] = True
                except Exception as error:
                    evidence["capability_call"] = {
                        "status": "failed",
                        "call_executed": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    evidence["checks"]["capability_call"] = False
        elif require_capability:
            evidence["capability_call"] = {
                "status": "unverified",
                "call_executed": False,
                "error": "缺少精确 capability oracle",
            }
            evidence["checks"]["capability_call"] = False
        checks = evidence["checks"]
        evidence["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as error:
        evidence["status"] = "failed"
        evidence["error"] = f"{type(error).__name__}: {error}"
    finally:
        if manager is not None:
            try:
                await manager.terminate_all()
            except Exception as error:
                evidence["terminate_error"] = f"{type(error).__name__}: {error}"
        if event_bus is not None:
            await event_bus.aclose()
    return evidence


async def _exercise_fleet(
    *,
    jobs: list[dict[str, Any]],
    repo_root: Path,
    marketplace: str,
    workspace: Path,
    plugins_home: Path,
    core_root: Path,
    capability_calls: dict[str, dict[str, Any]],
    require_capability: bool,
    distribution: dict[str, Any] | None,
) -> dict[str, Any]:
    """在一个合法组合中 install 全部 bundle，再观察每个真实 generation。"""

    _ensure_empty_directory(workspace, "workspace")
    _ensure_empty_directory(plugins_home, "plugins-home")
    core_root = _validate_core_root(core_root, repo_root)
    valid_jobs: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for job in jobs:
        try:
            checkout = _source_checkout(job["source"], repo_root)
            job = {**job, "source_checkout": checkout}
            valid_jobs.append(job)
        except Exception as error:
            reports.append(
                {
                    "plugin": job.get("label"),
                    "source": job.get("source"),
                    "status": "failed",
                    "checks": {"source_checkout_is_external": False},
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not valid_jobs:
        return {
            "mode": "all",
            "reports": reports,
            "status": "failed",
            "error": "没有可安装的 external source",
        }
    first_source = valid_jobs[0]["source_checkout"]
    runtime = _prepare_runtime(
        repo_root=repo_root,
        source_checkout=first_source if isinstance(first_source, Path) and first_source.is_dir() else None,
        core_root=core_root,
        workspace=workspace,
    )
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus

    installed: list[dict[str, Any]] = []
    for job in valid_jobs:
        row: dict[str, Any] = {
            "plugin": job.get("label"),
            "source": job["source"],
            "checks": {"source_checkout_is_external": True},
        }
        try:
            result = install_git_plugin(
                workspace=workspace,
                source=job["source"],
                marketplace=marketplace,
                plugins_home=plugins_home,
            )
            artifact = result.installed_path.resolve(strict=True)
            plugin_id, entrypoint = _plugin_id_from_manifest(artifact, marketplace)
            expected_name = str(job.get("label", ""))
            installed_name = plugin_id.split("@", 1)[0]
            if expected_name and installed_name != expected_name:
                raise ValueError(
                    "安装 manifest 身份与 distribution/inventory 不一致: "
                    f"expected={expected_name} actual={installed_name}"
                )
            row.update(
                {
                    "plugin_id": plugin_id,
                    "source_revision": result.source_revision,
                    "installed_artifact": str(artifact),
                    "installed_entrypoint": str(artifact / entrypoint),
                    "entrypoint": entrypoint,
                    "artifact": artifact,
                    "source_checkout": job.get("source_checkout"),
                    "expected_source_revision": job.get("source_revision"),
                }
            )
            row["checks"].update(
                {
                    "formal_install_artifact": artifact.is_relative_to(
                        (plugins_home / "cache").resolve(strict=False)
                    ),
                    "installed_manifest_identity": installed_name == expected_name,
                    "installed_source_has_no_sibling_plugins": not (artifact / "plugins").exists(),
                    "source_revision_matches_distribution": (
                        job.get("source_revision") in (None, result.source_revision)
                    ),
                }
            )
            installed.append(row)
        except Exception as error:
            row["status"] = "failed"
            row["error"] = f"{type(error).__name__}: {error}"
            reports.append(row)
    manager: Any = None
    event_bus: Any = None
    load_error: str | None = None
    try:
        event_bus = EventBus()
        manager = PluginManager(
            [],
            event_bus=event_bus,
            workspace=workspace,
            installed_cache_root=plugins_home / "cache",
        )
        try:
            await manager.load_all()
        except Exception as error:
            load_error = f"{type(error).__name__}: {error}"
        snapshot = manager.current_snapshot
        visible = _visible_checkout_modules(repo_root, None)
        core_violations = _core_module_violations(core_root)
        for row in installed:
            plugin_id = row["plugin_id"]
            generation = None if snapshot is None else snapshot.generations.get(plugin_id)
            if generation is not None:
                row["checks"]["apply_attempted"] = True
                row.update(
                    _generation_evidence(
                        generation=generation,
                        artifact=row["artifact"],
                        entrypoint=row["entrypoint"],
                        workspace=workspace,
                        repo_root=repo_root,
                        source_checkout=row.get("source_checkout"),
                    )
                )
            else:
                row["checks"]["apply"] = False
                gate = manager.latest_gate(plugin_id)
                row["checks"]["apply_attempted"] = gate is not None
                row["gate"] = (
                    None
                    if gate is None
                    else {
                        "status": gate.status,
                        "failure_reason": gate.failure_reason,
                        "checks": [
                            {"id": item.check_id, "status": item.status, "evidence": item.evidence}
                            for item in gate.checks
                        ],
                    }
                )
                row["error"] = (
                    (gate.failure_reason if gate is not None else None)
                    or load_error
                    or "formal install 后未形成 stable generation/module"
                )
            row["checkout_modules_visible"] = visible
            row["checks"]["checkout_invisible"] = not visible
            row["core_module_violations"] = core_violations
            row["checks"]["core_modules_from_artifact"] = not core_violations
        calls_to_run = [
            row
            for row in installed
            if _capability_for(
                capability_calls,
                str(row.get("plugin")),
                str(row.get("plugin_id")),
                str(row.get("plugin_id", "")).split("@", 1)[0],
            )
            is not None
        ]
        if require_capability:
            calls_to_run = installed
        if calls_to_run and snapshot is not None:
            from agent.plugins.snapshot import lease_runtime_snapshot

            async with lease_runtime_snapshot(manager.snapshot_store) as leased:
                if leased.composition_root is None:
                    raise RuntimeError("snapshot 缺少 composition root")
                for row in calls_to_run:
                    plugin_id = row["plugin_id"]
                    generation = snapshot.generations.get(plugin_id)
                    spec = _capability_for(
                        capability_calls,
                        str(row.get("plugin")),
                        plugin_id,
                        plugin_id.split("@", 1)[0],
                    )
                    if generation is None:
                        row["capability_call"] = {
                            "status": "not_run",
                            "call_executed": False,
                            "error": "apply 未成功，不能调用能力",
                        }
                        row["checks"]["capability_call"] = False
                    elif spec is None:
                        row["capability_call"] = {
                            "status": "unverified",
                            "call_executed": False,
                            "error": "缺少精确 capability oracle",
                        }
                        row["checks"]["capability_call"] = False
                    else:
                        try:
                            row["capability_call"] = await _invoke_capability(
                                root=leased.composition_root,
                                plugin_id=plugin_id,
                                spec=spec,
                            )
                            row["checks"]["capability_call"] = True
                        except Exception as error:
                            row["capability_call"] = {
                                "status": "failed",
                                "call_executed": False,
                                "error": f"{type(error).__name__}: {error}",
                            }
                            row["checks"]["capability_call"] = False
        elif require_capability:
            for row in installed:
                row["capability_call"] = {
                    "status": "unverified",
                    "call_executed": False,
                    "error": "缺少精确 capability oracle 或 stable generation",
                }
                row["checks"]["capability_call"] = False
        for row in installed:
            row["status"] = "passed" if all(row["checks"].values()) else "failed"
            row.pop("artifact", None)
            row.pop("source_checkout", None)
            reports.append(row)
    finally:
        if manager is not None:
            try:
                await manager.terminate_all()
            except Exception as error:
                load_error = load_error or f"terminate: {type(error).__name__}: {error}"
        if event_bus is not None:
            await event_bus.aclose()
    return {
        "mode": "all",
        "runtime": runtime,
        "distribution_source_commit": None if distribution is None else distribution.get("source_commit"),
        "load_error": load_error,
        "installed_count": len(installed),
        "applied_count": sum(item.get("checks", {}).get("apply", False) for item in installed),
        "reports": reports,
        "status": "passed" if reports and all(item.get("status") == "passed" for item in reports) else "failed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="一个真实 external Git checkout、bundle 或 Git URL")
    parser.add_argument("--plugin", help="单插件结果标签")
    parser.add_argument("--all", action="store_true", help="按 inventory 或 distribution 执行全量组合")
    parser.add_argument("--inventory", type=Path, help="plugin_inventory.py 生成的 JSON")
    parser.add_argument("--sources-json", type=Path, help="插件名称到 external checkout/URL 的 JSON 映射")
    parser.add_argument("--distribution", type=Path, help="build_plugin_distribution.py 生成的 distribution.json")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--marketplace", default="external-acceptance")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--plugins-home", type=Path)
    parser.add_argument("--capability-service", help="兼容旧参数；没有精确 oracle 时必然失败")
    parser.add_argument("--capability-calls-json", type=Path)
    parser.add_argument("--require-capabilities", action="store_true")
    parser.add_argument("--core-root", type=Path, help="已解包、位于 checkout 外且不含业务源码的 Core 制品")
    parser.add_argument("--core-tar", type=Path, help="build distribution 生成的 Core tar")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-temporary", action="store_true")
    return parser


def _failure_report(label: str, error: str, *, source: str | None = None) -> dict[str, Any]:
    return {
        "plugin": label,
        **({"source": source} if source is not None else {}),
        "status": "failed",
        "checks": {"external_source_mapped": False},
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source and args.all:
        print("必须二选一：--source 或 --all", file=sys.stderr)
        return 2
    if args.distribution and args.source:
        print("--distribution 不能与 --source 同时使用", file=sys.stderr)
        return 2
    if not args.source and not args.all and not args.distribution:
        print("必须提供 --source 或 --all/--distribution", file=sys.stderr)
        return 2
    if args.distribution:
        args.all = True
    original_cwd = Path.cwd()
    temp_root: Path | None = None
    reports: list[dict[str, Any]] = []
    distribution: dict[str, Any] | None = None
    fleet_result: dict[str, Any] | None = None
    try:
        repo_root = args.repo_root.resolve(strict=True)
        capability_calls = _load_capability_calls(args.capability_calls_json)
        jobs: list[dict[str, Any]] = []
        if args.source:
            spec = capability_calls.get(args.plugin or Path(args.source).stem)
            if args.capability_service is not None and spec is None:
                spec = {
                    "service": args.capability_service,
                    "entrypoint": "",
                    "input": None,
                    "expect": {},
                }
            jobs.append(
                {
                    "label": args.plugin or Path(args.source).stem,
                    "source": args.source,
                    "capability_spec": spec,
                }
            )
        elif args.distribution:
            distribution = _load_distribution(args.distribution)
            if args.inventory is not None:
                inventory = _load_inventory(args.inventory)
                expected = {
                    str(entry["manifest"]["name"])
                    for entry in inventory["packages"]
                    if entry.get("classification") == "manifest-plugin"
                    and isinstance(entry.get("manifest"), dict)
                    and isinstance(entry["manifest"].get("name"), str)
                }
                actual = {str(row["name"]) for row in distribution["plugins"]}
                for missing in sorted(expected - actual):
                    reports.append(
                        _failure_report(
                            missing,
                            "distribution 缺少 inventory manifest-plugin；不能以部分 fleet 代替全量验收",
                        )
                    )
                for extra in sorted(actual - expected):
                    reports.append(
                        _failure_report(
                            extra,
                            "distribution 含 inventory 之外的插件；请先对齐固定 inventory",
                        )
                    )
            for row in distribution["plugins"]:
                jobs.append(
                    {
                        "label": row["name"],
                        "source": str(Path(distribution["_root"]) / row["file"]),
                        "source_revision": row.get("source_revision"),
                        "capability_spec": _capability_for(capability_calls, row["name"]),
                    }
                )
        else:
            if args.inventory is None or args.sources_json is None:
                print("--all 必须同时提供 --inventory 和 --sources-json，或使用 --distribution", file=sys.stderr)
                return 2
            inventory = _load_inventory(args.inventory)
            sources = _load_source_map(args.sources_json)
            for entry in inventory["packages"]:
                label = str(entry.get("package", ""))
                if entry.get("classification") != "manifest-plugin":
                    reports.append(
                        _failure_report(
                            label,
                            "inventory row is a support-package without an install manifest",
                        )
                    )
                    continue
                source = _source_for_entry(entry, sources)
                if source is None:
                    reports.append(
                        _failure_report(
                            label,
                            "no external source mapping; one example cannot certify the inventory",
                        )
                    )
                    continue
                manifest = entry.get("manifest")
                manifest_name = manifest.get("name") if isinstance(manifest, dict) else None
                jobs.append(
                    {
                        "label": label,
                        "source": source,
                        "capability_spec": _capability_for(capability_calls, label, manifest_name),
                    }
                )
        if not jobs:
            result = {
                "schema_version": 2,
                "repository": str(repo_root),
                "mode": "all" if args.all else "single",
                "reports": reports,
                "status": "failed",
                "distribution_source_commit": None if distribution is None else distribution.get("source_commit"),
            }
            rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
            return 1
        if args.workspace is None or args.plugins_home is None or args.core_root is None:
            temp_root = Path(tempfile.mkdtemp(prefix="akashic-external-acceptance-"))
        root = temp_root
        workspace = args.workspace or root / "workspace"
        plugins_home = args.plugins_home or root / "plugins-home"
        if workspace.resolve(strict=False) == plugins_home.resolve(strict=False):
            raise ValueError("workspace 和 plugins-home 必须是不同目录")
        core_root = args.core_root
        if core_root is None:
            core_tar = args.core_tar
            if core_tar is None and distribution is not None:
                core_tar = Path(distribution["_root"]) / str(distribution["core"]["file"])
            if core_tar is not None:
                core_root = _extract_core_tar(core_tar, temp_root, repo_root)
        if not args.all:
            job = jobs[0]
            if core_root is None:
                reports.append(
                    _failure_report(
                        str(job["label"]),
                        "外置验收必须提供仓库外的 Core 制品目录",
                        source=str(job["source"]),
                    )
                )
            else:
                reports.append(
                    asyncio.run(
                        _exercise(
                            source=str(job["source"]),
                            repo_root=repo_root,
                            marketplace=args.marketplace,
                            workspace=workspace,
                            plugins_home=plugins_home,
                            capability_service=args.capability_service,
                            core_root=core_root,
                            capability_spec=job.get("capability_spec"),
                            require_capability=args.require_capabilities,
                        )
                    )
                )
        elif core_root is None:
            reports.extend(
                _failure_report(
                    str(job["label"]),
                    "全量 external acceptance 缺少 Core 制品；不能用 checkout 兜底",
                    source=str(job["source"]),
                )
                for job in jobs
            )
        else:
            fleet_result = asyncio.run(
                _exercise_fleet(
                    jobs=jobs,
                    repo_root=repo_root,
                    marketplace=args.marketplace,
                    workspace=workspace,
                    plugins_home=plugins_home,
                    core_root=core_root,
                    capability_calls=capability_calls,
                    require_capability=args.require_capabilities,
                    distribution=distribution,
                )
            )
            reports.extend(fleet_result.pop("reports"))
        result = {
            "schema_version": 2,
            "repository": str(repo_root),
            "workspace": str(workspace),
            "plugins_home": str(plugins_home),
            "mode": "all" if args.all else "single",
            "distribution_source_commit": None if distribution is None else distribution.get("source_commit"),
            "reports": reports,
            "status": "passed" if reports and all(item.get("status") == "passed" for item in reports) else "failed",
        }
        if fleet_result is not None:
            result["fleet"] = fleet_result
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0 if result["status"] == "passed" else 1
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"external acceptance setup failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    finally:
        os.chdir(original_cwd)
        if temp_root is not None and not args.keep_temporary:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
