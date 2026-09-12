#!/usr/bin/env python3
"""Run a real V3 external-plugin install and isolated runtime smoke.

This helper deliberately refuses a source checkout below the repository and
never synthesizes a fixture plugin.  ``--all`` consumes a source mapping for
every manifest inventory row; missing mappings are failed evidence rather than
an implicit pass.  The normal Core archive is expected to be the module source
at runtime, so the report compares its bytes with the formally installed
artifact while separately proving that the checkout is not visible.
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
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


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


def _source_checkout(path_or_url: str, repo_root: Path) -> Path | None:
    """Validate a local external checkout; return None for a remote URL."""

    if "://" in path_or_url or path_or_url.startswith("git@"):
        return None
    path = Path(path_or_url).expanduser().resolve(strict=True)
    if _under(path, repo_root):
        raise ValueError(
            "source checkout 在本仓库内；外置验收拒绝把 builtin checkout 当 external artifact"
        )
    if path.is_file():
        import subprocess
        subprocess.run(["git", "bundle", "list-heads", str(path)], check=True, capture_output=True)
        return path
    if not path.is_dir():
        raise ValueError(f"external source 不是 Git 目录或 bundle: {path}")
    try:
        import subprocess

        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"external source 不是可复现 Git checkout: {path}") from error
    return path


def _hide_checkouts(repo_root: Path, source_checkout: Path | None) -> None:
    """Remove checkout roots from import search and purge loaded plugin files."""

    roots = [repo_root.resolve(strict=False)]
    if source_checkout is not None:
        roots.append(source_checkout.resolve(strict=False))
    filtered: list[str] = []
    for raw in sys.path:
        try:
            candidate = Path(raw or os.curdir).resolve(strict=False)
        except OSError:
            filtered.append(raw)
            continue
        if any(_under(candidate, root) for root in roots):
            continue
        filtered.append(raw)
    sys.path[:] = filtered
    for module_name, module in tuple(sys.modules.items()):
        if module_name == "plugins" or module_name.startswith("plugins."):
            sys.modules.pop(module_name, None)
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        path = Path(module_file).resolve(strict=False)
        if source_checkout is not None and _under(path, source_checkout):
            sys.modules.pop(module_name, None)
        elif _under(path, repo_root / "plugins"):
            sys.modules.pop(module_name, None)


def _visible_checkout_modules(
    repo_root: Path, source_checkout: Path | None
) -> list[dict[str, str]]:
    roots = [repo_root / "plugins"]
    if source_checkout is not None:
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


def _plugin_id_from_manifest(artifact: Path, marketplace: str) -> tuple[str, str]:
    from agent.plugins.static_manifest import load_static_plugin_manifest

    manifest = load_static_plugin_manifest(artifact)
    return f"{manifest.name}@{marketplace}", manifest.entrypoint


async def _exercise(
    *,
    source: str,
    repo_root: Path,
    marketplace: str,
    workspace: Path,
    plugins_home: Path,
    capability_service: str | None,
    core_root: Path | None = None,
) -> dict[str, Any]:
    """Install one real checkout and execute its V3 apply path."""

    source_checkout = _source_checkout(source, repo_root)
    evidence: dict[str, Any] = {
        "source": source,
        "source_checkout": None if source_checkout is None else str(source_checkout),
        "checks": {},
    }
    evidence["checks"]["source_checkout_is_external"] = True
    try:
        if core_root is None or _under(core_root, repo_root) or (core_root / "plugins").exists():
            raise ValueError("外置验收必须提供仓库之外、不含 plugins 的 Core 制品目录")
        _hide_checkouts(repo_root, source_checkout)
        sys.path.insert(0, str(core_root))
        if importlib.util.find_spec("plugins") is not None:
            raise ValueError("运行环境仍能解析 checkout 的 plugins 命名空间")
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] in {"agent", "bootstrap", "bus", "core", "infra", "session", "utils"}:
                location = getattr(module, "__file__", None)
                if location is not None and not _under(Path(location), core_root):
                    raise ValueError(f"Core 模块预先从制品之外加载: {name}: {location}")
        from agent.plugins.install import install_git_plugin
        from agent.plugins.manager import PluginManager
        from agent.plugins.static_manifest import load_static_plugin_manifest
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
        evidence["checks"]["formal_install_artifact"] = artifact.is_relative_to(
            (plugins_home / "cache").resolve(strict=False)
        )
        evidence["checks"]["installed_manifest_identity"] = (
            load_static_plugin_manifest(artifact).name == plugin_id.split("@", 1)[0]
        )

        event_bus = EventBus()
        manager = PluginManager(
            [],
            event_bus=event_bus,
            workspace=workspace,
            installed_cache_root=plugins_home / "cache",
        )
        try:
            await manager.load_all()
            snapshot = manager.current_snapshot
            generation = (
                None if snapshot is None else snapshot.generations.get(plugin_id)
            )
            module = (
                None if generation is None else sys.modules.get(generation.module_path)
            )
            module_file = getattr(module, "__file__", None)
            evidence["generation_id"] = (
                None if generation is None else generation.generation_id
            )
            evidence["module_name"] = None if module is None else module.__name__
            evidence["module_file"] = module_file
            evidence["checks"]["apply"] = generation is not None and module is not None
            if (
                not isinstance(module_file, str)
                or generation is None
                or snapshot is None
            ):
                raise RuntimeError("formal install 后未形成 stable generation/module")
            module_path = Path(module_file).resolve(strict=True)
            archive_root = (workspace / "runtime" / "plugin-archives").resolve(
                strict=False
            )
            evidence["checks"]["module_file_is_core_archive"] = _under(
                module_path, archive_root
            )
            evidence["checks"]["module_file_not_checkout"] = not any(
                _under(module_path, root)
                for root in (repo_root / "plugins", source_checkout)
                if root is not None
            )
            installed_entrypoint = artifact / entrypoint
            evidence["module_file_sha256"] = _sha256(module_path)
            evidence["installed_entrypoint_sha256"] = _sha256(installed_entrypoint)
            evidence["checks"]["module_bytes_match_installed_artifact"] = (
                evidence["module_file_sha256"]
                == evidence["installed_entrypoint_sha256"]
            )
            visible = _visible_checkout_modules(repo_root, source_checkout)
            evidence["checkout_modules_visible"] = visible
            evidence["checks"]["checkout_invisible"] = not visible

            # 服务目录只证明注册；指定可调用能力的真实执行单独记录。
            from agent.plugins.snapshot import lease_runtime_snapshot

            async with lease_runtime_snapshot(manager.snapshot_store) as leased:
                context = leased.composition_root.context
                services = context.provided_services(plugin_ids=frozenset({plugin_id}))
                capability = {
                    "operation": "CompositionSnapshotRoot.context.provided_services",
                    "service_keys": sorted(key.name for key in services),
                    "service_count": len(services),
                }
                evidence["checks"]["capability_call"] = False
                if capability_service:
                    from agent.plugin_composition import ServiceKey

                    key = ServiceKey(capability_service)
                    if key not in services:
                        raise ValueError("选定能力不是目标插件提供的服务")
                    value = context.require(key)
                    capability["requested_service"] = capability_service
                    capability["requested_type"] = type(value).__name__
                    if callable(value):
                        called = value()
                        if inspect.isawaitable(called):
                            called = await called
                        capability["call_result_type"] = type(called).__name__
                        evidence["checks"]["capability_call"] = True
                    else:
                        raise TypeError("选定能力不可调用；读取对象不能充当行为验收")
                else:
                    capability["status"] = "unverified: 必须提供真实能力调用，服务枚举不是调用"
                evidence["capability_call"] = capability
        finally:
            try:
                await manager.terminate_all()
            finally:
                await event_bus.aclose()
        checks = evidence["checks"]
        evidence["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as error:
        evidence["status"] = "failed"
        evidence["error"] = f"{type(error).__name__}: {error}"
    return evidence


def _load_source_map(path: Path) -> Mapping[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source mapping 必须是 JSON object")
    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise ValueError("source mapping 的 key/value 必须是非空字符串")
        result[key] = value
    return result


def _load_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("packages"), list):
        raise ValueError("inventory JSON 缺少 packages 数组")
    return payload


def _source_for_entry(
    entry: Mapping[str, Any], sources: Mapping[str, str]
) -> str | None:
    package = str(entry.get("package", ""))
    manifest = entry.get("manifest")
    manifest_name = manifest.get("name") if isinstance(manifest, Mapping) else None
    for key in (package, str(manifest_name) if manifest_name else ""):
        if key in sources:
            return sources[key]
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="一个真实 external Git checkout 或 Git URL")
    parser.add_argument("--plugin", help="单插件结果标签；默认从 manifest 读取")
    parser.add_argument(
        "--all", action="store_true", help="按 inventory 对所有 manifest 插件执行"
    )
    parser.add_argument(
        "--inventory", type=Path, help="plugin_inventory.py 生成的 JSON"
    )
    parser.add_argument(
        "--sources-json",
        type=Path,
        help="plugin 名称到 external checkout/URL 的 JSON 映射",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--marketplace", default="external-acceptance")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--plugins-home", type=Path)
    parser.add_argument("--capability-service")
    parser.add_argument("--core-root", type=Path, help="已解包、位于 checkout 外且不含业务源码的 Core 制品")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-temporary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.all == bool(args.source):
        print("必须二选一：--source 或 --all", file=sys.stderr)
        return 2
    temp_root: Path | None = None
    if args.workspace is None or args.plugins_home is None:
        import tempfile

        temp_root = Path(tempfile.mkdtemp(prefix="akashic-external-acceptance-"))
        root = temp_root
        workspace = args.workspace or root / "workspace"
        plugins_home = args.plugins_home or root / "plugins-home"
    else:
        workspace = args.workspace
        plugins_home = args.plugins_home
    workspace.mkdir(parents=True, exist_ok=True)
    plugins_home.mkdir(parents=True, exist_ok=True)
    repo_root = args.repo_root.resolve(strict=True)

    reports: list[dict[str, Any]] = []
    try:
        if args.source:
            reports.append(
                asyncio.run(
                    _exercise(
                        source=args.source,
                        repo_root=repo_root,
                        marketplace=args.marketplace,
                        workspace=workspace,
                        plugins_home=plugins_home,
                        capability_service=args.capability_service,
                        core_root=args.core_root,
                    )
                )
            )
        else:
            if args.inventory is None or args.sources_json is None:
                print(
                    "--all 必须同时提供 --inventory 和 --sources-json", file=sys.stderr
                )
                return 2
            inventory = _load_inventory(args.inventory)
            sources = _load_source_map(args.sources_json)
            for entry in inventory["packages"]:
                if entry.get("classification") != "manifest-plugin":
                    classification = entry.get("classification")
                    reports.append(
                        {
                            "plugin": entry.get("package"),
                            "status": "failed",
                            "checks": {"manifest_plugin": False},
                            "error": (
                                "inventory row has an invalid V3 manifest"
                                if classification == "invalid-manifest"
                                else "inventory row is a support-package without an install manifest"
                            ),
                        }
                    )
                    continue
                source = _source_for_entry(entry, sources)
                if source is None:
                    reports.append(
                        {
                            "plugin": entry.get("package"),
                            "status": "failed",
                            "checks": {"external_source_mapped": False},
                            "error": "no external source mapping; one example cannot certify the inventory",
                        }
                    )
                    continue
                reports.append(
                    asyncio.run(
                        _exercise(
                            source=source,
                            repo_root=repo_root,
                            marketplace=args.marketplace,
                            workspace=workspace / str(entry["package"]),
                            plugins_home=plugins_home / str(entry["package"]),
                            capability_service=args.capability_service,
                        core_root=args.core_root,
                        )
                    )
                )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            f"external acceptance setup failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    finally:
        if temp_root is not None and not args.keep_temporary:
            shutil.rmtree(temp_root)

    result = {
        "schema_version": 1,
        "repository": str(repo_root),
        "workspace": str(workspace),
        "plugins_home": str(plugins_home),
        "mode": "all" if args.all else "single",
        "reports": reports,
        "status": (
            "passed"
            if reports and all(item.get("status") == "passed" for item in reports)
            else "failed"
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
