"""Support helper for external plugin composition acceptance tests."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4


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
    path = Path(path_or_url).expanduser().resolve(strict=True)
    if _under(path, repo_root):
        raise ValueError("source checkout 在本仓库内；外置验收拒绝把 builtin checkout 当 external artifact")
    if path.is_file():
        subprocess.run(["git", "bundle", "list-heads", str(path)], check=True, capture_output=True, text=True)
        return path
    if not path.is_dir():
        raise ValueError(f"external source 不是 Git 目录或 bundle: {path}")
    subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path, check=True, capture_output=True, text=True)
    return path


def _module_locations(module: Any) -> tuple[Path, ...]:
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        locations.append(Path(module_file).resolve(strict=False))
    package_path = getattr(module, "__path__", ())
    if isinstance(package_path, str):
        package_path = (package_path,)
    try:
        locations.extend(
            Path(item).resolve(strict=False)
            for item in package_path
            if isinstance(item, (str, os.PathLike))
        )
    except TypeError:
        pass
    return tuple(dict.fromkeys(locations))


def _purge_modules(roots: tuple[Path, ...]) -> None:
    modules = sorted(tuple(sys.modules.items()), key=lambda item: item[0].count("."), reverse=True)
    for module_name, module in modules:
        locations = _module_locations(module)
        if not locations:
            if module_name == "plugins" or module_name.startswith("plugins."):
                sys.modules.pop(module_name, None)
            continue
        if any(_under(path, root) for path in locations for root in roots):
            sys.modules.pop(module_name, None)


def _visible_checkout_modules(repo_root: Path, source_checkout: Path | None) -> list[dict[str, str]]:
    roots = [repo_root / "plugins"]
    if source_checkout is not None and source_checkout.is_dir():
        roots.append(source_checkout)
    evidence: list[dict[str, str]] = []
    for module_name, module in sorted(sys.modules.items()):
        for path in _module_locations(module):
            if any(_under(path, root) for root in roots):
                evidence.append({"module": module_name, "file": str(path)})
                break
    return evidence


def _ensure_empty_directory(path: Path, label: str) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"{label} 必须是目录而不能是文件或符号链接: {path}")
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} 必须为空的一次性目录: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _plugin_id_from_manifest(artifact: Path, marketplace: str) -> str:
    from agent.plugins.static_manifest import load_static_plugin_manifest

    return f"{load_static_plugin_manifest(artifact).name}@{marketplace}"


async def _invoke_capability(*, root: Any, plugin_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    from agent.plugin_composition import ServiceKey
    from plugins.ledger.contract import MESSAGE_CATALOG

    service = spec["service"]
    entrypoint = spec["entrypoint"]
    input_value = spec["input"]
    expected = spec["expect"]
    if entrypoint != "external_acceptance.message_roundtrip" or service not in {
        "acceptance.messages.roundtrip.v1",
        "acceptance.consumer.roundtrip.v1",
    }:
        raise ValueError(f"未实现安全能力 oracle: {service} {entrypoint}")
    key = ServiceKey(service)
    provided = root.provided_services(plugin_ids=frozenset({plugin_id}))
    if key not in provided:
        raise ValueError(f"选定能力不是目标插件提供的服务: {service}")
    provider_context, value = root._service_provider(key)
    async with provider_context.runtime_scope():
        actual = value(input_value)
        if inspect.isawaitable(actual):
            actual = await actual
    if "type" in expected and type(actual).__name__ != expected["type"]:
        raise AssertionError(f"能力返回类型不符: expected={expected['type']} actual={type(actual).__name__}")
    if actual != expected["value"]:
        raise AssertionError(f"能力返回值不符: expected={expected['value']!r} actual={actual!r}")
    assert isinstance(actual, dict)
    session_id = actual["session_id"]
    message_id = actual["message_id"]
    message = root.context.require(MESSAGE_CATALOG).reader(session_id).get(message_id)
    if message is None:
        raise AssertionError("Message roundtrip 返回的 identity 在独立 catalog reader 中不存在")
    text_parts = tuple(
        part.value for part in getattr(message.body, "parts", ()) if getattr(part, "kind", None) == "text"
    )
    if text_parts != (actual.get("read_text"),):
        raise AssertionError(f"Message roundtrip 独立 reader 读取正文不匹配: actual={text_parts!r}")
    return {
        "service": service,
        "entrypoint": entrypoint,
        "actual": actual,
        "message_readback": {
            "session_id": session_id,
            "message_id": message_id,
            "source": message.source,
            "seq": message.seq,
            "text_parts": text_parts,
        },
        "durable_message_readback": True,
        "call_executed": True,
        "status": "passed",
    }


def _generation_evidence(*, generation: Any, artifact: Path, repo_root: Path, source_checkout: Path | None) -> dict[str, Any]:
    module = sys.modules.get(generation.module_path)
    module_file = getattr(module, "__file__", None)
    checks: dict[str, bool] = {"apply": generation is not None and module is not None}
    if not isinstance(module_file, str):
        checks.update({"module_file_is_installed_entrypoint": False, "module_file_not_checkout": False, "module_bytes_match_installed_artifact": False})
    else:
        resolved = Path(module_file).resolve(strict=False)
        entrypoint = (artifact / "plugin.py").resolve(strict=False)
        checks["module_file_is_installed_entrypoint"] = resolved == entrypoint
        checks["module_file_not_checkout"] = not _under(resolved, repo_root) and (
            source_checkout is None or not _under(resolved, source_checkout)
        )
        checks["module_bytes_match_installed_artifact"] = (
            resolved.is_file() and entrypoint.is_file() and _sha256(resolved) == _sha256(entrypoint)
        )
    return {"generation_id": generation.generation_id, "module_file": module_file, "checks": checks}


async def _exercise_business_composition(
    *,
    jobs: list[dict[str, Any]],
    repo_root: Path,
    marketplace: str,
    workspace: Path,
    plugins_home: Path,
    api_sources: tuple[Path, ...] = (),
    replacement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_empty_directory(workspace, "workspace")
    _ensure_empty_directory(plugins_home, "plugins-home")
    _purge_modules((repo_root / "plugins",))

    from agent.plugin_composition import FiberState
    from agent.plugins.bundles import set_plugin_choice
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection

    PluginSelection(workspace).initialize()
    log = None
    manager = PluginManager([], workspace=workspace, installed_cache_root=plugins_home / "cache")
    reports: list[dict[str, Any]] = []
    installed_by_id: dict[str, dict[str, Any]] = {}
    replacement_evidence: dict[str, Any] | None = None
    initial_root: Any = None
    initial_generations: dict[str, Any] = {}
    source_restore: tuple[Path, Path] | None = None

    async def run_call(*, plugin_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        from plugins.ledger.contract import SessionAttributes

        raw_input = spec.get("input")
        if isinstance(raw_input, dict) and isinstance((session_id := raw_input.get("session_id")), str) and session_id:
            assert log is not None
            log.ensure_session(session_id, SessionAttributes(visibility="internal", learning="excluded"))
        root = manager.live_root
        if root is None:
            raise RuntimeError("business composition 缺少 live Root")
        generation = manager.generation(plugin_id)
        if generation is None:
            raise RuntimeError(f"business capability 缺少 generation: {plugin_id}")
        evidence = await _invoke_capability(root=root, plugin_id=plugin_id, spec=spec)
        evidence["live_root_id"] = root.generation_id
        evidence["generation_id"] = generation.generation_id
        return evidence

    try:
        for source in api_sources:
            installed = install_git_plugin(workspace=workspace, source=str(source), marketplace=marketplace, plugins_home=plugins_home)
            set_plugin_choice(workspace, _plugin_id_from_manifest(installed.installed_path, marketplace), enabled=False)

        for job in jobs:
            row: dict[str, Any] = {
                "plugin": job.get("label"),
                "source": str(job.get("source", "")),
                "checks": {
                    "source_checkout_is_external": True,
                    "formal_install_artifact": False,
                    "installed_manifest_identity": False,
                    "installed_source_has_no_sibling_plugins": False,
                    "apply": False,
                },
            }
            try:
                installed = install_git_plugin(
                    workspace=workspace, source=str(job["source"]), marketplace=marketplace, plugins_home=plugins_home,
                )
                artifact = installed.installed_path.resolve(strict=True)
                plugin_id = _plugin_id_from_manifest(artifact, marketplace)
                expected_name = str(job.get("label", ""))
                row.update({
                    "plugin_id": plugin_id,
                    "installed_artifact": str(artifact),
                    "capability_spec": job.get("capability_spec"),
                })
                row["checks"].update({
                    "formal_install_artifact": artifact.is_relative_to((plugins_home / "cache").resolve(strict=False)),
                    "installed_manifest_identity": not expected_name or plugin_id.split("@", 1)[0] == expected_name,
                    "installed_source_has_no_sibling_plugins": not (artifact / "plugins").exists(),
                })
                installed_by_id[plugin_id] = row
            except Exception as error:
                row["status"] = "failed"
                row["error"] = f"{type(error).__name__}: {error}"
            reports.append(row)

        await manager.load_all()
        from plugins.ledger.contract import MESSAGE_CATALOG, SessionAttributes

        initial_root = manager.live_root
        if initial_root is None:
            raise RuntimeError("business composition 未形成 live Root")
        log = initial_root.context.require(MESSAGE_CATALOG)._log
        initial_generations = {pid: manager.generation(pid) for pid in installed_by_id}
        visible_checkout_modules: list[dict[str, str]] = []
        for job in jobs:
            visible_checkout_modules.extend(
                _visible_checkout_modules(repo_root, _source_checkout(str(job["source"]), repo_root))
            )
        for row in reports:
            row["checkout_modules_visible"] = visible_checkout_modules
            row["checks"]["checkout_invisible"] = not visible_checkout_modules
            plugin_id = row.get("plugin_id")
            if not isinstance(plugin_id, str):
                continue
            generation = initial_generations.get(plugin_id)
            row["checks"]["apply"] = (
                generation is not None and generation.fiber is not None and generation.fiber.state is FiberState.ACTIVE
            )
            if generation is None:
                row["error"] = row.get("error", "apply 后缺少 live generation")
                continue
            if generation.fiber is not None and generation.fiber.error is not None:
                row["error"] = str(generation.fiber.error)
            gen_ev = _generation_evidence(
                generation=generation,
                artifact=Path(str(row["installed_artifact"])),
                repo_root=repo_root,
                source_checkout=_source_checkout(str(row["source"]), repo_root),
            )
            row["checks"].update({
                k: bool(gen_ev["checks"].get(k, False))
                for k in ("module_file_is_installed_entrypoint", "module_file_not_checkout", "module_bytes_match_installed_artifact")
            })
            if (spec := row.get("capability_spec")) is not None:
                row["capability_call"] = await run_call(plugin_id=plugin_id, spec=spec)
                row["checks"]["capability_call"] = True
                row["checks"]["durable_message_readback"] = bool(row["capability_call"].get("durable_message_readback", False))

        if replacement is not None:
            provider_id = f"{replacement['plugin']}@{marketplace}"
            consumer_id = f"{replacement['consumer']}@{marketplace}"
            old_provider = initial_generations[provider_id]
            old_consumer = initial_generations[consumer_id]
            old_provider_artifact = Path(str(installed_by_id[provider_id]["installed_artifact"]))
            original_provider_source = Path(str(installed_by_id[provider_id]["source"]))
            if replacement.get("remove_original_source", False):
                backup = original_provider_source.with_name(original_provider_source.name + ".before-acceptance-" + uuid4().hex)
                original_provider_source.rename(backup)
                source_restore = (original_provider_source, backup)
            old_consumer_fiber = old_consumer.fiber
            assert old_consumer_fiber is not None
            old_consumer_token = old_consumer_fiber.context.fiber.activation_token
            before_spec = replacement["before"]
            if isinstance((raw_input := before_spec.get("input")), dict) and isinstance((sid := raw_input.get("session_id")), str) and sid:
                log.ensure_session(sid, SessionAttributes(visibility="internal", learning="excluded"))
            update_id = "acceptance-" + uuid4().hex
            async with old_consumer_fiber.context.runtime_scope():
                old_consumer_call = await _invoke_capability(root=initial_root, plugin_id=consumer_id, spec=before_spec)
                old_consumer_call["live_root_id"] = initial_root.generation_id
                old_consumer_call["generation_id"] = old_consumer.generation_id
                installation = asyncio.create_task(manager.install(
                    source=replacement["source"], marketplace=marketplace, ref_name="", sparse_paths=[], update_id=update_id,
                ))
                await asyncio.wait_for(old_consumer_fiber._admission_closed.wait(), 10)
                waited_for_old = old_consumer_fiber._admission_closed.is_set()
                operation = manager._operation
                if operation is None or operation.task.done():
                    raise RuntimeError("旧 consumer 调用未排空，安装却已结束")
            accepted = await installation
            await operation.task
            update_status = manager.read_update(update_id)
            if manager.live_root is not initial_root:
                raise RuntimeError("replacement 改换了 live Root")
            new_provider = manager.generation(provider_id)
            new_consumer = manager.generation(consumer_id)
            assert new_provider is not None and new_consumer is not None
            new_consumer_call = await run_call(plugin_id=consumer_id, spec=replacement["after"])
            new_module = sys.modules.get(new_provider.module_path)
            new_module_file = getattr(new_module, "__file__", None)
            replacement_artifact = Path(new_provider.plugin_dir).resolve(strict=True)
            replacement_entrypoint = replacement_artifact / "plugin.py"
            new_module_sha256 = _sha256(Path(new_module_file)) if isinstance(new_module_file, str) else None
            replacement_sha256 = _sha256(replacement_entrypoint) if replacement_entrypoint.is_file() else None
            replacement_evidence = {
                "old_consumer_call": old_consumer_call,
                "new_consumer_call": new_consumer_call,
                "old_generation_id": old_provider.generation_id,
                "new_generation_id": new_provider.generation_id,
                "checks": {
                    "publication_committed": update_status.state == "active" and accepted.selection == "selected",
                    "install_waited_for_old_consumer": waited_for_old,
                    "provider_generation_changed": old_provider.generation_id != new_provider.generation_id,
                    "consumer_reactivated_after_provider_change": (
                        new_consumer.fiber is not None and new_consumer.fiber.context.fiber.activation_token is not old_consumer_token
                    ),
                    "old_consumer_readback": bool(old_consumer_call.get("durable_message_readback", False)),
                    "replacement_consumer_readback": bool(new_consumer_call.get("durable_message_readback", False)),
                    "replacement_module_from_new_artifact": new_module_sha256 is not None and new_module_sha256 == replacement_sha256,
                    "replacement_module_not_old_artifact": not isinstance(new_module_file, str) or not _under(Path(new_module_file), old_provider_artifact),
                    "original_source_not_required": not original_provider_source.exists(),
                },
            }
            replacement_evidence["status"] = "passed" if all(replacement_evidence["checks"].values()) else "failed"
    except Exception as error:
        reports.append({"plugin": "business-composition", "status": "failed", "checks": {"composition_completed": False}, "error": f"{type(error).__name__}: {error}"})
    finally:
        try:
            await manager.terminate_all()
        finally:
            if source_restore is not None:
                original, backup = source_restore
                if original.exists():
                    raise RuntimeError(f"原源码路径被重新占用，备份保留在 {backup}")
                backup.rename(original)

    resource_close = {"live_root_closed": manager.live_root is None, "message_log_closed": log is not None and log._closed}
    for row in reports:
        row["status"] = "passed" if all(row.get("checks", {}).values()) else "failed"
    checks = {
        "composition_loaded": initial_root is not None,
        "all_reports_passed": bool(reports) and all(row.get("status") == "passed" for row in reports),
        "business_calls_executed": any(row.get("checks", {}).get("capability_call", False) for row in reports),
        "durable_message_readback": any(row.get("checks", {}).get("durable_message_readback", False) for row in reports),
        **resource_close,
    }
    if replacement is not None:
        checks["replacement_verified"] = replacement_evidence is not None and replacement_evidence["status"] == "passed"
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "reports": reports,
        "replacement": replacement_evidence,
        "checks": checks,
    }
