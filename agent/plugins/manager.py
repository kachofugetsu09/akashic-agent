from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import secrets
import sys
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from agent.plugins.manifest import (
    load_package_manifest,
    load_plugin_manifest,
    write_package_manifest,
    write_plugin_manifest,
)
from agent.plugins.packages import discover_plugin_packages, enabled_plugin_packages
from agent.plugins.specs import (
    ManagedServiceSpec,
    McpServerSpec,
    ProactiveSourceSpec,
    RegisteredProactiveSource,
)
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterToolResultCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeToolCallCtx,
    BeforeTurnCtx,
    PreToolCtx,
    PromptRenderCtx,
)
from agent.plugins.registry import MetadataKind, PluginEventType, plugin_registry
from agent.plugins.source_resolver import resolve_plugin_sources
from agent.plugins.jobs import (
    IntervalTrigger,
    PluginJobSpec,
    PluginLlmService,
    RegisteredPluginJob,
)
from agent.plugins.scope import CleanupFailure, PluginScope, ScopedEventBus
from agent.plugins.generation import (
    GateCheckResult,
    GateResult,
    PluginContributions,
    PluginGeneration,
    PluginReadinessContext,
    PluginSemanticCheck,
)
from agent.plugins.importer import FreshPluginImporter
from agent.plugins.skill_host import PluginSkillHost, PreparedSkillCatalog
from agent.plugins.mcp_host import PluginMcpHost, PreparedMcpCatalog
from agent.plugins.activity_host import (
    PluginJobHost,
    PluginProactiveHost,
    PreparedJobCatalog,
    PreparedProactiveCatalog,
)
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    plugin_is_active,
)
from proactive_v2.lifecycle import ProactiveLifecycleSpec
from proactive_v2.lifecycle import ProactiveLifecycleBuilder
from agent.tool_hooks.base import ToolHook
from agent.tool_hooks.types import HookContext, HookOutcome
from bus.event_bus import EventBus
from infra.channels.contract import Channel

logger = logging.getLogger(__name__)
U = TypeVar("U")


def _package_project_root(plugin_dirs: list[Path]) -> Path | None:
    for plugin_dir in plugin_dirs:
        root = plugin_dir.parent if plugin_dir.name == "plugins" else None
        if root is not None and (root / "plugin_packages").is_dir():
            return root
    return None


async def _complete_critical(awaitable: Awaitable[U]) -> tuple[U, bool]:
    """在外部取消后完成关键异步操作，并返回是否收到取消。"""

    # 1. 将关键操作放入独立任务，避免调用方取消传播进去
    task = asyncio.ensure_future(awaitable)
    cancelled = False

    # 2. 屏蔽等待并记录外部取消，直到操作本身结束
    while not task.done():
        try:
            _ = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True

    # 3. 读取操作结果，保留其真实异常
    result = await task
    return result, cancelled

_EVENT_TYPE_MAP: dict[PluginEventType, type] = {
    PluginEventType.BEFORE_TURN: BeforeTurnCtx,
    PluginEventType.BEFORE_REASONING: BeforeReasoningCtx,
    PluginEventType.PROMPT_RENDER: PromptRenderCtx,
    PluginEventType.BEFORE_STEP: BeforeStepCtx,
    PluginEventType.AFTER_STEP: AfterStepCtx,
    PluginEventType.AFTER_REASONING: AfterReasoningCtx,
    PluginEventType.AFTER_TURN: AfterTurnCtx,
    PluginEventType.BEFORE_TOOL_CALL: BeforeToolCallCtx,
    PluginEventType.AFTER_TOOL_RESULT: AfterToolResultCtx,
}


@dataclass(frozen=True)
class ActivePluginInfo:
    plugin_id: str
    plugin_dir: Path
    manifest: dict[str, object]
    module_path: str
    skill_roots: tuple[Path, ...] = ()
    drift_skill_roots: tuple[Path, ...] = ()
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)


class PluginManager:
    POST_PUBLISH_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: EventBus,
        tool_registry: Any = None,
        workspace: Path | None = None,
        session_manager: Any = None,
        memory_engine: Any = None,
        llm: PluginLlmService | None = None,
        installed_cache_root: Path | None = None,
        user_mcp_server_names: Callable[[], set[str]] | None = None,
    ) -> None:
        self._dirs = plugin_dirs
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._workspace = workspace
        self._session_manager = session_manager
        self._memory_engine = memory_engine
        self._llm = llm
        self._installed_cache_root = installed_cache_root
        self._user_mcp_server_names = user_mcp_server_names or set
        self._channel_switcher: Callable[
            [str, tuple[Channel, ...], tuple[Channel, ...]],
            Awaitable[None],
        ] | None = None
        self._dashboard_preparer: Callable[[RuntimeSnapshot], None] | None = None
        self._service_switcher: Callable[
            [str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]],
            Awaitable[None],
        ] | None = None
        self._endpoint_quiescer: Callable[[], Awaitable[None]] | None = None
        self._endpoint_resumer: Callable[[], Awaitable[None]] | None = None
        self._endpoint_switcher: Callable[
            [
                str,
                dict[str, dict[str, Any]],
                dict[str, dict[str, Any]],
                tuple[Channel, ...],
                tuple[Channel, ...],
            ],
            Awaitable[None],
        ] | None = None
        self._loaded: set[str] = set()
        self._channels: list[Channel] = []
        self._tool_hooks: list[ToolHook] = []
        self._before_turn_modules: list[object] = []
        self._before_reasoning_modules: list[object] = []
        self._prompt_render_modules: list[object] = []
        self._before_step_modules: list[object] = []
        self._after_step_modules: list[object] = []
        self._after_reasoning_modules: list[object] = []
        self._after_turn_modules: list[object] = []
        self._proactive_modules: list[object] = []
        self._proactive_lifecycles: list[object] = []
        self._proactive_module_factories: list[object] = []
        self._proactive_runtime_factories: list[object] = []
        self._proactive_sources: list[RegisteredProactiveSource] = []
        self._jobs: list[RegisteredPluginJob] = []
        self._active_plugins: dict[str, ActivePluginInfo] = {}
        self._scopes: dict[str, PluginScope] = {}
        self._cleanup_failures: list[CleanupFailure] = []
        self._active_generations: dict[str, PluginGeneration] = {}
        self._prepared_generations: dict[str, PluginGeneration] = {}
        self._gate_results: dict[str, GateResult] = {}
        self._stable_aliases: dict[str, str] = {}
        self._generation_sequence = 0
        self._candidate_prepare_lock = asyncio.Lock()
        self._fresh_importer = FreshPluginImporter()
        self._manager_namespace = secrets.token_hex(4)
        self._skill_host = PluginSkillHost(workspace)
        self._mcp_host = PluginMcpHost()
        self._job_host = PluginJobHost()
        self._proactive_host = PluginProactiveHost()
        self._snapshot_compiler = RuntimeSnapshotCompiler()
        self._snapshot_store = RuntimeSnapshotStore(self._on_snapshot_drained)
        self._snapshot_skill_catalogs: dict[str, str] = {}
        self._event_bus.bind_runtime_snapshot_store(self._snapshot_store)

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    @property
    def tool_hooks(self) -> list[ToolHook]:
        return list(self._tool_hooks)

    @property
    def channels(self) -> list[Channel]:
        return list(self._channels)

    @property
    def before_turn_modules(self) -> list[object]:
        return list(self._before_turn_modules)

    @property
    def before_reasoning_modules(self) -> list[object]:
        return list(self._before_reasoning_modules)

    @property
    def prompt_render_modules(self) -> list[object]:
        return list(self._prompt_render_modules)

    @property
    def before_step_modules(self) -> list[object]:
        return list(self._before_step_modules)

    @property
    def after_step_modules(self) -> list[object]:
        return list(self._after_step_modules)

    @property
    def after_reasoning_modules(self) -> list[object]:
        return list(self._after_reasoning_modules)

    @property
    def after_turn_modules(self) -> list[object]:
        return list(self._after_turn_modules)

    @property
    def proactive_modules(self) -> list[object]:
        return list(self._proactive_modules)

    @property
    def proactive_lifecycles(self) -> list[object]:
        return list(self._proactive_lifecycles)

    @property
    def proactive_module_factories(self) -> list[object]:
        return list(self._proactive_module_factories)

    @property
    def proactive_runtime_factories(self) -> list[object]:
        return list(self._proactive_runtime_factories)

    @property
    def proactive_sources(self) -> list[RegisteredProactiveSource]:
        return list(self._proactive_sources)

    @property
    def jobs(self) -> list[RegisteredPluginJob]:
        return list(self._jobs)

    @property
    def llm(self) -> PluginLlmService | None:
        return self._llm

    @property
    def plugin_dirs(self) -> list[Path]:
        return list(self._dirs)

    def active_plugins(self) -> list[ActivePluginInfo]:
        return [
            self._active_plugins[generation.module_path]
            for generation in self._active_generations.values()
            if self._registry_active(generation.module_path)
        ]

    @property
    def cleanup_failures(self) -> list[CleanupFailure]:
        return list(self._cleanup_failures)

    def generation(self, plugin_id: str) -> PluginGeneration | None:
        return self._active_generations.get(plugin_id)

    def latest_gate(self, plugin_id: str) -> GateResult | None:
        return self._gate_results.get(plugin_id)

    def prepared_generation(self, plugin_id: str) -> PluginGeneration | None:
        return self._prepared_generations.get(plugin_id)

    def skill_catalog(self, generation_id: str) -> PreparedSkillCatalog | None:
        return self._skill_host.get(generation_id)

    def mcp_catalog(self, generation_id: str) -> PreparedMcpCatalog | None:
        return self._mcp_host.get(generation_id)

    def bind_channel_switcher(
        self,
        switcher: Callable[
            [str, tuple[Channel, ...], tuple[Channel, ...]],
            Awaitable[None],
        ],
    ) -> None:
        self._channel_switcher = switcher

    def bind_dashboard_preparer(
        self,
        preparer: Callable[[RuntimeSnapshot], None],
    ) -> None:
        self._dashboard_preparer = preparer

    def bind_service_switcher(
        self,
        switcher: Callable[
            [str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]],
            Awaitable[None],
        ],
    ) -> None:
        self._service_switcher = switcher

    def bind_endpoint_admission(
        self,
        *,
        quiesce: Callable[[], Awaitable[None]],
        resume: Callable[[], Awaitable[None]],
    ) -> None:
        self._endpoint_quiescer = quiesce
        self._endpoint_resumer = resume

    def bind_endpoint_switcher(
        self,
        switcher: Callable[
            [
                str,
                dict[str, dict[str, Any]],
                dict[str, dict[str, Any]],
                tuple[Channel, ...],
                tuple[Channel, ...],
            ],
            Awaitable[None],
        ],
    ) -> None:
        self._endpoint_switcher = switcher

    def job_catalog(self, generation_id: str) -> PreparedJobCatalog | None:
        return self._job_host.get(generation_id)

    def proactive_catalog(
        self,
        generation_id: str,
    ) -> PreparedProactiveCatalog | None:
        return self._proactive_host.get(generation_id)

    @property
    def current_snapshot(self) -> RuntimeSnapshot | None:
        return self._snapshot_store.current

    @property
    def snapshot_store(self) -> RuntimeSnapshotStore:
        return self._snapshot_store

    def sync_manifest(self, *, plugins_home: Path | None = None) -> Path:
        entries = load_plugin_manifest(plugins_home)
        project_root = _package_project_root(self._dirs)
        if project_root is not None:
            packages = discover_plugin_packages(project_root)
            package_entries = load_package_manifest(plugins_home)
            for package_id, package in packages.items():
                if package_id not in package_entries:
                    package_entries[package_id] = any(
                        entries.get(member, False) for member in package.members
                    )
                for member in package.members:
                    entries.pop(member, None)
            _ = write_package_manifest(package_entries, plugins_home=plugins_home)
        for mod in self.discover():
            if mod.get("package_id"):
                continue
            _ = entries.setdefault(_resolve_plugin_id(mod), True)
        return write_plugin_manifest(entries, plugins_home=plugins_home)

    def watch_revision(self) -> str:
        digest = hashlib.sha256()
        home = _plugins_home(self._installed_cache_root)
        digest.update(_path_metadata(home / "manifest.toml"))
        for mod in self.discover():
            plugin_id = _resolve_plugin_id(mod)
            plugin_dir = Path(mod["plugin_root"])
            data_dir = _resolve_plugin_data_dir(
                mod["name"],
                mod,
                self._installed_cache_root,
            )
            digest.update(plugin_id.encode())
            digest.update(_source_metadata_revision(plugin_dir))
            digest.update(_path_metadata(data_dir / "config.local.toml"))
        return digest.hexdigest()

    def _registry_active(self, module_path: str) -> bool:
        if module_path not in self._active_plugins:
            return False
        instance = plugin_registry.get_instance(module_path)
        if instance is None:
            return True
        return plugin_is_active(instance, plugin_id=module_path)

    @property
    def telegram_bot_commands(self) -> list[tuple[str, str]]:
        commands: list[tuple[str, str]] = []
        for generation in self._active_generations.values():
            if not self._registry_active(generation.module_path):
                continue
            instance = generation.instance
            getter = getattr(instance, "telegram_bot_commands", None)
            if getter is None:
                continue
            typed_getter = cast(Callable[[], list[tuple[str, str]]], getter)
            for command, description in typed_getter():
                commands.append((str(command), str(description)))
        return commands

    # 扫描所有 plugin_dirs，返回可加载的插件描述列表
    def discover(self) -> list[dict[str, str]]:
        mods: list[dict[str, str]] = []
        seen_names: set[str] = set()
        project_root = _package_project_root(self._dirs)
        packages = discover_plugin_packages(project_root) if project_root else {}
        enabled_packages = (
            enabled_plugin_packages(
                project_root,
                load_package_manifest(_plugins_home(self._installed_cache_root)),
            )
            if project_root
            else {}
        )
        member_packages = {
            member: package.id
            for package in packages.values()
            for member in package.members
        }
        enabled_members = {
            member
            for package in enabled_packages.values()
            for member in package.members
        }
        for source in resolve_plugin_sources(
            self._dirs,
            installed_cache_root=self._installed_cache_root,
        ):
            name = source.plugin_root.parent.name if source.source_type == "installed" else source.plugin_root.name
            package_id = member_packages.get(name, "")
            if package_id and name not in enabled_members:
                continue
            if name in seen_names and source.source_type == "builtin":
                logger.warning("插件名重复，跳过: %s (%s)", name, source.plugin_root)
                continue
            seen_names.add(name)
            import_suffix = name.replace("-", "_").replace("@", "_")
            import_source = source.marketplace or source.plugin_root.parent.name
            module_path = source.plugin_root / "plugin.py"
            mods.append({
                "name": name,
                "plugin_root": str(source.plugin_root),
                "module_path": str(module_path) if module_path is not None else "",
                "import_path": f"akasic_plugin_{import_source}_{import_suffix}",
                "marketplace": source.marketplace,
                "source_type": source.source_type,
                "package_id": package_id,
            })
        return mods

    async def load_all(self) -> None:
        for mod in self.discover():
            _ = await self._load_one(mod)

    async def prepare_candidate(self, plugin_id: str) -> PluginGeneration | None:
        await self.discard_prepared(plugin_id)
        for mod in self.discover():
            if _resolve_plugin_id(mod) == plugin_id:
                return await self._load_one(mod, activate=False)
        raise KeyError(f"插件不存在: {plugin_id}")

    async def discard_prepared(self, plugin_id: str) -> None:
        generation = self._prepared_generations.pop(plugin_id, None)
        if generation is None:
            return
        await self._dispose_generation(generation, state="discarded")

    async def _dispose_generation(
        self,
        generation: PluginGeneration,
        *,
        state: str,
        preserve_stable_alias: bool = False,
    ) -> None:
        """完成插件终止、作用域清理和注册表卸载。"""

        # 1. 终止生命周期对象，并在调用方取消后继续完成它
        externally_cancelled = False
        if generation.initialization_started:
            terminator = getattr(generation.instance, "terminate", None)
            if callable(terminator):
                try:
                    _, terminator_cancelled = await _complete_critical(
                        cast(Callable[[], Awaitable[None]], terminator)()
                    )
                    externally_cancelled = terminator_cancelled
                except (asyncio.CancelledError, Exception) as error:
                    current = asyncio.current_task()
                    externally_cancelled = (
                        current is not None and current.cancelling() > 0
                    )
                    self._cleanup_failures.append(
                        CleanupFailure(
                            resource=f"plugin:{generation.plugin_id}:terminate",
                            error=str(error) or type(error).__name__,
                        )
                    )

        # 2. 收集作用域失败，确保外部取消不会截断资源清理
        cleanup_failures, cleanup_cancelled = await _complete_critical(
            generation.scope.aclose()
        )
        self._cleanup_failures.extend(cleanup_failures)
        externally_cancelled = externally_cancelled or cleanup_cancelled

        # 3. 清理注册表和模块树
        _ = self._scopes.pop(generation.module_path, None)
        self._loaded.discard(generation.module_path)
        _ = self._active_plugins.pop(generation.module_path, None)
        for metadata in plugin_registry.get_handlers_by_module_path(
            generation.module_path
        ):
            if metadata.kind == MetadataKind.TOOL and self._tool_registry is not None:
                self._tool_registry.unregister(
                    metadata.tool_name or metadata.handler_name
                )
        self._remove_module_tree(generation.module_path)
        stable_alias = self._stable_aliases.get(generation.module_path)
        if stable_alias is not None and not preserve_stable_alias:
            _ = self._stable_aliases.pop(generation.module_path, None)
            if plugin_registry.get_instance(stable_alias) is generation.instance:
                self._remove_module_tree(stable_alias)
            else:
                self._fresh_importer.unregister(stable_alias)
        generation.state = state
        if externally_cancelled:
            raise asyncio.CancelledError

    async def _on_snapshot_drained(self, snapshot: RuntimeSnapshot) -> None:
        catalog_id = self._snapshot_skill_catalogs.pop(snapshot.snapshot_id, None)
        if catalog_id is not None:
            self._skill_host.close(catalog_id)
        state = "aborted" if snapshot.state == "aborted" else "retired"
        current = self._snapshot_store.current
        for generation in snapshot.generations.values():
            if self._snapshot_store.generation_is_referenced_elsewhere(
                generation,
                excluding_snapshot_id=snapshot.snapshot_id,
            ):
                continue
            replacement = (
                current.generations.get(generation.plugin_id)
                if current is not None
                else None
            )
            await self._dispose_generation(
                generation,
                state=state,
                preserve_stable_alias=(
                    replacement is not None and replacement is not generation
                ),
            )

    async def prepare_changed(self) -> list[dict[str, object]]:
        async with self._candidate_prepare_lock:
            discovered = {
                _resolve_plugin_id(mod): mod
                for mod in self.discover()
            }
            return await self._prepare_changed(discovered=discovered)

    async def reconcile_changed(self) -> list[dict[str, object]]:
        async with self._candidate_prepare_lock:
            results: list[dict[str, object]] = []
            discovered = {
                _resolve_plugin_id(mod): mod
                for mod in self.discover()
            }
            manifest = load_plugin_manifest(
                _plugins_home(self._installed_cache_root)
            )
            desired = {
                plugin_id
                for plugin_id, mod in discovered.items()
                if mod.get("package_id") or manifest.get(plugin_id, True)
            }
            for plugin_id in sorted(set(self._active_generations) - desired):
                results.append(await self._deactivate_plugin(plugin_id))
            for plugin_id in sorted(desired.intersection(self._active_generations)):
                prepared = await self._prepare_changed(
                    discovered=discovered,
                    plugin_ids={plugin_id},
                    force_reprepare=True,
                )
                if not prepared:
                    continue
                result = prepared[0]
                if result.get("prepared_generation") is None:
                    results.append(result)
                    continue
                results.append(await self._publish_prepared(plugin_id))
            for plugin_id in sorted(desired - set(self._active_generations)):
                generation = await self._load_one(discovered[plugin_id], activate=False)
                if generation is None:
                    continue
                results.append(await self._publish_prepared(plugin_id))
            return results

    async def reconcile_disabled_and_drain(self, plugin_id: str) -> None:
        async with self._candidate_prepare_lock:
            manifest = load_plugin_manifest(
                _plugins_home(self._installed_cache_root)
            )
            if manifest.get(plugin_id, False):
                raise RuntimeError(f"插件尚未禁用: {plugin_id}")
            active = self._active_generations.get(plugin_id)
            if active is None:
                return
            await self._deactivate_plugin(plugin_id)
            await self._snapshot_store.wait_for_generation_drained(active)
            if not active.scope.closed:
                raise RuntimeError(f"插件旧代资源尚未关闭: {plugin_id}")

    async def _deactivate_plugin(self, plugin_id: str) -> dict[str, object]:
        active = self._active_generations[plugin_id]
        generations = {
            key: generation
            for key, generation in self._active_generations.items()
            if key != plugin_id
        }
        snapshot, catalog_id = self._compile_topology_snapshot(generations)
        try:
            self._compile_snapshot_event_handlers(snapshot)
            if self._dashboard_preparer is not None:
                self._dashboard_preparer(snapshot)
        except BaseException:
            self._skill_host.close(catalog_id)
            raise

        old_services = active.contributions.managed_services
        old_channels = active.contributions.channels
        from agent.plugins.snapshot import get_current_runtime_lease

        if (old_services or old_channels) and get_current_runtime_lease() is not None:
            self._skill_host.close(catalog_id)
            raise RuntimeError("持有 RuntimeSnapshot lease 时不能切换独占端点")
        quiesced = (
            self._snapshot_store.pause_admission()
            if old_services or old_channels
            else None
        )
        endpoints_switched = False
        transaction = None
        try:
            if quiesced is not None:
                if self._endpoint_quiescer is not None:
                    await self._endpoint_quiescer()
                await self._snapshot_store.wait_for_no_leases(quiesced)
            if old_services or old_channels:
                await self._switch_plugin_endpoints(
                    plugin_id,
                    old_services,
                    {},
                    old_channels,
                    (),
                )
                endpoints_switched = True
            self._snapshot_skill_catalogs[snapshot.snapshot_id] = catalog_id
            transaction = self._snapshot_store.begin_publish(
                snapshot,
                admission_gated=quiesced is not None,
            )
            await self._post_snapshot_invariants(snapshot)
        except BaseException:
            endpoint_error: BaseException | None = None
            if endpoints_switched:
                try:
                    await self._switch_plugin_endpoints(
                        plugin_id,
                        {},
                        old_services,
                        (),
                        old_channels,
                    )
                except BaseException as error:
                    endpoint_error = error
            if transaction is not None and self._snapshot_store.current is snapshot:
                await self._snapshot_store.abort(transaction)
            else:
                await self._snapshot_store.resume(quiesced)
                _ = self._snapshot_skill_catalogs.pop(snapshot.snapshot_id, None)
                self._skill_host.close(catalog_id)
            if self._endpoint_resumer is not None and quiesced is not None:
                await self._endpoint_resumer()
            if endpoint_error is not None:
                raise RuntimeError("禁用插件后旧端点恢复失败") from endpoint_error
            raise

        commit_error: BaseException | None = None
        commit_cancelled = False
        try:
            assert transaction is not None
            _, commit_cancelled = await _complete_critical(
                self._snapshot_store.commit(transaction)
            )
        except BaseException as error:
            commit_error = error
        _ = self._active_generations.pop(plugin_id)
        self._channels = [
            channel
            for generation in self._active_generations.values()
            for channel in generation.contributions.channels
        ]
        resume_cancelled = False
        if self._endpoint_resumer is not None and quiesced is not None:
            _, resume_cancelled = await _complete_critical(self._endpoint_resumer())
        if commit_error is not None:
            raise commit_error
        if commit_cancelled or resume_cancelled:
            raise asyncio.CancelledError
        result = {
            "plugin_id": plugin_id,
            "old_generation": active.generation_id,
            "new_generation": None,
            "snapshot_id": snapshot.snapshot_id,
            "publication_state": "disabled",
        }
        logger.info(
            "plugin_snapshot_status %s",
            json.dumps(result, ensure_ascii=False, sort_keys=True),
        )
        return result

    async def _switch_plugin_endpoints(
        self,
        plugin_id: str,
        old_services: dict[str, dict[str, Any]],
        new_services: dict[str, dict[str, Any]],
        old_channels: tuple[Channel, ...],
        new_channels: tuple[Channel, ...],
    ) -> None:
        services_changed = old_services != new_services
        channels_changed = old_channels != new_channels
        if self._endpoint_switcher is not None:
            await self._endpoint_switcher(
                plugin_id,
                old_services,
                new_services,
                old_channels,
                new_channels,
            )
            return
        if services_changed and channels_changed:
            raise RuntimeError("同时切换 managed service 与 Channel 需要统一端点宿主")
        if services_changed:
            if self._service_switcher is None:
                raise RuntimeError("managed service 宿主未绑定")
            await self._service_switcher(plugin_id, old_services, new_services)
        if channels_changed:
            if self._channel_switcher is None:
                raise RuntimeError("Channel 宿主未绑定")
            await self._channel_switcher(plugin_id, old_channels, new_channels)

    def _compile_topology_snapshot(
        self,
        generations: dict[str, PluginGeneration],
    ) -> tuple[RuntimeSnapshot, str]:
        self._generation_sequence += 1
        catalog_id = f"topology:{self._generation_sequence}:{secrets.token_hex(4)}"
        ordered = list(generations.values())
        catalog = self._skill_host.prepare(
            catalog_id,
            normal_roots=PluginSkillHost.roots_for(ordered, drift=False),
            drift_roots=PluginSkillHost.roots_for(ordered, drift=True),
            ignored_normal_roots=tuple(
                root
                for generation in ordered
                for root in generation.contributions.skill_roots
            ),
            ignored_drift_roots=tuple(
                root
                for generation in ordered
                for root in generation.contributions.drift_skill_roots
            ),
        )
        try:
            snapshot = self._snapshot_compiler.compile(
                generations,
                snapshot_revision=catalog_id,
            )
            snapshot.skill_catalog_generation_id = catalog_id
            snapshot.plugin_skill_index = catalog.normal_plugins
            snapshot.tool_registry = self._compile_snapshot_tools(generations)
            snapshot.tool_hooks = self._compile_snapshot_tool_hooks(generations)
            return snapshot, catalog_id
        except BaseException:
            self._skill_host.close(catalog_id)
            raise

    async def publish_prepared(self, plugin_id: str) -> dict[str, object]:
        async with self._candidate_prepare_lock:
            return await self._publish_prepared(plugin_id)

    async def _publish_prepared(self, plugin_id: str) -> dict[str, object]:
        generation = self._prepared_generations.get(plugin_id)
        if generation is None:
            raise KeyError(f"插件没有待发布候选: {plugin_id}")
        snapshot = generation.runtime_snapshot
        if snapshot is None:
            raise RuntimeError(f"插件候选缺少 RuntimeSnapshot: {plugin_id}")
        active = self._active_generations.get(plugin_id)
        try:
            await self._initialize_prepared_generation(generation)
        except (asyncio.CancelledError, Exception) as error:
            self._record_failed_gate(
                plugin_id=plugin_id,
                revision=generation.source_revision,
                check_id="initialize",
                reason=str(error) or type(error).__name__,
            )
            await self.discard_prepared(plugin_id)
            if isinstance(error, asyncio.CancelledError):
                raise
            result = self._publication_status(
                plugin_id,
                active=active,
                candidate=generation,
                publication_state="failed",
            )
            logger.info(
                "plugin_snapshot_status %s",
                json.dumps(result, ensure_ascii=False, sort_keys=True),
            )
            return result

        old_services = (
            active.contributions.managed_services if active is not None else {}
        )
        new_services = generation.contributions.managed_services
        old_channels = active.contributions.channels if active is not None else ()
        new_channels = generation.contributions.channels
        endpoint_changed = (
            old_services != new_services or old_channels != new_channels
        )
        self._compile_snapshot_event_handlers(snapshot)
        if self._dashboard_preparer is not None:
            try:
                self._dashboard_preparer(snapshot)
            except Exception as error:
                self._record_failed_gate(
                    plugin_id=plugin_id,
                    revision=generation.source_revision,
                    check_id="dashboard",
                    reason=str(error) or type(error).__name__,
                )
                await self.discard_prepared(plugin_id)
                return self._publication_status(
                    plugin_id,
                    active=active,
                    candidate=generation,
                    publication_state="failed",
                )

        quiesced_snapshot: RuntimeSnapshot | None = None
        if endpoint_changed:
            from agent.plugins.snapshot import get_current_runtime_lease

            if get_current_runtime_lease() is not None:
                await self.discard_prepared(plugin_id)
                raise RuntimeError("持有 RuntimeSnapshot lease 时不能切换独占端点")
            quiesced_snapshot = self._snapshot_store.pause_admission()
            try:
                if self._endpoint_quiescer is not None:
                    await self._endpoint_quiescer()
                if quiesced_snapshot is not None:
                    await self._snapshot_store.wait_for_no_leases(
                        quiesced_snapshot
                    )
            except BaseException:
                await self._snapshot_store.resume(quiesced_snapshot)
                if self._endpoint_resumer is not None:
                    await self._endpoint_resumer()
                await self.discard_prepared(plugin_id)
                raise
        endpoints_switched = False
        if endpoint_changed:
            try:
                await self._switch_plugin_endpoints(
                    plugin_id,
                    old_services,
                    new_services,
                    old_channels,
                    new_channels,
                )
                endpoints_switched = True
            except (asyncio.CancelledError, Exception) as error:
                self._record_failed_gate(
                    plugin_id=plugin_id,
                    revision=generation.source_revision,
                    check_id="endpoints",
                    reason=str(error) or type(error).__name__,
                )
                await self._snapshot_store.resume(quiesced_snapshot)
                if self._endpoint_resumer is not None:
                    await self._endpoint_resumer()
                await self.discard_prepared(plugin_id)
                if isinstance(error, asyncio.CancelledError):
                    raise
                return self._publication_status(
                    plugin_id,
                    active=active,
                    candidate=generation,
                    publication_state="failed",
                )
        if generation.staged_event_bus is not None:
            generation.staged_event_bus.publish()
        transaction = self._snapshot_store.begin_publish(
            snapshot,
            admission_gated=quiesced_snapshot is not None,
        )
        try:
            await asyncio.wait_for(
                self._post_publish_invariants(generation, snapshot),
                timeout=self.POST_PUBLISH_TIMEOUT_SECONDS,
            )
        except (asyncio.CancelledError, Exception):
            _ = self._prepared_generations.pop(plugin_id, None)
            generation.state = "aborted"
            endpoint_error: BaseException | None = None
            if endpoints_switched:
                try:
                    await self._switch_plugin_endpoints(
                        plugin_id,
                        new_services,
                        old_services,
                        new_channels,
                        old_channels,
                    )
                except BaseException as error:
                    endpoint_error = error
            await self._snapshot_store.abort(transaction)
            if self._endpoint_resumer is not None:
                await self._endpoint_resumer()
            if endpoint_error is not None:
                raise RuntimeError("Snapshot abort 后旧端点恢复失败") from endpoint_error
            raise

        commit_error: BaseException | None = None
        commit_cancelled = False
        try:
            _, commit_cancelled = await _complete_critical(
                self._snapshot_store.commit(transaction)
            )
        except BaseException as error:
            commit_error = error

        _ = self._prepared_generations.pop(plugin_id)
        self._scopes[generation.module_path] = generation.scope
        self._loaded.add(generation.module_path)
        generation.state = "active"
        self._active_generations[plugin_id] = generation
        if active is not None:
            active.state = "retired"
        self._activate_published_generation(generation, active)
        self._channels = [
            channel
            for item in self._active_generations.values()
            for channel in item.contributions.channels
        ]
        resume_cancelled = False
        if self._endpoint_resumer is not None and quiesced_snapshot is not None:
            _, resume_cancelled = await _complete_critical(self._endpoint_resumer())
        if commit_error is not None:
            raise commit_error
        if commit_cancelled or resume_cancelled:
            raise asyncio.CancelledError
        result = self._publication_status(
            plugin_id,
            active=active,
            candidate=generation,
            publication_state="committed",
        )
        logger.info(
            "plugin_snapshot_status %s",
            json.dumps(result, ensure_ascii=False, sort_keys=True),
        )
        return result

    def _activate_published_generation(
        self,
        generation: PluginGeneration,
        previous: PluginGeneration | None,
    ) -> None:
        plugin_dir = Path(generation.instance.context.plugin_dir)
        self._active_plugins[generation.module_path] = ActivePluginInfo(
            plugin_id=generation.plugin_id,
            plugin_dir=plugin_dir,
            manifest=generation.contributions.manifest,
            module_path=generation.module_path,
            skill_roots=generation.contributions.skill_roots,
            drift_skill_roots=generation.contributions.drift_skill_roots,
            mcp_servers=generation.contributions.mcp_servers,
        )
        stable_alias = None
        if previous is not None:
            stable_alias = self._stable_aliases.pop(previous.module_path, None)
        if stable_alias is None:
            retired_module = next(
                (
                    module_path
                    for module_path, info in self._active_plugins.items()
                    if module_path != generation.module_path
                    and info.plugin_id == generation.plugin_id
                ),
                None,
            )
            if retired_module is not None:
                stable_alias = self._stable_aliases.pop(retired_module, None)
        if stable_alias is None:
            stable_alias = generation.module_path.rsplit("__g", 1)[0]
        self._stable_aliases[generation.module_path] = stable_alias
        self._remove_module_tree(stable_alias)
        self._fresh_importer.register(stable_alias, plugin_dir)
        plugin_registry.register_instance(stable_alias, generation.instance)
        sys.modules[stable_alias] = sys.modules[generation.module_path]

    async def _initialize_prepared_generation(
        self,
        generation: PluginGeneration,
    ) -> None:
        if generation.initialization_started:
            return
        from agent.plugins.context import PluginKVStore

        instance = cast(Any, generation.instance)
        context = instance.context
        staged_event_bus = ScopedEventBus(
            self._event_bus,
            generation.scope,
            staged=True,
        )
        generation.staged_event_bus = staged_event_bus
        context.event_bus = staged_event_bus
        context.kv_store = PluginKVStore(context.data_dir / ".kv.json")
        context.session_manager = self._session_manager
        context.memory_engine = self._memory_engine
        context.llm = self._llm
        context.scope = generation.scope
        context.tool_registry = generation.runtime_snapshot.tool_registry
        generation.initialization_started = True
        await instance.initialize()
        generation.minimum_resource_count = generation.scope.resource_count

    def _compile_snapshot_event_handlers(self, snapshot: RuntimeSnapshot) -> None:
        handlers: dict[type[object], list[Any]] = {}
        for generation in snapshot.generations.values():
            for metadata in plugin_registry.get_handlers_by_module_path(
                generation.module_path
            ):
                if metadata.kind != MetadataKind.LIFECYCLE:
                    continue
                event_type = _EVENT_TYPE_MAP.get(metadata.event_type)  # type: ignore[arg-type]
                if event_type is None:
                    continue
                handlers.setdefault(event_type, []).append(
                    functools.partial(metadata.handler, generation.instance)
                )
            staged = generation.staged_event_bus
            if staged is None:
                continue
            for event_type, handler in staged.staged_handlers():
                handlers.setdefault(event_type, []).append(handler)
        snapshot.event_handlers = MappingProxyType(
            {
                event_type: tuple(event_handlers)
                for event_type, event_handlers in handlers.items()
            }
        )

    async def _post_publish_invariants(
        self,
        generation: PluginGeneration,
        snapshot: RuntimeSnapshot,
    ) -> None:
        await self._post_snapshot_invariants(snapshot)
        if snapshot.generations.get(generation.plugin_id) is not generation:
            raise RuntimeError("RuntimeSnapshot generation 不一致")

    async def _post_snapshot_invariants(
        self,
        snapshot: RuntimeSnapshot,
    ) -> None:
        await asyncio.sleep(0)
        if self.current_snapshot is not snapshot:
            raise RuntimeError("RuntimeSnapshot 发布指针不一致")
        catalog_id = snapshot.skill_catalog_generation_id
        if catalog_id is not None and self._skill_host.get(catalog_id) is None:
            raise RuntimeError("RuntimeSnapshot skill catalog 不可用")
        for generation_id in snapshot.mcp_catalog_generation_ids.values():
            catalog = self._mcp_host.get(generation_id)
            if catalog is None:
                raise RuntimeError("RuntimeSnapshot MCP catalog 不可用")
            if any(not server.client.connected for server in catalog.servers.values()):
                raise RuntimeError("RuntimeSnapshot MCP client 已断开")
        for item in snapshot.generations.values():
            if item.scope.closed:
                raise RuntimeError("RuntimeSnapshot 插件作用域已关闭")
            if item.scope.resource_count < item.minimum_resource_count:
                raise RuntimeError("RuntimeSnapshot 插件资源数量不足")
            if item.job_catalog is not None and self._job_host.get(
                item.generation_id
            ) is not item.job_catalog:
                raise RuntimeError("RuntimeSnapshot Job catalog 不可用")
            if item.proactive_catalog is not None and self._proactive_host.get(
                item.generation_id
            ) is not item.proactive_catalog:
                raise RuntimeError("RuntimeSnapshot proactive catalog 不可用")

    def _publication_status(
        self,
        plugin_id: str,
        *,
        active: PluginGeneration | None,
        candidate: PluginGeneration,
        publication_state: str,
    ) -> dict[str, object]:
        return {
            "plugin_id": plugin_id,
            "old_generation": active.generation_id if active is not None else None,
            "new_generation": candidate.generation_id,
            "snapshot_id": (
                self.current_snapshot.snapshot_id
                if self.current_snapshot is not None
                else None
            ),
            "publication_state": publication_state,
        }

    async def _prepare_changed(
        self,
        *,
        discovered: dict[str, dict[str, str]],
        plugin_ids: set[str] | None = None,
        force_reprepare: bool = False,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for plugin_id, active in tuple(self._active_generations.items()):
            if plugin_ids is not None and plugin_id not in plugin_ids:
                continue
            mod = discovered.get(plugin_id)
            if mod is None:
                continue
            plugin_dir = Path(mod["plugin_root"])
            try:
                source_revision = _source_revision(plugin_dir)
                config_revision = _file_revision(
                    _resolve_plugin_data_dir(
                        mod["name"],
                        mod,
                        self._installed_cache_root,
                    )
                    / "config.local.toml"
                )
            except Exception:
                source_revision = ""
                config_revision = ""
            current_prepared = self._prepared_generations.get(plugin_id)
            if force_reprepare and current_prepared is not None:
                await self.discard_prepared(plugin_id)
                current_prepared = None
            matches_active = (
                source_revision == active.source_revision
                and config_revision == active.config_revision
            )
            if matches_active:
                if current_prepared is None:
                    continue
                await self.discard_prepared(plugin_id)
                result = {
                    "plugin_id": plugin_id,
                    "active_generation": active.generation_id,
                    "prepared_generation": None,
                    "gate_status": "active",
                    "candidate_revision": source_revision,
                    "skills": (
                        list(active.skill_catalog.names)
                        if active.skill_catalog is not None
                        else []
                    ),
                    "skill_descriptions": _skill_descriptions(active),
                    "drift_skill_descriptions": _drift_skill_descriptions(active),
                    "skill_body_hashes": _skill_body_hashes(active, drift=False),
                    "drift_skill_body_hashes": _skill_body_hashes(
                        active,
                        drift=True,
                    ),
                    "mcp_tools": _mcp_tool_names(active),
                    "readiness_checks": _gate_check_evidence(
                        active,
                        "readiness_semantic_checks",
                    ),
                    "jobs": _job_keys(active),
                    "proactive_sources": _proactive_source_keys(active),
                    "job_specs": _job_spec_evidence(active),
                    "proactive_source_specs": _proactive_source_spec_evidence(active),
                    "snapshot_id": (
                        self.current_snapshot.snapshot_id
                        if self.current_snapshot is not None
                        else None
                    ),
                }
                results.append(result)
                _log_candidate_status(result)
                continue
            if (
                current_prepared is not None
                and source_revision == current_prepared.source_revision
                and config_revision == current_prepared.config_revision
            ):
                continue
            await self.discard_prepared(plugin_id)
            prepared = await self._load_one(mod, activate=False)
            gate = self.latest_gate(plugin_id)
            result: dict[str, object] = {
                "plugin_id": plugin_id,
                "active_generation": active.generation_id,
                "prepared_generation": (
                    prepared.generation_id if prepared is not None else None
                ),
                "gate_status": gate.status if gate is not None else "failed",
                "candidate_revision": (
                    gate.candidate_revision if gate is not None else ""
                ),
                "skills": (
                    list(prepared.skill_catalog.names)
                    if prepared is not None and prepared.skill_catalog is not None
                    else []
                ),
                "skill_descriptions": (
                    _skill_descriptions(prepared) if prepared is not None else {}
                ),
                "drift_skill_descriptions": (
                    _drift_skill_descriptions(prepared)
                    if prepared is not None
                    else {}
                ),
                "skill_body_hashes": (
                    _skill_body_hashes(prepared, drift=False)
                    if prepared is not None
                    else {}
                ),
                "drift_skill_body_hashes": (
                    _skill_body_hashes(prepared, drift=True)
                    if prepared is not None
                    else {}
                ),
                "mcp_tools": _mcp_tool_names(prepared) if prepared is not None else [],
                "readiness_checks": (
                    _gate_check_evidence(prepared, "readiness_semantic_checks")
                    if prepared is not None
                    else []
                ),
                "jobs": _job_keys(prepared) if prepared is not None else [],
                "proactive_sources": (
                    _proactive_source_keys(prepared)
                    if prepared is not None
                    else []
                ),
                "job_specs": (
                    _job_spec_evidence(prepared) if prepared is not None else {}
                ),
                "proactive_source_specs": (
                    _proactive_source_spec_evidence(prepared)
                    if prepared is not None
                    else {}
                ),
                "snapshot_id": (
                    self.current_snapshot.snapshot_id
                    if self.current_snapshot is not None
                    else None
                ),
            }
            results.append(result)
            _log_candidate_status(result)
        return results

    async def _load_one(
        self,
        mod: dict[str, str],
        *,
        activate: bool = True,
    ) -> PluginGeneration | None:
        stable_module_path = mod["import_path"]
        plugin_dir = Path(mod["plugin_root"])
        initial_plugin_id = _resolve_plugin_id(mod)
        if activate and initial_plugin_id in self._active_generations:
            return self._active_generations[initial_plugin_id]
        plugin_manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
        if not mod.get("package_id") and plugin_manifest.get(initial_plugin_id, True) is False:
            logger.info("插件已禁用（manifest.toml）: %s", initial_plugin_id)
            return None
        tool_names: list[str] = []
        hook_count_before = len(self._tool_hooks)
        before_turn_count_before = len(self._before_turn_modules)
        before_reasoning_count_before = len(self._before_reasoning_modules)
        prompt_render_count_before = len(self._prompt_render_modules)
        before_step_count_before = len(self._before_step_modules)
        after_step_count_before = len(self._after_step_modules)
        after_reasoning_count_before = len(self._after_reasoning_modules)
        after_turn_count_before = len(self._after_turn_modules)
        proactive_module_count_before = len(self._proactive_modules)
        proactive_lifecycle_count_before = len(self._proactive_lifecycles)
        proactive_factory_count_before = len(self._proactive_module_factories)
        proactive_runtime_factory_count_before = len(self._proactive_runtime_factories)
        proactive_source_count_before = len(self._proactive_sources)
        job_count_before = len(self._jobs)
        channel_count_before = len(self._channels)
        module_path = mod["module_path"].strip()
        if not module_path:
            raise RuntimeError(f"插件缺少 plugin.py: {plugin_dir}")
        try:
            source_revision = _source_revision(plugin_dir)
        except Exception as error:
            revision = hashlib.sha256(
                f"{plugin_dir}:{error}".encode()
            ).hexdigest()
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=revision,
                check_id="source_boundary",
                reason=str(error),
            )
            return None
        data_dir = _resolve_plugin_data_dir(
            mod["name"],
            mod,
            self._installed_cache_root,
        )
        config_revision = _file_revision(data_dir / "config.local.toml")
        self._generation_sequence += 1
        generation_id = (
            f"{initial_plugin_id}:{source_revision[:12]}:{self._generation_sequence}"
        )
        mp = (
            f"{stable_module_path}__g{self._generation_sequence}_"
            f"{source_revision[:8]}_{self._manager_namespace}"
        )
        try:
            self._import_plugin(mp, Path(module_path))
        except Exception as error:
            logger.warning("插件 %s 导入失败: %s", mod["name"], error)
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id="import",
                reason=str(error),
            )
            return None
        cls = plugin_registry.get_class(mp)
        if cls is None:
            logger.warning("插件 %s 未注册类", mod["name"])
            self._remove_module_tree(mp)
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id="plugin_class",
                reason="plugin.py 未注册 Plugin 子类",
            )
            return None
        try:
            instance = cls()
            name = str(instance.name or mod["name"]).strip()
            if not name:
                raise RuntimeError("插件缺少 name")
            plugin_id = f"{name}@{mod['marketplace']}" if mod["marketplace"] else name
            if plugin_id != initial_plugin_id:
                raise RuntimeError(
                    f"插件目录身份与声明不一致: directory={initial_plugin_id} declared={plugin_id}"
                )
            plugin_config = _load_plugin_config(
                data_dir,
                getattr(cls, "ConfigModel", None),
            )
        except Exception as error:
            self._remove_module_tree(mp)
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id=("config" if isinstance(error, _PluginConfigError) else "identity"),
                reason=str(error),
            )
            return None
        from agent.plugins.context import PluginContext, PluginKVStore
        scope = PluginScope(plugin_id)
        instance.context = PluginContext(  # type: ignore[attr-defined]
            event_bus=None,  # type: ignore[arg-type]
            tool_registry=None,
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            data_dir=data_dir,
            kv_store=PluginKVStore(data_dir / ".kv.json", writable=False),
            config=plugin_config,
            workspace=self._workspace,
            session_manager=None,
            memory_engine=None,
            llm=None,
            scope=None,
            generation_id=generation_id,
        )
        plugin_registry.register_instance(mp, instance)
        initialization_started = False

        async def rollback_load() -> None:
            terminator = getattr(instance, "terminate", None)
            if initialization_started and callable(terminator):
                try:
                    typed_terminator = cast(
                        Callable[[], Awaitable[None]],
                        terminator,
                    )
                    await typed_terminator()
                except (asyncio.CancelledError, Exception) as terminate_error:
                    self._cleanup_failures.append(
                        CleanupFailure(
                            resource=f"plugin:{plugin_id}:terminate",
                            error=str(terminate_error) or type(terminate_error).__name__,
                        )
                    )
            self._cleanup_failures.extend(await scope.aclose())
            self._remove_module_tree(mp)
            for tool_name in tool_names:
                if self._tool_registry is not None:
                    self._tool_registry.unregister(tool_name)
            del self._tool_hooks[hook_count_before:]
            del self._before_turn_modules[before_turn_count_before:]
            del self._before_reasoning_modules[before_reasoning_count_before:]
            del self._prompt_render_modules[prompt_render_count_before:]
            del self._before_step_modules[before_step_count_before:]
            del self._after_step_modules[after_step_count_before:]
            del self._after_reasoning_modules[after_reasoning_count_before:]
            del self._after_turn_modules[after_turn_count_before:]
            del self._proactive_modules[proactive_module_count_before:]
            del self._proactive_lifecycles[proactive_lifecycle_count_before:]
            del self._proactive_module_factories[proactive_factory_count_before:]
            del self._proactive_runtime_factories[proactive_runtime_factory_count_before:]
            del self._proactive_sources[proactive_source_count_before:]
            del self._jobs[job_count_before:]
            del self._channels[channel_count_before:]

        try:
            load_phase = "declarations"
            contributions = self._collect_candidate_contributions(
                instance=instance,
                plugin_id=plugin_id,
                plugin_dir=plugin_dir,
                data_dir=data_dir,
                module_path=mp,
                source_revision=source_revision,
            )
            gate_result = self._validate_candidate(
                instance=instance,
                plugin_id=plugin_id,
                revision=source_revision,
                contributions=contributions,
            )
            self._gate_results[plugin_id] = gate_result
            if gate_result.status == "failed":
                raise _CandidateRejected(gate_result)
            generation = PluginGeneration(
                plugin_id=plugin_id,
                generation_id=generation_id,
                module_path=mp,
                source_revision=source_revision,
                config_revision=config_revision,
                instance=instance,
                scope=scope,
                contributions=contributions,
                gate_result=gate_result,
                state="prepared" if not activate else "activating",
            )
            catalog_generations = [
                active_generation
                for active_generation in self._active_generations.values()
                if active_generation.plugin_id != plugin_id
            ]
            catalog_generations.append(generation)
            ignored_generations = [*self._active_generations.values(), generation]
            try:
                skill_catalog = self._skill_host.prepare(
                    generation_id,
                    normal_roots=PluginSkillHost.roots_for(
                        catalog_generations,
                        drift=False,
                    ),
                    drift_roots=PluginSkillHost.roots_for(
                        catalog_generations,
                        drift=True,
                    ),
                    ignored_normal_roots=tuple(
                        root
                        for item in ignored_generations
                        for root in item.contributions.skill_roots
                    ),
                    ignored_drift_roots=tuple(
                        root
                        for item in ignored_generations
                        for root in item.contributions.drift_skill_roots
                    ),
                )
            except Exception as error:
                gate_result = _with_gate_check(
                    gate_result,
                    check_id="skill_catalog",
                    passed=False,
                    evidence=str(error),
                )
                self._gate_results[plugin_id] = gate_result
                raise _CandidateRejected(gate_result) from error
            gate_result = _with_gate_check(
                gate_result,
                check_id="skill_catalog",
                passed=True,
                evidence=list(skill_catalog.names),
            )
            self._gate_results[plugin_id] = gate_result
            generation.gate_result = gate_result
            generation.skill_catalog = skill_catalog
            scope.defer(
                "skill_catalog",
                lambda: self._skill_host.close(generation_id),
            )
            try:
                job_catalog = self._job_host.prepare(
                    generation_id,
                    contributions.jobs,
                )
                scope.defer(
                    "job_catalog",
                    lambda: self._job_host.close(generation_id),
                )
                proactive_catalog = self._proactive_host.prepare(
                    generation_id,
                    contributions.proactive_sources,
                )
                scope.defer(
                    "proactive_catalog",
                    lambda: self._proactive_host.close(generation_id),
                )
            except Exception as error:
                gate_result = _with_gate_check(
                    gate_result,
                    check_id="activity_catalogs",
                    passed=False,
                    evidence=str(error),
                )
                self._gate_results[plugin_id] = gate_result
                raise _CandidateRejected(gate_result) from error
            generation.job_catalog = job_catalog
            generation.proactive_catalog = proactive_catalog
            contributions = replace(
                contributions,
                jobs=tuple(job_catalog.jobs.values()),
                proactive_sources=tuple(proactive_catalog.sources.values()),
            )
            generation.contributions = contributions
            gate_result = _with_gate_check(
                gate_result,
                check_id="activity_catalogs",
                passed=True,
                evidence={
                    "jobs": sorted(job_catalog.jobs),
                    "proactive_sources": sorted(proactive_catalog.sources),
                },
            )
            self._gate_results[plugin_id] = gate_result
            generation.gate_result = gate_result
            if (
                not activate
                or contributions.mcp_servers
                or contributions.proactive_sources
            ):
                try:
                    mcp_catalog = await self._mcp_host.prepare(
                        generation_id,
                        server_specs=contributions.mcp_servers,
                        proactive_sources=contributions.proactive_sources,
                        scope=scope,
                    )
                except Exception as error:
                    gate_result = _with_gate_check(
                        gate_result,
                        check_id="mcp_readiness",
                        passed=False,
                        evidence=str(error),
                        gate_id="G1/G2/G3-readiness",
                    )
                    self._gate_results[plugin_id] = gate_result
                    raise _CandidateRejected(gate_result) from error
                generation.mcp_catalog = mcp_catalog
                scope.defer(
                    "mcp_catalog",
                    lambda: self._mcp_host.close(generation_id),
                )
                try:
                    raw_readiness_checks: object = await instance.readiness_semantic_checks(
                        PluginReadinessContext(
                            generation_id=generation_id,
                            mcp_catalog=mcp_catalog,
                            job_catalog=job_catalog,
                            proactive_catalog=proactive_catalog,
                        )
                    )
                    if not isinstance(raw_readiness_checks, list):
                        raise RuntimeError(
                            "readiness_semantic_checks 返回值不是 list"
                        )
                    readiness_checks = cast(list[object], raw_readiness_checks)
                except Exception as error:
                    readiness_passed = False
                    readiness_evidence: object = str(error)
                else:
                    invalid_readiness = [
                        check
                        for check in readiness_checks
                        if not isinstance(check, PluginSemanticCheck) or not check.passed
                    ]
                    readiness_passed = not invalid_readiness
                    normalized_readiness: list[dict[str, object]] = []
                    for check in readiness_checks:
                        if isinstance(check, PluginSemanticCheck):
                            normalized_readiness.append(
                                {
                                    "check_id": check.check_id,
                                    "passed": check.passed,
                                    "evidence": check.evidence,
                                }
                            )
                        else:
                            normalized_readiness.append(
                                {
                                    "check_id": "invalid",
                                    "passed": False,
                                    "evidence": repr(check),
                                }
                            )
                    readiness_evidence = normalized_readiness
                gate_result = _with_gate_check(
                    gate_result,
                    check_id="mcp_readiness",
                    passed=True,
                    evidence=list(mcp_catalog.tool_names),
                    gate_id="G1/G2/G3-readiness",
                )
                gate_result = _with_gate_check(
                    gate_result,
                    check_id="readiness_semantic_checks",
                    passed=readiness_passed,
                    evidence=readiness_evidence,
                )
                self._gate_results[plugin_id] = gate_result
                generation.gate_result = gate_result
                if gate_result.status == "failed":
                    raise _CandidateRejected(gate_result)
                if not activate:
                    generation.runtime_snapshot = self._compile_generation_snapshot(
                        generation
                    )
                    generation.minimum_resource_count = scope.resource_count
                    self._prepared_generations[plugin_id] = generation
                    return generation
            generation.runtime_snapshot = self._compile_generation_snapshot(generation)
            staged_event_bus = ScopedEventBus(self._event_bus, scope, staged=True)
            generation.staged_event_bus = staged_event_bus
            instance.context.event_bus = staged_event_bus
            instance.context.kv_store = PluginKVStore(data_dir / ".kv.json")
            instance.context.session_manager = self._session_manager
            instance.context.memory_engine = self._memory_engine
            instance.context.llm = self._llm
            instance.context.scope = scope
            instance.context.tool_registry = generation.runtime_snapshot.tool_registry
            load_phase = "initialize"
            initialization_started = True
            generation.initialization_started = True
            await instance.initialize()
            load_phase = "publish"
            self._register_tools(instance, mp, tool_names)
            self._bind_tool_hooks(instance, mp)
            self._publish_contributions(contributions)
            self._channels.extend(contributions.channels)
            staged_event_bus.publish()
            generation.minimum_resource_count = scope.resource_count
        except asyncio.CancelledError:
            rollback_task = asyncio.create_task(
                rollback_load(),
                name=f"plugin_rollback:{plugin_id}",
            )
            while not rollback_task.done():
                try:
                    await asyncio.shield(rollback_task)
                except asyncio.CancelledError:
                    continue
            await rollback_task
            raise
        except _CandidateRejected as error:
            logger.warning(
                "插件 %s 候选验证失败: %s",
                mod["name"],
                error.gate.failure_reason,
            )
            await rollback_load()
            return None
        except Exception as error:
            logger.warning("插件 %s 加载失败，回滚: %s", mod["name"], error)
            self._record_failed_gate(
                plugin_id=plugin_id,
                revision=source_revision,
                check_id=load_phase,
                reason=str(error),
            )
            await rollback_load()
            return None
        self._scopes[mp] = scope
        self._loaded.add(mp)
        self._active_plugins[mp] = ActivePluginInfo(
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            manifest=contributions.manifest,
            module_path=mp,
            skill_roots=contributions.skill_roots,
            drift_skill_roots=contributions.drift_skill_roots,
            mcp_servers=contributions.mcp_servers,
        )
        generation.state = "active"
        self._active_generations[plugin_id] = generation
        self._stable_aliases[mp] = stable_module_path
        self._remove_module_tree(stable_module_path)
        self._fresh_importer.register(stable_module_path, plugin_dir)
        plugin_registry.register_instance(stable_module_path, instance)
        sys.modules[stable_module_path] = sys.modules[mp]
        assert generation.runtime_snapshot is not None
        self._compile_snapshot_event_handlers(generation.runtime_snapshot)
        await self._publish_committed_snapshot(generation.runtime_snapshot)
        logger.info("插件已加载: %s", mod["name"])
        return generation

    def _compile_generation_snapshot(
        self,
        generation: PluginGeneration,
    ) -> RuntimeSnapshot:
        generations = dict(self._active_generations)
        generations[generation.plugin_id] = generation
        try:
            snapshot = self._snapshot_compiler.compile(
                generations,
                catalog_generation=generation,
            )
            snapshot.tool_registry = self._compile_snapshot_tools(generations)
            snapshot.tool_hooks = self._compile_snapshot_tool_hooks(generations)
            return snapshot
        except Exception as error:
            gate = _with_gate_check(
                generation.gate_result,
                check_id="runtime_snapshot",
                passed=False,
                evidence=str(error),
            )
            generation.gate_result = gate
            self._gate_results[generation.plugin_id] = gate
            raise _CandidateRejected(gate) from error

    def _compile_snapshot_tools(
        self,
        generations: dict[str, PluginGeneration],
    ) -> Any:
        if self._tool_registry is None:
            return None
        plugin_mcp_sources = {
            ("mcp", server_name)
            for generation in generations.values()
            for server_name in generation.contributions.mcp_servers
        }
        registry = self._tool_registry.fork(
            excluded_source_types={"plugin"},
            excluded_sources=plugin_mcp_sources,
        )
        for generation in sorted(generations.values(), key=lambda item: item.plugin_id):
            plugin_name = getattr(
                generation.instance,
                "name",
                generation.plugin_id,
            )
            for md in plugin_registry.get_handlers_by_module_path(
                generation.module_path
            ):
                if md.kind != MetadataKind.TOOL:
                    continue
                tool = _build_plugin_tool(generation.instance, md)
                if registry.has_tool(tool.name):
                    raise RuntimeError(f"插件工具名称重复: {tool.name}")
                registry.register(
                    tool,
                    risk=md.tool_risk or "read-write",
                    always_on=bool(md.tool_always_on),
                    search_hint=md.tool_search_hint,
                    source_type="plugin",
                    source_name=plugin_name,
                )
            if generation.mcp_catalog is None:
                continue
            for server in generation.mcp_catalog.servers.values():
                for tool in server.tools:
                    if registry.has_tool(tool.name):
                        raise RuntimeError(f"MCP 工具名称重复: {tool.name}")
                    registry.register(
                        tool,
                        risk="external-side-effect",
                        source_type="mcp",
                        source_name=server.name,
                    )
        return registry

    def _compile_snapshot_tool_hooks(
        self,
        generations: dict[str, PluginGeneration],
    ) -> tuple[ToolHook, ...]:
        hooks: list[ToolHook] = []
        for generation in sorted(generations.values(), key=lambda item: item.plugin_id):
            for metadata in plugin_registry.get_handlers_by_module_path(
                generation.module_path
            ):
                if metadata.kind != MetadataKind.TOOL_HOOK:
                    continue
                hooks.append(
                    _PluginToolHook(
                        name=(
                            f"plugin:{getattr(generation.instance, 'name', generation.module_path)}:"
                            f"{metadata.handler_name}"
                        ),
                        handler=functools.partial(metadata.handler, generation.instance),
                        tool_name_filter=metadata.hook_tool_name,
                    )
                )
        return tuple(hooks)

    async def _publish_committed_snapshot(
        self,
        snapshot: RuntimeSnapshot,
    ) -> None:
        if self._snapshot_store.current is None:
            self._snapshot_store.install(snapshot)
            return
        transaction = self._snapshot_store.begin_publish(snapshot)
        await self._snapshot_store.commit(transaction)

    def _collect_candidate_contributions(
        self,
        *,
        instance: Any,
        plugin_id: str,
        plugin_dir: Path,
        data_dir: Path,
        module_path: str,
        source_revision: str,
    ) -> PluginContributions:
        cls = type(instance)
        sources: list[RegisteredProactiveSource] = []
        for source in _load_module_list(instance, "proactive_sources"):
            if not isinstance(source, ProactiveSourceSpec):
                raise RuntimeError(
                    f"插件 {plugin_id}.proactive_sources 返回值不是 ProactiveSourceSpec"
                )
            sources.append(RegisteredProactiveSource(plugin_id=plugin_id, spec=source))
        jobs: list[RegisteredPluginJob] = []
        for spec in _load_module_list(instance, "jobs"):
            if not isinstance(spec, PluginJobSpec):
                raise RuntimeError(
                    f"插件 {plugin_id}.jobs 返回值不是 PluginJobSpec"
                )
            jobs.append(
                RegisteredPluginJob(
                    plugin_id=plugin_id,
                    plugin_context=instance.context,
                    spec=spec,
                )
            )
        return PluginContributions(
            manifest={
                "name": str(instance.name or ""),
                "version": str(instance.version or ""),
                "desc": str(instance.desc or ""),
                "author": str(instance.author or ""),
            },
            skill_roots=_resolve_declared_roots(plugin_dir, cls.skill_roots()),
            drift_skill_roots=_resolve_declared_roots(
                plugin_dir,
                cls.drift_skill_roots(),
            ),
            mcp_servers=_resolve_mcp_servers(
                plugin_dir,
                data_dir,
                cls.mcp_servers(),
            ),
            managed_services=_resolve_managed_services(
                plugin_dir,
                data_dir,
                cls.managed_services(),
                source_revision=source_revision,
            ),
            before_turn_modules=tuple(
                _load_module_list(instance, "before_turn_modules")
            ),
            before_reasoning_modules=tuple(
                _load_module_list(instance, "before_reasoning_modules")
            ),
            prompt_render_modules=tuple(
                _load_module_list(instance, "prompt_render_modules")
            ),
            before_step_modules=tuple(
                _load_module_list(instance, "before_step_modules")
            ),
            after_step_modules=tuple(
                _load_module_list(instance, "after_step_modules")
            ),
            after_reasoning_modules=tuple(
                _load_module_list(instance, "after_reasoning_modules")
            ),
            after_turn_modules=tuple(
                _load_module_list(instance, "after_turn_modules")
            ),
            proactive_modules=tuple(
                _load_module_list(instance, "proactive_modules")
            ),
            proactive_lifecycles=tuple(
                _load_module_list(instance, "proactive_lifecycles")
            ),
            proactive_module_factories=tuple(
                _load_module_list(instance, "proactive_module_factories")
            ),
            proactive_runtime_factories=tuple(
                _load_module_list(instance, "proactive_runtime_factories")
            ),
            proactive_sources=tuple(sources),
            jobs=tuple(jobs),
            channels=cast(
                tuple[Channel, ...],
                tuple(_load_module_list(instance, "channels")),
            ),
            dashboard_module=_resolve_dashboard_module(
                plugin_dir,
                cls.dashboard_module(),
            ),
        )

    def _validate_candidate(
        self,
        *,
        instance: Any,
        plugin_id: str,
        revision: str,
        contributions: PluginContributions,
    ) -> GateResult:
        checks: list[GateCheckResult] = []
        current = self._active_generations.get(plugin_id)
        other_generations = [
            generation
            for generation in self._active_generations.values()
            if generation.plugin_id != plugin_id
        ]

        def check(check_id: str, passed: bool, evidence: object = "") -> None:
            checks.append(
                GateCheckResult(
                    check_id=check_id,
                    status="passed" if passed else "failed",
                    evidence=evidence,
                )
            )

        check(
            "api_version",
            getattr(instance, "api_version", None) == 1,
            getattr(instance, "api_version", None),
        )
        metadata = plugin_registry.get_handlers_by_module_path(type(instance).__module__)
        tool_names = [
            md.tool_name or md.handler_name
            for md in metadata
            if md.kind == MetadataKind.TOOL
        ]
        duplicate_tools = _duplicates(tool_names)
        current_tool_names = (
            {
                metadata.tool_name or metadata.handler_name
                for metadata in plugin_registry.get_handlers_by_module_path(
                    current.module_path
                )
                if metadata.kind == MetadataKind.TOOL
            }
            if current is not None
            else set()
        )
        occupied_tools = (
            sorted(
                name
                for name in tool_names
                if self._tool_registry.has_tool(name) and name not in current_tool_names
            )
            if self._tool_registry is not None
            else []
        )
        check(
            "tool_names",
            not duplicate_tools and not occupied_tools,
            {"duplicates": duplicate_tools, "occupied": occupied_tools},
        )
        source_ids = [source.spec.id for source in contributions.proactive_sources]
        source_errors = [
            source.spec.id
            for source in contributions.proactive_sources
            if not source.spec.id
            or not source.spec.channels
            or not set(source.spec.channels).issubset({"alert", "content", "context"})
            or not source.spec.server
            or not source.spec.fetch_tool
            or source.spec.fetch_page_size < 0
            or source.spec.server not in contributions.mcp_servers
        ]
        check(
            "proactive_sources",
            not _duplicates(source_ids) and not source_errors,
            {"duplicates": _duplicates(source_ids), "invalid": source_errors},
        )
        occupied_servers = {
            server_name
            for generation in other_generations
            for server_name in generation.contributions.mcp_servers
        }
        occupied_servers.update(self._user_mcp_server_names())
        check(
            "mcp_servers",
            not occupied_servers.intersection(contributions.mcp_servers),
            sorted(occupied_servers.intersection(contributions.mcp_servers)),
        )
        job_ids = [job.spec.id for job in contributions.jobs]
        check(
            "job_ids",
            all(job_ids) and not _duplicates(job_ids) if job_ids else True,
            _duplicates(job_ids),
        )
        channel_names = [
            str(getattr(channel, "name", "")).strip()
            for channel in contributions.channels
        ]
        occupied_channels = {
            str(getattr(channel, "name", "")).strip()
            for generation in other_generations
            for channel in generation.contributions.channels
        }
        check(
            "channel_names",
            (
                all(channel_names)
                and not _duplicates(channel_names)
                and not occupied_channels.intersection(channel_names)
            )
            if channel_names
            else True,
            {
                "duplicates": _duplicates(channel_names),
                "occupied": sorted(occupied_channels.intersection(channel_names)),
            },
        )
        phase_groups = (
            ("before_turn_modules", contributions.before_turn_modules),
            ("before_reasoning_modules", contributions.before_reasoning_modules),
            ("prompt_render_modules", contributions.prompt_render_modules),
            ("before_step_modules", contributions.before_step_modules),
            ("after_step_modules", contributions.after_step_modules),
            ("after_reasoning_modules", contributions.after_reasoning_modules),
            ("after_turn_modules", contributions.after_turn_modules),
        )
        try:
            for field_name, candidate_modules in phase_groups:
                active_modules = [
                    module
                    for generation in other_generations
                    for module in getattr(generation.contributions, field_name)
                ]
                _ = RuntimeSnapshotCompiler.order_plugin_modules(
                    tuple([*active_modules, *candidate_modules])
                )
        except RuntimeError as error:
            check("phase_graph", False, str(error))
        else:
            check("phase_graph", True)
        lifecycle_ids = [
            lifecycle.id
            for lifecycle in contributions.proactive_lifecycles
            if isinstance(lifecycle, ProactiveLifecycleSpec)
        ]
        check(
            "proactive_lifecycles",
            len(lifecycle_ids) == len(contributions.proactive_lifecycles)
            and not _duplicates(lifecycle_ids)
            and not {
                lifecycle.id
                for generation in other_generations
                for lifecycle in generation.contributions.proactive_lifecycles
                if isinstance(lifecycle, ProactiveLifecycleSpec)
            }.intersection(lifecycle_ids),
            {
                "duplicates": _duplicates(lifecycle_ids),
                "occupied": sorted(
                    {
                        lifecycle.id
                        for generation in other_generations
                        for lifecycle in generation.contributions.proactive_lifecycles
                        if isinstance(lifecycle, ProactiveLifecycleSpec)
                    }.intersection(lifecycle_ids)
                ),
            },
        )
        lifecycle_structure_errors: list[str] = []
        for lifecycle in contributions.proactive_lifecycles:
            if not isinstance(lifecycle, ProactiveLifecycleSpec):
                continue
            if (
                not lifecycle.id
                or any(not value for value in lifecycle.initial_slots)
                or any(not value for value in lifecycle.terminal_slots)
                or len(set(lifecycle.initial_slots)) != len(lifecycle.initial_slots)
                or len(set(lifecycle.terminal_slots)) != len(lifecycle.terminal_slots)
            ):
                lifecycle_structure_errors.append(f"{lifecycle.id}: slots")
                continue
            try:
                _ = ProactiveLifecycleBuilder().build(
                    ProactiveLifecycleSpec(
                        id=lifecycle.id,
                        modules=lifecycle.modules,
                        initial_slots=lifecycle.initial_slots,
                    )
                )
            except RuntimeError as error:
                lifecycle_structure_errors.append(f"{lifecycle.id}: {error}")
        check(
            "proactive_lifecycle_structure",
            not lifecycle_structure_errors,
            lifecycle_structure_errors,
        )
        try:
            semantic_checks = instance.static_semantic_checks()
        except Exception as error:
            check("semantic_checks", False, str(error))
        else:
            invalid_semantic = [
                semantic
                for semantic in semantic_checks
                if not isinstance(semantic, PluginSemanticCheck) or not semantic.passed
            ]
            check(
                "semantic_checks",
                not invalid_semantic,
                [
                    getattr(semantic, "evidence", repr(semantic))
                    for semantic in invalid_semantic
                ],
            )
        failed = [item for item in checks if item.status == "failed"]
        return GateResult(
            gate_id="G1/G3-static",
            plugin_id=plugin_id,
            candidate_revision=revision,
            status="failed" if failed else "passed",
            checks=tuple(checks),
            failure_reason="; ".join(item.check_id for item in failed),
        )

    def _publish_contributions(self, contributions: PluginContributions) -> None:
        self._before_turn_modules.extend(contributions.before_turn_modules)
        self._before_reasoning_modules.extend(contributions.before_reasoning_modules)
        self._prompt_render_modules.extend(contributions.prompt_render_modules)
        self._before_step_modules.extend(contributions.before_step_modules)
        self._after_step_modules.extend(contributions.after_step_modules)
        self._after_reasoning_modules.extend(contributions.after_reasoning_modules)
        self._after_turn_modules.extend(contributions.after_turn_modules)
        self._proactive_modules.extend(contributions.proactive_modules)
        self._proactive_lifecycles.extend(contributions.proactive_lifecycles)
        self._proactive_module_factories.extend(
            contributions.proactive_module_factories
        )
        self._proactive_runtime_factories.extend(
            contributions.proactive_runtime_factories
        )
        self._proactive_sources.extend(contributions.proactive_sources)
        self._jobs.extend(contributions.jobs)

    def _record_failed_gate(
        self,
        *,
        plugin_id: str,
        revision: str,
        check_id: str,
        reason: str,
    ) -> None:
        self._gate_results[plugin_id] = GateResult(
            gate_id="G1/G3-static",
            plugin_id=plugin_id,
            candidate_revision=revision,
            status="failed",
            checks=(
                GateCheckResult(
                    check_id=check_id,
                    status="failed",
                    evidence=reason,
                ),
            ),
            failure_reason=reason,
        )

    def _import_plugin(self, module_name: str, path: Path) -> None:
        self._fresh_importer.register(module_name, path.parent)
        spec = self._fresh_importer.root_spec(module_name, path)
        if spec is None or spec.loader is None:
            self._fresh_importer.unregister(module_name)
            raise ImportError(f"无法加载插件文件: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except BaseException:
            self._remove_module_tree(module_name)
            raise

    def _remove_module_tree(self, module_name: str) -> None:
        self._fresh_importer.unregister(module_name)
        plugin_registry.remove_module_tree(module_name)
        for imported_name in tuple(sys.modules):
            if imported_name == module_name or imported_name.startswith(f"{module_name}."):
                _ = sys.modules.pop(imported_name, None)

    def _register_tools(
        self,
        instance: Any,
        module_path: str,
        tool_names: list[str],
    ) -> None:
        if self._tool_registry is None:
            return
        for md in plugin_registry.get_handlers_by_module_path(module_path):
            # 1. 只处理 TOOL 类型元数据
            if md.kind != MetadataKind.TOOL:
                continue
            tool = _build_plugin_tool(instance, md)
            tool_name = tool.name
            # 3. 注册到 ToolRegistry，标记来源为 plugin
            plugin_name = getattr(instance, "name", None) or module_path
            if self._tool_registry.has_tool(tool_name):
                raise RuntimeError(f"插件工具名称重复: {tool_name}")
            tool_names.append(tool_name)
            self._tool_registry.register(
                tool,
                risk=md.tool_risk or "read-write",
                always_on=bool(md.tool_always_on),
                search_hint=md.tool_search_hint,
                source_type="plugin",
                source_name=plugin_name,
            )
            logger.info("插件工具已注册: %s (来自 %s)", tool_name, plugin_name)

    def _bind_handlers(
        self,
        instance: Any,
        module_path: str,
        scope: PluginScope,
    ) -> None:
        for md in plugin_registry.get_handlers_by_module_path(module_path):
            # 1. Phase 1 只绑定生命周期 handler，TOOL 类型留给后续 phase
            if md.kind != MetadataKind.LIFECYCLE:
                continue
            # 2. 跳过当前 phase 尚未支持的事件类型
            ctx_type = _EVENT_TYPE_MAP.get(md.event_type)  # type: ignore[arg-type]
            if ctx_type is None:
                continue
            # 3. 绑定 instance 为第一个参数，EventBus 已处理 sync/async，直接注册
            bound = functools.partial(md.handler, instance)
            _ = scope.subscribe(self._event_bus, ctx_type, bound)

    def _bind_tool_hooks(self, instance: Any, module_path: str) -> None:
        for md in plugin_registry.get_handlers_by_module_path(module_path):
            if md.kind != MetadataKind.TOOL_HOOK:
                continue
            bound = functools.partial(md.handler, instance)
            hook = _PluginToolHook(
                name=f"plugin:{getattr(instance, 'name', module_path)}:{md.handler_name}",
                handler=bound,
                tool_name_filter=md.hook_tool_name,
            )
            self._tool_hooks.append(hook)
            logger.info("插件 tool hook 已注册: %s", hook.name)

    async def terminate_all(self) -> None:
        """完成快照、插件生命周期和作用域资源的全量关闭。"""

        # 1. 先完成快照回收，再处理候选代际
        _, externally_cancelled = await _complete_critical(
            self._snapshot_store.close()
        )
        for plugin_id in tuple(self._prepared_generations):
            _, cancelled = await _complete_critical(self.discard_prepared(plugin_id))
            externally_cancelled = externally_cancelled or cancelled

        # 2. 逐插件终止并消费全部 cleanup failures
        for mp in list(self._loaded):
            active_info = self._active_plugins.get(mp)
            instance = plugin_registry.get_instance(mp)
            terminator = getattr(instance, "terminate", None)
            if callable(terminator):
                try:
                    typed_terminator = cast(
                        Callable[[], Awaitable[None]],
                        terminator,
                    )
                    _, cancelled = await _complete_critical(typed_terminator())
                    externally_cancelled = externally_cancelled or cancelled
                except (asyncio.CancelledError, Exception) as error:
                    current = asyncio.current_task()
                    externally_cancelled = externally_cancelled or (
                        current is not None and current.cancelling() > 0
                    )
                    error_text = str(error) or type(error).__name__
                    logger.warning("插件 terminate 失败 (%s): %s", mp, error_text)
                    self._cleanup_failures.append(
                        CleanupFailure(
                            resource=f"plugin:{mp}:terminate",
                            error=error_text,
                        )
                    )
            scope = self._scopes.pop(mp, None)
            if scope is not None:
                cleanup_failures, cancelled = await _complete_critical(scope.aclose())
                self._cleanup_failures.extend(cleanup_failures)
                externally_cancelled = externally_cancelled or cancelled

            # 3. 注销工具、模块和运行时注册
            for md in plugin_registry.get_handlers_by_module_path(mp):
                if md.kind == MetadataKind.TOOL and self._tool_registry is not None:
                    self._tool_registry.unregister(md.tool_name or md.handler_name)
            self._remove_module_tree(mp)
            stable_alias = self._stable_aliases.pop(mp, None)
            if stable_alias is not None:
                active_alias = plugin_registry.get_instance(stable_alias)
                if active_alias is instance:
                    self._remove_module_tree(stable_alias)
                else:
                    self._fresh_importer.unregister(stable_alias)
            if active_info is not None:
                generation = self._active_generations.get(active_info.plugin_id)
                if generation is not None and generation.module_path == mp:
                    _ = self._active_generations.pop(active_info.plugin_id)
                    generation.state = "retired"
            _ = self._active_plugins.pop(mp, None)
        self._loaded.clear()
        self._active_plugins.clear()
        self._tool_hooks.clear()
        self._before_turn_modules.clear()
        self._before_reasoning_modules.clear()
        self._prompt_render_modules.clear()
        self._before_step_modules.clear()
        self._after_step_modules.clear()
        self._after_reasoning_modules.clear()
        self._after_turn_modules.clear()
        self._proactive_modules.clear()
        self._proactive_lifecycles.clear()
        self._proactive_module_factories.clear()
        self._proactive_runtime_factories.clear()
        self._proactive_sources.clear()
        self._jobs.clear()
        self._channels.clear()
        self._scopes.clear()
        self._active_generations.clear()
        self._prepared_generations.clear()
        self._stable_aliases.clear()
        if externally_cancelled:
            raise asyncio.CancelledError


class _PluginConfigError(Exception):
    pass


class _CandidateRejected(Exception):
    def __init__(self, gate: GateResult) -> None:
        super().__init__(gate.failure_reason)
        self.gate = gate


def _with_gate_check(
    gate: GateResult,
    *,
    check_id: str,
    passed: bool,
    evidence: object,
    gate_id: str | None = None,
) -> GateResult:
    check = GateCheckResult(
        check_id=check_id,
        status="passed" if passed else "failed",
        evidence=evidence,
    )
    checks = (*gate.checks, check)
    failed = [item.check_id for item in checks if item.status == "failed"]
    return GateResult(
        gate_id=gate_id or gate.gate_id,
        plugin_id=gate.plugin_id,
        candidate_revision=gate.candidate_revision,
        status="failed" if failed else "passed",
        checks=checks,
        failure_reason="; ".join(failed),
    )


def _load_plugin_config(
    data_dir: Path,
    config_model: type[BaseModel] | None = None,
) -> Any:
    config_path = data_dir / "config.local.toml"
    raw_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise _PluginConfigError(str(e)) from e
    if config_model is not None:
        if not isinstance(config_model, type) or not issubclass(config_model, BaseModel):
            raise _PluginConfigError("ConfigModel 必须继承 pydantic.BaseModel")
        try:
            return config_model.model_validate(raw_config)
        except ValidationError as e:
            raise _PluginConfigError(_format_validation_error(e)) from e
    from agent.plugins.config import PluginConfig
    return PluginConfig(raw_config) if raw_config else None


def _format_validation_error(error: ValidationError) -> str:
    parts: list[str] = []
    for item in error.errors():
        path = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        parts.append(f"{path}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def _load_module_list(instance: Any, method_name: str) -> list[object]:
    provider = getattr(instance, method_name, None)
    if provider is None:
        return []
    if not callable(provider):
        raise RuntimeError(
            f"插件 {type(instance).__name__}.{method_name} 不是可调用对象"
        )
    try:
        loaded = provider()
    except Exception as e:
        raise RuntimeError(
            f"插件 {type(instance).__name__}.{method_name} 声明失败: {e}"
        ) from e
    if loaded is None:
        raise RuntimeError(
            f"插件 {type(instance).__name__}.{method_name} 返回值不能为 None"
        )
    if not isinstance(loaded, list):
        raise RuntimeError(
            f"插件 {type(instance).__name__}.{method_name} 返回值不是 list"
        )
    return loaded


def _resolve_plugin_id(mod: dict[str, str]) -> str:
    name = mod["name"]
    marketplace = mod.get("marketplace", "").strip()
    if not marketplace:
        return name
    return f"{name}@{marketplace}"


def _resolve_plugin_data_dir(
    name: str,
    mod: dict[str, str],
    installed_cache_root: Path | None,
) -> Path:
    marketplace = mod.get("marketplace", "").strip()
    suffix = marketplace or "builtin"
    return _plugins_home(installed_cache_root) / "data" / f"{name}-{suffix}"


def _plugins_home(installed_cache_root: Path | None) -> Path:
    if installed_cache_root is not None:
        return installed_cache_root.parent
    return Path.home() / ".akashic-plugin"


def _resolve_declared_roots(
    plugin_dir: Path,
    declared: tuple[str, ...],
) -> tuple[Path, ...]:
    plugin_root = plugin_dir.resolve(strict=False)
    roots: list[Path] = []
    for raw_path in declared:
        path = (plugin_dir / raw_path).resolve(strict=False)
        _require_plugin_path(plugin_root, path, "能力目录")
        if not path.is_dir():
            raise RuntimeError(f"插件能力目录不存在: {path}")
        roots.append(path)
    return tuple(roots)


def _resolve_dashboard_module(plugin_dir: Path, declared: str | None) -> Path | None:
    if declared is None:
        return None
    path = (plugin_dir / declared).resolve(strict=False)
    root = plugin_dir.resolve(strict=False)
    if not path.is_relative_to(root) or path.suffix != ".py" or not path.is_file():
        raise RuntimeError(f"插件 dashboard module 无效: {declared}")
    return path


def _resolve_managed_services(
    plugin_dir: Path,
    data_dir: Path,
    declared: list[ManagedServiceSpec],
    *,
    source_revision: str,
) -> dict[str, dict[str, Any]]:
    services: dict[str, dict[str, Any]] = {}
    plugin_root = plugin_dir.resolve(strict=False)
    for spec in declared:
        if (
            not isinstance(spec, ManagedServiceSpec)
            or not spec.id
            or not spec.command
            or spec.startup_timeout_seconds <= 0
            or not all(isinstance(item, str) and item for item in spec.command)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in spec.env.items()
            )
            or not isinstance(spec.readiness_url, str)
        ):
            raise RuntimeError(f"插件 managed service 声明无效: {spec!r}")
        if spec.id in services:
            raise RuntimeError(f"插件 managed service 名称重复: {spec.id}")
        command = [
            _resolve_command_item(plugin_root, item, executable=index == 0)
            for index, item in enumerate(spec.command)
        ]
        cwd_path = Path(spec.cwd)
        resolved_cwd = (
            cwd_path.resolve(strict=False)
            if cwd_path.is_absolute()
            else (plugin_root / cwd_path).resolve(strict=False)
        )
        _require_plugin_path(plugin_root, resolved_cwd, "managed service cwd")
        cwd = str(resolved_cwd)
        if _is_python_command(command[0]):
            runtime_root = _resolve_mcp_runtime_root(plugin_dir, cwd, command)
            if runtime_root is not None:
                venv_python = _venv_python(runtime_root / ".venv")
                if venv_python.exists():
                    command[0] = str(venv_python)
        services[spec.id] = {
            "command": command,
            "cwd": cwd,
            "env": {**spec.env, "AKA_PLUGIN_DATA_DIR": str(data_dir)},
            "readiness_url": spec.readiness_url,
            "startup_timeout_seconds": spec.startup_timeout_seconds,
            "revision": source_revision,
        }
    return services


def _resolve_mcp_servers(
    plugin_dir: Path,
    data_dir: Path,
    declared: list[McpServerSpec],
) -> dict[str, dict[str, Any]]:
    servers: dict[str, dict[str, Any]] = {}
    plugin_root = plugin_dir.resolve(strict=False)
    for spec in declared:
        if not isinstance(spec, McpServerSpec) or not spec.name or not spec.command:
            raise RuntimeError(f"插件 MCP server 声明无效: {spec!r}")
        if not all(isinstance(item, str) and item for item in spec.command):
            raise RuntimeError(f"插件 MCP command 声明无效: {spec.name}")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in spec.env.items()
        ):
            raise RuntimeError(f"插件 MCP env 声明无效: {spec.name}")
        if spec.name in servers:
            raise RuntimeError(f"插件 MCP server 名称重复: {spec.name}")
        command = [
            _resolve_command_item(plugin_root, item, executable=index == 0)
            for index, item in enumerate(spec.command)
        ]
        cwd_path = Path(spec.cwd)
        resolved_cwd = (
            cwd_path.resolve(strict=False)
            if cwd_path.is_absolute()
            else (plugin_root / cwd_path).resolve(strict=False)
        )
        _require_plugin_path(plugin_root, resolved_cwd, "MCP cwd")
        cwd = str(resolved_cwd)
        env = {**spec.env, "AKA_PLUGIN_DATA_DIR": str(data_dir)}
        if _is_python_command(command[0]):
            runtime_root = _resolve_mcp_runtime_root(plugin_dir, cwd, command)
            if runtime_root is not None:
                venv_python = _venv_python(runtime_root / ".venv")
                if venv_python.exists():
                    command[0] = str(venv_python)
        servers[spec.name] = {"command": command, "env": env, "cwd": cwd}
    return servers


def _resolve_command_item(
    plugin_dir: Path,
    item: str,
    *,
    executable: bool,
) -> str:
    path = Path(item)
    if executable and path.is_absolute():
        return item
    if "/" not in item and "\\" not in item and not item.startswith("."):
        return item
    resolved = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (plugin_dir / path).resolve(strict=False)
    )
    _require_plugin_path(plugin_dir, resolved, "MCP command")
    return str(resolved)


def _require_plugin_path(plugin_dir: Path, path: Path, label: str) -> None:
    try:
        _ = path.relative_to(plugin_dir)
    except ValueError as error:
        raise RuntimeError(f"插件 {label} 越界: {path}") from error


def _is_python_command(value: str) -> bool:
    return Path(value).name.lower() in {"python", "python3", "python.exe"}


def _resolve_mcp_runtime_root(
    plugin_dir: Path,
    cwd: str,
    command: list[str],
) -> Path | None:
    candidates: list[Path] = []
    if len(command) >= 2:
        script_path = Path(command[1])
        if script_path.is_absolute():
            candidates.append(script_path.parent)
    candidates.extend([Path(cwd), plugin_dir])
    for candidate in candidates:
        if (candidate / "requirements.txt").exists():
            return candidate
    return None


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _collect_skill_names(skill_roots: tuple[Path, ...]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for root in skill_roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not (child / "SKILL.md").exists():
                continue
            if child.name in seen:
                continue
            seen.add(child.name)
            names.append(child.name)
    return names


def _build_plugin_tool(instance: Any, metadata: Any) -> Any:
    from agent.tools.base import Tool as AgentTool

    bound = functools.partial(metadata.handler, instance, None)
    tool_name = metadata.tool_name or metadata.handler_name
    tool_class = type(
        f"PluginTool_{tool_name}",
        (AgentTool,),
        {
            "name": tool_name,
            "description": (metadata.handler.__doc__ or "").strip(),
            "parameters": metadata.tool_schema
            or {"type": "object", "properties": {}, "required": []},
            "execute": _make_execute(bound),
        },
    )
    return tool_class()


def _make_execute(bound: Any) -> Any:
    # 预先提取插件函数接受的参数名（排除 self/event），用于过滤 Registry 注入的 context 字段
    sig = inspect.signature(bound)
    accepted = frozenset(
        name for name in sig.parameters if name not in ("self", "event")
    )

    # 工厂函数把 bound 和 accepted 锁进闭包，避免动态 type() 时 self 顶掉 bound
    async def execute(self: Any, **kwargs: Any) -> str:
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        result = bound(**filtered)
        if inspect.isawaitable(result):
            result = await result
        return str(result)
    return execute


class _PluginToolHook(ToolHook):
    """将插件的 @on_tool_pre handler 适配为 ToolExecutor 的 ToolHook 接口。"""

    event = "pre_tool_use"
    snapshot_managed = True

    def __init__(
        self,
        name: str,
        handler: Any,
        tool_name_filter: str | None = None,
    ) -> None:
        self.name = name
        self._handler = handler
        self._tool_name_filter = tool_name_filter

    def matches(self, ctx: HookContext) -> bool:
        if self._tool_name_filter is None:
            return True
        return ctx.request.tool_name == self._tool_name_filter

    async def run(self, ctx: HookContext) -> HookOutcome:
        # 1. 构造 PreToolCtx（复制 arguments，避免插件直接改原对象）
        event = PreToolCtx(
            session_key=ctx.request.session_key,
            channel=ctx.request.channel,
            chat_id=ctx.request.chat_id,
            tool_name=ctx.request.tool_name,
            arguments=dict(ctx.current_arguments),
            call_id=ctx.request.call_id,
            source=ctx.request.source,
            request_text=ctx.request.request_text,
            tool_batch=ctx.request.tool_batch,
            tool_batch_index=ctx.request.tool_batch_index,
        )
        # 2. 调插件 handler，返回值决定行为
        result = self._handler(event)
        if inspect.isawaitable(result):
            result = await result
        # 3. None → 不改参；dict → 新 arguments；HookOutcome → 允许插件直接 deny
        if result is None:
            return HookOutcome()
        if isinstance(result, HookOutcome):
            return result
        if isinstance(result, dict):
            return HookOutcome(updated_input=cast("dict[str, Any]", result))
        return HookOutcome()


def _file_revision(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve(strict=False)).encode())
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        digest.update(b"<missing>")
    return digest.hexdigest()


def _source_revision(plugin_dir: Path) -> str:
    digest = hashlib.sha256()
    root = plugin_dir.resolve(strict=False)
    excluded = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    for current, directories, filenames in os.walk(plugin_dir, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in excluded
        )
        current_path = Path(current)
        for name in [*directories, *sorted(filenames)]:
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


def _source_metadata_revision(plugin_dir: Path) -> bytes:
    digest = hashlib.sha256()
    excluded = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    for current, directories, filenames in os.walk(plugin_dir, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in excluded
        )
        current_path = Path(current)
        for name in [*directories, *sorted(filenames)]:
            path = current_path / name
            relative = path.relative_to(plugin_dir)
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            digest.update(str(relative).encode())
            digest.update(str(stat.st_mtime_ns).encode())
            digest.update(str(stat.st_size).encode())
            if path.is_symlink():
                digest.update(os.readlink(path).encode())
    return digest.digest()


def _path_metadata(path: Path) -> bytes:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return f"{path}:missing".encode()
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _skill_descriptions(generation: PluginGeneration) -> dict[str, str]:
    catalog = generation.skill_catalog
    if catalog is None:
        return {}
    return {
        name: record.description
        for name, record in sorted(catalog.normal.records.items())
    }


def _drift_skill_descriptions(generation: PluginGeneration) -> dict[str, str]:
    catalog = generation.skill_catalog
    if catalog is None:
        return {}
    return {
        name: record.description
        for name, record in sorted(catalog.drift.records.items())
    }


def _skill_body_hashes(
    generation: PluginGeneration,
    *,
    drift: bool,
) -> dict[str, str]:
    catalog = generation.skill_catalog
    if catalog is None:
        return {}
    records = catalog.drift.records if drift else catalog.normal.records
    return {
        name: hashlib.sha256(record.content.encode()).hexdigest()
        for name, record in sorted(records.items())
    }


def _mcp_tool_names(generation: PluginGeneration) -> list[str]:
    catalog = generation.mcp_catalog
    return list(catalog.tool_names) if catalog is not None else []


def _log_candidate_status(result: dict[str, object]) -> None:
    logger.info(
        "plugin_candidate_status plugin=%s gate=%s active=%s prepared=%s "
        "revision=%s counts=skills:%d,drift_skills:%d,mcp:%d,jobs:%d,sources:%d",
        result["plugin_id"],
        result["gate_status"],
        result["active_generation"],
        result["prepared_generation"] or "-",
        str(result["candidate_revision"])[:12],
        len(cast(list[object], result["skills"])),
        len(cast(dict[object, object], result["drift_skill_descriptions"])),
        len(cast(list[object], result["mcp_tools"])),
        len(cast(list[object], result["jobs"])),
        len(cast(list[object], result["proactive_sources"])),
    )
    logger.debug(
        "plugin_candidate_status_detail %s",
        json.dumps(result, ensure_ascii=False, sort_keys=True),
    )


def _job_keys(generation: PluginGeneration) -> list[str]:
    catalog = generation.job_catalog
    return sorted(catalog.jobs) if catalog is not None else []


def _proactive_source_keys(generation: PluginGeneration) -> list[str]:
    catalog = generation.proactive_catalog
    return sorted(catalog.sources) if catalog is not None else []


def _job_spec_evidence(generation: PluginGeneration) -> dict[str, object]:
    catalog = generation.job_catalog
    if catalog is None:
        return {}
    return {
        key: [
            (
                {"type": "interval", "seconds": trigger.seconds}
                if isinstance(trigger, IntervalTrigger)
                else {
                    "type": "event",
                    "event": trigger.event_type.__name__,
                }
            )
            for trigger in job.spec.triggers
        ]
        for key, job in sorted(catalog.jobs.items())
    }


def _proactive_source_spec_evidence(
    generation: PluginGeneration,
) -> dict[str, object]:
    catalog = generation.proactive_catalog
    if catalog is None:
        return {}
    return {
        key: {
            "server": source.spec.server,
            "fetch_tool": source.spec.fetch_tool,
            "ack_tool": source.spec.ack_tool,
            "fetch_page_size": source.spec.fetch_page_size,
        }
        for key, source in sorted(catalog.sources.items())
    }


def _gate_check_evidence(
    generation: PluginGeneration,
    check_id: str,
) -> object:
    for check in reversed(generation.gate_result.checks):
        if check.check_id == check_id:
            return check.evidence
    return []
