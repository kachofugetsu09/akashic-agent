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
import shutil
import socket
import sys
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ValidationError

from agent.plugins.manifest import (
    ensure_workspace_plugin_data_dir,
    load_package_manifest,
    load_plugin_manifest,
    plugins_root,
    workspace_plugin_data_dir,
    write_package_manifest,
    write_plugin_manifest,
)
from agent.plugins.packages import (
    _select_enabled_plugin_packages,  # pyright: ignore[reportPrivateUsage]
    discover_plugin_packages,
)
from agent.plugins.specs import (
    ManagedServiceSpec,
    McpServerSpec,
    MobileUiContribution,
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
from agent.plugins.artifacts import (
    ArtifactPointer,
    ArtifactSelector,
    discard_latest_pointer,
    pointer_state_path,
    read_pointer,
    read_pointers,
    relative_artifact_pointer,
    resolve_pointer,
    write_pointers,
)
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
    MobileUiAsset,
    PluginContributions,
    PluginGeneration,
    PluginReadinessContext,
    PluginSemanticCheck,
)
from agent.plugins.importer import FreshPluginImporter
from agent.plugins.install import PluginInstallResult, install_git_plugin
from agent.plugins.reload_journal import (
    ReloadJournal,
    ReloadPhase,
    ReloadRecoveryAction,
)
from agent.plugins.skill_host import PluginSkillHost, PreparedSkillCatalog
from agent.mcp.generation import WorkspaceMcpGeneration
from agent.mcp.host import McpGenerationHost, PreparedMcpCatalog
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
    SnapshotTransaction,
    get_current_runtime_snapshot,
    plugin_is_active,
)
from proactive_v2.lifecycle import ProactiveLifecycleSpec
from proactive_v2.lifecycle import ProactiveLifecycleBuilder
from agent.tool_hooks.base import ToolHook
from agent.tool_hooks.types import HookContext, HookOutcome
from bus.event_bus import EventBus
from infra.channels.contract import Channel
from infra.persistence.json_store import atomic_save_json

logger = logging.getLogger(__name__)
U = TypeVar("U")


def _generation_can_write(generation: PluginGeneration) -> bool:
    if generation.state in {"activating", "active"}:
        return True
    snapshot = get_current_runtime_snapshot()
    return (
        snapshot is not None
        and snapshot.generations.get(generation.plugin_id) is generation
    )


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


@dataclass(frozen=True)
class _ReadyPluginCandidate:
    plugin_id: str
    previous: PluginGeneration | None
    candidate: PluginGeneration
    snapshot: RuntimeSnapshot


class PluginManager:
    POST_PUBLISH_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: EventBus,
        workspace: Path,
        tool_registry: Any = None,
        session_manager: Any = None,
        memory_engine: Any = None,
        llm: PluginLlmService | None = None,
        installed_cache_root: Path | None = None,
    ) -> None:
        self._dirs = plugin_dirs
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._workspace = workspace
        self._session_manager = session_manager
        self._memory_engine = memory_engine
        self._llm = llm
        self._installed_cache_root = installed_cache_root
        self._channel_switcher: (
            Callable[
                [str, tuple[Channel, ...], tuple[Channel, ...]],
                Awaitable[None],
            ]
            | None
        ) = None
        self._dashboard_preparer: Callable[[RuntimeSnapshot], None] | None = None
        self._service_switcher: (
            Callable[
                [str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]],
                Awaitable[None],
            ]
            | None
        ) = None
        self._candidate_service_starter: (
            Callable[[str, dict[str, dict[str, Any]]], Awaitable[None]] | None
        ) = None
        self._candidate_service_stopper: Callable[[str], Awaitable[None]] | None = None
        self._candidate_service_health_check: (
            Callable[[str], Awaitable[None]] | None
        ) = None
        self._endpoint_quiescer: Callable[[], Awaitable[None]] | None = None
        self._endpoint_resumer: Callable[[], Awaitable[None]] | None = None
        self._endpoint_switcher: (
            Callable[
                [
                    str,
                    dict[str, dict[str, Any]],
                    dict[str, dict[str, Any]],
                    tuple[Channel, ...],
                    tuple[Channel, ...],
                ],
                Awaitable[None],
            ]
            | None
        ) = None
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
        self._draining_generations: dict[str, list[PluginGeneration]] = {}
        self._prepared_generations: dict[str, PluginGeneration] = {}
        self._ready_candidate: _ReadyPluginCandidate | None = None
        self._gate_results: dict[str, GateResult] = {}
        self._stable_aliases: dict[str, str] = {}
        self._generation_sequence = 0
        self._candidate_prepare_lock = asyncio.Lock()
        self._fresh_importer = FreshPluginImporter()
        self._manager_namespace = secrets.token_hex(4)
        self._skill_host = PluginSkillHost(workspace)
        self._mcp_host = McpGenerationHost()
        self._active_workspace_mcp: WorkspaceMcpGeneration | None = None
        self._prepared_workspace_mcp: WorkspaceMcpGeneration | None = None
        self._job_host = PluginJobHost()
        self._proactive_host = PluginProactiveHost()
        self._snapshot_compiler = RuntimeSnapshotCompiler()
        self._snapshot_store = RuntimeSnapshotStore(self._on_snapshot_drained)
        self._snapshot_skill_catalogs: dict[str, str] = {}
        self._reload_journal = ReloadJournal(workspace)
        self._drain_transactions: dict[str, str] = {}
        self._drained_before_commit: set[str] = set()
        self._event_bus.bind_runtime_snapshot_store(self._snapshot_store)

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    @property
    def tool_hooks(self) -> list[ToolHook]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.tool_hooks)
        return list(self._tool_hooks)

    @property
    def channels(self) -> list[Channel]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.channels.values())
        return list(self._channels)

    @property
    def before_turn_modules(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.before_turn_modules)
        return list(self._before_turn_modules)

    @property
    def before_reasoning_modules(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.before_reasoning_modules)
        return list(self._before_reasoning_modules)

    @property
    def prompt_render_modules(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.prompt_render_modules)
        return list(self._prompt_render_modules)

    @property
    def before_step_modules(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.before_step_modules)
        return list(self._before_step_modules)

    @property
    def after_step_modules(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.after_step_modules)
        return list(self._after_step_modules)

    @property
    def after_reasoning_modules(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.after_reasoning_modules)
        return list(self._after_reasoning_modules)

    @property
    def after_turn_modules(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.after_turn_modules)
        return list(self._after_turn_modules)

    @property
    def proactive_modules(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.proactive_modules)
        return list(self._proactive_modules)

    @property
    def proactive_lifecycles(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.proactive_lifecycles)
        return list(self._proactive_lifecycles)

    @property
    def proactive_module_factories(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.proactive_module_factories)
        return list(self._proactive_module_factories)

    @property
    def proactive_runtime_factories(self) -> list[object]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.proactive_runtime_factories)
        return list(self._proactive_runtime_factories)

    @property
    def proactive_sources(self) -> list[RegisteredProactiveSource]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.proactive_sources.values())
        return list(self._proactive_sources)

    @property
    def jobs(self) -> list[RegisteredPluginJob]:
        if self.current_snapshot is not None:
            return list(self.current_snapshot.jobs.values())
        return list(self._jobs)

    @property
    def llm(self) -> PluginLlmService | None:
        return self._llm

    @property
    def plugin_dirs(self) -> list[Path]:
        return list(self._dirs)

    @property
    def skill_projection_roots(self) -> list[Path]:
        roots = self.plugin_dirs
        if self._installed_cache_root is not None:
            roots.append(self._installed_cache_root)
        return roots

    def sync_skill_links(self):
        """Rebuild workspace links from the active stable plugin generations."""

        from agent.plugins.skill_links import PluginSkillLinker

        return PluginSkillLinker(
            workspace=self._workspace,
            plugin_roots=self.skill_projection_roots,
            memory_engine=self._memory_engine,
        ).sync(self.active_plugins())

    def _prepare_skill_links_for_promotion(
        self,
        generation: PluginGeneration,
    ) -> tuple[Any, list[ActivePluginInfo], list[ActivePluginInfo]]:
        """Build and validate both sides of the stable skill projection switch."""

        from agent.plugins.skill_links import PluginSkillLinker

        contributions = generation.production_contributions or generation.contributions
        plugin_dir = Path(cast(Any, generation.instance).context.plugin_dir).resolve(
            strict=False
        )
        target = ActivePluginInfo(
            plugin_id=generation.plugin_id,
            plugin_dir=plugin_dir,
            manifest=contributions.manifest,
            module_path=generation.module_path,
            skill_roots=contributions.skill_roots,
            drift_skill_roots=contributions.drift_skill_roots,
            mcp_servers=contributions.mcp_servers,
        )
        stable = self.active_plugins()
        post_promotion = [
            plugin
            for plugin in stable
            if plugin.plugin_id != generation.plugin_id
        ]
        post_promotion.append(target)
        linker = PluginSkillLinker(
            workspace=self._workspace,
            plugin_roots=self.skill_projection_roots,
            memory_engine=self._memory_engine,
        )
        linker.validate(post_promotion)
        return linker, stable, post_promotion

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

    @property
    def active_workspace_mcp(self) -> WorkspaceMcpGeneration | None:
        return self._active_workspace_mcp

    @property
    def prepared_workspace_mcp(self) -> WorkspaceMcpGeneration | None:
        return self._prepared_workspace_mcp

    def assert_no_workspace_mcp_plugin_conflicts(self) -> None:
        """拒绝启动扫描中发现的 workspace/plugin MCP 名称冲突。"""

        workspace = self._active_workspace_mcp
        if workspace is None:
            return
        names = set(workspace.catalog.servers)
        conflicts: list[str] = []
        for plugin_id, gate in self._gate_results.items():
            for check in gate.checks:
                if check.check_id != "mcp_servers" or check.status != "failed":
                    continue
                evidence = check.evidence
                if isinstance(evidence, list) and names.intersection(
                    item for item in evidence if isinstance(item, str)
                ):
                    conflicts.append(plugin_id)
        if conflicts:
            raise RuntimeError(
                "workspace MCP 与插件声明冲突: " + ", ".join(sorted(conflicts))
            )

    async def prepare_workspace_mcp(
        self,
        server_specs: dict[str, dict[str, Any]],
        *,
        revision: str,
    ) -> WorkspaceMcpGeneration:
        """准备 workspace MCP 候选，不改变当前运行快照。"""

        async with self._candidate_prepare_lock:
            await self._discard_workspace_mcp_candidate()
            self._check_workspace_mcp_name_conflicts(server_specs)

            # 1. 完整连接候选 catalog，任何失败都回收候选作用域
            self._generation_sequence += 1
            generation_id = (
                f"workspace-mcp:{self._generation_sequence}:{secrets.token_hex(4)}"
            )
            scope = PluginScope("workspace-mcp")
            try:
                catalog = await self._mcp_host.prepare(
                    generation_id,
                    server_specs=server_specs,
                    required_tools={},
                    scope=scope,
                )
                scope.defer(
                    "mcp_catalog",
                    lambda: self._mcp_host.close(generation_id),
                )
            except BaseException:
                cleanup_failures, _ = await _complete_critical(scope.aclose())
                self._cleanup_failures.extend(cleanup_failures)
                raise

            # 2. 基于锁内最新插件 generations 编译候选 snapshot
            generation = WorkspaceMcpGeneration(
                generation_id=generation_id,
                revision=revision,
                scope=scope,
                catalog=catalog,
            )
            try:
                generation.runtime_snapshot = self._compile_workspace_mcp_snapshot(
                    generation
                )
            except BaseException:
                await self._dispose_workspace_mcp(generation, state="rejected")
                raise
            self._prepared_workspace_mcp = generation
            return generation

    async def publish_workspace_mcp(self) -> WorkspaceMcpGeneration:
        """原子发布 workspace MCP 候选，并让旧代际随 lease 排空。"""

        async with self._candidate_prepare_lock:
            generation = self._prepared_workspace_mcp
            if generation is None or generation.runtime_snapshot is None:
                raise RuntimeError("workspace MCP 没有可发布候选")
            try:
                self._check_workspace_mcp_name_conflicts(generation.catalog.servers)
                snapshot = self._compile_workspace_mcp_snapshot(generation)
                generation.runtime_snapshot = snapshot
                self._validate_workspace_mcp_generation(generation)
            except BaseException:
                await self._discard_workspace_mcp_candidate()
                raise

            # 1. 首个快照直接安装；已有快照使用可回滚发布事务
            previous = self._active_workspace_mcp
            if self._snapshot_store.current is None:
                self._snapshot_store.install(snapshot)
            else:
                transaction = self._snapshot_store.begin_publish(snapshot)
                try:
                    await self._post_snapshot_invariants(snapshot)
                except BaseException:
                    self._prepared_workspace_mcp = None
                    await _complete_critical(self._snapshot_store.abort(transaction))
                    raise
                self._active_workspace_mcp = generation
                self._prepared_workspace_mcp = None
                generation.state = "active"
                _, commit_cancelled = await _complete_critical(
                    self._snapshot_store.commit(transaction)
                )
                self._mcp_host.mark_active(generation.generation_id)
                if previous is not None:
                    self._mcp_host.mark_draining(previous.generation_id)
                if commit_cancelled:
                    raise asyncio.CancelledError
                return generation

            self._active_workspace_mcp = generation
            self._prepared_workspace_mcp = None
            generation.state = "active"
            self._mcp_host.mark_active(generation.generation_id)
            return generation

    async def discard_workspace_mcp_candidate(self) -> None:
        async with self._candidate_prepare_lock:
            await self._discard_workspace_mcp_candidate()

    async def _discard_workspace_mcp_candidate(self) -> None:
        generation = self._prepared_workspace_mcp
        self._prepared_workspace_mcp = None
        if generation is None:
            return
        await self._dispose_workspace_mcp(generation, state="discarded")

    def _check_workspace_mcp_name_conflicts(
        self,
        server_specs: Mapping[str, object],
    ) -> None:
        occupied = {
            server_name
            for generation in self._active_generations.values()
            for server_name in generation.contributions.mcp_servers
        }
        occupied.update(
            server_name
            for generation in self._prepared_generations.values()
            for server_name in generation.contributions.mcp_servers
        )
        conflicts = sorted(occupied.intersection(server_specs))
        if conflicts:
            raise RuntimeError(
                f"workspace MCP 与插件 server 名称冲突: {', '.join(conflicts)}"
            )

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

    def bind_candidate_service_host(
        self,
        *,
        start: Callable[[str, dict[str, dict[str, Any]]], Awaitable[None]],
        stop: Callable[[str], Awaitable[None]],
        assert_healthy: Callable[[str], Awaitable[None]],
    ) -> None:
        """Bind isolated managed-service ownership for validation candidates."""

        self._candidate_service_starter = start
        self._candidate_service_stopper = stop
        self._candidate_service_health_check = assert_healthy

    async def wait_mcp_fatal_failure(self) -> None:
        """Escalate exhausted active MCP recovery to the runtime owner."""

        await self._mcp_host.wait_fatal_failure()

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
    def latest_snapshot(self) -> RuntimeSnapshot | None:
        return self._snapshot_store.latest

    @property
    def ready_candidate(self) -> PluginGeneration | None:
        return (
            None if self._ready_candidate is None else self._ready_candidate.candidate
        )

    @property
    def installed_plugins_home(self) -> Path:
        return _plugins_home(self._installed_cache_root)

    @property
    def snapshot_store(self) -> RuntimeSnapshotStore:
        return self._snapshot_store

    @property
    def reload_journal(self) -> ReloadJournal:
        return self._reload_journal

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
        for mod in self.discover(installed_selector="latest"):
            if mod.get("package_id"):
                continue
            _ = entries.setdefault(_resolve_plugin_id(mod), True)
        return write_plugin_manifest(entries, plugins_home=plugins_home)

    def watch_revision(self) -> str:
        digest = hashlib.sha256()
        home = _plugins_home(self._installed_cache_root)
        digest.update(_path_metadata(home / "manifest.toml"))
        for mod in self.discover(installed_selector="latest"):
            plugin_id = _resolve_plugin_id(mod)
            plugin_dir = Path(mod["plugin_root"])
            data_dir = _resolve_plugin_data_dir(
                mod["name"],
                mod,
                self._workspace,
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
        return self._declared_bot_commands("telegram_bot_commands")

    @property
    def mobile_bot_commands(self) -> list[tuple[str, str]]:
        return self._declared_bot_commands("mobile_bot_commands")

    def _declared_bot_commands(self, getter_name: str) -> list[tuple[str, str]]:
        """聚合当前插件代际为指定渠道显式声明的命令。"""

        commands: list[tuple[str, str]] = []
        for generation in self._active_generations.values():
            if not self._registry_active(generation.module_path):
                continue
            instance = generation.instance
            getter = getattr(instance, getter_name, None)
            if getter is None:
                continue
            typed_getter = cast(Callable[[], list[tuple[str, str]]], getter)
            for command, description in typed_getter():
                commands.append((str(command), str(description)))
        return commands

    # 扫描所有 plugin_dirs，返回可加载的插件描述列表
    def discover(
        self,
        *,
        installed_selector: ArtifactSelector = "stable",
    ) -> list[dict[str, str]]:
        mods: list[dict[str, str]] = []
        seen_names: set[str] = set()
        project_root = _package_project_root(self._dirs)
        packages = discover_plugin_packages(project_root) if project_root else {}
        enabled_packages = (
            _select_enabled_plugin_packages(
                packages,
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
            installed_selector=installed_selector,
        ):
            name = (
                source.plugin_name
                if source.source_type == "installed"
                else source.plugin_root.name
            )
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
            mods.append(
                {
                    "name": name,
                    "plugin_root": str(source.plugin_root),
                    "module_path": str(module_path) if module_path is not None else "",
                    "import_path": f"akasic_plugin_{import_source}_{import_suffix}",
                    "marketplace": source.marketplace,
                    "source_type": source.source_type,
                    "package_id": package_id,
                }
            )
        return mods

    async def load_all(self) -> None:
        """Load stable plugins and reconstruct any durable latest candidate."""

        # 1. 先处理尚未进入 latest_ready 的残留事务，恢复磁盘 pointer。
        recovery = self._reload_journal.pending_recovery()
        self._require_unique_recovery_plugins(recovery)
        stable_by_id = self._discovered_by_id(installed_selector="stable")
        latest_by_id = self._discovered_by_id(installed_selector="latest")
        for action in recovery:
            if action.action != "discard_candidate":
                continue
            self._discard_recovery_pointer(
                action.plugin_id,
                action.source_revision,
                stable_by_id=stable_by_id,
                latest_by_id=latest_by_id,
            )
            self._reload_journal.finish_recovery(action)
            self._write_startup_recovery_fact(action, committed=False)

        # 2. 根据 durable pointer 判定 promoting 崩溃发生在切换前还是切换后。
        stable_by_id = self._discovered_by_id(installed_selector="stable")
        latest_by_id = self._discovered_by_id(installed_selector="latest")
        restore_candidates, restore_committed, restore_discarded = (
            self._classify_reload_recovery(
                recovery,
                stable_by_id=stable_by_id,
                latest_by_id=latest_by_id,
            )
        )
        for action in restore_discarded:
            self._reload_journal.finish_recovery(action)
            self._write_startup_recovery_fact(action, committed=False)

        # 3. stable 先恢复服务；latest 候选随后以新事务重新准备和验证。
        for mod in stable_by_id.values():
            _ = await self._load_one(mod)
        self._finish_committed_recovery(restore_committed)
        await self._restore_latest_candidates(restore_candidates, latest_by_id)

    @staticmethod
    def _require_unique_recovery_plugins(
        recovery: tuple[ReloadRecoveryAction, ...],
    ) -> None:
        seen: set[str] = set()
        for action in recovery:
            if action.plugin_id in seen:
                raise RuntimeError(
                    f"同一插件存在多个未完成 ReloadTransaction: {action.plugin_id}"
                )
            seen.add(action.plugin_id)

    def _classify_reload_recovery(
        self,
        recovery: tuple[ReloadRecoveryAction, ...],
        *,
        stable_by_id: dict[str, dict[str, str]],
        latest_by_id: dict[str, dict[str, str]],
    ) -> tuple[
        list[ReloadRecoveryAction],
        list[ReloadRecoveryAction],
        list[ReloadRecoveryAction],
    ]:
        """Classify durable transactions by the pointer switch already on disk."""

        restore_candidates: list[ReloadRecoveryAction] = []
        restore_committed: list[ReloadRecoveryAction] = []
        restore_discarded: list[ReloadRecoveryAction] = []
        for action in recovery:
            if action.action == "discard_candidate":
                continue
            stable_revision = _mod_source_revision(stable_by_id.get(action.plugin_id))
            latest_revision = _mod_source_revision(latest_by_id.get(action.plugin_id))
            if action.action == "restore_candidate":
                if latest_revision != action.source_revision:
                    raise RuntimeError(
                        "ReloadTransaction latest 恢复源码不一致: "
                        f"{action.plugin_id} expected={action.source_revision} "
                        f"actual={latest_revision}"
                    )
                restore_candidates.append(action)
                continue
            if stable_revision == action.source_revision:
                restore_committed.append(action)
                continue
            if (
                action.phase in {"commit_started", "promoting"}
                and latest_revision == action.source_revision
            ):
                self._discard_recovery_pointer(
                    action.plugin_id,
                    action.source_revision,
                    stable_by_id=stable_by_id,
                    latest_by_id=latest_by_id,
                )
                restore_discarded.append(replace(action, action="discard_candidate"))
                continue
            if (
                action.phase in {"commit_started", "promoting"}
                and stable_revision == latest_revision
                and self._has_installed_pointer_state(action.plugin_id)
            ):
                restore_discarded.append(action)
                continue
            raise RuntimeError(
                "ReloadTransaction 恢复源码不一致: "
                f"{action.plugin_id} expected={action.source_revision} "
                f"stable={stable_revision} latest={latest_revision}"
            )
        return restore_candidates, restore_committed, restore_discarded

    def _has_installed_pointer_state(self, plugin_id: str) -> bool:
        plugin_name, separator, marketplace = plugin_id.rpartition("@")
        if not separator:
            return False
        plugin_base = (
            _plugins_home(self._installed_cache_root)
            / "cache"
            / marketplace
            / plugin_name
        )
        state_path = pointer_state_path(plugin_base)
        return state_path.exists() or state_path.is_symlink()

    def _finish_committed_recovery(
        self,
        recovery: list[ReloadRecoveryAction],
    ) -> None:
        """Confirm that every disk-committed generation became active stable."""

        for action in recovery:
            generation = self._active_generations.get(action.plugin_id)
            if generation is None:
                raise RuntimeError(
                    f"ReloadTransaction 恢复缺少插件: {action.plugin_id}"
                )
            assert generation.source_revision == action.source_revision
            self._reload_journal.finish_recovery(action)
            self._write_startup_recovery_fact(action, committed=True)

    def _write_startup_recovery_fact(
        self,
        action: ReloadRecoveryAction,
        *,
        committed: bool,
    ) -> None:
        message = (
            f"{action.plugin_id} 更新已在 Core 重启后确认提交；当前使用新版本。"
            if committed
            else f"{action.plugin_id} 更新在 Core 重启时没有完成；候选已丢弃，原版本保持可用。"
        )
        atomic_save_json(
            self._workspace / "runtime" / "plugin-rollout-fact.json",
            {"message": message},
            ensure_ascii=False,
            domain="plugin_rollout_fact",
        )

    async def _restore_latest_candidates(
        self,
        recovery: list[ReloadRecoveryAction],
        latest_by_id: dict[str, dict[str, str]],
    ) -> None:
        """Rebuild latest candidates; reject a bad candidate without losing stable."""

        for action in recovery:
            self._reload_journal.finish_recovery(action)
            mod = latest_by_id.get(action.plugin_id)
            if mod is None:
                raise RuntimeError(
                    f"ReloadTransaction latest 恢复缺少插件: {action.plugin_id}"
                )
            generation = await self._load_one(mod, activate=False)
            if generation is None:
                _discard_installed_candidate_mod(mod)
                logger.error(
                    "ReloadTransaction latest 候选恢复失败，保留 stable: %s",
                    action.plugin_id,
                )
                continue
            try:
                result = await self._publish_prepared(action.plugin_id)
            except Exception:
                await self.discard_prepared(action.plugin_id)
                _discard_installed_candidate_mod(mod)
                logger.exception(
                    "ReloadTransaction latest 候选发布失败，保留 stable: %s",
                    action.plugin_id,
                )
                continue
            if result["publication_state"] != "latest_ready":
                _discard_installed_candidate_mod(mod)
                logger.error(
                    "ReloadTransaction latest 候选被拒绝，保留 stable: %s",
                    action.plugin_id,
                )

    def _discovered_by_id(
        self,
        *,
        installed_selector: ArtifactSelector,
    ) -> dict[str, dict[str, str]]:
        return {
            _resolve_plugin_id(mod): mod
            for mod in self.discover(installed_selector=installed_selector)
        }

    @staticmethod
    def _discard_recovery_pointer(
        plugin_id: str,
        source_revision: str,
        *,
        stable_by_id: dict[str, dict[str, str]],
        latest_by_id: dict[str, dict[str, str]],
    ) -> None:
        latest = latest_by_id.get(plugin_id)
        stable = stable_by_id.get(plugin_id)
        if latest is None or latest.get("source_type") != "installed":
            return
        latest_revision = _mod_source_revision(latest)
        stable_revision = _mod_source_revision(stable)
        if latest_revision == stable_revision:
            return
        if latest_revision != source_revision:
            raise RuntimeError(
                "ReloadTransaction discard 源码不一致: "
                f"{plugin_id} expected={source_revision} actual={latest_revision}"
            )
        plugin_base = _installed_artifact_base_from_root(Path(latest["plugin_root"]))
        _ = discard_latest_pointer(plugin_base)

    async def prepare_candidate(self, plugin_id: str) -> PluginGeneration | None:
        if self._ready_candidate is not None:
            raise RuntimeError(
                f"已有 latest 等待 promote/discard: {self._ready_candidate.plugin_id}"
            )
        await self.discard_prepared(plugin_id, preserve_latest=True)
        for mod in self.discover(installed_selector="latest"):
            if _resolve_plugin_id(mod) == plugin_id:
                generation = await self._load_one(mod, activate=False)
                if generation is None:
                    _discard_installed_candidate_mod(mod)
                return generation
        raise KeyError(f"插件不存在: {plugin_id}")

    async def discard_prepared(
        self,
        plugin_id: str,
        *,
        preserve_latest: bool = False,
        error: str = "candidate discarded",
    ) -> None:
        generation = self._prepared_generations.pop(plugin_id, None)
        if generation is None:
            return
        if not preserve_latest:
            _discard_generation_candidate_pointer(generation)
        self._abort_reload(generation, error=error)
        await self._dispose_generation(generation, state="discarded")

    def _begin_reload_attempt(
        self,
        *,
        plugin_id: str,
        generation_id: str,
        source_revision: str,
        config_revision: str,
    ) -> str:
        base = self.current_snapshot
        return self._reload_journal.begin(
            plugin_id=plugin_id,
            base_snapshot_id=base.snapshot_id if base is not None else None,
            generation_id=generation_id,
            source_revision=source_revision,
            config_revision=config_revision,
        )

    def _abort_reload_attempt(self, tx_id: str | None, *, error: str) -> None:
        if tx_id is not None:
            self._reload_journal.advance(tx_id, "aborted", error=error)

    async def _dispose_generation(
        self,
        generation: PluginGeneration,
        *,
        state: str,
        preserve_stable_alias: bool = False,
    ) -> None:
        """完成插件终止、作用域清理和注册表卸载。"""

        from agent.plugins.context import allow_plugin_cleanup_writes

        # 1. 终止生命周期对象，并在调用方取消后继续完成它
        externally_cancelled = False
        if generation.prepare_started:
            terminator = getattr(generation.instance, "terminate", None)
            if callable(terminator):
                try:
                    with allow_plugin_cleanup_writes(generation.generation_id):
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
        with allow_plugin_cleanup_writes(generation.generation_id):
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

    def _retire_generation(self, generation: PluginGeneration) -> None:
        """通知已关闭 admission 的 generation 进入退役状态。"""

        if generation.retire_started:
            return
        generation.retire_started = True
        generation.state = "retired"
        if generation.mcp_catalog is not None:
            self._mcp_host.mark_draining(generation.generation_id)
        self._draining_generations.setdefault(generation.plugin_id, []).append(
            generation
        )
        try:
            cast(Any, generation.instance).retire()
        except Exception as error:
            error_text = str(error) or type(error).__name__
            logger.warning(
                "插件 retire 失败 (%s): %s",
                generation.plugin_id,
                error_text,
            )
            self._cleanup_failures.append(
                CleanupFailure(
                    resource=f"plugin:{generation.plugin_id}:retire",
                    error=error_text,
                )
            )

    def _forget_drained_generation(self, generation: PluginGeneration) -> None:
        tracked = self._draining_generations.get(generation.plugin_id)
        if tracked is None:
            return
        remaining = [item for item in tracked if item is not generation]
        if remaining:
            self._draining_generations[generation.plugin_id] = remaining
        else:
            _ = self._draining_generations.pop(generation.plugin_id, None)

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
            self._forget_drained_generation(generation)
        workspace_mcp = snapshot.workspace_mcp_generation
        if (
            workspace_mcp is not None
            and not self._snapshot_store.workspace_mcp_is_referenced_elsewhere(
                workspace_mcp,
                excluding_snapshot_id=snapshot.snapshot_id,
            )
        ):
            await self._dispose_workspace_mcp(workspace_mcp, state=state)
        self._finish_drained_reload(snapshot.snapshot_id)

    async def _dispose_workspace_mcp(
        self,
        generation: WorkspaceMcpGeneration,
        *,
        state: str,
    ) -> None:
        cleanup_failures, _ = await _complete_critical(generation.scope.aclose())
        self._cleanup_failures.extend(cleanup_failures)
        generation.state = state

    async def prepare_changed(self) -> list[dict[str, object]]:
        async with self._candidate_prepare_lock:
            if self._ready_candidate is not None:
                return [self._ready_candidate_status()]
            discovered = {
                _resolve_plugin_id(mod): mod
                for mod in self.discover(installed_selector="latest")
            }
            return await self._prepare_changed(discovered=discovered)

    async def reconcile_changed(self) -> list[dict[str, object]]:
        async with self._candidate_prepare_lock:
            return await self._reconcile_changed_locked()

    async def install_candidate(
        self,
        *,
        source: str,
        marketplace: str,
        ref_name: str,
        sparse_paths: list[str],
    ) -> tuple[PluginInstallResult, dict[str, object]]:
        """Stage one immutable artifact and publish its latest runtime atomically."""

        # 1. 与 watcher 共用 candidate owner，写 cache 前拒绝未决候选。
        async with self._candidate_prepare_lock:
            _, preflight_cancelled = await _complete_critical(
                self._reconcile_changed_locked()
            )
            if preflight_cancelled:
                raise asyncio.CancelledError
            status = self.candidate_status()
            if status["candidate_state"] in {
                "preparing",
                "prepared",
                "validating",
                "commit_started",
                "latest_ready",
                "discarding",
                "promoting",
            }:
                raise RuntimeError(
                    "已有插件候选等待处理: "
                    f"plugin={status['candidate_plugin_id']} "
                    f"phase={status['candidate_state']} "
                    f"tx={status['candidate_reload_tx_id']}"
                )

            # 2. 持锁完成 artifact 发布与 runtime reconcile，不留 watcher 插入窗口。
            result, install_cancelled = await _complete_critical(
                asyncio.to_thread(
                    install_git_plugin,
                    workspace=self._workspace,
                    source=source,
                    marketplace=marketplace,
                    ref_name=ref_name,
                    sparse_paths=sparse_paths,
                    plugins_home=self.installed_plugins_home,
                    stage_candidate=True,
                )
            )
            _, reconcile_cancelled = await _complete_critical(
                self._reconcile_changed_locked()
            )
            plugin_id = f"{result.plugin_name}@{result.marketplace}"
            status = self.candidate_status()
            if result.staged_candidate and (
                status["candidate_plugin_id"] != plugin_id
                or status["candidate_state"] != "latest_ready"
            ):
                raise RuntimeError(
                    "插件候选未进入 latest_ready: "
                    f"requestedPlugin={plugin_id} "
                    f"installedGitRevision={result.source_revision} "
                    f"actualPlugin={status['candidate_plugin_id']} "
                    f"actualRuntimeRevision={status['candidate_source_revision']} "
                    f"phase={status['candidate_state']} "
                    f"tx={status['candidate_reload_tx_id']} "
                    f"error={status['candidate_error']}"
                )
            if install_cancelled or reconcile_cancelled:
                raise asyncio.CancelledError
            return result, status

    def annotate_reload(self, tx_id: str, details: dict[str, object]) -> None:
        """Append turn lineage evidence to an existing reload transaction."""

        self._reload_journal.annotate(tx_id, details)

    def require_installed_plugin(self, plugin_id: str) -> None:
        """Fail before registering uninstall when the plugin has no installed owner."""

        manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
        if plugin_id not in manifest:
            raise RuntimeError(f"插件未安装: {plugin_id}")

    async def _reconcile_changed_locked(self) -> list[dict[str, object]]:
        """Reconcile discovered latest artifacts while candidate ownership is held."""

        await self._snapshot_store.retry_drains()
        results: list[dict[str, object]] = []
        ready = self._ready_candidate
        if ready is not None:
            manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
            if manifest.get(ready.plugin_id, True):
                return [self._ready_candidate_status()]
            results.append(await self._drop_ready(ready.plugin_id))
        discovered = {
            _resolve_plugin_id(mod): mod
            for mod in self.discover(installed_selector="latest")
        }
        manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
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
            publication = await self._publish_prepared(plugin_id)
            results.append(publication)
            if publication.get("publication_state") == "latest_ready":
                return results
        for plugin_id in sorted(desired - set(self._active_generations)):
            generation = await self._load_one(discovered[plugin_id], activate=False)
            if generation is None:
                _discard_installed_candidate_mod(discovered[plugin_id])
                continue
            publication = await self._publish_prepared(plugin_id)
            results.append(publication)
            if publication.get("publication_state") == "latest_ready":
                return results
        return results

    async def reconcile_disabled_and_drain(self, plugin_id: str) -> None:
        async with self._candidate_prepare_lock:
            manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
            if manifest.get(plugin_id, False):
                raise RuntimeError(f"插件尚未禁用: {plugin_id}")
            if self._ready_candidate is not None:
                if self._ready_candidate.plugin_id != plugin_id:
                    raise RuntimeError(
                        "存在其他插件 latest，必须先 promote/discard: "
                        f"{self._ready_candidate.plugin_id}"
                    )
                _ = await self._drop_ready(plugin_id)
            active = self._active_generations.get(plugin_id)
            draining = self._draining_generations.get(plugin_id, [])
            if active is None and not draining:
                return
            if active is not None:
                _ = await self._deactivate_plugin(plugin_id)
                draining = self._draining_generations[plugin_id]
            for generation in draining:
                await self._snapshot_store.wait_for_generation_drained(generation)
                if not generation.scope.closed:
                    raise RuntimeError(f"插件旧代资源尚未关闭: {plugin_id}")
            _ = self._draining_generations.pop(plugin_id, None)

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
            if transaction is not None:
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
                self._snapshot_store.commit(
                    transaction,
                    after_open=lambda: self._retire_generation(active),
                )
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
        result: dict[str, object] = {
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
                workspace_mcp_generation=self._active_workspace_mcp,
            )
            snapshot.skill_catalog_generation_id = catalog_id
            snapshot.plugin_skill_index = catalog.normal_plugins
            snapshot.tool_registry = self._compile_snapshot_tools(
                generations,
                self._active_workspace_mcp,
            )
            snapshot.tool_hooks = self._compile_snapshot_tool_hooks(generations)
            return snapshot, catalog_id
        except BaseException:
            self._skill_host.close(catalog_id)
            raise

    async def publish_prepared(self, plugin_id: str) -> dict[str, object]:
        async with self._candidate_prepare_lock:
            return await self._publish_prepared(plugin_id)

    async def switch_ready(self, plugin_id: str) -> dict[str, object]:
        """Promote the one ready installed candidate without rebuilding it."""

        async with self._candidate_prepare_lock:
            ready = self._require_ready_candidate(plugin_id)
            generation = ready.candidate
            tx_id = generation.reload_tx_id
            if tx_id is None:
                raise RuntimeError("latest candidate 缺少 reload transaction")
            from agent.plugins.context import PreparedPluginKVStore

            context = cast(Any, generation.instance).context
            kv_store = context.kv_store
            if isinstance(kv_store, PreparedPluginKVStore) and kv_store.dirty:
                raise RuntimeError("候选插件修改了 KV，read-only 验证不能 promote")

            old_services = (
                ready.previous.contributions.managed_services
                if ready.previous is not None
                else {}
            )
            target_contributions = (
                generation.production_contributions or generation.contributions
            )
            new_services = target_contributions.managed_services
            old_channels = (
                ready.previous.contributions.channels
                if ready.previous is not None
                else ()
            )
            new_channels = target_contributions.channels
            endpoint_changed = (
                old_services != new_services or old_channels != new_channels
            )

            skill_linker, stable_skill_plugins, target_skill_plugins = (
                self._prepare_skill_links_for_promotion(generation)
            )

            # 1. Seal both stable and validation leases before touching ownership.
            candidate_snapshot = self._snapshot_store.pause_candidate_admission(
                ready.snapshot
            )
            quiesced_snapshot = (
                self._snapshot_store.pause_admission() if endpoint_changed else None
            )
            endpoints_switched = False
            if endpoint_changed:
                try:
                    if self._endpoint_quiescer is not None:
                        await self._endpoint_quiescer()
                    if quiesced_snapshot is not None:
                        await self._snapshot_store.wait_for_no_leases(quiesced_snapshot)
                    await self._snapshot_store.wait_for_no_leases(ready.snapshot)
                    await self._restore_ready_runtime(ready)
                    generation = ready.candidate
                    kv_store = cast(Any, generation.instance).context.kv_store
                    new_services = generation.contributions.managed_services
                    new_channels = generation.contributions.channels
                    await self._switch_plugin_endpoints(
                        plugin_id,
                        old_services,
                        new_services,
                        old_channels,
                        new_channels,
                    )
                    endpoints_switched = True
                except BaseException:
                    await self._snapshot_store.resume(quiesced_snapshot)
                    await self._snapshot_store.resume(candidate_snapshot)
                    if self._endpoint_resumer is not None:
                        await self._endpoint_resumer()
                    if self._ready_candidate is ready:
                        await self._drop_ready(plugin_id)
                    raise
            else:
                try:
                    await self._snapshot_store.wait_for_no_leases(ready.snapshot)
                    await self._restore_ready_runtime(ready)
                    generation = ready.candidate
                    kv_store = cast(Any, generation.instance).context.kv_store
                except asyncio.CancelledError:
                    await self._snapshot_store.resume(candidate_snapshot)
                    raise
                except Exception:
                    await self._snapshot_store.resume(candidate_snapshot)
                    if self._ready_candidate is ready:
                        await self._drop_ready(plugin_id)
                    raise

            # 2. 先切可回滚的 Skill 投影，再提交持久 pointer；整个回调不跨 await。
            skill_links_switched = False
            link_result = None

            def before_open() -> None:
                nonlocal link_result, skill_links_switched
                try:
                    link_result = skill_linker.sync(target_skill_plugins)
                except BaseException:
                    skill_linker.sync(stable_skill_plugins)
                    raise
                skill_links_switched = True
                phase = self._reload_journal.get(tx_id).phase
                if phase == "latest_ready":
                    self._advance_reload(generation, "promoting")
                artifact_base = _installed_artifact_base(generation)
                if artifact_base is not None:
                    _switch_ready_pointer(ready, artifact_base)
                if isinstance(kv_store, PreparedPluginKVStore):
                    kv_store.commit()

            # 3. Snapshot pointer 切换后再替换 manager 的 stable generation owner。
            def after_open() -> None:
                self._activate_published_generation(generation, ready.previous)
                generation.state = "active"
                if generation.mcp_catalog is not None:
                    self._mcp_host.mark_active(generation.generation_id)
                self._scopes[generation.module_path] = generation.scope
                self._loaded.add(generation.module_path)
                self._active_generations[plugin_id] = generation
                if ready.previous is not None:
                    self._retire_generation(ready.previous)
                self._channels = [
                    channel
                    for item in self._active_generations.values()
                    for channel in item.contributions.channels
                ]

            previous_snapshot = self.current_snapshot
            if previous_snapshot is not None:
                self._drain_transactions[previous_snapshot.snapshot_id] = tx_id
            try:
                transaction, cancelled = await _complete_critical(
                    self._snapshot_store.promote_latest(
                        before_open=before_open,
                        after_open=after_open,
                    )
                )
            except BaseException:
                endpoint_error: BaseException | None = None
                skill_error: BaseException | None = None
                if skill_links_switched:
                    try:
                        skill_linker.sync(stable_skill_plugins)
                    except BaseException as error:
                        skill_error = error
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
                if (
                    previous_snapshot is not None
                    and self.current_snapshot is previous_snapshot
                ):
                    _ = self._drain_transactions.pop(
                        previous_snapshot.snapshot_id,
                        None,
                    )
                await self._snapshot_store.resume(quiesced_snapshot)
                await self._snapshot_store.resume(candidate_snapshot)
                if self._endpoint_resumer is not None and endpoint_changed:
                    await self._endpoint_resumer()
                if endpoint_error is not None:
                    raise RuntimeError(
                        "插件 promote 失败后旧 endpoint 恢复失败"
                    ) from endpoint_error
                if skill_error is not None:
                    raise RuntimeError(
                        "插件 promote 失败后 stable skill 投影恢复失败"
                    ) from skill_error
                if endpoint_changed and self._ready_candidate is ready:
                    await self._drop_ready(plugin_id)
                raise
            self._ready_candidate = None
            self._track_reload_drain(generation, transaction.previous)
            if self._endpoint_resumer is not None and endpoint_changed:
                await self._endpoint_resumer()
            assert link_result is not None
            logger.info(
                "插件 stable skill 投影同步完成: expected=%d created=%d repaired=%d removed=%d skipped=%d",
                link_result.expected,
                link_result.created,
                link_result.repaired,
                link_result.removed,
                link_result.skipped,
            )
            result = self._publication_status(
                plugin_id,
                active=ready.previous,
                candidate=generation,
                publication_state="promoted",
            )
            validation_data_dir = (
                self._workspace
                / "runtime"
                / "plugin-validation"
                / generation.generation_id
            )
            try:
                await asyncio.to_thread(
                    _remove_validation_data_dir, validation_data_dir
                )
            except Exception as error:
                logger.error(
                    "候选隔离 plugin-data 清理失败: %s: %s",
                    validation_data_dir,
                    error,
                )
            logger.info(
                "plugin_snapshot_status %s",
                json.dumps(result, ensure_ascii=False, sort_keys=True),
            )
            if cancelled:
                raise asyncio.CancelledError
            return result

    async def _restore_ready_runtime(
        self,
        ready: _ReadyPluginCandidate,
    ) -> None:
        """Replace isolated validation resources before formal endpoint commit."""

        generation = ready.candidate
        production = generation.production_contributions
        if production is None:
            return
        production_data_dir = generation.production_data_dir
        if production_data_dir is None:
            raise RuntimeError("候选缺少 production plugin-data identity")

        # 1. Stop isolated services after every validation lease has ended.
        if generation.validation_managed_services:
            if self._candidate_service_stopper is None:
                raise RuntimeError("候选 managed service 隔离宿主未绑定")
            if self._candidate_service_health_check is None:
                raise RuntimeError("候选 managed service 健康检查未绑定")
            await self._candidate_service_health_check(generation.generation_id)
        if generation.mcp_catalog is not None:
            self._mcp_host.assert_healthy(generation.generation_id)
        if generation.validation_managed_services:
            assert self._candidate_service_stopper is not None
            await self._candidate_service_stopper(generation.generation_id)

        # 2. Reconnect MCP with formal endpoint env, then refresh snapshot payload.
        if generation.mcp_catalog is not None:
            await self._mcp_host.close(generation.generation_id)
            generation.mcp_catalog = None
        generation.contributions = production
        generation.data_dir = production_data_dir
        from agent.plugins.context import PreparedPluginKVStore

        context = cast(Any, generation.instance).context
        context.data_dir = production_data_dir
        context.kv_store = PreparedPluginKVStore(
            production_data_dir / ".kv.json",
            can_write=lambda: _generation_can_write(generation),
            writer_id=generation.generation_id,
        )
        if production.mcp_servers or production.proactive_sources:
            generation.mcp_catalog = await self._mcp_host.prepare(
                generation.generation_id,
                server_specs=production.mcp_servers,
                required_tools=_required_mcp_tools(production.proactive_sources),
                scope=generation.scope,
            )
            generation.scope.defer(
                "production_mcp_catalog",
                lambda: self._mcp_host.close(generation.generation_id),
            )
        replacement = self._compile_generation_snapshot(generation)
        if replacement.snapshot_id != ready.snapshot.snapshot_id:
            raise RuntimeError(
                "候选隔离资源恢复后 snapshot identity 发生变化: "
                f"{ready.snapshot.snapshot_id} -> {replacement.snapshot_id}"
            )
        _replace_snapshot_payload(ready.snapshot, replacement)
        self._compile_snapshot_event_handlers(ready.snapshot)
        if self._dashboard_preparer is not None:
            self._dashboard_preparer(ready.snapshot)
        cast(Any, generation.instance).context.tool_registry = (
            ready.snapshot.tool_registry
        )
        generation.production_contributions = None
        generation.validation_managed_services = {}
        generation.production_data_dir = None

    async def drop_candidate(self, plugin_id: str) -> dict[str, object]:
        """Discard the one ready installed candidate and preserve stable."""

        async with self._candidate_prepare_lock:
            return await self._drop_ready(plugin_id)

    async def _drop_ready(self, plugin_id: str) -> dict[str, object]:
        ready = self._require_ready_candidate(plugin_id)
        tx_id = ready.candidate.reload_tx_id
        if tx_id is None:
            raise RuntimeError("latest candidate 缺少 reload transaction")
        phase = self._reload_journal.get(tx_id).phase
        if phase in {"latest_ready", "promoting"}:
            self._advance_reload(
                ready.candidate,
                "discarding",
                error="candidate behavior rejected",
            )
        elif phase != "discarding":
            raise RuntimeError(f"latest candidate 不能从 {phase} discard")
        artifact_base = _installed_artifact_base(ready.candidate)
        if artifact_base is not None:
            _restore_ready_pointer(ready, artifact_base)
        _, cancelled = await _complete_critical(
            self._snapshot_store.discard_latest(ready.snapshot)
        )
        self._advance_reload(
            ready.candidate,
            "aborted",
            error="candidate behavior rejected",
        )
        self._ready_candidate = None
        result = self._publication_status(
            plugin_id,
            active=ready.previous,
            candidate=ready.candidate,
            publication_state="discarded",
        )
        logger.info(
            "plugin_snapshot_status %s",
            json.dumps(result, ensure_ascii=False, sort_keys=True),
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    def candidate_status(self, plugin_id: str | None = None) -> dict[str, object]:
        ready = self._ready_candidate
        transaction = None
        if ready is not None and (plugin_id is None or ready.plugin_id == plugin_id):
            tx_id = ready.candidate.reload_tx_id
            if tx_id is None:
                raise RuntimeError("latest candidate 缺少 reload transaction")
            transaction = self._reload_journal.get(tx_id)
        else:
            latest = self._reload_journal.latest(plugin_id=plugin_id)
            if latest is not None and latest.phase not in {"complete", "recovered"}:
                transaction = latest
        return {
            "stable_snapshot_id": (
                self.current_snapshot.snapshot_id
                if self.current_snapshot is not None
                else None
            ),
            "latest_snapshot_id": (
                self.latest_snapshot.snapshot_id
                if self.latest_snapshot is not None
                else None
            ),
            "candidate_plugin_id": (
                transaction.plugin_id if transaction is not None else None
            ),
            "candidate_generation_id": (
                transaction.generation_id if transaction is not None else None
            ),
            "candidate_state": None if transaction is None else transaction.phase,
            "candidate_source_revision": (
                None if transaction is None else transaction.source_revision
            ),
            "candidate_reload_tx_id": (
                None if transaction is None else transaction.tx_id
            ),
            "candidate_error": None if transaction is None else transaction.error,
        }

    def candidate_child_evidence(
        self,
        plugin_id: str,
        generation_id: str,
        items: tuple[object, ...],
    ) -> tuple[str, ...]:
        """返回 child 真实成功使用的候选 Tool 或 Skill 证据。"""

        # 1. 从冻结的 latest snapshot 判定 owner，不信任 child 自报。
        ready = self._require_ready_candidate(plugin_id)
        generation = ready.candidate
        if generation.generation_id != generation_id:
            raise RuntimeError(
                "candidate child generation 身份不一致: "
                f"expected={generation.generation_id} actual={generation_id}"
            )
        registry = ready.snapshot.tool_registry
        if registry is None:
            raise RuntimeError("candidate RuntimeSnapshot 缺少 ToolRegistry")
        plugin_name = str(getattr(generation.instance, "name", plugin_id))
        owned_tools = registry.get_source_tool_names(
            "plugin", plugin_name, risk="read-only"
        )
        for server_name in generation.contributions.mcp_servers:
            owned_tools.update(
                registry.get_source_tool_names(
                    "mcp", server_name, risk="read-only"
                )
            )
        owned_skills = {
            skill_dir.name
            for root in generation.contributions.skill_roots
            for skill_dir in root.iterdir()
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").is_file()
        }
        skill_catalog_generation_id = (
            generation.skill_catalog.generation_id
            if generation.skill_catalog is not None
            else None
        )

        # 2. 只接受成功工具 item 或本轮真实注入的候选 Skill。
        evidence: set[str] = set()
        for item in items:
            kind = getattr(getattr(item, "kind", None), "value", None)
            data = getattr(item, "data", None)
            if not isinstance(data, dict):
                continue
            if kind == "toolCall":
                name = data.get("name")
                if data.get("status") == "success" and name in owned_tools:
                    evidence.add(f"tool:{name}")
                provenance = data.get("runtimeProvenance")
                if (
                    data.get("status") == "success"
                    and name == "load_skill"
                    and isinstance(provenance, dict)
                    and provenance.get("kind") == "plugin-skill"
                    and provenance.get("pluginId") == plugin_id
                    and provenance.get("skillName") in owned_skills
                    and provenance.get("skillCatalogGenerationId")
                    == skill_catalog_generation_id
                    and provenance.get("runtimeSnapshotId") == ready.snapshot.snapshot_id
                ):
                    evidence.add(f"skill:{provenance['skillName']}")
        return tuple(sorted(evidence))

    def _ready_candidate_status(self) -> dict[str, object]:
        ready = self._ready_candidate
        if ready is None:
            raise RuntimeError("没有等待 promote/discard 的插件候选")
        return self._publication_status(
            ready.plugin_id,
            active=ready.previous,
            candidate=ready.candidate,
            publication_state="latest_ready",
        )

    def _require_ready_candidate(self, plugin_id: str) -> _ReadyPluginCandidate:
        ready = self._ready_candidate
        if ready is None:
            raise RuntimeError("没有等待 promote/discard 的插件候选")
        if ready.plugin_id != plugin_id:
            raise RuntimeError(f"latest 属于其他插件: {ready.plugin_id}")
        return ready

    async def _publish_prepared(self, plugin_id: str) -> dict[str, object]:
        generation = self._prepared_generations.get(plugin_id)
        if generation is None:
            raise KeyError(f"插件没有待发布候选: {plugin_id}")
        active = self._active_generations.get(plugin_id)
        stage_latest = _installed_generation_is_candidate(generation)
        production = generation.production_contributions or generation.contributions
        production_endpoint_changed = (
            active.contributions.managed_services if active is not None else {}
        ) != production.managed_services or (
            active.contributions.channels if active is not None else ()
        ) != production.channels
        try:
            if stage_latest:
                if (
                    generation.production_contributions is None
                    or generation.production_data_dir is None
                ):
                    raise RuntimeError("installed candidate 缺少隔离 plugin-data 身份")
                if (
                    generation.mcp_catalog is not None
                    and not generation.contributions.mcp_servers
                    and not generation.contributions.proactive_sources
                ):
                    await self._mcp_host.close(generation.generation_id)
                    generation.mcp_catalog = None
                if generation.validation_managed_services:
                    if self._candidate_service_starter is None:
                        raise RuntimeError("候选 managed service 隔离宿主未绑定")
                    await self._candidate_service_starter(
                        generation.generation_id,
                        generation.validation_managed_services,
                    )
                    assert self._candidate_service_stopper is not None
                    generation.scope.defer(
                        "validation_managed_services",
                        lambda: self._candidate_service_stopper(
                            generation.generation_id
                        ),
                    )
            workspace_generations = tuple(
                item
                for item in (
                    self._active_workspace_mcp,
                    self._prepared_workspace_mcp,
                )
                if item is not None
            )
            conflicts = sorted(
                set(generation.contributions.mcp_servers).intersection(
                    server_name
                    for item in workspace_generations
                    for server_name in item.catalog.servers
                )
            )
            if conflicts:
                raise RuntimeError(
                    f"插件 MCP 与 workspace server 名称冲突: {', '.join(conflicts)}"
                )
            generation.runtime_snapshot = self._compile_generation_snapshot(generation)
            snapshot = generation.runtime_snapshot
            cast(Any, generation.instance).context.tool_registry = (
                snapshot.tool_registry
            )
        except (asyncio.CancelledError, Exception) as error:
            error_text = str(error) or type(error).__name__
            self._record_failed_gate(
                plugin_id=plugin_id,
                revision=generation.source_revision,
                check_id="publish_rebase",
                reason=error_text,
            )
            await self.discard_prepared(
                plugin_id,
                error=f"publish_rebase: {error_text}",
            )
            raise
        try:
            await self._prepare_generation(generation)
        except (asyncio.CancelledError, Exception) as error:
            error_text = str(error) or type(error).__name__
            self._record_failed_gate(
                plugin_id=plugin_id,
                revision=generation.source_revision,
                check_id="prepare",
                reason=error_text,
            )
            await self.discard_prepared(
                plugin_id,
                error=f"prepare: {error_text}",
            )
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
        endpoint_changed = not stage_latest and (
            old_services != new_services or old_channels != new_channels
        )
        if stage_latest and production_endpoint_changed:
            self._reload_journal.annotate(
                cast(str, generation.reload_tx_id),
                {
                    "event": "exclusive_endpoints_deferred",
                    "managed_services_changed": (
                        active is None
                        or active.contributions.managed_services
                        != production.managed_services
                    ),
                    "channels_changed": (
                        active is None
                        and bool(production.channels)
                        or active is not None
                        and active.contributions.channels != production.channels
                    ),
                },
            )
        self._compile_snapshot_event_handlers(snapshot)
        if self._dashboard_preparer is not None:
            try:
                self._dashboard_preparer(snapshot)
            except Exception as error:
                error_text = str(error) or type(error).__name__
                self._record_failed_gate(
                    plugin_id=plugin_id,
                    revision=generation.source_revision,
                    check_id="dashboard",
                    reason=error_text,
                )
                await self.discard_prepared(
                    plugin_id,
                    error=f"dashboard: {error_text}",
                )
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
                error_text = "持有 RuntimeSnapshot lease 时不能切换独占端点"
                await self.discard_prepared(
                    plugin_id,
                    error=f"endpoint_lease: {error_text}",
                )
                raise RuntimeError(error_text)
            quiesced_snapshot = self._snapshot_store.pause_admission()
            try:
                if self._endpoint_quiescer is not None:
                    await self._endpoint_quiescer()
                if quiesced_snapshot is not None:
                    await self._snapshot_store.wait_for_no_leases(quiesced_snapshot)
            except BaseException as error:
                error_text = str(error) or type(error).__name__
                await self._snapshot_store.resume(quiesced_snapshot)
                if self._endpoint_resumer is not None:
                    await self._endpoint_resumer()
                await self.discard_prepared(
                    plugin_id,
                    error=f"endpoint_quiesce: {error_text}",
                )
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
                error_text = str(error) or type(error).__name__
                self._record_failed_gate(
                    plugin_id=plugin_id,
                    revision=generation.source_revision,
                    check_id="endpoints",
                    reason=error_text,
                )
                await self._snapshot_store.resume(quiesced_snapshot)
                if self._endpoint_resumer is not None:
                    await self._endpoint_resumer()
                await self.discard_prepared(
                    plugin_id,
                    error=f"endpoints: {error_text}",
                )
                if isinstance(error, asyncio.CancelledError):
                    raise
                return self._publication_status(
                    plugin_id,
                    active=active,
                    candidate=generation,
                    publication_state="failed",
                )
        transaction = self._snapshot_store.begin_publish(
            snapshot,
            admission_gated=quiesced_snapshot is not None,
        )
        self._advance_reload(
            generation,
            "validating",
            candidate_snapshot_id=snapshot.snapshot_id,
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
            await self._abort_failed_publication(
                generation,
                transaction,
                error="post-publish invariant failed",
            )
            if self._endpoint_resumer is not None:
                await self._endpoint_resumer()
            if endpoint_error is not None:
                raise RuntimeError(
                    "Snapshot abort 后旧端点恢复失败"
                ) from endpoint_error
            raise

        commit_error: BaseException | None = None
        commit_cancelled = False
        from agent.plugins.context import PreparedPluginKVStore

        def open_candidate() -> None:
            self._advance_reload(generation, "commit_started")
            context = cast(Any, generation.instance).context
            context.data_dir = generation.data_dir
            context.session_manager = self._session_manager
            context.memory_engine = self._memory_engine
            context.llm = self._llm
            generation.state = "activating"
            try:
                cast(Any, generation.instance).activate()
            except BaseException:
                context.data_dir = None
                raise
            if isinstance(context.kv_store, PreparedPluginKVStore) and not stage_latest:
                context.kv_store.commit()
            if generation.staged_event_bus is not None:
                generation.staged_event_bus.publish()
            generation.state = "candidate" if stage_latest else "active"

        previous_snapshot = transaction.previous
        if (
            not stage_latest
            and generation.reload_tx_id is not None
            and previous_snapshot is not None
        ):
            self._drain_transactions[previous_snapshot.snapshot_id] = (
                generation.reload_tx_id
            )
        try:
            if stage_latest:
                _, commit_cancelled = await _complete_critical(
                    self._snapshot_store.commit_latest(
                        transaction,
                        before_open=open_candidate,
                    )
                )
            else:
                _, commit_cancelled = await _complete_critical(
                    self._snapshot_store.commit(
                        transaction,
                        before_open=open_candidate,
                        after_open=(
                            None
                            if active is None
                            else lambda: self._retire_generation(active)
                        ),
                    )
                )
        except BaseException as error:
            commit_error = error

        if (
            commit_error is not None
            and self._snapshot_store.pending_candidate is snapshot
        ):
            if previous_snapshot is not None:
                _ = self._drain_transactions.pop(
                    previous_snapshot.snapshot_id,
                    None,
                )
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
            await self._abort_failed_publication(
                generation,
                transaction,
                error=str(commit_error) or type(commit_error).__name__,
            )
            if self._endpoint_resumer is not None:
                await self._endpoint_resumer()
            if endpoint_error is not None:
                raise RuntimeError(
                    "Snapshot commit 失败后旧端点恢复失败"
                ) from endpoint_error
            raise commit_error

        _ = self._prepared_generations.pop(plugin_id)
        if stage_latest:
            self._ready_candidate = _ReadyPluginCandidate(
                plugin_id=plugin_id,
                previous=active,
                candidate=generation,
                snapshot=snapshot,
            )
            self._advance_reload(generation, "latest_ready")
            if commit_error is not None:
                raise commit_error
            if commit_cancelled:
                raise asyncio.CancelledError
            result = self._publication_status(
                plugin_id,
                active=active,
                candidate=generation,
                publication_state="latest_ready",
            )
            logger.info(
                "plugin_snapshot_status %s",
                json.dumps(result, ensure_ascii=False, sort_keys=True),
            )
            return result

        self._track_reload_drain(generation, previous_snapshot)
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
        plugin_dir = Path(cast(Any, generation.instance).context.plugin_dir).resolve(
            strict=False
        )
        published_module = sys.modules[generation.module_path]
        stable_alias = None
        if previous is not None:
            stable_alias = self._stable_aliases.get(previous.module_path)
        retired_module = None
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
                stable_alias = self._stable_aliases.get(retired_module)
        if stable_alias is None:
            stable_alias = generation.module_path.rsplit("__g", 1)[0]

        # 先完成可能失败的查找，再替换 stable import alias。
        self._remove_module_tree(stable_alias)
        self._fresh_importer.register(stable_alias, plugin_dir)
        plugin_registry.register_instance(stable_alias, generation.instance)
        sys.modules[stable_alias] = published_module
        if previous is not None:
            _ = self._stable_aliases.pop(previous.module_path, None)
        if retired_module is not None:
            _ = self._stable_aliases.pop(retired_module, None)
        self._stable_aliases[generation.module_path] = stable_alias
        self._active_plugins[generation.module_path] = ActivePluginInfo(
            plugin_id=generation.plugin_id,
            plugin_dir=plugin_dir,
            manifest=generation.contributions.manifest,
            module_path=generation.module_path,
            skill_roots=generation.contributions.skill_roots,
            drift_skill_roots=generation.contributions.drift_skill_roots,
            mcp_servers=generation.contributions.mcp_servers,
        )

    async def _prepare_generation(
        self,
        generation: PluginGeneration,
    ) -> None:
        if generation.prepare_started:
            return
        from agent.plugins.context import PreparedPluginKVStore

        instance = cast(Any, generation.instance)
        context = instance.context
        staged_event_bus = ScopedEventBus(
            self._event_bus,
            generation.scope,
            staged=True,
        )
        generation.staged_event_bus = staged_event_bus
        context.event_bus = staged_event_bus
        context.kv_store = PreparedPluginKVStore(
            generation.data_dir / ".kv.json",
            can_write=lambda: _generation_can_write(generation),
            writer_id=generation.generation_id,
        )
        context._can_start_tasks = lambda: generation.state in {
            "activating",
            "active",
            "candidate",
        }
        context.scope = generation.scope
        context.tool_registry = generation.runtime_snapshot.tool_registry
        generation.prepare_started = True
        await instance.prepare()
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
        if snapshot.state == "committed":
            if self.current_snapshot is not snapshot:
                raise RuntimeError("RuntimeSnapshot 已提交指针不一致")
        elif (
            snapshot.state != "validating"
            or self._snapshot_store.pending_candidate is not snapshot
        ):
            raise RuntimeError("RuntimeSnapshot 候选事务不一致")
        catalog_id = snapshot.skill_catalog_generation_id
        if catalog_id is not None and self._skill_host.get(catalog_id) is None:
            raise RuntimeError("RuntimeSnapshot skill catalog 不可用")
        for generation_id in snapshot.mcp_catalog_generation_ids.values():
            catalog = self._mcp_host.get(generation_id)
            if catalog is None:
                raise RuntimeError("RuntimeSnapshot MCP catalog 不可用")
            if any(not server.client.connected for server in catalog.servers.values()):
                raise RuntimeError("RuntimeSnapshot MCP client 已断开")
        workspace_mcp = snapshot.workspace_mcp_generation
        if workspace_mcp is not None:
            if (
                self._mcp_host.get(workspace_mcp.generation_id)
                is not workspace_mcp.catalog
            ):
                raise RuntimeError("RuntimeSnapshot workspace MCP catalog 不可用")
            self._validate_workspace_mcp_generation(workspace_mcp)
        for item in snapshot.generations.values():
            if item.scope.closed:
                raise RuntimeError("RuntimeSnapshot 插件作用域已关闭")
            if item.scope.resource_count < item.minimum_resource_count:
                raise RuntimeError("RuntimeSnapshot 插件资源数量不足")
            if (
                item.job_catalog is not None
                and self._job_host.get(item.generation_id) is not item.job_catalog
            ):
                raise RuntimeError("RuntimeSnapshot Job catalog 不可用")
            if (
                item.proactive_catalog is not None
                and self._proactive_host.get(item.generation_id)
                is not item.proactive_catalog
            ):
                raise RuntimeError("RuntimeSnapshot proactive catalog 不可用")

    def _advance_reload(
        self,
        generation: PluginGeneration,
        phase: ReloadPhase,
        *,
        candidate_snapshot_id: str | None = None,
        error: str = "",
    ) -> None:
        tx_id = generation.reload_tx_id
        if tx_id is None:
            return
        self._reload_journal.advance(
            tx_id,
            phase,
            candidate_snapshot_id=candidate_snapshot_id,
            error=error,
        )

    def _abort_reload(
        self,
        generation: PluginGeneration,
        *,
        error: str,
    ) -> None:
        tx_id = generation.reload_tx_id
        if tx_id is None:
            return
        phase = self._reload_journal.get(tx_id).phase
        if phase in {"complete", "aborted", "recovered"}:
            return
        self._advance_reload(generation, "aborted", error=error)

    async def _abort_failed_publication(
        self,
        generation: PluginGeneration,
        transaction: SnapshotTransaction,
        *,
        error: str,
    ) -> None:
        """撤销失败发布，并留下可被启动恢复判定的持久状态。"""

        # 1. pointer 失败时保留未完成 journal，让重启按磁盘事实恢复。
        pointer_error: BaseException | None = None
        try:
            _discard_generation_candidate_pointer(generation)
        except BaseException as caught:
            pointer_error = caught

        # 2. snapshot drain 失败不能阻止已恢复 pointer 的 journal 终态。
        snapshot_error: BaseException | None = None
        try:
            _, _ = await _complete_critical(self._snapshot_store.abort(transaction))
        except BaseException as caught:
            snapshot_error = caught
        if pointer_error is None:
            self._abort_reload(generation, error=error)

        # 3. 清理异常优先暴露，避免把半完成恢复伪装成原始发布失败。
        if pointer_error is not None:
            raise RuntimeError(
                "候选发布失败后 artifact pointer 恢复失败"
            ) from pointer_error
        if snapshot_error is not None:
            raise RuntimeError(
                "候选发布失败后 RuntimeSnapshot 回收失败"
            ) from snapshot_error

    def _track_reload_drain(
        self,
        generation: PluginGeneration,
        previous_snapshot: RuntimeSnapshot | None,
    ) -> None:
        tx_id = generation.reload_tx_id
        if tx_id is None:
            return
        phase = self._reload_journal.get(tx_id).phase
        if phase == "latest_ready":
            self._advance_reload(generation, "promoting")
            phase = "promoting"
        if phase in {"commit_started", "promoting"}:
            self._advance_reload(generation, "committed")
        if previous_snapshot is None:
            self._advance_reload(generation, "complete")
            return
        snapshot_id = previous_snapshot.snapshot_id
        if snapshot_id in self._drained_before_commit:
            self._drained_before_commit.remove(snapshot_id)
            self._advance_reload(generation, "complete")
            return
        self._advance_reload(generation, "draining")
        self._drain_transactions[snapshot_id] = tx_id

    def _finish_drained_reload(self, snapshot_id: str) -> None:
        tx_id = self._drain_transactions.pop(snapshot_id, None)
        if tx_id is None:
            return
        record = self._reload_journal.get(tx_id)
        if record.phase in {"commit_started", "promoting"}:
            self._drained_before_commit.add(snapshot_id)
            return
        if record.phase == "committed":
            self._reload_journal.advance(tx_id, "draining")
            record = self._reload_journal.get(tx_id)
        if record.phase == "draining":
            self._reload_journal.advance(tx_id, "complete")

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
                self.latest_snapshot.snapshot_id
                if publication_state == "latest_ready"
                and self.latest_snapshot is not None
                else (
                    self.current_snapshot.snapshot_id
                    if self.current_snapshot is not None
                    else None
                )
            ),
            "stable_snapshot_id": (
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
                        self._workspace,
                    )
                    / "config.local.toml"
                )
            except Exception:
                source_revision = ""
                config_revision = ""
            current_prepared = self._prepared_generations.get(plugin_id)
            if force_reprepare and current_prepared is not None:
                await self.discard_prepared(plugin_id, preserve_latest=True)
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
            await self.discard_prepared(plugin_id, preserve_latest=True)
            prepared = await self._load_one(mod, activate=False)
            if prepared is None:
                _discard_installed_candidate_mod(mod)
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
                    _drift_skill_descriptions(prepared) if prepared is not None else {}
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
                    _proactive_source_keys(prepared) if prepared is not None else []
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
        plugin_manifest = load_plugin_manifest(
            _plugins_home(self._installed_cache_root)
        )
        if (
            not mod.get("package_id")
            and plugin_manifest.get(initial_plugin_id, True) is False
        ):
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
        self._generation_sequence += 1
        generation_sequence = self._generation_sequence
        module_path = mod["module_path"].strip()
        try:
            source_revision = _source_revision(plugin_dir)
        except Exception as error:
            revision = hashlib.sha256(f"{plugin_dir}:{error}".encode()).hexdigest()
            generation_id = f"{initial_plugin_id}:{revision[:12]}:{generation_sequence}"
            reload_tx_id = (
                self._begin_reload_attempt(
                    plugin_id=initial_plugin_id,
                    generation_id=generation_id,
                    source_revision=revision,
                    config_revision="",
                )
                if not activate
                else None
            )
            error_text = str(error) or type(error).__name__
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=revision,
                check_id="source_boundary",
                reason=error_text,
            )
            self._abort_reload_attempt(
                reload_tx_id,
                error=f"source_boundary: {error_text}",
            )
            return None
        data_dir = _resolve_plugin_data_dir(
            mod["name"],
            mod,
            self._workspace,
        )
        ensure_workspace_plugin_data_dir(data_dir, self._workspace)
        config_revision = _file_revision(data_dir / "config.local.toml")
        generation_id = (
            f"{initial_plugin_id}:{source_revision[:12]}:{generation_sequence}"
        )
        reload_tx_id = (
            self._begin_reload_attempt(
                plugin_id=initial_plugin_id,
                generation_id=generation_id,
                source_revision=source_revision,
                config_revision=config_revision,
            )
            if not activate
            else None
        )
        mp = (
            f"{stable_module_path}__g{generation_sequence}_"
            f"{source_revision[:8]}_{self._manager_namespace}"
        )
        if not module_path:
            error_text = f"插件缺少 plugin.py: {plugin_dir}"
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id="plugin_module",
                reason=error_text,
            )
            self._abort_reload_attempt(
                reload_tx_id,
                error=f"plugin_module: {error_text}",
            )
            raise RuntimeError(error_text)
        try:
            self._import_plugin(mp, Path(module_path))
        except Exception as error:
            logger.warning("插件 %s 导入失败: %s", mod["name"], error)
            error_text = str(error) or type(error).__name__
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id="import",
                reason=error_text,
            )
            self._abort_reload_attempt(
                reload_tx_id,
                error=f"import: {error_text}",
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
            self._abort_reload_attempt(
                reload_tx_id,
                error="plugin_class: plugin.py 未注册 Plugin 子类",
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
            error_text = str(error) or type(error).__name__
            check_id = "config" if isinstance(error, _PluginConfigError) else "identity"
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id=check_id,
                reason=error_text,
            )
            self._abort_reload_attempt(
                reload_tx_id,
                error=f"{check_id}: {error_text}",
            )
            return None
        from agent.plugins.context import PluginContext, PluginKVStore

        scope = PluginScope(plugin_id)
        instance.context = PluginContext(  # type: ignore[attr-defined]
            event_bus=None,  # type: ignore[arg-type]
            tool_registry=None,
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            data_dir=None,
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
        prepare_started = False

        async def rollback_load(error: str) -> None:
            if reload_tx_id is not None:
                phase = self._reload_journal.get(reload_tx_id).phase
                if phase not in {"complete", "aborted", "recovered"}:
                    self._reload_journal.advance(
                        reload_tx_id,
                        "aborted",
                        error=error,
                    )
            terminator = getattr(instance, "terminate", None)
            if prepare_started and callable(terminator):
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
                            error=str(terminate_error)
                            or type(terminate_error).__name__,
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
            del self._proactive_runtime_factories[
                proactive_runtime_factory_count_before:
            ]
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
                data_dir=data_dir,
                instance=instance,
                scope=scope,
                contributions=contributions,
                gate_result=gate_result,
                source_type=cast(
                    Literal["builtin", "installed"],
                    mod["source_type"],
                ),
                state="prepared",
                reload_tx_id=reload_tx_id,
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
            if not activate and _installed_generation_is_candidate(generation):
                generation.production_contributions = contributions
                generation.production_data_dir = generation.data_dir
                validation_root = (
                    self._workspace
                    / "runtime"
                    / "plugin-validation"
                    / generation.generation_id
                )
                if validation_root.exists():
                    raise RuntimeError(
                        f"候选验证目录已存在: {validation_root}"
                    )
                validation_workspace = validation_root / "workspace"
                validation_data_dir = (
                    validation_workspace
                    / "plugin-data"
                    / generation.data_dir.name
                )
                validation_data_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(generation.data_dir, validation_data_dir)
                generation.scope.defer(
                    "validation_plugin_data",
                    lambda: asyncio.to_thread(
                        _remove_validation_data_dir, validation_root
                    ),
                )
                generation.data_dir = validation_data_dir
                generation.contributions = _validation_contributions(
                    generation,
                    self._active_generations.get(plugin_id),
                    validation_workspace=validation_workspace,
                )
                contributions = generation.contributions
            if (
                not activate
                or contributions.mcp_servers
                or contributions.proactive_sources
            ):
                try:
                    mcp_catalog = await self._mcp_host.prepare(
                        generation_id,
                        server_specs=contributions.mcp_servers,
                        required_tools=_required_mcp_tools(
                            contributions.proactive_sources
                        ),
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
                    raw_readiness_checks: object = (
                        await instance.readiness_semantic_checks(
                            PluginReadinessContext(
                                generation_id=generation_id,
                                mcp_catalog=mcp_catalog,
                                job_catalog=job_catalog,
                                proactive_catalog=proactive_catalog,
                            )
                        )
                    )
                    if not isinstance(raw_readiness_checks, list):
                        raise RuntimeError("readiness_semantic_checks 返回值不是 list")
                    readiness_checks = cast(list[object], raw_readiness_checks)
                except Exception as error:
                    readiness_passed = False
                    readiness_evidence: object = str(error)
                else:
                    invalid_readiness = [
                        check
                        for check in readiness_checks
                        if not isinstance(check, PluginSemanticCheck)
                        or not check.passed
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
                    self._advance_reload(
                        generation,
                        "prepared",
                        candidate_snapshot_id=generation.runtime_snapshot.snapshot_id,
                    )
                    generation.minimum_resource_count = scope.resource_count
                    self._prepared_generations[plugin_id] = generation
                    return generation
            generation.runtime_snapshot = self._compile_generation_snapshot(generation)
            from agent.plugins.context import PreparedPluginKVStore

            load_phase = "prepare"
            prepare_started = True
            await self._prepare_generation(generation)
            instance.context.data_dir = data_dir
            instance.context.session_manager = self._session_manager
            instance.context.memory_engine = self._memory_engine
            instance.context.llm = self._llm
            generation.state = "activating"
            instance.activate()
            if isinstance(instance.context.kv_store, PreparedPluginKVStore):
                instance.context.kv_store.commit()
            load_phase = "publish"
            self._register_tools(instance, mp, tool_names)
            self._bind_tool_hooks(instance, mp)
            self._publish_contributions(contributions)
            self._channels.extend(contributions.channels)
            assert generation.staged_event_bus is not None
            generation.staged_event_bus.publish()
            generation.minimum_resource_count = scope.resource_count
        except asyncio.CancelledError:
            rollback_task = asyncio.create_task(
                rollback_load(f"candidate {load_phase} cancelled"),
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
            await rollback_load(_gate_failure_details(error.gate))
            return None
        except Exception as error:
            logger.warning("插件 %s 加载失败，回滚: %s", mod["name"], error)
            self._record_failed_gate(
                plugin_id=plugin_id,
                revision=source_revision,
                check_id=load_phase,
                reason=str(error),
            )
            await rollback_load(str(error) or type(error).__name__)
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
        if generation.mcp_catalog is not None:
            self._mcp_host.mark_active(generation.generation_id)
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
                workspace_mcp_generation=self._active_workspace_mcp,
            )
            snapshot.tool_registry = self._compile_snapshot_tools(
                generations,
                self._active_workspace_mcp,
            )
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
        workspace_mcp: WorkspaceMcpGeneration | None = None,
    ) -> Any:
        if self._tool_registry is None:
            return None
        plugin_mcp_sources = {
            ("mcp", server_name)
            for generation in generations.values()
            for server_name in generation.contributions.mcp_servers
        }
        workspace_mcp_sources: set[tuple[str, str]] = (
            {("mcp", server_name) for server_name in workspace_mcp.catalog.servers}
            if workspace_mcp is not None
            else set()
        )
        registry = self._tool_registry.fork(
            excluded_source_types={"plugin"},
            excluded_sources=plugin_mcp_sources | workspace_mcp_sources,
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
                server_spec = generation.contributions.mcp_servers[server.name]
                candidate_read_only_tools = _candidate_mcp_read_only_tools(
                    generation,
                    server.name,
                    server_spec,
                    {tool.name for tool in server.tools},
                )
                for tool in server.tools:
                    if registry.has_tool(tool.name):
                        raise RuntimeError(f"MCP 工具名称重复: {tool.name}")
                    registry.register(
                        tool,
                        risk=(
                            "read-only"
                            if tool.name in candidate_read_only_tools
                            else "external-side-effect"
                        ),
                        source_type="mcp",
                        source_name=server.name,
                    )
        if workspace_mcp is not None:
            for server in workspace_mcp.catalog.servers.values():
                for tool in server.tools:
                    if registry.has_tool(tool.name):
                        raise RuntimeError(f"workspace MCP 工具名称重复: {tool.name}")
                    registry.register(
                        tool,
                        risk="external-side-effect",
                        source_type="mcp",
                        source_name=server.name,
                    )
        return registry

    def _compile_workspace_mcp_snapshot(
        self,
        generation: WorkspaceMcpGeneration,
    ) -> RuntimeSnapshot:
        snapshot = self._snapshot_compiler.compile(
            self._active_generations,
            snapshot_revision=generation.revision,
            workspace_mcp_generation=generation,
        )
        snapshot.tool_registry = self._compile_snapshot_tools(
            self._active_generations,
            generation,
        )
        snapshot.tool_hooks = self._compile_snapshot_tool_hooks(
            self._active_generations
        )
        self._compile_snapshot_event_handlers(snapshot)
        return snapshot

    @staticmethod
    def _validate_workspace_mcp_generation(
        generation: WorkspaceMcpGeneration,
    ) -> None:
        if generation.scope.closed:
            raise RuntimeError("workspace MCP 候选作用域已关闭")
        if any(
            not server.client.connected
            for server in generation.catalog.servers.values()
        ):
            raise RuntimeError("workspace MCP 候选 client 已断开")

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
                        handler=functools.partial(
                            metadata.handler, generation.instance
                        ),
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
        cls = cast(type[Any], type(instance))
        _reject_legacy_mobile_ui_api(cls, plugin_id)
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
                raise RuntimeError(f"插件 {plugin_id}.jobs 返回值不是 PluginJobSpec")
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
                self._workspace,
                cls.mcp_servers(),
            ),
            managed_services=_resolve_managed_services(
                plugin_dir,
                data_dir,
                self._workspace,
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
            after_step_modules=tuple(_load_module_list(instance, "after_step_modules")),
            after_reasoning_modules=tuple(
                _load_module_list(instance, "after_reasoning_modules")
            ),
            after_turn_modules=tuple(_load_module_list(instance, "after_turn_modules")),
            proactive_modules=tuple(_load_module_list(instance, "proactive_modules")),
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
            mobile_ui_asset=_resolve_mobile_ui_asset(
                plugin_dir,
                cls.mobile_ui(),
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
        other_generations.extend(
            generation
            for prepared_id, generation in self._prepared_generations.items()
            if prepared_id != plugin_id
        )

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
            getattr(instance, "api_version", None) == 2,
            getattr(instance, "api_version", None),
        )
        lifecycle_type = type(instance)
        legacy_lifecycle = [
            name for name in ("initialize",) if name in lifecycle_type.__dict__
        ]
        check(
            "lifecycle_api",
            not legacy_lifecycle
            and inspect.iscoroutinefunction(instance.prepare)
            and not inspect.iscoroutinefunction(instance.activate)
            and not inspect.iscoroutinefunction(instance.retire)
            and inspect.iscoroutinefunction(instance.terminate),
            {"legacy": legacy_lifecycle},
        )
        metadata = plugin_registry.get_handlers_by_module_path(
            type(instance).__module__
        )
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
        if self._active_workspace_mcp is not None:
            occupied_servers.update(self._active_workspace_mcp.catalog.servers)
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
                (
                    all(channel_names)
                    and not _duplicates(channel_names)
                    and not occupied_channels.intersection(channel_names)
                )
                if channel_names
                else True
            ),
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
            if imported_name == module_name or imported_name.startswith(
                f"{module_name}."
            ):
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

        from agent.plugins.context import allow_plugin_cleanup_writes

        # 1. 先关闭当前 generation admission，再完成快照回收
        for generation in self._active_generations.values():
            self._retire_generation(generation)
        _, externally_cancelled = await _complete_critical(self._snapshot_store.close())
        self._ready_candidate = None
        for plugin_id in tuple(self._prepared_generations):
            _, cancelled = await _complete_critical(self.discard_prepared(plugin_id))
            externally_cancelled = externally_cancelled or cancelled
        if self._prepared_workspace_mcp is not None:
            _, cancelled = await _complete_critical(
                self._discard_workspace_mcp_candidate()
            )
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
                    generation = (
                        None
                        if active_info is None
                        else self._active_generations.get(active_info.plugin_id)
                    )
                    writer_id = "" if generation is None else generation.generation_id
                    with allow_plugin_cleanup_writes(writer_id):
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
                generation = (
                    None
                    if active_info is None
                    else self._active_generations.get(active_info.plugin_id)
                )
                writer_id = "" if generation is None else generation.generation_id
                with allow_plugin_cleanup_writes(writer_id):
                    cleanup_failures, cancelled = await _complete_critical(
                        scope.aclose()
                    )
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
        self._draining_generations.clear()
        self._prepared_generations.clear()
        self._active_workspace_mcp = None
        self._prepared_workspace_mcp = None
        self._stable_aliases.clear()
        if externally_cancelled:
            raise asyncio.CancelledError


class _PluginConfigError(Exception):
    pass


class _CandidateRejected(Exception):
    def __init__(self, gate: GateResult) -> None:
        super().__init__(gate.failure_reason)
        self.gate = gate


def _gate_failure_details(gate: GateResult) -> str:
    """把失败 Gate 的 check 与证据压成可持久诊断文本。"""
    return (
        "; ".join(
            f"{check.check_id}: {check.evidence}"
            for check in gate.checks
            if check.status == "failed"
        )
        or gate.failure_reason
    )


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
        if not isinstance(config_model, type) or not issubclass(
            config_model, BaseModel
        ):
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
    workspace: Path,
) -> Path:
    """把插件可写数据固定到当前 workspace 的独立目录。"""

    # 1. 交给统一路径边界校验插件身份
    marketplace = mod.get("marketplace", "").strip()
    suffix = marketplace or "builtin"
    return workspace_plugin_data_dir(workspace, name, suffix)


def _plugins_home(installed_cache_root: Path | None) -> Path:
    if installed_cache_root is not None:
        return installed_cache_root.parent
    return plugins_root()


def _installed_artifact_base(generation: PluginGeneration) -> Path | None:
    if generation.source_type != "installed":
        return None
    plugin_dir = Path(cast(Any, generation.instance).context.plugin_dir)
    plugin_base = (
        plugin_dir.parent.parent
        if plugin_dir.parent.name == ".artifacts"
        else plugin_dir.parent
    )
    state_path = pointer_state_path(plugin_base)
    if not state_path.exists() and not state_path.is_symlink():
        return None
    return plugin_base


def _installed_generation_is_candidate(generation: PluginGeneration) -> bool:
    """Return whether this installed generation is the explicit latest pointer."""

    if generation.source_type != "installed":
        return False
    plugin_dir = Path(cast(Any, generation.instance).context.plugin_dir)
    return _installed_candidate_base_from_root(plugin_dir) is not None


def _installed_candidate_base(generation: PluginGeneration) -> Path | None:
    if generation.source_type != "installed":
        return None
    plugin_dir = Path(cast(Any, generation.instance).context.plugin_dir)
    return _installed_candidate_base_from_root(plugin_dir)


def _discard_generation_candidate_pointer(generation: PluginGeneration) -> None:
    plugin_base = _installed_candidate_base(generation)
    if plugin_base is not None:
        _ = discard_latest_pointer(plugin_base)


def _installed_candidate_base_from_root(plugin_dir: Path) -> Path | None:
    """Resolve the candidate pointer owner for an exact installed latest root."""

    # 1. Legacy installs and stable==latest keep immediate publish compatibility.
    plugin_base = _installed_artifact_base_from_root(plugin_dir)
    stable = read_pointer(plugin_base, "stable")
    latest = read_pointer(plugin_base, "latest")
    if stable is None and latest is None:
        return None
    if stable is None or latest is None:
        raise RuntimeError(f"插件 artifact pointer 必须成对存在: {plugin_base}")
    if stable == latest:
        return None

    # 2. Candidate operations must own the exact durable latest root.
    latest_root = resolve_pointer(plugin_base, latest)
    if latest_root is None or plugin_dir.resolve() != latest_root.resolve():
        raise RuntimeError(f"插件 generation 与 latest pointer 不一致: {plugin_dir}")
    return plugin_base


def _switch_ready_pointer(
    ready: _ReadyPluginCandidate,
    plugin_base: Path,
) -> None:
    """在候选仍拥有磁盘 pointer 时原子提升它。"""

    previous, candidate = _ready_artifact_pointers(ready, plugin_base)
    pointers = read_pointers(plugin_base)
    if pointers is None or (pointers.stable, pointers.latest) not in {
        (previous, candidate),
        (candidate, candidate),
    }:
        raise RuntimeError(f"插件 artifact pointer 已被其他发布改变: {plugin_base}")
    _ = write_pointers(plugin_base, stable=candidate, latest=candidate)


def _restore_ready_pointer(
    ready: _ReadyPluginCandidate,
    plugin_base: Path,
) -> None:
    """把 ready candidate 的完整指针对恢复到先前 stable。"""

    previous, candidate = _ready_artifact_pointers(ready, plugin_base)
    pointers = read_pointers(plugin_base)
    if pointers is None or (pointers.stable, pointers.latest) not in {
        (previous, candidate),
        (candidate, candidate),
        (previous, previous),
    }:
        raise RuntimeError(f"插件 artifact pointer 已被其他发布改变: {plugin_base}")
    _ = write_pointers(plugin_base, stable=previous, latest=previous)


def _ready_artifact_pointers(
    ready: _ReadyPluginCandidate,
    plugin_base: Path,
) -> tuple[ArtifactPointer, ArtifactPointer]:
    """解析 ready candidate 事务拥有的前后 artifact pointer。"""

    candidate_root = Path(cast(Any, ready.candidate.instance).context.plugin_dir)
    candidate = relative_artifact_pointer(plugin_base, candidate_root)
    if ready.previous is None:
        return ArtifactPointer(None), candidate
    previous_root = Path(cast(Any, ready.previous.instance).context.plugin_dir)
    previous_base = _installed_artifact_base_from_root(previous_root)
    if previous_base.resolve() != plugin_base.resolve():
        raise RuntimeError("latest candidate 与 stable 不属于同一插件 artifact")
    return relative_artifact_pointer(plugin_base, previous_root), candidate


def _discard_installed_candidate_mod(mod: dict[str, str]) -> None:
    if mod.get("source_type") != "installed":
        return
    plugin_base = _installed_candidate_base_from_root(Path(mod["plugin_root"]))
    if plugin_base is not None:
        _ = discard_latest_pointer(plugin_base)


def _installed_artifact_base_from_root(plugin_dir: Path) -> Path:
    return (
        plugin_dir.parent.parent
        if plugin_dir.parent.name == ".artifacts"
        else plugin_dir.parent
    )


def _mod_source_revision(mod: dict[str, str] | None) -> str | None:
    if mod is None:
        return None
    return _source_revision(Path(mod["plugin_root"]))


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


def _reject_legacy_mobile_ui_api(cls: type[Any], plugin_id: str) -> None:
    """拒绝已移除的移动 UI v1 声明，避免插件被静默降级。"""

    legacy_methods = tuple(
        name
        for name in (
            "mobile_ui_module",
            "mobile_ui_stylesheet",
            "mobile_ui_call",
        )
        if inspect.getattr_static(cls, name, None) is not None
    )
    if legacy_methods:
        raise RuntimeError(
            f"插件 {plugin_id} 使用已移除的 Mobile UI v1 API: "
            f"{', '.join(legacy_methods)}；请迁移到 mobile_ui 和 mobile_ui_query"
        )


def _resolve_mobile_ui_asset(
    plugin_dir: Path,
    declared: MobileUiContribution | None,
) -> MobileUiAsset | None:
    """在插件激活边界固化并校验移动 UI 资产。"""

    if declared is None:
        return None
    if not isinstance(declared, MobileUiContribution):
        raise RuntimeError("插件 mobile UI 声明必须是 MobileUiContribution")
    root = plugin_dir.resolve(strict=False)
    module_path = (plugin_dir / declared.module).resolve(strict=False)
    if (
        not module_path.is_relative_to(root)
        or module_path.suffix != ".js"
        or not module_path.is_file()
    ):
        raise RuntimeError(f"插件 mobile UI module 无效: {declared.module}")
    allowed_slots = {
        "turn.before_reasoning",
        "turn.before_tool",
        "turn.after_answer",
        "drawer.panel",
    }
    if len(set(declared.slots)) != len(declared.slots) or any(
        slot not in allowed_slots for slot in declared.slots
    ):
        raise RuntimeError("插件 mobile UI slots 无效")
    navigation = declared.navigation
    if navigation is not None and (
        not navigation.label.strip()
        or len(navigation.label) > 64
        or not navigation.description.strip()
        or len(navigation.description) > 160
    ):
        raise RuntimeError("插件 mobile UI navigation 无效")
    stylesheet = ""
    if declared.stylesheet is not None:
        stylesheet_path = (plugin_dir / declared.stylesheet).resolve(strict=False)
        if (
            not stylesheet_path.is_relative_to(root)
            or stylesheet_path.suffix != ".css"
            or not stylesheet_path.is_file()
        ):
            raise RuntimeError(f"插件 mobile UI stylesheet 无效: {declared.stylesheet}")
        stylesheet = stylesheet_path.read_text(encoding="utf-8")
    module = module_path.read_text(encoding="utf-8")
    module_encoded = module.encode("utf-8")
    stylesheet_encoded = stylesheet.encode("utf-8")
    if len(module_encoded) + len(stylesheet_encoded) > 240 * 1024:
        raise RuntimeError("插件 mobile UI 资产超过协议安全预算")
    return MobileUiAsset(
        module=module,
        module_sha256=hashlib.sha256(module_encoded).hexdigest(),
        module_bytes=len(module_encoded),
        stylesheet=stylesheet,
        stylesheet_sha256=(
            hashlib.sha256(stylesheet_encoded).hexdigest() if stylesheet else None
        ),
        stylesheet_bytes=len(stylesheet_encoded),
        navigation_label=None if navigation is None else navigation.label.strip(),
        navigation_description=(
            None if navigation is None else navigation.description.strip()
        ),
        slots=tuple(declared.slots),
    )


def _resolve_managed_services(
    plugin_dir: Path,
    data_dir: Path,
    workspace: Path,
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
            or not isinstance(spec.validation_port_env, str)
            or (
                spec.validation_port_env
                and not spec.validation_port_env.replace("_", "A").isalnum()
            )
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
            "env": {
                **spec.env,
                "AKA_PLUGIN_DATA_DIR": str(data_dir),
                "AKASHIC_WORKSPACE": str(workspace),
            },
            "readiness_url": spec.readiness_url,
            "startup_timeout_seconds": spec.startup_timeout_seconds,
            "revision": source_revision,
            "validation_port_env": spec.validation_port_env,
        }
    return services


def _resolve_mcp_servers(
    plugin_dir: Path,
    data_dir: Path,
    workspace: Path,
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
        if (
            not isinstance(spec.candidate_read_only_tools, tuple)
            or not all(
                isinstance(value, str) and value
                for value in spec.candidate_read_only_tools
            )
            or len(set(spec.candidate_read_only_tools))
            != len(spec.candidate_read_only_tools)
        ):
            raise RuntimeError(f"插件 MCP candidate 只读工具声明无效: {spec.name}")
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
        env = {
            **spec.env,
            "AKA_PLUGIN_DATA_DIR": str(data_dir),
            "AKASHIC_WORKSPACE": str(workspace),
        }
        if _is_python_command(command[0]):
            runtime_root = _resolve_mcp_runtime_root(plugin_dir, cwd, command)
            if runtime_root is not None:
                venv_python = _venv_python(runtime_root / ".venv")
                if venv_python.exists():
                    command[0] = str(venv_python)
        servers[spec.name] = {
            "command": command,
            "env": env,
            "cwd": cwd,
            "candidate_read_only_tools": spec.candidate_read_only_tools,
        }
    return servers


def _validation_contributions(
    candidate: PluginGeneration,
    previous: PluginGeneration | None,
    *,
    validation_workspace: Path,
) -> PluginContributions:
    """构造候选路径隔离视图；同 UID 进程仍可绕过它，它不是安全沙箱。"""

    # 1. Allocate one loopback port for every changed managed service.
    production = candidate.contributions
    old_services = (
        previous.contributions.managed_services if previous is not None else {}
    )
    validation_services: dict[str, dict[str, Any]] = {}
    validation_env: dict[str, str] = {}
    for service_id, spec in production.managed_services.items():
        if old_services.get(service_id) == spec:
            continue
        port_env = str(spec.get("validation_port_env") or "")
        readiness_url = str(spec.get("readiness_url") or "")
        if not port_env or not readiness_url:
            raise RuntimeError(
                "候选插件改变独占 managed service，但未声明通用隔离端口: "
                f"plugin={candidate.plugin_id} service={service_id} "
                "需要 ManagedServiceSpec.validation_port_env 和 readiness_url"
            )
        port = _allocate_validation_port()
        validation_env[port_env] = str(port)
        isolated = dict(spec)
        isolated["env"] = {
            **dict(spec.get("env") or {}),
            port_env: str(port),
            "AKA_PLUGIN_DATA_DIR": str(candidate.data_dir),
            "AKASHIC_WORKSPACE": str(validation_workspace),
        }
        isolated["readiness_url"] = _replace_url_port(readiness_url, port)
        validation_services[service_id] = isolated

    # 2. Candidate MCP processes inherit only declared endpoint variables.
    mcp_servers = {
        name: {
            **spec,
            "env": {
                **dict(spec.get("env") or {}),
                **validation_env,
                "AKA_PLUGIN_DATA_DIR": str(candidate.data_dir),
                "AKASHIC_WORKSPACE": str(validation_workspace),
            },
        }
        for name, spec in production.mcp_servers.items()
    }
    candidate.validation_managed_services = validation_services
    return replace(
        production,
        mcp_servers=mcp_servers,
        managed_services=validation_services,
        channels=(previous.contributions.channels if previous is not None else ()),
    )


def _candidate_mcp_read_only_tools(
    generation: PluginGeneration,
    server_name: str,
    server_spec: dict[str, Any],
    available_tools: set[str],
) -> frozenset[str]:
    """严格校验并返回只对候选开放的 MCP 只读工具集合。"""

    # 1. 正式 snapshot 不继承候选验证专用的只读声明。
    if (
        generation.production_data_dir is None
        or generation.data_dir == generation.production_data_dir
    ):
        return frozenset()

    # 2. 候选声明精确匹配且默认拒绝；未知工具直接失败。
    raw_names = server_spec.get("candidate_read_only_tools", ())
    if not isinstance(raw_names, tuple) or not all(
        isinstance(name, str) and name for name in raw_names
    ):
        raise RuntimeError(
            f"MCP candidate 只读工具声明无效: server={server_name}"
        )
    declared = frozenset(raw_names)
    public_names = frozenset(
        f"mcp_{server_name}__{remote_name}" for remote_name in declared
    )
    unknown = public_names.difference(available_tools)
    if unknown:
        raise RuntimeError(
            "MCP candidate 只读工具不存在: "
            f"server={server_name} tools={', '.join(sorted(declared))}"
        )
    return public_names


def _allocate_validation_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


def _remove_validation_data_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _replace_url_port(url: str, port: int) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise RuntimeError(f"managed service readiness_url 无效: {url}")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return urlunsplit(
        (parsed.scheme, f"{host}:{port}", parsed.path, parsed.query, parsed.fragment)
    )


def _replace_snapshot_payload(
    target: RuntimeSnapshot,
    source: RuntimeSnapshot,
) -> None:
    """Refresh an unleased candidate while preserving store-owned lifecycle fields."""

    if target.lease_count or target.state != "committed":
        raise RuntimeError("只能刷新无 lease 的 committed candidate snapshot")
    for name in (
        "generations",
        "before_turn_modules",
        "before_reasoning_modules",
        "prompt_render_modules",
        "before_step_modules",
        "after_step_modules",
        "after_reasoning_modules",
        "after_turn_modules",
        "jobs",
        "proactive_sources",
        "proactive_modules",
        "proactive_lifecycles",
        "proactive_module_factories",
        "proactive_runtime_factories",
        "tool_hooks",
        "channels",
        "skill_catalog_generation_id",
        "mcp_catalog_generation_ids",
        "workspace_mcp_generation",
        "managed_services",
        "dashboard_bindings",
        "tool_registry",
        "plugin_skill_index",
        "event_handlers",
    ):
        setattr(target, name, getattr(source, name))


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
        directories[:] = sorted(name for name in directories if name not in excluded)
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
        directories[:] = sorted(name for name in directories if name not in excluded)
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


def _required_mcp_tools(
    sources: tuple[RegisteredProactiveSource, ...],
) -> dict[str, tuple[str, ...]]:
    required: dict[str, list[str]] = {}
    for source in sources:
        names = required.setdefault(source.spec.server, [])
        names.append(source.spec.fetch_tool)
        if source.spec.ack_tool:
            names.append(source.spec.ack_tool)
    return {
        server_name: tuple(tool_names) for server_name, tool_names in required.items()
    }


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
