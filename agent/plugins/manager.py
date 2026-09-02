from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import secrets
import shutil
import sqlite3
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType, UnionType
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, TypeVar, Union, cast, get_args, get_origin

from pydantic import AliasChoices, AliasPath, BaseModel, ValidationError

from agent.plugin_composition import (
    CHANNELS,
    COMMANDS,
    INTERACTION_UNDO,
    CompositionError,
    MANAGED_PROCESSES,
    WORKLOADS,
    MCP_SERVERS,
    SESSION_READ,
    SESSION_COMPACTION_STORAGE,
    SCOPED_TURNS,
    CONTINUATIONS,
    DELIVERIES,
    DURABLE_DELIVERIES,
    TIMERS,
    BACKGROUND_JOBS,
    TOOL_CATALOG,
    UI_SLOTS,
    CompositionOverlay,
    CompositionRoot,
    CompositionSnapshotRoot,
    CredentialRef,
    FiberState,
    PluginChannels,
    PluginUiSlots,
    PluginCommands,
    InteractionUndoService,
    PluginBackgroundJobs,
    PluginToolBinding,
    PluginToolCatalog,
    PluginTools,
    PluginRuntime,
    SessionReadService,
    SessionCompactionStorage,
    PluginScopedTurns,
    PluginContinuations,
    PluginDeliveries,
    PluginDurableDeliveries,
    PluginTimers,
    ServiceView,
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    SNAPSHOT_SEALING,
    RuntimeStarted,
    RuntimeStopping,
    SnapshotSealing,
)
from agent.plugin_composition.channels import (
    CoreChannelDefinition,
    ChannelRegistrySnapshot,
    CredentialRef,
    ProviderClient,
    ProviderClientFactory,
)
from agent.plugin_composition.mcp_slots import PluginMcpServers
from agent.plugin_composition.process_slots import PluginManagedProcesses
from agent.plugin_composition.workload_slots import PluginWorkloads
from agent.plugin_composition.commands import command_discovery_catalog
from agent.plugin_composition.model import (
    resolve_declared_workspace_file,
    resolve_declared_workspace_root,
)
from agent.control.timer import AsyncioOneShotTimer
from agent.plugin_composition.durable_deliveries import (
    DurableProjector,
    DurableDeliveryRequest,
    DurableSender,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from bus.events import ChannelMessage
from agent.plugin_composition.channels import ChannelDeliveryReceipt
from agent.plugins.composable import ComposablePlugin
from agent.plugins.interaction_undo import InteractionUndoCoordinator
from agent.plugins.composition_generation_host import (
    CompositionGenerationHost,
    CompositionRuntimeFailure,
    CompositionRuntimeGeneration,
)
from agent.plugins.channel_generation_host import (
    ChannelCleanupTombstone,
    ChannelGeneration,
    ChannelGenerationHost,
    ChannelStartRecord,
)
from agent.plugins.channel_credentials import CoreProviderClientFactory

from agent.plugins.manifest import (
    ensure_workspace_plugin_data_dir,
    load_plugin_manifest,
    plugins_root,
    validate_workspace_plugin_data_path,
    workspace_plugin_data_dir,
    write_plugin_manifest,
)
from infra.channels.base import SessionIdentityIndex
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from session.store import ChannelIdentityWriteReceipt
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
from agent.plugins.scope import CleanupFailure, PluginScope
from agent.plugins.generation import (
    GateCheckResult,
    GateResult,
    PluginContributions,
    PluginGeneration,
    PluginSemanticCheck,
)
from agent.plugins.importer import FreshPluginImporter
from agent.plugins.install import PluginInstallResult, install_git_plugin
from agent.plugins.static_manifest import (
    StaticPluginManifest,
    load_static_plugin_manifest,
    materialize_static_command,
    staged_python_interpreter,
    validate_module_exports,
)
from agent.plugins.reload_journal import (
    RecoveryActionName,
    RecoveryTarget,
    ReloadJournal,
    ReloadPhase,
    ReloadRecoveryAction,
)
from agent.plugins.skill_host import PluginSkillHost
from agent.plugins.web_ui import resolve_web_module
from agent.workloads.client import UnixWorkloadController, WorkloadController
from agent.plugins.generation_activity_host import (
    ActivityCatalog,
    ActivityHost,
    ActivityTransaction,
)
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotLease,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    SnapshotTransaction,
    get_current_runtime_snapshot,
)
from bus.event_bus import EventBus
from infra.persistence.json_store import atomic_save_json

logger = logging.getLogger(__name__)
U = TypeVar("U")


def _snapshot_command_catalog(
    snapshot: RuntimeSnapshot | None,
) -> tuple[tuple[str, str], ...]:
    registry = None if snapshot is None else snapshot.command_registry
    return command_discovery_catalog(registry)


class _NoopProviderClient:
    def credential(self, ref: CredentialRef) -> str:
        raise RuntimeError(f"Core channel 未声明 credential: {'.'.join(ref.path)}")

    async def aclose(self) -> None:
        return None


class _NoopProviderClientFactory:
    async def create(
        self,
        credentials: Mapping[str, CredentialRef],
    ) -> ProviderClient:
        if credentials:
            raise RuntimeError("Core channel 不得解析未声明的 provider credential")
        return _NoopProviderClient()

    async def aclose(self) -> None:
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


@dataclass(frozen=True)
class ActivePluginInfo:
    plugin_id: str
    plugin_dir: Path
    manifest: dict[str, object]
    module_path: str
    skill_roots: tuple[Path, ...] = ()
    drift_skill_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _ReadyPluginCandidate:
    plugin_id: str
    previous: PluginGeneration | None
    candidate: PluginGeneration
    snapshot: RuntimeSnapshot


class _PublicationParticipantSwitchError(RuntimeError):
    """Report a forward participant switch rejected before publication opened."""


class _PublicationParticipantRestoreError(RuntimeError):
    """Keep the old snapshot closed when an external owner cannot be restored."""

    def __init__(self, message: str, *, resources: tuple[str, ...]) -> None:
        super().__init__(message)
        self.resources = resources


@dataclass
class _ChannelPublicationState:
    previous: RuntimeSnapshot | None
    candidate: RuntimeSnapshot
    previous_identity: str | None
    candidate_identity: str | None
    old_runtime: ChannelGeneration | None
    old_factories: Mapping[str, ProviderClientFactory]
    new_factories: Mapping[str, ProviderClientFactory]
    changed: bool
    old_closed: bool = False
    old_stopped: bool = False
    new_runtime: ChannelGeneration | None = None


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
        installed_cache_root: Path | None = None,
        channel_attachment_store: ChannelAttachmentArtifactStore | None = None,
        disabled_builtin_plugins: frozenset[str] = frozenset(),
        workload_controller: WorkloadController | None = None,
    ) -> None:
        self._dirs = plugin_dirs
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._workspace = workspace
        self._session_manager = session_manager
        self._interaction_undo = (
            InteractionUndoCoordinator(session_manager)
            if session_manager is not None
            else None
        )
        self._conversation_runtime: object | None = None
        self._programmatic_session_creator: Callable[..., object] | None = None
        self._programmatic_session_reader: Callable[[str], object] | None = None
        self._installed_cache_root = installed_cache_root
        self._disabled_builtin_plugins = disabled_builtin_plugins
        self._dashboard_preparer: Callable[[RuntimeSnapshot], None] | None = None
        self._dashboard_validation_releaser: (
            Callable[[RuntimeSnapshot], Awaitable[None]] | None
        ) = None
        self._endpoint_quiescer: Callable[[], Awaitable[None]] | None = None
        self._endpoint_resumer: Callable[[], Awaitable[None]] | None = None
        self._endpoint_switcher: (
            Callable[
                [
                    tuple[tuple[str, str], ...],
                    tuple[tuple[str, str], ...],
                ],
                Awaitable[None],
            ]
            | None
        ) = None
        self._loaded: set[str] = set()
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
        self._composition_pending: tuple[str, ...] = ()
        self._candidate_prepare_lock = asyncio.Lock()
        self._fresh_importer = FreshPluginImporter()
        self._manager_namespace = secrets.token_hex(4)
        self._skill_host = PluginSkillHost(workspace)
        self._composition_runtime_generations: dict[str, PluginGeneration] = {}
        if workload_controller is None:
            workload_socket = os.environ.get("AKASHIC_WORKLOAD_SOCKET", "").strip()
            if workload_socket:
                workload_controller = UnixWorkloadController(Path(workload_socket))
        workload_workspace_id = hashlib.sha256(
            str(workspace.resolve(strict=False)).encode("utf-8")
        ).hexdigest()[:16]
        self._composition_generation_host = CompositionGenerationHost(
            workload_controller=workload_controller,
            workspace_id=(
                workload_workspace_id if workload_controller is not None else None
            ),
            on_failure=self._on_composition_runtime_failure,
        )
        self._snapshot_compiler = RuntimeSnapshotCompiler()
        self._snapshot_store = RuntimeSnapshotStore(self._on_snapshot_drained)
        self._runtime_started_roots: set[object] = set()
        self._runtime_lifecycle_lock = asyncio.Lock()
        self._runtime_services_enabled = False
        self._snapshot_skill_catalogs: dict[str, str] = {}
        self._reload_journal = ReloadJournal(workspace)
        self._channel_provider_factory_resolver: (
            Callable[
                [RuntimeSnapshot],
                Mapping[str, ProviderClientFactory],
            ]
            | None
        ) = self._default_channel_provider_factories
        self._channel_identity_indexes: dict[str, SessionIdentityIndex] = {}
        self._channel_generation_host = ChannelGenerationHost(
            on_before_start=self._reserve_channel_binding,
            config_revision_checker=self._check_channel_config_revision,
            on_failure=self._on_channel_cleanup_failure,
            snapshot_lease_acquirer=self._snapshot_store.lease,
            identity_resolver=self._resolve_channel_identity,
            identity_rememberer=self._remember_channel_identity,
            identity_rollbacker=self._rollback_channel_identity,
            attachment_import=channel_attachment_store,
            attachment_read=channel_attachment_store,
        )
        self._active_channel_generation: ChannelGeneration | None = None
        self._active_channel_catalog_identity: str | None = None
        self._core_channel_definitions: tuple[CoreChannelDefinition, ...] = ()
        self._channel_boot_transactions: set[str] = set()
        self._activity_host: ActivityHost | None = None
        self._drain_transactions: dict[str, str] = {}
        self._drained_before_commit: set[str] = set()
        self._event_bus.bind_runtime_snapshot_store(self._snapshot_store)
        self._continuation_publisher: Callable[[Any], Awaitable[None]] | None = None
        self._delivery_sender: (
            Callable[[ChannelMessage], Awaitable[ChannelDeliveryReceipt]] | None
        ) = None
        self._durable_delivery_sender: DurableSender | None = None
        self._durable_delivery_recovered = False

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    def bind_conversation_runtime(
        self,
        runtime: object,
        *,
        programmatic_session_creator: Callable[..., object],
        programmatic_session_reader: Callable[[str], object] | None = None,
    ) -> None:
        """Bind formal scoped Turn admission before plugin topology is loaded."""

        if self._conversation_runtime is not None:
            raise RuntimeError("PluginManager ConversationRuntime 已绑定")
        self._conversation_runtime = runtime
        self._programmatic_session_creator = programmatic_session_creator
        self._programmatic_session_reader = programmatic_session_reader

    def bind_continuation_publisher(
        self,
        publisher: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Bind the narrow internal Message publisher before loading plugins."""

        if self._continuation_publisher is not None:
            raise RuntimeError("PluginManager continuation publisher 已绑定")
        self._continuation_publisher = publisher

    def bind_delivery_sender(
        self,
        sender: Callable[[ChannelMessage], Awaitable[ChannelDeliveryReceipt]],
    ) -> None:
        """Bind the narrow committed Channel sender before loading plugins."""

        if self._delivery_sender is not None:
            raise RuntimeError("PluginManager delivery sender 已绑定")
        self._delivery_sender = sender

    def bind_durable_delivery_sender(self, sender: DurableSender) -> None:
        """Bind the two-stage provider boundary before loading durable consumers."""

        if self._durable_delivery_sender is not None:
            raise RuntimeError("PluginManager durable delivery sender 已绑定")
        self._durable_delivery_sender = sender

    async def run_runtime_services(self) -> None:
        """Follow stable Roots without retaining Turn admission across reloads."""

        if self._runtime_services_enabled:
            raise RuntimeError("plugin runtime services 已启动")
        self._runtime_services_enabled = True
        snapshot: RuntimeSnapshot | None = None
        try:
            while True:
                async with self._candidate_prepare_lock:
                    lease = await self._snapshot_store.acquire()
                    snapshot = lease.snapshot
                    try:
                        await self._start_runtime_snapshot(snapshot)
                    finally:
                        await lease.release()
                _ = await self._snapshot_store.wait_for_stable_change(snapshot)
                await self._snapshot_store.wait_for_snapshot_drained(snapshot)
        finally:
            self._runtime_services_enabled = False
            if snapshot is not None:
                _ = await _complete_critical(self._stop_runtime_snapshot(snapshot))

    async def _start_runtime_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        """Start one exact Root once without retaining an admission lease."""

        async with self._runtime_lifecycle_lock:
            if snapshot is not self.current_snapshot or not snapshot.accepting_leases:
                return
            root = snapshot.composition_root
            if root is None or root.instance_token in self._runtime_started_roots:
                return
            result, cancelled = await _complete_critical(
                root.context.serial(RUNTIME_STARTED, RuntimeStarted())
            )
            if result is not None:
                raise CompositionError(
                    "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED",
                    "runtime.started 接入点不接受 Bail",
                )
            self._runtime_started_roots.add(root.instance_token)
        if cancelled:
            raise asyncio.CancelledError

    async def _start_current_runtime_snapshot(self) -> None:
        """Start lifecycle only after the exact Root is public and leasable."""

        snapshot = self.current_snapshot
        if snapshot is None:
            return
        if not snapshot.accepting_leases:
            raise RuntimeError("current RuntimeSnapshot 尚未开放")
        await self._start_runtime_snapshot(snapshot)

    async def _stop_runtime_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        """Settle one started Root once before its effects are disposed."""

        async with self._runtime_lifecycle_lock:
            root = snapshot.composition_root
            if root is None or root.instance_token not in self._runtime_started_roots:
                return
            result, cancelled = await _complete_critical(
                root.context.serial(RUNTIME_STOPPING, RuntimeStopping())
            )
            if result is not None:
                raise CompositionError(
                    "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED",
                    "runtime.stopping 接入点不接受 Bail",
                )
            self._runtime_started_roots.remove(root.instance_token)
        if cancelled:
            raise asyncio.CancelledError

    @property
    def skill_projection_roots(self) -> list[Path]:
        roots = list(self._dirs)
        if self._installed_cache_root is not None:
            roots.append(self._installed_cache_root)
        return roots

    def _sync_skill_links(self):
        """Rebuild workspace links from the active v3 generations."""

        from agent.plugins.skill_links import PluginSkillLinker

        return PluginSkillLinker(
            workspace=self._workspace,
            plugin_roots=self.skill_projection_roots,
        ).sync(self.active_plugins())

    def _prepare_skill_links_for_promotion(
        self,
        generation: PluginGeneration,
        candidate_snapshot: RuntimeSnapshot,
    ) -> tuple[Any, list[ActivePluginInfo], list[ActivePluginInfo]]:
        """Build and validate both sides of the stable skill projection switch."""

        from agent.plugins.skill_links import PluginSkillLinker

        contributions = generation.production_contributions or generation.contributions
        plugin_dir = generation.plugin_dir.resolve(strict=False)
        target = ActivePluginInfo(
            plugin_id=generation.plugin_id,
            plugin_dir=plugin_dir,
            manifest=contributions.manifest,
            module_path=generation.module_path,
            skill_roots=contributions.skill_roots,
            drift_skill_roots=contributions.drift_skill_roots,
        )
        stable = self.active_plugins()
        post_promotion = [
            plugin for plugin in stable if plugin.plugin_id != generation.plugin_id
        ]
        if any(item is generation for item in candidate_snapshot.active_generations()):
            post_promotion.append(target)
        linker = PluginSkillLinker(
            workspace=self._workspace,
            plugin_roots=self.skill_projection_roots,
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

    def workload_urls(self, generation_id: str) -> Mapping[tuple[str, str], str]:
        """Return ready workload URLs for one exact plugin generation."""

        return self._composition_generation_host.workload_urls(generation_id)

    def bind_dashboard_preparer(
        self,
        preparer: Callable[[RuntimeSnapshot], None],
        *,
        validation_releaser: Callable[[RuntimeSnapshot], Awaitable[None]],
    ) -> None:
        self._dashboard_preparer = preparer
        self._dashboard_validation_releaser = validation_releaser

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
                tuple[tuple[str, str], ...],
                tuple[tuple[str, str], ...],
            ],
            Awaitable[None],
        ],
    ) -> None:
        self._endpoint_switcher = switcher

    def bind_channel_provider_factory_resolver(
        self,
        resolver: Callable[
            [RuntimeSnapshot],
            Mapping[str, ProviderClientFactory],
        ],
    ) -> None:
        """Bind Core's formal-only provider factory projection."""

        if not callable(resolver):
            raise TypeError("channel provider factory resolver 必须可调用")
        self._channel_provider_factory_resolver = resolver

    def bind_activity_host(self, host: ActivityHost) -> None:
        """Bind the single Core owner for background activity."""

        if self._activity_host is not None:
            raise RuntimeError("ActivityHost 已绑定")
        self._activity_host = host

    @staticmethod
    def _activity_catalog_identity(snapshot: RuntimeSnapshot | None) -> str | None:
        if snapshot is None:
            return None
        jobs = snapshot.background_job_catalog
        if jobs is None:
            return None
        descriptors = jobs.descriptors
        owners = sorted({descriptor.owner for descriptor in descriptors})
        bindings: list[str] = []
        for owner in owners:
            generation = snapshot.generations.get(owner)
            if generation is None:
                raise RuntimeError(f"Activity catalog owner generation 缺失: {owner}")
            bindings.append(
                f"{owner}:{generation.generation_id}:{generation.source_revision}"
            )
        return "|".join(
            (
                "jobs:" + jobs.identity,
                "bindings:" + ",".join(bindings),
            )
        )

    def _channel_identity_index(self, channel: str) -> SessionIdentityIndex:
        """Return the Core-owned durable identity index for one channel."""

        current = self._channel_identity_indexes.get(channel)
        if current is not None:
            return current
        if self._session_manager is None:
            raise RuntimeError("v3 Channel identity 需要 SessionManager")
        metadata_key = {
            "feishu": "feishu_open_id",
            "telegram": "username",
            "qq": "user_id",
        }.get(channel, "provider_identity")
        normalizer = str.lower if channel == "telegram" else None
        current = SessionIdentityIndex(
            self._session_manager,
            channel=channel,
            metadata_key=metadata_key,
            normalizer=normalizer,
        )
        _ = current.rebuild()
        self._channel_identity_indexes[channel] = current
        return current

    def _resolve_channel_identity(
        self,
        channel: str,
        provider_identity: str,
    ) -> str | None:
        """Resolve a proactive recipient without exposing SessionManager."""

        return self._channel_identity_index(channel).resolve(provider_identity)

    async def _remember_channel_identity(
        self,
        channel: str,
        provider_identity: str,
        recipient: str,
    ) -> ChannelIdentityWriteReceipt | None:
        """Persist identity mapping before accepting the inbound envelope."""

        return await self._channel_identity_index(channel).remember(
            provider_identity,
            recipient,
        )

    async def _rollback_channel_identity(self, receipt: object) -> bool:
        """Route one failed acceptance rollback to its exact Channel index."""

        if not isinstance(receipt, ChannelIdentityWriteReceipt):
            raise TypeError("channel identity rollback receipt 类型无效")
        return await self._channel_identity_index(receipt.channel).rollback(receipt)

    @property
    def channel_generation_host(self) -> ChannelGenerationHost:
        return self._channel_generation_host

    @property
    def composition_generation_host(self) -> CompositionGenerationHost:
        return self._composition_generation_host

    @staticmethod
    def _channel_catalog_identity(snapshot: RuntimeSnapshot | None) -> str | None:
        if snapshot is None:
            return None
        catalog = snapshot.channel_catalog
        registry = (
            catalog.registry if catalog is not None else snapshot.channel_registry
        )
        return None if registry is None else registry.identity

    def _channel_provider_factories(
        self,
        snapshot: RuntimeSnapshot | None,
    ) -> Mapping[str, ProviderClientFactory]:
        """Resolve provider factories only for a non-empty frozen catalog."""

        if snapshot is None:
            return {}
        catalog = snapshot.channel_catalog
        registry = (
            catalog.registry if catalog is not None else snapshot.channel_registry
        )
        if registry is None or not registry.descriptors:
            return {}
        resolver = self._channel_provider_factory_resolver
        if resolver is None:
            raise RuntimeError("v3 Channel provider factory resolver 尚未绑定")
        factories = resolver(cast(RuntimeSnapshot, snapshot))
        if not isinstance(factories, Mapping):
            raise TypeError("channel provider factory resolver 必须返回 mapping")
        return factories

    @staticmethod
    def _default_channel_provider_factories(
        snapshot: RuntimeSnapshot,
    ) -> Mapping[str, ProviderClientFactory]:
        """Build one formal credential owner for every frozen channel."""

        catalog = snapshot.channel_catalog
        registry = (
            catalog.registry if catalog is not None else snapshot.channel_registry
        )
        if registry is None:
            return {}
        result: dict[str, ProviderClientFactory] = {}
        for descriptor in registry.descriptors:
            if descriptor.owner == "core":
                if catalog is None:
                    raise RuntimeError("Core channel descriptor 缺少 committed catalog")
                definition = catalog.definition(descriptor.name)
                if definition is None:
                    raise RuntimeError(
                        f"Core channel definition 缺失: {descriptor.name}"
                    )
                result[descriptor.name] = _NoopProviderClientFactory()
                continue
            generation = snapshot.generations.get(descriptor.owner)
            if generation is None:
                raise RuntimeError(f"channel owner generation 缺失: {descriptor.owner}")
            result[descriptor.name] = CoreProviderClientFactory(
                generation.data_dir / "config.local.toml",
                descriptor.credential_paths,
                generation.config_revision,
            )
        return result

    def _prepare_channel_publication(
        self,
        previous: RuntimeSnapshot | None,
        candidate: RuntimeSnapshot,
    ) -> _ChannelPublicationState:
        """Freeze the exact old/new channel owners for one provisional switch."""

        previous_identity = self._channel_catalog_identity(previous)
        candidate_identity = self._channel_catalog_identity(candidate)
        changed = self._channel_binding_changed(previous, candidate)
        old_runtime = self._active_channel_generation
        if changed and previous_identity is not None:
            if (
                old_runtime is None
                or self._active_channel_catalog_identity != previous_identity
                or previous is None
            ):
                raise RuntimeError("旧 stable Channel runtime owner 不一致")
        return _ChannelPublicationState(
            previous=previous,
            candidate=candidate,
            previous_identity=previous_identity,
            candidate_identity=candidate_identity,
            old_runtime=old_runtime,
            old_factories=(
                self._channel_provider_factories(previous) if changed else {}
            ),
            new_factories=(
                self._channel_provider_factories(candidate) if changed else {}
            ),
            changed=changed,
        )

    def _channel_binding_changed(
        self,
        previous: RuntimeSnapshot | None,
        candidate: RuntimeSnapshot,
    ) -> bool:
        """判断 exact Channel binding 是否必须随候选 snapshot 换代。"""

        previous_identity = self._channel_catalog_identity(previous)
        candidate_identity = self._channel_catalog_identity(candidate)
        return previous_identity != candidate_identity or (
            candidate_identity is not None
            and (previous is None or previous.snapshot_id != candidate.snapshot_id)
        )

    async def _close_channel_publication(
        self,
        state: _ChannelPublicationState,
    ) -> None:
        """Close, drain, and stop the old runtime before switching endpoints."""

        if not state.changed:
            return
        if state.old_runtime is not None:
            state.old_runtime.close_admission()
            state.old_closed = True
            await state.old_runtime.drain()
            await state.old_runtime.stop()
            state.old_stopped = True
            self._active_channel_generation = None
            self._active_channel_catalog_identity = None

    async def _start_channel_publication(
        self,
        state: _ChannelPublicationState,
    ) -> None:
        """Start the new exact runtime with admission still closed."""

        if not state.changed:
            return
        if state.candidate_identity is not None:
            state.new_runtime = await self._channel_generation_host.start_formal(
                state.candidate,
                state.new_factories,
            )

    def _open_channel_publication(self, state: _ChannelPublicationState) -> None:
        """Publish and open the new exact runtime after the stable pointer moved."""

        if not state.changed:
            return
        self._active_channel_generation = state.new_runtime
        self._active_channel_catalog_identity = state.candidate_identity
        if state.new_runtime is not None:
            state.new_runtime.open_admission()
        self._finish_channel_boot_transactions(state.candidate)

    def _finish_channel_boot_transactions(
        self,
        snapshot: RuntimeSnapshot,
    ) -> None:
        """Finish only journal rows created for a fresh stable Channel boot."""

        registry = snapshot.channel_registry
        owners = (
            set()
            if registry is None
            else {descriptor.owner for descriptor in registry.descriptors}
        )
        for plugin_id in owners:
            generation = snapshot.generations[plugin_id]
            tx_id = generation.reload_tx_id
            if tx_id is None or tx_id not in self._channel_boot_transactions:
                continue
            self._advance_reload(generation, "committed")
            self._advance_reload(generation, "complete")
            self._channel_boot_transactions.remove(tx_id)

    def _abort_channel_boot_transactions(
        self,
        snapshot: RuntimeSnapshot,
        error: BaseException,
    ) -> None:
        """Abort clean fresh-boot rows while preserving cleanup tombstones."""

        for generation in snapshot.generations.values():
            tx_id = generation.reload_tx_id
            if tx_id is None or tx_id not in self._channel_boot_transactions:
                continue
            phase = self._reload_journal.get(tx_id).phase
            if phase not in {"cleanup_failed", "degraded"}:
                self._abort_reload(
                    generation,
                    error=str(error) or type(error).__name__,
                )
            self._channel_boot_transactions.remove(tx_id)

    async def _stop_staged_channel_publication(
        self,
        state: _ChannelPublicationState,
    ) -> None:
        """Stop the staged new runtime before restoring other participants."""

        if not state.changed or state.new_runtime is None:
            return
        await state.new_runtime.stop()

    async def _restore_old_channel_publication(
        self,
        state: _ChannelPublicationState,
    ) -> None:
        """Reconstruct the old runtime after all other owners rolled back."""

        if not state.changed:
            return
        restored = state.old_runtime
        if state.old_stopped and state.previous is not None:
            try:
                restored = await self._channel_generation_host.start_formal(
                    state.previous,
                    state.old_factories,
                    boot_owner="plugin-manager-rollback",
                )
            except BaseException:
                self._active_channel_generation = None
                self._active_channel_catalog_identity = None
                raise
        self._active_channel_generation = restored
        self._active_channel_catalog_identity = state.previous_identity

    def _reopen_restored_channel_publication(
        self,
        state: _ChannelPublicationState,
    ) -> None:
        """Reopen the restored runtime only after the old snapshot is restored."""

        if not state.changed:
            return
        runtime = self._active_channel_generation
        if runtime is not None:
            runtime.open_admission()
        if state.previous is not None:
            self._finish_channel_boot_transactions(state.previous)

    async def _reserve_channel_binding(self, record: ChannelStartRecord) -> None:
        """Persist an exact binding reservation before plugin code can run."""

        if record.plugin_id == "core":
            return
        generation = self._channel_generation(record.plugin_id, record.generation_id)
        tx_id = self._ensure_runtime_recovery_transaction(generation)
        if self._reload_journal.get(tx_id).phase == "preparing":
            self._reload_journal.advance(tx_id, "prepared")
            self._reload_journal.advance(tx_id, "validating")
            self._reload_journal.advance(tx_id, "commit_started")
            self._channel_boot_transactions.add(tx_id)
        self._reload_journal.annotate(
            tx_id,
            {
                "event": "channel_binding_reserved",
                "snapshot_id": record.snapshot_id,
                "catalog_identity": record.catalog_identity,
                "plugin_id": record.plugin_id,
                "generation_id": record.generation_id,
                "channel_name": record.channel_name,
                "binding_token": record.binding_token,
                "artifact_pointer": record.artifact_pointer,
                "factory_export": record.factory_export,
                "source_revision": record.source_revision,
                "config_revision": record.config_revision,
                "raw_config_revision": record.raw_config_revision,
                "descriptor_digest": record.descriptor_digest,
                "target": record.target,
                "boot_owner": record.boot_owner,
                "attempt": record.attempt,
            },
        )

    async def _check_channel_config_revision(
        self,
        record: ChannelStartRecord,
    ) -> None:
        """Fence formal credential resolution to the frozen raw config bytes."""

        if record.plugin_id == "core":
            return
        generation = self._channel_generation(record.plugin_id, record.generation_id)
        if str(generation.plugin_dir) != record.artifact_pointer:
            raise RuntimeError("channel artifact pointer 已漂移")
        current_revision = _file_revision(generation.data_dir / "config.local.toml")
        if current_revision != record.raw_config_revision:
            raise RuntimeError("channel credential config revision 已漂移")

    async def _on_channel_cleanup_failure(
        self,
        failure: ChannelCleanupTombstone,
    ) -> None:
        """Persist one retained channel binding without touching plugin Fiber state."""

        if failure.plugin_id == "core":
            logger.error(
                "Core channel cleanup pending: channel=%s binding=%s error=%s",
                failure.channel_name,
                failure.binding_token,
                failure.error,
            )
            return
        try:
            generation = self._channel_generation(
                failure.plugin_id,
                failure.generation_id,
            )
        except RuntimeError:
            generation = None
        if generation is None:
            actions = tuple(
                action
                for action in self._reload_journal.pending_recovery()
                if action.plugin_id == failure.plugin_id
                and action.failure_resource
                == f"channel-binding:{failure.binding_token}"
            )
            if len(actions) != 1:
                raise RuntimeError("channel cleanup failure 缺少 durable exact owner")
            tx_id = actions[0].tx_id
            recovery_target = actions[0].recovery_target
        else:
            tx_id = self._ensure_runtime_recovery_transaction(generation)
            recovery_target = self._composition_recovery_target(
                generation,
                tx_id=tx_id,
            )
        self._reload_journal.advance(
            tx_id,
            "cleanup_failed",
            error=failure.error,
            resource=f"channel-binding:{failure.binding_token}",
            formal_effects=("channel_binding_cleanup_pending",),
            recovery_action="retry_generation_cleanup",
            recovery_target=recovery_target,
            details={
                "event": "channel_binding_cleanup_failed",
                "snapshot_id": failure.snapshot_id,
                "catalog_identity": failure.catalog_identity,
                "channel_name": failure.channel_name,
                "binding_token": failure.binding_token,
                "artifact_pointer": failure.artifact_pointer,
                "factory_export": failure.factory_export,
                "source_revision": failure.source_revision,
                "config_revision": failure.config_revision,
                "raw_config_revision": failure.raw_config_revision,
                "descriptor_digest": failure.descriptor_digest,
                "target": failure.target,
                "boot_owner": failure.boot_owner,
                "attempt": failure.attempt_count,
            },
        )

    def _channel_generation(
        self,
        plugin_id: str,
        generation_id: str,
    ) -> PluginGeneration:
        """Find one exact retained generation without consulting a same-name replacement."""

        candidates: list[PluginGeneration] = []
        for snapshot in (self.current_snapshot, self.latest_snapshot):
            if snapshot is None:
                continue
            snapshot_generation = snapshot.generations.get(plugin_id)
            if snapshot_generation is not None:
                candidates.append(snapshot_generation)
        active = self._active_generations.get(plugin_id)
        if active is not None:
            candidates.append(active)
        prepared = self._prepared_generations.get(plugin_id)
        if prepared is not None:
            candidates.append(prepared)
        ready = self._ready_candidate
        if ready is not None and ready.plugin_id == plugin_id:
            candidates.append(ready.candidate)
            if ready.previous is not None:
                candidates.append(ready.previous)
        candidates.extend(self._draining_generations.get(plugin_id, ()))
        for generation in candidates:
            if generation.generation_id == generation_id:
                return generation
        raise RuntimeError(
            "channel binding 缺少 exact generation owner: "
            f"{plugin_id}/{generation_id}"
        )

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
        for mod in self.discover(installed_selector="latest"):
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
        if self.current_snapshot is None:
            return False
        return any(
            generation.module_path == module_path
            for generation in self.current_snapshot.active_generations()
        )

    def stable_channel_catalog(self) -> ChannelRegistrySnapshot | None:
        """Return the exact committed stable merged channel declaration catalog."""

        snapshot = self.current_snapshot
        if snapshot is None:
            return None
        catalog = snapshot.channel_catalog
        return catalog.registry if catalog is not None else snapshot.channel_registry

    async def bind_core_channel_definitions(
        self,
        definitions: tuple[CoreChannelDefinition, ...],
    ) -> None:
        """Commit Core channel projections and publish their exact Host bindings."""

        normalized = tuple(definitions)
        if any(not isinstance(item, CoreChannelDefinition) for item in normalized):
            raise TypeError("Core channel definitions 类型无效")
        if not normalized:
            return
        if self._core_channel_definitions:
            raise RuntimeError("Core channel definitions 已绑定")
        self._core_channel_definitions = normalized
        snapshot: RuntimeSnapshot | None = None
        try:
            snapshot, _ = await self._compile_topology_snapshot(
                dict(self._active_generations)
            )
            await self._publish_committed_snapshot(snapshot)
        except BaseException:
            if self.current_snapshot is not snapshot:
                self._core_channel_definitions = ()
            raise

    # 扫描所有 plugin_dirs，返回可加载的插件描述列表
    def discover(
        self,
        *,
        installed_selector: ArtifactSelector = "stable",
    ) -> list[dict[str, str]]:
        mods: list[dict[str, str]] = []
        seen_names: set[str] = set()
        for source in resolve_plugin_sources(
            self._dirs,
            installed_cache_root=self._installed_cache_root,
            installed_selector=installed_selector,
        ):
            name = source.plugin_name or source.plugin_root.name
            if (
                source.source_type == "builtin"
                and name in self._disabled_builtin_plugins
            ):
                continue
            if name in seen_names and source.source_type == "builtin":
                logger.warning("插件名重复，跳过: %s (%s)", name, source.plugin_root)
                continue
            seen_names.add(name)
            import_suffix = name.replace("-", "_").replace("@", "_")
            import_source = source.marketplace or source.plugin_root.parent.name
            module_path = source.plugin_root / source.entrypoint
            mods.append(
                {
                    "name": name,
                    "plugin_root": str(source.plugin_root),
                    "module_path": str(module_path) if module_path is not None else "",
                    "entrypoint": source.entrypoint,
                    "manifest_digest": (
                        source.static_manifest.identity_digest
                        if source.static_manifest is not None
                        else ""
                    ),
                    "import_path": f"akasic_plugin_{import_source}_{import_suffix}",
                    "marketplace": source.marketplace,
                    "source_type": source.source_type,
                }
            )
        return mods

    async def load_all(self) -> None:
        """Load stable plugins and reconstruct any durable latest candidate."""

        # 1. A prior Core boot cannot retain a live candidate lease.
        await self._composition_generation_host.cleanup_candidates()

        # 2. 处理尚未进入 latest_ready 的残留事务，恢复磁盘 pointer。
        recovery = self._reload_journal.pending_recovery()
        self._require_unique_recovery_plugins(recovery)
        stable_by_id = self._discovered_by_id(installed_selector="stable")
        latest_by_id = self._discovered_by_id(installed_selector="latest")
        runtime_recovery = tuple(
            action
            for action in recovery
            if action.action
            in {
                "retry_generation_cleanup",
                "retry_runtime_recovery",
            }
        )
        runtime_receipts = await self._prepare_boot_runtime_recovery(runtime_recovery)
        recovery = tuple(
            action for action in recovery if action not in runtime_recovery
        )
        if runtime_recovery:
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

        # 3. 根据 durable pointer 判定 promoting 崩溃发生在切换前还是切换后。
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

        # 4. stable 在未发布事务中完整装配；latest 随后以新事务恢复。
        if self._active_generations:
            for mod in stable_by_id.values():
                _ = await self._load_one(mod)
        else:
            await self._load_stable_batch(tuple(stable_by_id.values()))
        self._finish_committed_recovery(restore_committed)
        self._finish_boot_runtime_recovery(
            runtime_recovery,
            runtime_receipts,
        )
        await self._restore_latest_candidates(restore_candidates, latest_by_id)

    async def _prepare_boot_runtime_recovery(
        self,
        actions: tuple[ReloadRecoveryAction, ...],
    ) -> dict[str, str]:
        """Clean exact previous boots and normalize their durable artifact targets."""

        if not actions:
            return {}
        current_boot_id = os.environ.get("AKASHIC_BOOT_ID", "").strip()
        if os.environ.get("AKASHIC_SUPERVISED") != "1" or not current_boot_id:
            raise RuntimeError("v3 runtime recovery 需要 supervised boot identity")
        from agent.background.boot_guardian import _cleanup_boot_processes

        cleaned_boots: set[str] = set()
        receipts: dict[str, str] = {}
        for action in actions:
            previous_boot_id = action.runtime_owner_boot_id
            if not previous_boot_id or previous_boot_id == current_boot_id:
                raise RuntimeError(
                    "v3 runtime recovery 缺少不同于当前进程的旧 boot identity"
                )
            if previous_boot_id not in cleaned_boots:
                await asyncio.to_thread(
                    _cleanup_boot_processes,
                    boot_id=previous_boot_id,
                    gateway_group_id=None,
                )
                cleaned_boots.add(previous_boot_id)
            self._normalize_runtime_recovery_pointer(action)
            receipts[action.tx_id] = (
                f"boot-reconcile:previous={previous_boot_id}:"
                f"current={current_boot_id}:cleanup=complete:"
                f"target={action.recovery_target}"
            )
        return receipts

    def _normalize_runtime_recovery_pointer(
        self,
        action: ReloadRecoveryAction,
    ) -> None:
        """Verify one exact pointer pair and select only its recorded target."""

        plugin_name, separator, marketplace = action.plugin_id.rpartition("@")
        if not separator:
            if (
                action.base_artifact_pointer is not None
                or action.candidate_artifact_pointer is not None
            ):
                raise RuntimeError("builtin runtime recovery 不接受 artifact pointer")
            if action.recovery_target != "candidate":
                raise RuntimeError(
                    "builtin runtime recovery 只能恢复当前 release 中的插件"
                )
            return
        plugin_base = (
            _plugins_home(self._installed_cache_root)
            / "cache"
            / marketplace
            / plugin_name
        )
        pointers = read_pointers(plugin_base)
        if pointers is None or action.recovery_target is None:
            raise RuntimeError("runtime recovery 缺少 durable pointer/target evidence")
        base = ArtifactPointer(action.base_artifact_pointer)
        candidate_pointer = action.candidate_artifact_pointer
        pair = (pointers.stable, pointers.latest)
        if action.recovery_target == "base":
            accepted = {(base, base)}
            if candidate_pointer is not None:
                candidate = ArtifactPointer(candidate_pointer)
                accepted.update(
                    {
                        (base, candidate),
                        (candidate, candidate),
                    }
                )
            if pair not in accepted:
                raise RuntimeError(
                    f"runtime recovery base pointer 漂移: {plugin_base}: {pair}"
                )
            latest = (
                ArtifactPointer(candidate_pointer)
                if candidate_pointer is not None
                else base
            )
            _ = write_pointers(plugin_base, stable=base, latest=latest)
            return
        if candidate_pointer is None:
            raise RuntimeError("runtime recovery candidate target 缺少 exact pointer")
        candidate = ArtifactPointer(candidate_pointer)
        if pair != (candidate, candidate):
            raise RuntimeError(
                f"runtime recovery candidate pointer 未提交: {plugin_base}: {pair}"
            )

    def _finish_boot_runtime_recovery(
        self,
        actions: tuple[ReloadRecoveryAction, ...],
        receipts: Mapping[str, str],
    ) -> None:
        """Seal boot reconciliation only after the authoritative stable Root is live."""

        snapshot = self.current_snapshot
        for action in actions:
            generation = self._active_generations.get(action.plugin_id)
            expected_pointer = (
                action.candidate_artifact_pointer
                if action.recovery_target == "candidate"
                else action.base_artifact_pointer
            )
            if expected_pointer is not None:
                if generation is None:
                    raise RuntimeError(
                        "runtime recovery 未重建 exact stable generation"
                    )
                plugin_base = _installed_artifact_base(generation)
                if plugin_base is None or (
                    generation.plugin_dir.relative_to(plugin_base).as_posix()
                    != expected_pointer
                ):
                    raise RuntimeError(
                        "runtime recovery stable artifact identity 不一致"
                    )
            elif "@" not in action.plugin_id:
                if (
                    action.recovery_target != "candidate"
                    or generation is None
                    or generation.source_type != "builtin"
                ):
                    raise RuntimeError(
                        "builtin runtime recovery 未重建当前 release 插件"
                    )
            elif generation is not None:
                raise RuntimeError("runtime recovery 应恢复为无插件 base")
            if (
                action.recovery_target == "candidate"
                and generation is not None
                and generation.source_type == "installed"
                and generation.source_revision != action.source_revision
            ):
                raise RuntimeError("candidate runtime recovery source revision 不一致")
            if generation is not None and snapshot is not None:
                if self._composition_runtime_declared(snapshot, action.plugin_id):
                    if (
                        self._composition_generation_host.get(generation.generation_id)
                        is None
                    ):
                        raise RuntimeError("boot runtime recovery stable Host 未就绪")
                catalog = snapshot.channel_catalog
                registry = (
                    catalog.registry
                    if catalog is not None
                    else snapshot.channel_registry
                )
                channel_declared = registry is not None and any(
                    descriptor.owner == action.plugin_id
                    for descriptor in registry.descriptors
                )
                if channel_declared:
                    channel_runtime = self._active_channel_generation
                    if (
                        channel_runtime is None
                        or channel_runtime.snapshot_id != snapshot.snapshot_id
                        or self._channel_generation_host.get(snapshot.snapshot_id)
                        is None
                        or self._active_channel_catalog_identity != registry.identity
                    ):
                        raise RuntimeError(
                            "boot runtime recovery stable Channel Host 未就绪"
                        )
            if "activity-publication" in (action.failure_resource or ""):
                if snapshot is None or self._activity_host is None:
                    raise RuntimeError(
                        "boot runtime recovery 缺少 stable Activity owner"
                    )
                activity = self._activity_host.active
                expected_activity = ActivityCatalog(
                    background_jobs=snapshot.background_job_catalog,
                ).identity
                if (
                    activity is None
                    or activity.snapshot_id != snapshot.snapshot_id
                    or activity.catalog_identity != expected_activity
                    or not activity.admission_open
                ):
                    raise RuntimeError(
                        "boot runtime recovery stable Activity Host 未就绪"
                    )
            receipt = receipts.get(action.tx_id)
            if receipt is None:
                raise RuntimeError("boot runtime recovery receipt 缺失")
            stable_identity = (
                "none"
                if generation is None
                else f"{generation.generation_id}:{generation.source_revision}"
            )
            snapshot_id = "none" if snapshot is None else snapshot.snapshot_id
            self._reload_journal.finish_recovery(
                action,
                retry_receipt=(
                    f"{receipt}:snapshot={snapshot_id}:stable={stable_identity}"
                ),
            )
            self._write_startup_recovery_fact(
                action,
                committed=action.recovery_target == "candidate",
            )

    async def _load_stable_batch(
        self,
        mods: tuple[dict[str, str], ...],
    ) -> None:
        """暂存全部 stable 插件并发布一个完整运行时快照。"""

        staged: list[PluginGeneration] = []
        snapshot: RuntimeSnapshot | None = None
        catalog_id: str | None = None
        try:
            # 1. 只导入、校验并准备声明，不开放任何 stable snapshot。
            for mod in mods:
                generation = await self._load_one(mod, stage_stable=True)
                if generation is not None:
                    staged.append(generation)
            if not staged:
                return
            snapshot, catalog_id = await self._compile_stable_batch_snapshot(staged)
            for generation in staged:
                generation.runtime_snapshot = snapshot

            # 2. Root declarations become live only after the whole batch settled.
            for generation in staged:
                await self._start_composition_generation_runtime(
                    generation,
                    snapshot,
                    mode="formal",
                )

            # 3. Root mount 已完成全部 v3 lifecycle，登记待发布 generation。
            await self._activate_stable_batch(staged)

            # 4. 全部准备成功后才登记 stable owner，并一次安装快照。
            assert catalog_id is not None
            await self._publish_stable_batch(staged, snapshot, catalog_id)
        except BaseException as error:
            # 5. 未发布事务失败时恢复所有进程内 owner，并反向释放资源。
            _, cleanup_cancelled = await _complete_critical(
                self._discard_stable_batch(
                    staged,
                    snapshot=snapshot,
                    catalog_id=catalog_id,
                )
            )
            if cleanup_cancelled:
                raise asyncio.CancelledError
            if isinstance(error, _StablePluginFailed):
                await self._retry_stable_batch_without_failed(mods, error)
                return
            raise

    async def _compile_stable_batch_snapshot(
        self,
        staged: list[PluginGeneration],
    ) -> tuple[RuntimeSnapshot, str]:
        """为 stable 启动批次编译一个完整的未发布快照。"""

        try:
            return await self._compile_topology_snapshot(
                {item.plugin_id: item for item in staged}
            )
        except Exception as error:
            if len(staged) == 1 and "missing_services=" not in str(error):
                raise _StablePluginFailed(
                    staged[0], "runtime_snapshot", error
                ) from error
            raise

    async def _activate_stable_batch(
        self,
        staged: list[PluginGeneration],
    ) -> None:
        """Mark every fully mounted v3 generation ready for publication."""

        for generation in staged:
            try:
                await self._prepare_generation(generation)
                generation.state = "activating"
            except Exception as error:
                raise _StablePluginFailed(generation, "prepare", error) from error

    async def _publish_stable_batch(
        self,
        staged: list[PluginGeneration],
        snapshot: RuntimeSnapshot,
        catalog_id: str,
    ) -> None:
        """登记全部 stable owner 并一次安装批次快照。"""

        for generation in staged:
            try:
                generation.minimum_resource_count = generation.scope.resource_count
                self._scopes[generation.module_path] = generation.scope
                self._loaded.add(generation.module_path)
                generation.state = "active"
                self._active_generations[generation.plugin_id] = generation
                self._activate_published_generation(generation, None)
            except Exception as error:
                raise _StablePluginFailed(generation, "publish", error) from error
        self._snapshot_skill_catalogs[snapshot.snapshot_id] = catalog_id
        await self._publish_committed_snapshot(snapshot)
        for generation in staged:
            generation.boot_created_data_dir = False
            logger.info("插件已加载: %s", generation.plugin_id)

    async def _discard_stable_batch(
        self,
        staged: list[PluginGeneration],
        *,
        snapshot: RuntimeSnapshot | None,
        catalog_id: str | None,
    ) -> None:
        """释放只归属于未发布启动批次的全部资源。"""

        pending = self._snapshot_store.pending_transaction
        store_owned_pending = (
            snapshot is not None
            and pending is not None
            and pending.candidate is snapshot
        )
        if snapshot is not None and not store_owned_pending:
            _ = self._snapshot_skill_catalogs.pop(snapshot.snapshot_id, None)
        for generation in reversed(staged):
            _ = self._active_generations.pop(generation.plugin_id, None)
            if not store_owned_pending:
                generation.runtime_snapshot = None
                await self._dispose_generation(generation, state="discarded")
        if store_owned_pending:
            assert pending is not None
            await self._snapshot_store.abort(pending)
            for generation in staged:
                generation.runtime_snapshot = None
                if generation.boot_created_data_dir:
                    _remove_validation_data_dir(generation.data_dir)
                    generation.boot_created_data_dir = False
        else:
            for generation in staged:
                if generation.boot_created_data_dir:
                    _remove_validation_data_dir(generation.data_dir)
                    generation.boot_created_data_dir = False
        if (
            not store_owned_pending
            and snapshot is not None
            and snapshot.composition_root is not None
        ):
            await snapshot.composition_root.dispose()
        if catalog_id is not None and not store_owned_pending:
            self._skill_host.close(catalog_id)

    async def _retry_stable_batch_without_failed(
        self,
        mods: tuple[dict[str, str], ...],
        failure: _StablePluginFailed,
    ) -> None:
        """记录被拒绝的 stable 参与者并重建剩余批次。"""

        generation = failure.generation
        self._record_failed_gate(
            plugin_id=generation.plugin_id,
            revision=generation.source_revision,
            check_id=failure.phase,
            reason=str(failure.cause) or type(failure.cause).__name__,
        )
        logger.warning(
            "插件 %s 加载失败，回滚整个未发布批次: %s",
            generation.plugin_id,
            failure.cause,
        )
        remaining = tuple(
            mod for mod in mods if _resolve_plugin_id(mod) != generation.plugin_id
        )
        await self._load_stable_batch(remaining)

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
        _, cancelled = await _complete_critical(
            self._dispose_generation(generation, state="discarded")
        )
        runtime_failure = self._composition_generation_host.failure(
            generation.generation_id
        )
        if runtime_failure is not None:
            raise RuntimeError("候选 runtime cleanup 未完成，必须显式 retry")
        self._abort_reload(generation, error=error)
        if cancelled:
            raise asyncio.CancelledError

    def _begin_reload_attempt(
        self,
        *,
        plugin_id: str,
        generation_id: str,
        source_revision: str,
        config_revision: str,
        plugin_dir: Path,
        source_type: str,
    ) -> str:
        base = self.current_snapshot
        base_generation = None if base is None else base.generations.get(plugin_id)
        base_pointer: str | None = None
        candidate_pointer: str | None = None
        if source_type == "installed":
            plugin_base = _installed_artifact_base_from_root(plugin_dir)
            pointers = read_pointers(plugin_base)
            if pointers is None:
                raise RuntimeError(
                    f"installed reload 缺少 artifact pointer state: {plugin_base}"
                )
            candidate_pointer = plugin_dir.relative_to(plugin_base).as_posix()
            if pointers.latest.path != candidate_pointer:
                raise RuntimeError(
                    "installed reload generation 与 latest pointer 不一致: "
                    f"generation={candidate_pointer} latest={pointers.latest.path}"
                )
            base_pointer = pointers.stable.path
        return self._reload_journal.begin(
            plugin_id=plugin_id,
            base_snapshot_id=base.snapshot_id if base is not None else None,
            base_generation_id=(
                None if base_generation is None else base_generation.generation_id
            ),
            generation_id=generation_id,
            source_revision=source_revision,
            config_revision=config_revision,
            base_artifact_pointer=base_pointer,
            candidate_artifact_pointer=candidate_pointer,
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
        skip_composition_runtime: bool = False,
    ) -> None:
        """完成插件终止、作用域清理和注册表卸载。"""

        # 1. Host 必须在 exact Root/Health observer 仍存活时先回收进程。
        externally_cancelled = False
        if not skip_composition_runtime:
            try:
                await self._stop_composition_generation_runtime(generation)
            except asyncio.CancelledError:
                externally_cancelled = True
            except Exception as error:
                self._cleanup_failures.append(
                    CleanupFailure(
                        resource=f"plugin:{generation.plugin_id}:composition-runtime",
                        error=str(error) or type(error).__name__,
                    )
                )

        # 2. 回收尚未交给 snapshot store 的组合 Root。
        if generation.runtime_snapshot is not None:
            await self._dispose_unreferenced_composition_root(
                generation.runtime_snapshot
            )

        # 3. 收集作用域失败，确保外部取消不会截断资源清理。
        cleanup_failures, cleanup_cancelled = await _complete_critical(
            generation.scope.aclose()
        )
        self._cleanup_failures.extend(cleanup_failures)
        externally_cancelled = externally_cancelled or cleanup_cancelled
        if (
            not skip_composition_runtime
            and self._composition_generation_host.failure(generation.generation_id)
            is not None
        ):
            self._record_composition_runtime_failure(
                generation,
                RuntimeError("generation runtime cleanup 未完成"),
                formal_effects=("generation_runtime_cleanup_pending",),
            )

        # 4. 清理注册表和模块树。
        _ = self._scopes.pop(generation.module_path, None)
        self._loaded.discard(generation.module_path)
        _ = self._active_plugins.pop(generation.module_path, None)
        self._remove_module_tree(generation.module_path)
        stable_alias = self._stable_aliases.get(generation.module_path)
        if stable_alias is not None and not preserve_stable_alias:
            _ = self._stable_aliases.pop(generation.module_path, None)
            self._remove_module_tree(stable_alias)
        generation.state = state
        if externally_cancelled:
            raise asyncio.CancelledError

    async def _dispose_unreferenced_composition_root(
        self,
        snapshot: RuntimeSnapshot,
    ) -> None:
        root = snapshot.composition_root
        if root is None or self._snapshot_store.composition_is_referenced_elsewhere(
            root,
            excluding_snapshot_id="",
        ):
            return
        if self._dashboard_validation_releaser is not None:
            await self._dashboard_validation_releaser(snapshot)
        await self._stop_runtime_snapshot(snapshot)
        await root.dispose()

    def _retire_generation(self, generation: PluginGeneration) -> None:
        """通知已关闭 admission 的 generation 进入退役状态。"""

        if generation.retire_started:
            return
        generation.retire_started = True
        generation.state = "retired"
        self._draining_generations.setdefault(generation.plugin_id, []).append(
            generation
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
        composition_root = snapshot.composition_root
        root_unreferenced = (
            composition_root is not None
            and not self._snapshot_store.composition_is_referenced_elsewhere(
                composition_root,
                excluding_snapshot_id=snapshot.snapshot_id,
            )
        )
        if root_unreferenced:
            await self._stop_runtime_snapshot(snapshot)
        unreferenced_generations = tuple(
            generation
            for generation in snapshot.generations.values()
            if not self._snapshot_store.generation_is_referenced_elsewhere(
                generation,
                excluding_snapshot_id=snapshot.snapshot_id,
            )
        )
        for generation in unreferenced_generations:
            try:
                await self._stop_composition_generation_runtime(generation)
            except Exception as error:
                self._record_drained_composition_runtime_failure(
                    snapshot,
                    generation,
                    error,
                )
                self._cleanup_failures.append(
                    CleanupFailure(
                        resource=(f"plugin:{generation.plugin_id}:composition-runtime"),
                        error=str(error) or type(error).__name__,
                    )
                )
        if root_unreferenced:
            assert composition_root is not None
            if self._dashboard_validation_releaser is not None:
                await self._dashboard_validation_releaser(snapshot)
            await composition_root.dispose()
        catalog_id = self._snapshot_skill_catalogs.pop(snapshot.snapshot_id, None)
        if catalog_id is not None:
            self._skill_host.close(catalog_id)
        state = "aborted" if snapshot.state == "aborted" else "retired"
        current = self._snapshot_store.current
        for generation in unreferenced_generations:
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
                skip_composition_runtime=True,
            )
            self._forget_drained_generation(generation)
        self._finish_drained_reload(snapshot.snapshot_id)

    async def reconcile_changed(self) -> list[dict[str, object]]:
        async with self._candidate_prepare_lock:
            results = await self._reconcile_changed_locked()
            _ = self._sync_skill_links()
            return results

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
            if install_cancelled or reconcile_cancelled:
                if (
                    result.staged_candidate
                    and status["candidate_plugin_id"] == plugin_id
                    and status["candidate_state"] == "latest_ready"
                ):
                    _ = await self._drop_ready(plugin_id)
                raise asyncio.CancelledError
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
            if manifest.get(plugin_id, True)
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
            _ = self._sync_skill_links()

    async def _deactivate_plugin(self, plugin_id: str) -> dict[str, object]:
        active = self._active_generations[plugin_id]
        generations = {
            key: generation
            for key, generation in self._active_generations.items()
            if key != plugin_id
        }
        snapshot, catalog_id = await self._compile_topology_snapshot(generations)
        try:
            if self._dashboard_preparer is not None:
                self._dashboard_preparer(snapshot)
        except BaseException:
            self._skill_host.close(catalog_id)
            await self._dispose_unreferenced_composition_root(snapshot)
            raise

        old_commands = _snapshot_command_catalog(self.current_snapshot)
        new_commands = _snapshot_command_catalog(snapshot)
        current_snapshot = self.current_snapshot
        exclusive_endpoint_changed = (
            current_snapshot is not None
            and self._composition_runtime_declared(current_snapshot, plugin_id)
        )
        command_catalog_changed = old_commands != new_commands
        v3_channel_catalog_changed = self._channel_catalog_identity(
            self.current_snapshot
        ) != self._channel_catalog_identity(snapshot)
        publication_gated = (
            exclusive_endpoint_changed
            or command_catalog_changed
            or v3_channel_catalog_changed
        )
        from agent.plugins.snapshot import get_current_runtime_lease

        if (
            exclusive_endpoint_changed or v3_channel_catalog_changed
        ) and get_current_runtime_lease() is not None:
            self._skill_host.close(catalog_id)
            await self._dispose_unreferenced_composition_root(snapshot)
            raise RuntimeError("持有 RuntimeSnapshot lease 时不能切换独占端点")
        quiesced = self._snapshot_store.pause_admission() if publication_gated else None
        transaction = None
        try:
            if quiesced is not None:
                if exclusive_endpoint_changed and self._endpoint_quiescer is not None:
                    await self._endpoint_quiescer()
                if exclusive_endpoint_changed or v3_channel_catalog_changed:
                    await self._snapshot_store.wait_for_no_leases(quiesced)
            self._snapshot_skill_catalogs[snapshot.snapshot_id] = catalog_id
            transaction = self._snapshot_store.begin_publish(snapshot)
            await self._post_snapshot_invariants(snapshot)
        except BaseException:
            if transaction is not None:
                await self._snapshot_store.abort(transaction)
            else:
                await self._snapshot_store.resume(quiesced)
                _ = self._snapshot_skill_catalogs.pop(snapshot.snapshot_id, None)
                self._skill_host.close(catalog_id)
                await self._dispose_unreferenced_composition_root(snapshot)
            if self._endpoint_resumer is not None and exclusive_endpoint_changed:
                await self._endpoint_resumer()
            raise

        commit_error: BaseException | None = None
        commit_cancelled = False
        try:
            assert transaction is not None
            _, commit_cancelled = await _complete_critical(
                self._commit_snapshot_with_publication_participants(
                    transaction,
                    old_commands=old_commands,
                    new_commands=new_commands,
                    promote_latest=False,
                    force_provisional=exclusive_endpoint_changed,
                    after_open=lambda: self._retire_generation(active),
                )
            )
        except BaseException as error:
            commit_error = error
        if commit_error is not None:
            if self._snapshot_store.pending_candidate is snapshot:
                await self._snapshot_store.abort(
                    transaction,
                    reopen_previous=not isinstance(
                        commit_error,
                        _PublicationParticipantRestoreError,
                    ),
                )
            if (
                self._endpoint_resumer is not None
                and exclusive_endpoint_changed
                and self.current_snapshot is not None
                and self.current_snapshot.accepting_leases
            ):
                await self._endpoint_resumer()
            raise commit_error

        _ = self._active_generations.pop(plugin_id)
        resume_cancelled = False
        if self._endpoint_resumer is not None and exclusive_endpoint_changed:
            _, resume_cancelled = await _complete_critical(self._endpoint_resumer())
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
        old_commands: tuple[tuple[str, str], ...],
        new_commands: tuple[tuple[str, str], ...],
    ) -> None:
        if self._endpoint_switcher is not None:
            await self._endpoint_switcher(
                old_commands,
                new_commands,
            )
            return
        if old_commands != new_commands:
            raise RuntimeError("command catalog host 尚未绑定")

    async def _commit_snapshot_with_publication_participants(
        self,
        transaction: SnapshotTransaction,
        *,
        old_commands: tuple[tuple[str, str], ...],
        new_commands: tuple[tuple[str, str], ...],
        promote_latest: bool,
        force_provisional: bool = False,
        provisional_started: bool = False,
        reopen_previous_on_failure: bool = True,
        before_open: Callable[[], None] | None = None,
        after_open: Callable[[], None] | None = None,
    ) -> SnapshotTransaction:
        """Publish one snapshot around a single closed external-participant step."""

        # 1. Snapshots without external participants retain the one-step path.
        endpoints_changed = old_commands != new_commands
        channel_binding_changed = self._channel_binding_changed(
            transaction.previous,
            transaction.candidate,
        )
        previous_activity_identity = self._activity_catalog_identity(
            transaction.previous
        )
        candidate_activity_identity = self._activity_catalog_identity(
            transaction.candidate
        )
        activity_catalog_changed = (
            previous_activity_identity != candidate_activity_identity
            or (
                candidate_activity_identity is not None
                and (
                    transaction.previous is None
                    or transaction.previous.snapshot_id
                    != transaction.candidate.snapshot_id
                )
            )
        )
        if (
            not endpoints_changed
            and not channel_binding_changed
            and not activity_catalog_changed
            and not force_provisional
            and not provisional_started
        ):
            if promote_latest:
                return await self._snapshot_store.promote_latest(
                    before_open=before_open,
                    after_open=after_open,
                )
            await self._snapshot_store.commit(
                transaction,
                before_open=before_open,
                after_open=after_open,
            )
            return transaction

        # 2. Close both snapshots before any service/channel/command side effect.
        provisional = transaction
        if not provisional_started:
            provisional = (
                await self._snapshot_store.promote_latest_provisional()
                if promote_latest
                else transaction
            )
            if not promote_latest:
                await self._snapshot_store.commit_provisional(provisional)

        channel_state: _ChannelPublicationState | None = None
        activity_transaction: ActivityTransaction | None = None
        participants_switch_attempted = False
        forward_error: BaseException | None = None
        try:
            if activity_catalog_changed:
                activity_host = self._activity_host
                if activity_host is None:
                    raise RuntimeError(
                        "v3 Activity catalog 已声明但 ActivityHost 尚未绑定"
                    )
                target_lease = self._snapshot_store.retain_publication_target(
                    provisional
                )
                activity_transaction = await activity_host.prepare_transaction(
                    target_lease
                )
                await activity_host.pause_and_drain(activity_transaction)
            channel_state = self._prepare_channel_publication(
                provisional.previous,
                provisional.candidate,
            )
            await self._close_channel_publication(channel_state)
            if endpoints_changed:
                participants_switch_attempted = True
                try:
                    await self._switch_plugin_endpoints(
                        old_commands,
                        new_commands,
                    )
                except BaseException as error:
                    forward_error = error
                    raise
            await self._start_channel_publication(channel_state)
            if activity_transaction is not None:
                assert self._activity_host is not None
                await self._activity_host.materialize_closed(activity_transaction)

            def open_participants() -> None:
                if after_open is not None:
                    after_open()
                if activity_transaction is not None:
                    assert self._activity_host is not None
                    self._activity_host.finalize(activity_transaction)
                assert channel_state is not None
                self._open_channel_publication(channel_state)

            await self._snapshot_store.finalize_provisional(
                provisional,
                before_open=before_open,
                after_open=open_participants,
            )
            if activity_transaction is not None:
                assert self._activity_host is not None
                await self._activity_host.open(activity_transaction)
        except BaseException as publication_error:
            if (
                activity_transaction is not None
                and activity_transaction.finalized
                and not activity_transaction.settled
                and self.current_snapshot is provisional.candidate
            ):
                provisional.candidate.accepting_leases = False
                raise _PublicationParticipantRestoreError(
                    "Activity 新 owner 已提交，但旧 child cleanup 尚未完成",
                    resources=("activity-publication",),
                ) from publication_error
            rollback_errors: list[BaseException] = []
            channel_cleanup_failed = False
            activity_cleanup_failed = False
            endpoint_restore_failed = False
            if activity_transaction is not None and not activity_transaction.settled:
                assert self._activity_host is not None
                try:
                    await self._activity_host.rollback(activity_transaction)
                except BaseException as caught:
                    rollback_errors.append(caught)
                    activity_cleanup_failed = True
            if channel_state is not None:
                old_snapshot_id = (
                    None
                    if channel_state.previous is None
                    else (
                        channel_state.old_runtime.snapshot_id
                        if channel_state.old_runtime is not None
                        else None
                    )
                )
                channel_cleanup_failed = self._channel_generation_host.failure(
                    channel_state.candidate.snapshot_id
                ) is not None or (
                    old_snapshot_id is not None
                    and self._channel_generation_host.failure(old_snapshot_id)
                    is not None
                )
                try:
                    await self._stop_staged_channel_publication(channel_state)
                except BaseException as caught:
                    rollback_errors.append(caught)
                    channel_cleanup_failed = True
                if channel_cleanup_failed and not rollback_errors:
                    rollback_errors.append(publication_error)
            if participants_switch_attempted and not channel_cleanup_failed:
                try:
                    await self._switch_plugin_endpoints(
                        new_commands,
                        old_commands,
                    )
                except BaseException as caught:
                    rollback_errors.append(caught)
                    endpoint_restore_failed = True
            if channel_state is not None and not channel_cleanup_failed:
                try:
                    await self._restore_old_channel_publication(channel_state)
                except BaseException as caught:
                    rollback_errors.append(caught)
                    channel_cleanup_failed = True
            await self._snapshot_store.rollback_provisional(
                provisional,
                keep_candidate_latest=promote_latest,
                reopen_previous=(reopen_previous_on_failure and not rollback_errors),
            )
            if not rollback_errors and channel_state is not None:
                self._reopen_restored_channel_publication(channel_state)
            self._abort_channel_boot_transactions(
                provisional.candidate,
                publication_error,
            )
            if rollback_errors:
                resources: list[str] = []
                if activity_cleanup_failed:
                    resources.append("activity-publication")
                if channel_cleanup_failed:
                    resources.extend(("plugin-endpoint", "channel-publication"))
                elif endpoint_restore_failed:
                    resources.append("plugin-endpoint")
                raise _PublicationParticipantRestoreError(
                    "外部 publication participant 失败后旧 owner 恢复失败: "
                    + "; ".join(
                        str(error) or type(error).__name__ for error in rollback_errors
                    ),
                    resources=tuple(resources),
                ) from rollback_errors[0]
            if forward_error is not None:
                if isinstance(forward_error, asyncio.CancelledError):
                    raise forward_error
                raise _PublicationParticipantSwitchError(
                    "外部 publication participant 拒绝切换: "
                    f"{str(forward_error) or type(forward_error).__name__}"
                ) from forward_error
            raise publication_error
        return provisional

    async def _compile_topology_snapshot(
        self,
        generations: dict[str, PluginGeneration],
    ) -> tuple[RuntimeSnapshot, str]:
        self._generation_sequence += 1
        catalog_id = f"topology:{self._generation_sequence}:{secrets.token_hex(4)}"
        ordered = list(generations.values())
        active_ordered = self._static_active_generations(ordered)
        catalog = self._skill_host.prepare(
            catalog_id,
            normal_roots=PluginSkillHost.roots_for(active_ordered, drift=False),
            drift_roots=PluginSkillHost.roots_for(active_ordered, drift=True),
            ignored_normal_roots=tuple(
                root
                for generation in active_ordered
                for root in generation.contributions.skill_roots
            ),
            ignored_drift_roots=tuple(
                root
                for generation in active_ordered
                for root in generation.contributions.drift_skill_roots
            ),
        )
        composition_root, created_root = await self._resolve_composition_root(
            generations
        )
        try:
            snapshot = self._snapshot_compiler.compile(
                generations,
                snapshot_revision=catalog_id,
                composition_root=composition_root,
                core_channel_definitions=self._core_channel_definitions,
            )
            _validate_static_manifest_runtime(snapshot, generations)
            snapshot.skill_catalog_generation_id = catalog_id
            snapshot.plugin_skill_index = catalog.normal_plugins
            self._refresh_composition_runtime_tools(snapshot)
            return snapshot, catalog_id
        except BaseException:
            self._skill_host.close(catalog_id)
            if created_root and composition_root is not None:
                await composition_root.dispose()
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
            if self._reload_journal.get(tx_id).phase != "latest_ready":
                raise RuntimeError("latest candidate 已被 runtime recovery 撤销准入")

            old_commands = _snapshot_command_catalog(self.current_snapshot)
            new_commands = _snapshot_command_catalog(ready.snapshot)
            stable_snapshot = self.current_snapshot
            v3_runtime_handoff = self._composition_runtime_declared(
                ready.snapshot,
                plugin_id,
            ) or (
                stable_snapshot is not None
                and self._composition_runtime_declared(
                    stable_snapshot,
                    plugin_id,
                )
            )
            exclusive_endpoint_changed = v3_runtime_handoff
            command_catalog_changed = old_commands != new_commands
            v3_channel_catalog_changed = self._channel_catalog_identity(
                stable_snapshot
            ) != self._channel_catalog_identity(ready.snapshot)
            formal_root_handoff = (
                generation.production_contributions is not None
                and stable_snapshot is not None
                and stable_snapshot.composition_root is not None
            )
            publication_gated = (
                exclusive_endpoint_changed
                or command_catalog_changed
                or v3_channel_catalog_changed
                or formal_root_handoff
            )
            from agent.plugins.snapshot import get_current_runtime_lease

            if (
                exclusive_endpoint_changed
                or v3_channel_catalog_changed
            ) and get_current_runtime_lease() is not None:
                raise RuntimeError(
                    "持有 RuntimeSnapshot lease 时不能切换 Channel runtime"
                )

            skill_linker, stable_skill_plugins, target_skill_plugins = (
                self._prepare_skill_links_for_promotion(generation, ready.snapshot)
            )

            # 1. Seal both stable and validation leases before touching ownership.
            candidate_snapshot = self._snapshot_store.pause_candidate_admission(
                ready.snapshot
            )
            quiesced_snapshot = (
                self._snapshot_store.pause_admission()
                if publication_gated
                else None
            )
            runtime_restore_started = False
            stable_root_stopped = False
            provisional_transaction: SnapshotTransaction | None = None
            provisional_cancelled = False
            if publication_gated:
                try:
                    if (
                        exclusive_endpoint_changed
                        and self._endpoint_quiescer is not None
                    ):
                        await self._endpoint_quiescer()
                    if quiesced_snapshot is not None and (
                        exclusive_endpoint_changed
                        or v3_channel_catalog_changed
                        or formal_root_handoff
                    ):
                        await self._snapshot_store.wait_for_no_leases(quiesced_snapshot)
                    await self._snapshot_store.wait_for_no_leases(ready.snapshot)
                    self._snapshot_store.seal_candidate_validation(ready.snapshot)
                    (
                        provisional_transaction,
                        provisional_cancelled,
                    ) = await _complete_critical(
                        self._snapshot_store.promote_latest_provisional()
                    )
                    runtime_restore_started = True
                    try:
                        await self._restore_ready_runtime(
                            ready,
                            stable_snapshot=quiesced_snapshot,
                        )
                    finally:
                        stable_root_stopped = generation.formal_root_stopped
                    generation = ready.candidate
                    new_commands = _snapshot_command_catalog(ready.snapshot)
                except BaseException:
                    gated_runtime_error: BaseException | None = None
                    if stable_root_stopped:
                        assert quiesced_snapshot is not None
                        try:
                            await self._recover_stable_root(
                                generation,
                                quiesced_snapshot,
                            )
                        except BaseException as error:
                            gated_runtime_error = error
                    elif runtime_restore_started:
                        try:
                            await self._rollback_composition_runtime_replacement(
                                generation
                            )
                        except BaseException as error:
                            gated_runtime_error = error
                    if provisional_transaction is not None:
                        _, rollback_cancelled = await _complete_critical(
                            self._snapshot_store.rollback_provisional(
                                provisional_transaction,
                                keep_candidate_latest=True,
                                reopen_previous=gated_runtime_error is None,
                            )
                        )
                        provisional_cancelled = (
                            provisional_cancelled or rollback_cancelled
                        )
                    if gated_runtime_error is None:
                        await self._snapshot_store.resume(quiesced_snapshot)
                    if (
                        gated_runtime_error is None
                        and exclusive_endpoint_changed
                        and self._endpoint_resumer is not None
                    ):
                        await self._endpoint_resumer()
                    if runtime_restore_started and self._ready_candidate is ready:
                        if gated_runtime_error is None:
                            await self._clear_ready_after_failed_promotion(ready)
                    else:
                        await self._snapshot_store.resume(candidate_snapshot)
                    if gated_runtime_error is not None:
                        self._record_composition_runtime_failure(
                            generation,
                            gated_runtime_error,
                            formal_effects=(
                                "candidate_validation_stopped",
                                "old_runtime_restore_uncertain",
                            ),
                            recovery_target="base",
                        )
                        raise RuntimeError(
                            "candidate formalization 失败后旧 v3 runtime 恢复失败"
                        ) from gated_runtime_error
                    raise
            else:
                try:
                    await self._snapshot_store.wait_for_no_leases(ready.snapshot)
                    self._snapshot_store.seal_candidate_validation(ready.snapshot)
                    runtime_restore_started = True
                    await self._restore_ready_runtime(
                        ready,
                        stable_snapshot=None,
                    )
                    generation = ready.candidate
                except BaseException:
                    formalization_runtime_error: BaseException | None = None
                    if runtime_restore_started:
                        try:
                            await self._rollback_composition_runtime_replacement(
                                generation
                            )
                        except BaseException as error:
                            formalization_runtime_error = error
                    if runtime_restore_started and self._ready_candidate is ready:
                        if formalization_runtime_error is None:
                            await self._clear_ready_after_failed_promotion(ready)
                    else:
                        await self._snapshot_store.resume(candidate_snapshot)
                    if formalization_runtime_error is not None:
                        self._record_composition_runtime_failure(
                            generation,
                            formalization_runtime_error,
                            formal_effects=(
                                "candidate_validation_stopped",
                                "old_runtime_restore_uncertain",
                            ),
                        )
                        raise RuntimeError(
                            "candidate formalization 失败后旧 v3 runtime 恢复失败"
                        ) from formalization_runtime_error
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
                if phase != "latest_ready":
                    raise RuntimeError(
                        "candidate runtime recovery 已阻止 pointer commit"
                    )
                self._advance_reload(generation, "promoting")
                artifact_base = _installed_artifact_base(generation)
                if artifact_base is not None:
                    _switch_ready_pointer(ready, artifact_base)

            # 3. Snapshot pointer 切换后再替换 manager 的 stable generation owner。
            def after_open() -> None:
                self._activate_published_generation(generation, ready.previous)
                generation.state = "active"
                self._scopes[generation.module_path] = generation.scope
                self._loaded.add(generation.module_path)
                self._active_generations[plugin_id] = generation
                if ready.previous is not None:
                    self._retire_generation(ready.previous)

            previous_snapshot = self.current_snapshot
            if previous_snapshot is not None:
                self._drain_transactions[previous_snapshot.snapshot_id] = tx_id
            try:
                transaction, final_cancelled = await _complete_critical(
                    self._commit_snapshot_with_publication_participants(
                        provisional_transaction
                        or SnapshotTransaction(
                            previous=previous_snapshot,
                            candidate=ready.snapshot,
                        ),
                        old_commands=old_commands,
                        new_commands=new_commands,
                        promote_latest=True,
                        force_provisional=exclusive_endpoint_changed,
                        provisional_started=provisional_transaction is not None,
                        reopen_previous_on_failure=not formal_root_handoff,
                        before_open=before_open,
                        after_open=after_open,
                    )
                )
                cancelled = provisional_cancelled or final_cancelled
            except BaseException as publication_error:
                skill_error: BaseException | None = None
                pointer_error: BaseException | None = None
                runtime_error: BaseException | None = None
                participant_restore_error = (
                    publication_error
                    if isinstance(
                        publication_error,
                        _PublicationParticipantRestoreError,
                    )
                    else None
                )
                artifact_base = _installed_artifact_base(generation)
                if artifact_base is not None:
                    try:
                        _preserve_ready_pointer(ready, artifact_base)
                    except BaseException as error:
                        pointer_error = error
                if skill_links_switched:
                    try:
                        skill_linker.sync(stable_skill_plugins)
                    except BaseException as error:
                        skill_error = error
                if stable_root_stopped:
                    assert quiesced_snapshot is not None
                    try:
                        await self._recover_stable_root(
                            generation,
                            quiesced_snapshot,
                        )
                    except BaseException as error:
                        runtime_error = error
                elif runtime_restore_started:
                    try:
                        await self._rollback_composition_runtime_replacement(generation)
                    except BaseException as error:
                        runtime_error = error
                if (
                    previous_snapshot is not None
                    and self.current_snapshot is previous_snapshot
                ):
                    _ = self._drain_transactions.pop(
                        previous_snapshot.snapshot_id,
                        None,
                    )
                if (
                    runtime_error is None
                    and skill_error is None
                    and pointer_error is None
                    and participant_restore_error is None
                ):
                    await self._snapshot_store.resume(quiesced_snapshot)
                    await self._start_current_runtime_snapshot()
                if (
                    runtime_error is None
                    and skill_error is None
                    and pointer_error is None
                    and participant_restore_error is None
                    and self._endpoint_resumer is not None
                    and exclusive_endpoint_changed
                ):
                    await self._endpoint_resumer()
                recovery_error = (
                    runtime_error
                    or participant_restore_error
                    or skill_error
                    or pointer_error
                )
                if self._ready_candidate is ready and recovery_error is None:
                    await self._clear_ready_after_failed_promotion(ready)
                if recovery_error is not None:
                    recovery_resources: list[str] = []
                    recovery_effects: list[str] = []
                    if runtime_error is not None:
                        recovery_resources.append("composition-runtime")
                        recovery_effects.extend(
                            (
                                "candidate_formal_started",
                                "old_runtime_restore_uncertain",
                            )
                        )
                    if participant_restore_error is not None:
                        recovery_resources.extend(participant_restore_error.resources)
                        if "plugin-endpoint" in participant_restore_error.resources:
                            recovery_effects.append("endpoint_restore_uncertain")
                        if "channel-publication" in participant_restore_error.resources:
                            recovery_effects.append("stable_channel_restore_uncertain")
                        if (
                            "activity-publication"
                            in participant_restore_error.resources
                        ):
                            recovery_effects.append("stable_activity_restore_uncertain")
                    if skill_error is not None:
                        recovery_resources.append("plugin-skill-projection")
                        recovery_effects.append("stable_skill_restore_uncertain")
                    if pointer_error is not None:
                        recovery_resources.append("plugin-artifact-pointer")
                        recovery_effects.append("stable_pointer_restore_uncertain")
                    self._record_composition_runtime_failure(
                        generation,
                        recovery_error,
                        resource=",".join(recovery_resources),
                        formal_effects=tuple(recovery_effects),
                        recovery_target="base",
                    )
                    raise RuntimeError(
                        "插件 promote 失败后存在未完成的 formal recovery: "
                        + ", ".join(recovery_resources)
                    ) from recovery_error
                raise
            await self._start_current_runtime_snapshot()
            self._ready_candidate = None
            generation.replaced_composition_runtime_generation = None
            generation.formal_root_stopped = False
            generation.formal_root_released = False
            self._track_reload_drain(generation, transaction.previous)
            if self._endpoint_resumer is not None and exclusive_endpoint_changed:
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

    async def retry_runtime_recovery(self, plugin_id: str) -> dict[str, object]:
        """Retry one durable v3 runtime owner and reconcile its exact pointer target."""

        result, cancelled = await _complete_critical(
            self._retry_runtime_recovery_critical(plugin_id)
        )
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _retry_runtime_recovery_critical(
        self,
        plugin_id: str,
    ) -> dict[str, object]:
        """Complete one runtime recovery transaction before exposing cancellation."""

        async with self._candidate_prepare_lock:
            actions = tuple(
                action
                for action in self._reload_journal.pending_recovery()
                if action.plugin_id == plugin_id
                and action.action
                in {"retry_generation_cleanup", "retry_runtime_recovery"}
            )
            if len(actions) != 1:
                raise RuntimeError("插件没有待执行的 runtime recovery")
            action = actions[0]
            ready = self._ready_candidate
            if ready is not None and (
                ready.plugin_id != plugin_id
                or ready.candidate.reload_tx_id != action.tx_id
            ):
                ready = None
            prepared = self._prepared_generations.get(plugin_id)
            if prepared is not None and prepared.reload_tx_id != action.tx_id:
                prepared = None

            # 1. Retry every exact retained Host owner before changing pointers.
            receipts: list[str] = []
            resource = action.failure_resource or ""
            if "runtime-snapshot-drain" in resource:
                await self._snapshot_store.retry_drains()
                receipts.append("runtime-snapshot-drain-complete")
            if "activity-publication" in resource:
                activity_host = self._activity_host
                if activity_host is None:
                    raise RuntimeError("Activity recovery 缺少 ActivityHost owner")
                await activity_host.retry_recovery()
                receipts.append("stable-activity-runtime-restored")
            channel_tokens = tuple(
                item.removeprefix("channel-binding:")
                for item in resource.split(",")
                if item.startswith("channel-binding:")
            )
            for binding_token in channel_tokens:
                await self._channel_generation_host.retry_generation_cleanup(
                    binding_token
                )
                receipts.append(f"channel-binding-cleanup-complete:{binding_token}")
            retained_generation_ids = tuple(
                dict.fromkeys(
                    generation_id
                    for generation_id in (
                        action.generation_id,
                        action.base_generation_id,
                    )
                    if generation_id is not None
                    and self._composition_generation_host.failure(generation_id)
                    is not None
                )
            )
            for generation_id in retained_generation_ids:
                if action.action == "retry_runtime_recovery":
                    receipts.append(
                        await self._composition_generation_host.retry_runtime_recovery(
                            generation_id
                        )
                    )
                if self._composition_generation_host.failure(generation_id) is None:
                    _ = self._composition_runtime_generations.pop(
                        generation_id,
                        None,
                    )
                else:
                    receipts.append(
                        await self._composition_generation_host.retry_generation_cleanup(
                            generation_id
                        )
                    )

            # 2. Rebuild the exact committed stable runtime when rollback left it absent.
            stable = self._active_generations.get(plugin_id)
            current = self.current_snapshot
            if (
                action.action == "retry_runtime_recovery"
                and stable is not None
                and current is not None
                and self._composition_runtime_declared(current, plugin_id)
                and self._composition_generation_host.get(stable.generation_id) is None
                and (
                    ready is None
                    or ready.candidate.replaced_composition_runtime_generation is None
                )
            ):
                await self._rebuild_stable_root(stable, current)
                receipts.append("stable-composition-runtime-restored")

            # 3. Restore non-runtime formal effects only while their candidate owner exists.
            if (
                action.action == "retry_runtime_recovery"
                and ready is not None
                and ready.candidate.replaced_composition_runtime_generation is not None
            ):
                if current is None:
                    raise RuntimeError("runtime recovery 缺少 stable snapshot")
                await self._recover_stable_root(ready.candidate, current)
                receipts.append("stable-composition-runtime-restored")
            if "plugin-skill-projection" in resource:
                if ready is None:
                    raise RuntimeError("runtime recovery 缺少 skill candidate owner")
                linker, stable_plugins, _target_plugins = (
                    self._prepare_skill_links_for_promotion(
                        ready.candidate,
                        ready.snapshot,
                    )
                )
                _ = linker.sync(stable_plugins)
                receipts.append("stable-skill-projection-restored")

            # 4. Normalize the exact durable target before acquiring new resources.
            if (
                action.base_artifact_pointer is not None
                or action.candidate_artifact_pointer is not None
            ):
                self._normalize_runtime_recovery_pointer(action)
            elif "@" in plugin_id:
                raise RuntimeError(
                    "installed runtime recovery 缺少 exact artifact pointer"
                )
            cancelled = False
            candidate_snapshot = self._snapshot_store.unpromoted_candidate
            if action.recovery_target == "base" and candidate_snapshot is not None:
                _, cancelled = await _complete_critical(
                    self._snapshot_store.discard_latest(candidate_snapshot)
                )
            if action.recovery_target == "base" and ready is not None:
                self._ready_candidate = None
            if action.recovery_target == "base" and prepared is not None:
                _ = self._prepared_generations.pop(plugin_id, None)
                _, prepared_cancelled = await _complete_critical(
                    self._dispose_generation(prepared, state="discarded")
                )
                cancelled = cancelled or prepared_cancelled
            if action.recovery_target == "candidate" and ready is not None:
                if self.current_snapshot is not ready.snapshot:
                    raise RuntimeError(
                        "runtime recovery candidate target 尚未成为 stable"
                    )
                self._ready_candidate = None

            # 5. Rebuild the exact stable Channel owner after identity normalization.
            restored_channel_runtime: ChannelGeneration | None = None
            current_channel_identity = self._channel_catalog_identity(current)
            channel_publication_failed = (
                bool(channel_tokens) or "channel-publication" in resource
            )
            if channel_publication_failed and current is None:
                self._active_channel_generation = None
                self._active_channel_catalog_identity = None
            elif channel_publication_failed and current is not None:
                active_runtime = self._active_channel_generation
                if current_channel_identity is None:
                    self._active_channel_generation = None
                    self._active_channel_catalog_identity = None
                elif (
                    active_runtime is None
                    or self._channel_generation_host.get(active_runtime.snapshot_id)
                    is None
                    or self._active_channel_catalog_identity != current_channel_identity
                ):
                    restored_channel_runtime = (
                        await self._channel_generation_host.start_formal(
                            current,
                            self._channel_provider_factories(current),
                            boot_owner="plugin-manager-recovery",
                        )
                    )

            # 6. Open the exact Channel owner before any public admission resumes.
            if restored_channel_runtime is not None:
                self._active_channel_generation = restored_channel_runtime
                self._active_channel_catalog_identity = current_channel_identity
                restored_channel_runtime.open_admission()
                receipts.append("stable-channel-runtime-restored")
            receipt = ";".join(receipts) or "runtime-owner-already-clean"
            _, resume_cancelled = await _complete_critical(
                self._snapshot_store.resume(self.current_snapshot)
            )
            await self._start_current_runtime_snapshot()
            endpoint_resume_cancelled = False
            participant_only_recovery = all(
                item.startswith("channel-binding:")
                or item.startswith("channel-publication:")
                or item.startswith("activity-publication")
                for item in resource.split(",")
                if item
            )
            if self._endpoint_resumer is not None and not participant_only_recovery:
                _, endpoint_resume_cancelled = await _complete_critical(
                    self._endpoint_resumer()
                )
            self._reload_journal.finish_recovery(
                action,
                retry_receipt=receipt,
            )
            if cancelled or resume_cancelled or endpoint_resume_cancelled:
                raise asyncio.CancelledError
            active = self._active_generations.get(plugin_id)
            return {
                "plugin_id": plugin_id,
                "publication_state": "recovered",
                "recovery_target": action.recovery_target,
                "generation_id": (None if active is None else active.generation_id),
                "snapshot_id": (
                    None
                    if self.current_snapshot is None
                    else self.current_snapshot.snapshot_id
                ),
                "retry_receipt": receipt,
            }

    async def _restore_ready_runtime(
        self,
        ready: _ReadyPluginCandidate,
        *,
        stable_snapshot: RuntimeSnapshot | None,
    ) -> None:
        """Replace validation and stable Roots without overlapping formal owners."""

        generation = ready.candidate
        production = generation.production_contributions
        if production is None:
            return
        production_data_dir = generation.production_data_dir
        if production_data_dir is None:
            raise RuntimeError("候选缺少 production plugin-data identity")
        candidate_runtime = self._composition_generation_host.get(
            generation.generation_id
        )
        expected_mcp_catalog_digests = (
            None if candidate_runtime is None else candidate_runtime.mcp_catalog_digests
        )

        # 1. 隔离 Root 已封存，先停止其任务，再进入任何 formal await。
        if self._dashboard_validation_releaser is not None:
            await self._dashboard_validation_releaser(ready.snapshot)
        await self._stop_runtime_snapshot(ready.snapshot)
        await self._stop_composition_generation_runtime(generation)
        validation_root = ready.snapshot.composition_root
        if validation_root is not None:
            await validation_root.dispose()

        # 2. Restore the formal data projection, then refresh the exact Root payload.
        generation.contributions = production
        generation.data_dir = production_data_dir
        await self._stop_stable_root(generation, stable_snapshot)
        replacement = await self._compile_generation_snapshot(
            generation,
        )
        try:
            _validate_candidate_formal_snapshot_identity(
                generation,
                candidate=ready.snapshot,
                formal=replacement,
            )
        except RuntimeError:
            await self._dispose_unreferenced_composition_root(replacement)
            raise
        try:
            await self._start_snapshot_composition_runtimes(
                replacement,
                candidate=generation,
                expected_mcp_catalog_digests=expected_mcp_catalog_digests,
            )
        except BaseException:
            await self._stop_snapshot_composition_runtimes(replacement)
            await self._dispose_unreferenced_composition_root(replacement)
            raise
        _replace_snapshot_payload(ready.snapshot, replacement)
        validation_workspace = generation.validation_workspace
        generation.validation_workspace = None
        if validation_workspace is not None:
            _remove_validation_data_dir(validation_workspace.parent)
        generation.validation_data_inventory = ()
        if self._dashboard_preparer is not None:
            self._dashboard_preparer(ready.snapshot)
        generation.production_contributions = None
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
        retained = self._reload_journal.get(tx_id)
        if retained.phase in {"cleanup_failed", "degraded"}:
            raise RuntimeError("candidate runtime cleanup 未完成，必须先执行 recovery")
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

    async def _clear_ready_after_failed_promotion(
        self,
        ready: _ReadyPluginCandidate,
    ) -> None:
        """Release invalid validation state while retaining durable latest."""

        artifact_base = _installed_artifact_base(ready.candidate)
        if artifact_base is not None:
            _preserve_ready_pointer(ready, artifact_base)
        _, cancelled = await _complete_critical(
            self._snapshot_store.discard_latest(ready.snapshot)
        )
        self._ready_candidate = None
        if cancelled:
            raise asyncio.CancelledError

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
        if generation.reload_tx_id is not None and (
            self._reload_journal.get(generation.reload_tx_id).phase != "prepared"
        ):
            raise RuntimeError("插件候选已被 runtime recovery 撤销准入")
        active = self._active_generations.get(plugin_id)
        stage_latest = _installed_generation_is_candidate(generation)
        try:
            if stage_latest:
                if (
                    generation.production_contributions is None
                    or generation.production_data_dir is None
                ):
                    raise RuntimeError("installed candidate 缺少隔离 plugin-data 身份")
            prepared_snapshot = generation.runtime_snapshot
            generation.runtime_snapshot = await self._compile_generation_snapshot(
                generation,
                candidate_owner=generation,
            )
            await self._start_composition_generation_runtime(
                generation,
                generation.runtime_snapshot,
                mode="candidate",
            )
            if (
                prepared_snapshot is not None
                and prepared_snapshot is not generation.runtime_snapshot
            ):
                await self._dispose_unreferenced_composition_root(prepared_snapshot)
            snapshot = generation.runtime_snapshot
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

        old_commands = _snapshot_command_catalog(self.current_snapshot)
        new_commands = _snapshot_command_catalog(snapshot)
        current = self.current_snapshot
        v3_runtime_handoff = self._composition_runtime_declared(
            snapshot,
            plugin_id,
        ) or (
            current is not None
            and self._composition_runtime_declared(current, plugin_id)
        )
        exclusive_endpoint_changed = v3_runtime_handoff
        command_catalog_changed = old_commands != new_commands
        v3_channel_catalog_changed = self._channel_catalog_identity(
            current
        ) != self._channel_catalog_identity(snapshot)
        formal_root_handoff = (
            not stage_latest
            and current is not None
            and current.composition_root is not None
            and snapshot.composition_root is not current.composition_root
        )
        publication_gated = not stage_latest and (
            exclusive_endpoint_changed
            or command_catalog_changed
            or v3_channel_catalog_changed
            or formal_root_handoff
        )
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
        if publication_gated:
            from agent.plugins.snapshot import get_current_runtime_lease

            if (
                exclusive_endpoint_changed or v3_channel_catalog_changed
            ) and get_current_runtime_lease() is not None:
                error_text = "持有 RuntimeSnapshot lease 时不能切换独占端点"
                await self.discard_prepared(
                    plugin_id,
                    error=f"endpoint_lease: {error_text}",
                )
                raise RuntimeError(error_text)
            quiesced_snapshot = self._snapshot_store.pause_admission()
            try:
                if exclusive_endpoint_changed and self._endpoint_quiescer is not None:
                    await self._endpoint_quiescer()
                if quiesced_snapshot is not None and (
                    exclusive_endpoint_changed
                    or v3_channel_catalog_changed
                    or formal_root_handoff
                ):
                    await self._snapshot_store.wait_for_no_leases(quiesced_snapshot)
            except BaseException as error:
                error_text = str(error) or type(error).__name__
                await self._snapshot_store.resume(quiesced_snapshot)
                if exclusive_endpoint_changed and self._endpoint_resumer is not None:
                    await self._endpoint_resumer()
                await self.discard_prepared(
                    plugin_id,
                    error=f"endpoint_quiesce: {error_text}",
                )
                raise
        transaction = self._snapshot_store.begin_publish(snapshot)
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
            await self._abort_failed_publication(
                generation,
                transaction,
                error="post-publish invariant failed",
            )
            if self._endpoint_resumer is not None and exclusive_endpoint_changed:
                await self._endpoint_resumer()
            raise

        provisional_started = False
        provisional_cancelled = False
        stable_root_stopped = False
        if not stage_latest:
            try:
                self._snapshot_store.seal_pending_validation(snapshot)
                if publication_gated:
                    _, provisional_cancelled = await _complete_critical(
                        self._snapshot_store.commit_provisional(transaction)
                    )
                    provisional_started = True
                try:
                    _ = await self._restore_direct_candidate_runtime(
                        generation,
                        validation_snapshot=snapshot,
                        stable_snapshot=quiesced_snapshot,
                    )
                finally:
                    stable_root_stopped = generation.formal_root_stopped
            except (asyncio.CancelledError, Exception) as error:
                error_text = str(error) or type(error).__name__
                stable_recovery_error: BaseException | None = None
                if stable_root_stopped:
                    assert quiesced_snapshot is not None
                    try:
                        await self._recover_stable_root(
                            generation,
                            quiesced_snapshot,
                        )
                    except BaseException as recovery_error:
                        stable_recovery_error = recovery_error
                self._record_failed_gate(
                    plugin_id=plugin_id,
                    revision=generation.source_revision,
                    check_id="production_rebuild",
                    reason=error_text,
                )
                _ = self._prepared_generations.pop(plugin_id, None)
                generation.state = "aborted"
                previous_runtime = generation.replaced_composition_runtime_generation
                if (
                    previous_runtime is not None
                    and self._composition_generation_host.get(
                        previous_runtime.generation_id
                    )
                    is None
                ):
                    self._record_composition_runtime_failure(
                        generation,
                        stable_recovery_error or error,
                        formal_effects=(
                            "candidate_pointer_restored",
                            "old_runtime_restore_uncertain",
                        ),
                        recovery_target="base",
                    )
                runtime_restore_uncertain = stable_recovery_error is not None or (
                    previous_runtime is not None
                    and self._composition_generation_host.get(
                        previous_runtime.generation_id
                    )
                    is None
                )
                if provisional_started:
                    _, rollback_cancelled = await _complete_critical(
                        self._snapshot_store.rollback_provisional(
                            transaction,
                            keep_candidate_latest=False,
                            reopen_previous=not runtime_restore_uncertain,
                        )
                    )
                    provisional_cancelled = provisional_cancelled or rollback_cancelled
                await self._abort_failed_publication(
                    generation,
                    transaction,
                    error=f"production_rebuild: {error_text}",
                    reopen_previous=not runtime_restore_uncertain,
                )
                if (
                    self._endpoint_resumer is not None
                    and exclusive_endpoint_changed
                    and not runtime_restore_uncertain
                    and self.current_snapshot is not None
                    and self.current_snapshot.accepting_leases
                ):
                    await self._endpoint_resumer()
                raise

        commit_error: BaseException | None = None
        commit_cancelled = provisional_cancelled

        def open_candidate() -> None:
            self._advance_reload(generation, "commit_started")
            generation.state = "activating"
            if not stage_latest:
                self._activate_published_generation(generation, active)
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
                _, final_commit_cancelled = await _complete_critical(
                    self._commit_snapshot_with_publication_participants(
                        transaction,
                        old_commands=old_commands,
                        new_commands=new_commands,
                        promote_latest=False,
                        force_provisional=exclusive_endpoint_changed,
                        provisional_started=provisional_started,
                        reopen_previous_on_failure=not formal_root_handoff,
                        before_open=open_candidate,
                        after_open=(
                            None
                            if active is None
                            else lambda: self._retire_generation(active)
                        ),
                    )
                )
                commit_cancelled = commit_cancelled or final_commit_cancelled
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
            await self._abort_failed_publication(
                generation,
                transaction,
                error=str(commit_error) or type(commit_error).__name__,
                finish_journal=False,
                reopen_previous=not isinstance(
                    commit_error,
                    _PublicationParticipantRestoreError,
                )
                and not stable_root_stopped,
            )
            runtime_restore_error: BaseException | None = None
            try:
                if stable_root_stopped:
                    assert quiesced_snapshot is not None
                    await self._recover_stable_root(
                        generation,
                        quiesced_snapshot,
                    )
                else:
                    await self._restore_replaced_composition_runtime(generation)
            except BaseException as error:
                runtime_restore_error = error
            if (
                runtime_restore_error is None
                and stable_root_stopped
                and not isinstance(
                    commit_error,
                    _PublicationParticipantRestoreError,
                )
            ):
                await self._snapshot_store.resume(quiesced_snapshot)
                await self._start_current_runtime_snapshot()
            participant_restore_error = isinstance(
                commit_error,
                _PublicationParticipantRestoreError,
            )
            if runtime_restore_error is not None:
                self._record_composition_runtime_failure(
                    generation,
                    runtime_restore_error,
                    formal_effects=(
                        "candidate_pointer_restored",
                        "old_runtime_restore_uncertain",
                    ),
                    recovery_target="base",
                )
            elif participant_restore_error:
                participant_resource = (
                    "plugin-endpoint,channel-publication"
                    if exclusive_endpoint_changed
                    else "channel-publication"
                )
                participant_effects = (
                    (
                        "endpoint_restore_uncertain",
                        "stable_channel_restore_uncertain",
                    )
                    if exclusive_endpoint_changed
                    else ("stable_channel_restore_uncertain",)
                )
                self._record_composition_runtime_failure(
                    generation,
                    cast(BaseException, commit_error),
                    resource=participant_resource,
                    formal_effects=participant_effects,
                    recovery_target="base",
                )
            else:
                self._abort_reload(
                    generation,
                    error=str(commit_error) or type(commit_error).__name__,
                )
            if (
                self._endpoint_resumer is not None
                and exclusive_endpoint_changed
                and self.current_snapshot is not None
                and self.current_snapshot.accepting_leases
            ):
                await self._endpoint_resumer()
            if runtime_restore_error is not None:
                raise RuntimeError(
                    "Snapshot commit 失败后旧 v3 runtime 恢复失败"
                ) from runtime_restore_error
            if exclusive_endpoint_changed and isinstance(
                commit_error, _PublicationParticipantSwitchError
            ):
                return self._publication_status(
                    plugin_id,
                    active=active,
                    candidate=generation,
                    publication_state="failed",
                )
            raise commit_error
        if commit_error is None:
            if not stage_latest:
                await self._start_current_runtime_snapshot()
            generation.publication_created_data_dir = False

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
        generation.replaced_composition_runtime_generation = None
        generation.formal_root_stopped = False
        generation.formal_root_released = False
        if active is not None:
            active.state = "retired"
        resume_cancelled = False
        if self._endpoint_resumer is not None and exclusive_endpoint_changed:
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

    async def _restore_direct_candidate_runtime(
        self,
        generation: PluginGeneration,
        *,
        validation_snapshot: RuntimeSnapshot,
        stable_snapshot: RuntimeSnapshot | None,
    ) -> RuntimeSnapshot:
        """Close validation and stable Roots before rebuilding the formal Root."""

        candidate_runtime = self._composition_generation_host.get(
            generation.generation_id
        )
        expected_mcp_catalog_digests = (
            None if candidate_runtime is None else candidate_runtime.mcp_catalog_digests
        )
        if self._dashboard_validation_releaser is not None:
            await self._dashboard_validation_releaser(validation_snapshot)
        await self._stop_runtime_snapshot(validation_snapshot)
        await self._stop_composition_generation_runtime(generation)
        previous_root = validation_snapshot.composition_root
        if previous_root is not None:
            await previous_root.dispose()
        created_data_dir = not generation.data_dir.exists()
        ensure_workspace_plugin_data_dir(generation.data_dir, self._workspace)
        production_snapshot: RuntimeSnapshot | None = None
        try:
            await self._stop_stable_root(generation, stable_snapshot)
            production_snapshot = await self._compile_generation_snapshot(
                generation,
            )
            try:
                _validate_candidate_formal_snapshot_identity(
                    generation,
                    candidate=validation_snapshot,
                    formal=production_snapshot,
                )
            except RuntimeError:
                await self._dispose_unreferenced_composition_root(production_snapshot)
                raise
            await self._start_snapshot_composition_runtimes(
                production_snapshot,
                candidate=generation,
                expected_mcp_catalog_digests=expected_mcp_catalog_digests,
            )
        except BaseException:
            if production_snapshot is not None:
                await self._stop_snapshot_composition_runtimes(production_snapshot)
            if production_snapshot is not None:
                await self._dispose_unreferenced_composition_root(production_snapshot)
            if created_data_dir:
                _remove_validation_data_dir(generation.data_dir)
            raise
        generation.publication_created_data_dir = created_data_dir
        _replace_snapshot_payload(validation_snapshot, production_snapshot)
        validation_workspace = generation.validation_workspace
        generation.validation_workspace = None
        if self._dashboard_preparer is not None:
            self._dashboard_preparer(validation_snapshot)
        generation.runtime_snapshot = validation_snapshot
        if validation_workspace is not None:
            _remove_validation_data_dir(validation_workspace.parent)
        generation.validation_data_inventory = ()
        return validation_snapshot

    def _activate_published_generation(
        self,
        generation: PluginGeneration,
        previous: PluginGeneration | None,
    ) -> None:
        plugin_dir = generation.plugin_dir.resolve(strict=False)
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
        )

    async def _prepare_generation(
        self,
        generation: PluginGeneration,
    ) -> None:
        if generation.prepare_started:
            return
        assert generation.runtime_snapshot is not None
        generation.prepare_started = True
        generation.minimum_resource_count = generation.scope.resource_count

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
        for item in snapshot.generations.values():
            if item.scope.closed:
                raise RuntimeError("RuntimeSnapshot 插件作用域已关闭")
            if item.scope.resource_count < item.minimum_resource_count:
                raise RuntimeError("RuntimeSnapshot 插件资源数量不足")

    def _advance_reload(
        self,
        generation: PluginGeneration,
        phase: ReloadPhase,
        *,
        candidate_snapshot_id: str | None = None,
        error: str = "",
        resource: str | None = None,
        formal_effects: tuple[str, ...] | None = None,
        recovery_action: RecoveryActionName | None = None,
        attempt_count: int | None = None,
        details: dict[str, object] | None = None,
        recovery_target: RecoveryTarget | None = None,
    ) -> None:
        tx_id = generation.reload_tx_id
        if tx_id is None:
            return
        self._reload_journal.advance(
            tx_id,
            phase,
            candidate_snapshot_id=candidate_snapshot_id,
            error=error,
            resource=resource,
            formal_effects=formal_effects,
            recovery_action=recovery_action,
            attempt_count=attempt_count,
            details=details,
            recovery_target=recovery_target,
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
        if phase in {
            "complete",
            "aborted",
            "recovered",
            "cleanup_failed",
            "degraded",
        }:
            return
        self._advance_reload(generation, "aborted", error=error)

    async def _abort_failed_publication(
        self,
        generation: PluginGeneration,
        transaction: SnapshotTransaction,
        *,
        error: str,
        finish_journal: bool = True,
        reopen_previous: bool = True,
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
            _, _ = await _complete_critical(
                self._snapshot_store.abort(
                    transaction,
                    reopen_previous=reopen_previous,
                )
            )
        except BaseException as caught:
            snapshot_error = caught
        tx_id = generation.reload_tx_id
        if snapshot_error is not None and tx_id is not None:
            phase = self._reload_journal.get(tx_id).phase
            if phase not in {"cleanup_failed", "degraded"}:
                self._record_composition_runtime_failure(
                    generation,
                    snapshot_error,
                    resource="runtime-snapshot-drain",
                    formal_effects=(
                        "candidate_pointer_restored",
                        "candidate_runtime_cleanup_pending",
                    ),
                )
        if (
            finish_journal
            and pointer_error is None
            and (
                tx_id is None
                or self._reload_journal.get(tx_id).phase
                not in {"cleanup_failed", "degraded"}
            )
        ):
            self._abort_reload(generation, error=error)

        # 3. Root 已排空后恢复本次 publication 才创建的正式数据身份。
        if snapshot_error is not None:
            raise RuntimeError(
                "候选发布失败后 RuntimeSnapshot 回收失败"
            ) from snapshot_error
        if generation.publication_created_data_dir:
            _remove_validation_data_dir(generation.data_dir)
            generation.publication_created_data_dir = False

        # 4. 清理异常优先暴露，避免把半完成恢复伪装成原始发布失败。
        if pointer_error is not None:
            raise RuntimeError(
                "候选发布失败后 artifact pointer 恢复失败"
            ) from pointer_error

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
        stage_stable: bool = False,
    ) -> PluginGeneration | None:
        stable_module_path = mod["import_path"]
        plugin_dir = Path(mod["plugin_root"])
        initial_plugin_id = _resolve_plugin_id(mod)
        if activate and initial_plugin_id in self._active_generations:
            return self._active_generations[initial_plugin_id]
        plugin_manifest = load_plugin_manifest(
            _plugins_home(self._installed_cache_root)
        )
        if plugin_manifest.get(initial_plugin_id, True) is False:
            logger.info("插件已禁用（manifest.toml）: %s", initial_plugin_id)
            return None
        created_activation_data_dir = False
        self._generation_sequence += 1
        generation_sequence = self._generation_sequence
        module_path = mod["module_path"].strip()
        static_manifest: StaticPluginManifest | None = None
        manifest_path = plugin_dir / "akashic.plugin.toml"
        if manifest_path.exists() or manifest_path.is_symlink():
            try:
                # Static identity is the admission source.  No plugin module is
                # imported until this parse and the discovered entrypoint agree.
                static_manifest = load_static_plugin_manifest(plugin_dir)
                expected_module_path = plugin_dir / static_manifest.entrypoint
                discovered_entrypoint = mod.get("entrypoint", "plugin.py")
                if discovered_entrypoint != static_manifest.entrypoint:
                    raise RuntimeError(
                        "source discovery entrypoint 与静态 manifest 不一致: "
                        f"discovered={discovered_entrypoint} "
                        f"manifest={static_manifest.entrypoint}"
                    )
                if mod.get("manifest_digest", "") != static_manifest.identity_digest:
                    raise RuntimeError("source discovery manifest identity 已漂移")
                if Path(module_path).resolve(
                    strict=False
                ) != expected_module_path.resolve(strict=False):
                    raise RuntimeError(
                        "source discovery module path 与静态 manifest 不一致: "
                        f"discovered={module_path} expected={expected_module_path}"
                    )
            except Exception as error:
                raise RuntimeError(
                    f"插件 {initial_plugin_id} 静态 manifest admission 失败: {error}"
                ) from error
        elif mod.get("source_type") == "installed":
            raise RuntimeError(
                f"installed 插件缺少静态 v3 manifest: {initial_plugin_id}"
            )
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
                    plugin_dir=plugin_dir,
                    source_type=mod.get("source_type", "builtin"),
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
        validate_workspace_plugin_data_path(data_dir, self._workspace)
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
                plugin_dir=plugin_dir,
                source_type=mod.get("source_type", "builtin"),
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
        # Builtin v3 may omit a manifest; installed artifacts were rejected above.
        try:
            self._import_plugin(mp, Path(module_path))
        except Exception as error:
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
            raise RuntimeError(
                f"插件 {initial_plugin_id} 导入失败: {error_text}"
            ) from error
        loaded_module = sys.modules.get(mp)
        if static_manifest is not None:
            try:
                if not isinstance(loaded_module, ModuleType):
                    raise RuntimeError("v3 插件模块未保留在 import registry")
                validate_module_exports(
                    static_manifest,
                    loaded_module,
                    plugin_root=plugin_dir,
                )
            except Exception as error:
                self._remove_module_tree(mp)
                error_text = str(error) or type(error).__name__
                self._record_failed_gate(
                    plugin_id=initial_plugin_id,
                    revision=source_revision,
                    check_id="static_manifest_exports",
                    reason=error_text,
                )
                self._abort_reload_attempt(
                    reload_tx_id,
                    error=f"static_manifest_exports: {error_text}",
                )
                return None
        is_v3 = (
            loaded_module is not None
            and getattr(loaded_module, "api_version", None) == 3
        )
        if not is_v3:
            self._remove_module_tree(mp)
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id="plugin_api",
                reason="plugin.py 必须声明 api_version = 3",
            )
            self._abort_reload_attempt(
                reload_tx_id,
                error="plugin_api: plugin.py 必须声明 api_version = 3",
            )
            raise RuntimeError(f"插件只接受 api_version = 3: {initial_plugin_id}")
        try:
            if not isinstance(loaded_module, ModuleType):
                raise RuntimeError("v3 插件模块未保留在 import registry")
            instance = ComposablePlugin.from_module(loaded_module)
            config_model = cast(type[BaseModel] | None, instance.ConfigModel)
            name = str(instance.name or mod["name"]).strip()
            if not name:
                raise RuntimeError("插件缺少 name")
            plugin_id = f"{name}@{mod['marketplace']}" if mod["marketplace"] else name
            if plugin_id != initial_plugin_id:
                raise RuntimeError(
                    f"插件目录身份与声明不一致: directory={initial_plugin_id} declared={plugin_id}"
                )
            credential_paths = (
                _static_channel_credential_paths(static_manifest)
                if static_manifest is not None
                else ()
            )
            credential_alias_groups = (
                _validate_channel_credential_schema(
                    config_model,
                    credential_paths=credential_paths,
                )
                if static_manifest is not None
                else ()
            )
            config_projection = _read_plugin_config_projection(
                data_dir,
                credential_paths=credential_paths,
                credential_alias_groups=credential_alias_groups,
            )
            plugin_config = _validate_plugin_config_projection(
                config_projection,
                config_model,
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
        if not stage_stable and activate:
            created_activation_data_dir = not data_dir.exists()
            ensure_workspace_plugin_data_dir(data_dir, self._workspace)
        scope = PluginScope(plugin_id, generation_id=generation_id)
        generation: PluginGeneration | None = None

        async def rollback_load(error: str) -> None:
            if reload_tx_id is not None:
                phase = self._reload_journal.get(reload_tx_id).phase
                if phase not in {"complete", "aborted", "recovered"}:
                    self._reload_journal.advance(
                        reload_tx_id,
                        "aborted",
                        error=error,
                    )
            if generation is not None and generation.runtime_snapshot is not None:
                await self._dispose_unreferenced_composition_root(
                    generation.runtime_snapshot
                )
            self._cleanup_failures.extend(await scope.aclose())
            if created_activation_data_dir:
                _remove_validation_data_dir(data_dir)
            elif generation is not None and generation.boot_created_data_dir:
                _remove_validation_data_dir(generation.data_dir)
                generation.boot_created_data_dir = False
            self._remove_module_tree(mp)

        try:
            load_phase = "declarations"
            instance.bind_static_services(self._composition_service_view())
            contributions = self._collect_candidate_contributions(
                instance=instance,
                plugin_id=plugin_id,
                plugin_dir=plugin_dir,
            )
            gate_result = self._validate_candidate(
                instance=instance,
                plugin_id=plugin_id,
                revision=source_revision,
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
                plugin_dir=plugin_dir,
                data_dir=data_dir,
                config=plugin_config,
                config_projection=config_projection,
                instance=instance,
                scope=scope,
                contributions=contributions,
                gate_result=gate_result,
                source_type=cast(
                    Literal["builtin", "installed"],
                    mod["source_type"],
                ),
                static_manifest=static_manifest,
                entrypoint=(
                    static_manifest.entrypoint
                    if static_manifest is not None
                    else "plugin.py"
                ),
                state="prepared",
                reload_tx_id=reload_tx_id,
            )
            if stage_stable:
                generation.boot_created_data_dir = not data_dir.exists()
                ensure_workspace_plugin_data_dir(data_dir, self._workspace)
            catalog_generations = [
                active_generation
                for active_generation in self._active_generations.values()
                if active_generation.plugin_id != plugin_id
            ]
            catalog_generations.append(generation)
            catalog_generations = self._static_active_generations(catalog_generations)
            ignored_generations = self._static_active_generations(
                [*self._active_generations.values(), generation]
            )
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
            if not activate:
                validation_root = (
                    self._workspace
                    / "runtime"
                    / "plugin-validation"
                    / generation.generation_id
                )
                if validation_root.exists():
                    raise RuntimeError(f"候选验证目录已存在: {validation_root}")
                validation_workspace = validation_root / "workspace"
                generation.validation_workspace = validation_workspace
                generation.scope.defer(
                    "validation_plugin_data",
                    lambda: asyncio.to_thread(
                        _remove_validation_data_dir, validation_root
                    ),
                )
            if not activate and _installed_generation_is_candidate(generation):
                generation.production_contributions = contributions
                generation.production_data_dir = generation.data_dir
                assert generation.validation_workspace is not None
                validation_data_dir = (
                    generation.validation_workspace
                    / "plugin-data"
                    / generation.data_dir.name
                )
                validation_data_dir.parent.mkdir(parents=True, exist_ok=True)
                generation.validation_data_inventory = _copy_validation_data(
                    generation.data_dir,
                    validation_data_dir,
                    _candidate_data_exclude_paths(generation),
                )
                generation.data_dir = validation_data_dir
            if not activate:
                generation.runtime_snapshot = await self._compile_generation_snapshot(
                    generation,
                    candidate_owner=generation,
                )
                self._advance_reload(
                    generation,
                    "prepared",
                    candidate_snapshot_id=generation.runtime_snapshot.snapshot_id,
                )
                generation.minimum_resource_count = scope.resource_count
                self._prepared_generations[plugin_id] = generation
                return generation
            if stage_stable:
                return generation
            generation.runtime_snapshot = await self._compile_generation_snapshot(
                generation,
                allow_pending_composition=True,
            )
            load_phase = "prepare"
            await self._prepare_generation(generation)
            generation.state = "activating"
            load_phase = "publish"
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
        )
        generation.state = "active"
        self._active_generations[plugin_id] = generation
        self._stable_aliases[mp] = stable_module_path
        self._remove_module_tree(stable_module_path)
        self._fresh_importer.register(stable_module_path, plugin_dir)
        sys.modules[stable_module_path] = sys.modules[mp]
        assert generation.runtime_snapshot is not None
        await self._publish_committed_snapshot(generation.runtime_snapshot)
        logger.info("插件已加载: %s", mod["name"])
        return generation

    async def _compile_generation_snapshot(
        self,
        generation: PluginGeneration,
        *,
        allow_pending_composition: bool = False,
        candidate_owner: PluginGeneration | None = None,
        force_fresh_composition: bool = False,
    ) -> RuntimeSnapshot:
        generations = dict(self._active_generations)
        generations[generation.plugin_id] = generation
        composition_root, created_root = await self._resolve_composition_root(
            generations,
            allow_pending=allow_pending_composition,
            candidate_owner=candidate_owner,
            force_fresh=force_fresh_composition,
        )
        try:
            uses_overlay = isinstance(composition_root, CompositionOverlay)
            snapshot = self._snapshot_compiler.compile(
                generations,
                catalog_generation=generation,
                composition_root=composition_root,
                base_snapshot=(
                    self.current_snapshot
                    if candidate_owner is not None and uses_overlay
                    else None
                ),
                replaced_plugin_ids=(
                    composition_root.replaced_plugin_ids
                    if uses_overlay
                    else frozenset()
                ),
                core_channel_definitions=self._core_channel_definitions,
                require_composition_ready=True,
            )
            _validate_static_manifest_runtime(snapshot, generations)
            if candidate_owner is not None:
                self._preflight_durable_delivery_targets(snapshot)
            snapshot.tool_registry = self._compile_snapshot_tools(
                generations,
                snapshot.plugin_tool_catalog,
            )
            return snapshot
        except Exception as error:
            if created_root and composition_root is not None:
                await composition_root.dispose()
            gate = _with_gate_check(
                generation.gate_result,
                check_id="runtime_snapshot",
                passed=False,
                evidence=str(error),
            )
            generation.gate_result = gate
            self._gate_results[generation.plugin_id] = gate
            raise _CandidateRejected(gate) from error

    def _read_existing_session_compaction(self, session_key: str):
        """读取同一 Session 的消息与 active compaction 语义。"""

        session_manager = self._session_manager
        if session_manager is None:
            raise RuntimeError("Session Read Service 缺少 SessionManager")
        session = session_manager.get_existing(session_key)
        compaction = session_manager.control_store.get_active_compaction(session_key)
        return session, compaction

    async def _resolve_composition_root(
        self,
        generations: dict[str, PluginGeneration],
        *,
        allow_pending: bool = False,
        candidate_owner: PluginGeneration | None = None,
        force_fresh: bool = False,
    ) -> tuple[CompositionSnapshotRoot | None, bool]:
        """复用 stable Root，或为 candidate 挂载一个增量拓扑。"""

        # 1. 只有 stable-to-stable 的纯 payload 变化可以复用 Root。
        ordered = tuple(
            generation
            for generation in sorted(
                generations.values(), key=lambda item: item.plugin_id
            )
        )
        current = self.current_snapshot
        current_ordered = (
            tuple(
                generation
                for generation in sorted(
                    current.generations.values(), key=lambda item: item.plugin_id
                )
            )
            if current is not None
            else ()
        )
        stable_root = None if current is None else current.composition_root
        candidate_plugin_ids: frozenset[str] = (
            frozenset()
            if candidate_owner is None
            else self._candidate_dependency_closure(
                ordered,
                stable_root,
                candidate_owner.plugin_id,
            )
        )
        mount_order = (
            ordered
            if candidate_owner is None
            else tuple(
                item for item in ordered if item.plugin_id in candidate_plugin_ids
            )
        )
        if (
            candidate_owner is None
            and not force_fresh
            and current is not None
            and len(ordered) == len(current_ordered)
            and all(
                left is right
                for left, right in zip(ordered, current_ordered, strict=True)
            )
        ):
            return current.composition_root, False
        if not ordered and not self._core_channel_definitions:
            return None, False

        # 2. stable 拓扑变化创建完整 Root；candidate Root 只拥有变更插件。
        identity = "|".join(
            f"{item.plugin_id}:{item.generation_id}" for item in ordered
        )
        if not identity:
            identity = "core-channels:" + "|".join(
                f"{item.name}:{item.generation_id}:{item.source_revision}:{item.config_revision}"
                for item in self._core_channel_definitions
            )
        root = CompositionRoot(
            "plugins:" + hashlib.sha256(identity.encode()).hexdigest()[:16],
            candidate_incident_limit=(1024 if candidate_owner is not None else None),
        )
        root._bind_runtime_scope_acquirer(
            lambda: self._snapshot_store.acquire_composition_root(root)
        )
        try:
            _ = await root.context.provide(COMMANDS, PluginCommands())
            if any(
                CHANNELS in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                _ = await root.context.provide(
                    CHANNELS,
                    PluginChannels(root.instance_token),
                )
            if any(
                MCP_SERVERS in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                _ = await root.context.provide(
                    MCP_SERVERS,
                    PluginMcpServers(root.instance_token),
                )
            if any(
                MANAGED_PROCESSES in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                _ = await root.context.provide(
                    MANAGED_PROCESSES,
                    PluginManagedProcesses(root.instance_token),
                )
            if any(
                WORKLOADS in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                _ = await root.context.provide(
                    WORKLOADS,
                    PluginWorkloads(root.instance_token),
                )
            if any(
                BACKGROUND_JOBS in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                _ = await root.context.provide(
                    BACKGROUND_JOBS,
                    PluginBackgroundJobs(root.instance_token),
                )
            if any(
                TOOL_CATALOG in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                _ = await root.context.provide(
                    TOOL_CATALOG,
                    PluginTools(root.instance_token),
                )
            if any(
                UI_SLOTS in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                _ = await root.context.provide(UI_SLOTS, PluginUiSlots())
            if self._session_manager is not None and any(
                SESSION_READ in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                session_read = (
                    SessionReadService(self._read_existing_session_compaction)
                    if candidate_owner is None
                    else SessionReadService.candidate_validation()
                )
                _ = await root.context.provide(SESSION_READ, session_read)
            if self._session_manager is not None and any(
                SESSION_COMPACTION_STORAGE
                in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                compaction_storage = (
                    SessionCompactionStorage(self._session_manager)
                    if candidate_owner is None
                    else SessionCompactionStorage.candidate_validation()
                )
                _ = await root.context.provide(
                    SESSION_COMPACTION_STORAGE,
                    compaction_storage,
                )
            if any(
                SCOPED_TURNS in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):

                async def acquire_root_scope() -> RuntimeSnapshotLease:
                    return await self._snapshot_store.acquire_composition_root(root)

                scoped_turns = (
                    PluginScopedTurns(
                        self._conversation_runtime,
                        self._programmatic_session_creator,
                        self._programmatic_session_reader,
                        acquire_root_scope,
                    )
                    if candidate_owner is None
                    else PluginScopedTurns.candidate_validation()
                )
                _ = await root.context.provide(SCOPED_TURNS, scoped_turns)
            if any(
                CONTINUATIONS in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                continuations = (
                    PluginContinuations(self._continuation_publisher)
                    if candidate_owner is None
                    else PluginContinuations.candidate_validation()
                )
                _ = await root.context.provide(CONTINUATIONS, continuations)
            if any(
                TIMERS in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                timers = (
                    PluginTimers(AsyncioOneShotTimer())
                    if candidate_owner is None
                    else PluginTimers.candidate_validation()
                )
                _ = await root.context.provide(TIMERS, timers)
            if any(
                DELIVERIES in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                deliveries = (
                    PluginDeliveries(self._delivery_sender)
                    if candidate_owner is None
                    else PluginDeliveries.candidate_validation()
                )
                _ = await root.context.provide(DELIVERIES, deliveries)
            if any(
                DURABLE_DELIVERIES in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                durable_deliveries = (
                    self._formal_durable_deliveries()
                    if candidate_owner is None
                    else PluginDurableDeliveries.candidate_validation()
                )
                _ = await root.context.provide(DURABLE_DELIVERIES, durable_deliveries)
            if any(
                INTERACTION_UNDO in cast(ComposablePlugin, item.instance).inject
                for item in mount_order
            ):
                if candidate_owner is None and self._interaction_undo is None:
                    raise RuntimeError("INTERACTION_UNDO 需要 Session owner")
                interaction_undo = (
                    InteractionUndoService(self._interaction_undo.undo_latest)
                    if candidate_owner is None and self._interaction_undo is not None
                    else InteractionUndoService.candidate_validation()
                )
                _ = await root.context.provide(INTERACTION_UNDO, interaction_undo)
            if candidate_owner is None:
                for item in ordered:
                    await self._mount_generation_composition(root, item)
                resolved_root: CompositionSnapshotRoot = root
            else:
                await self._mount_candidate_composition(
                    root,
                    mount_order,
                    candidate_owner=candidate_owner,
                )
                if stable_root is None and len(generations) == 1:
                    resolved_root = root
                elif isinstance(stable_root, CompositionRoot):
                    resolved_root = CompositionOverlay(
                        stable_root,
                        root,
                        plugin_ids=frozenset(generations),
                        replaced_plugin_ids=candidate_plugin_ids,
                    )
                else:
                    raise RuntimeError("candidate 增量验证需要一个正式 stable Root")
            receipt = resolved_root.receipt()
            if not receipt.ready:
                missing_services = tuple(
                    sorted(
                        {
                            service
                            for fiber in receipt.fibers
                            if fiber.name in receipt.required_pending
                            for service in fiber.missing_services
                        }
                    )
                )
                if (
                    allow_pending
                    and receipt.required_pending
                    and all(
                        fiber.state == FiberState.PENDING
                        for fiber in receipt.fibers
                        if fiber.name in receipt.required_pending
                    )
                    and not receipt.required_degraded
                    and not receipt.incident_overflowed
                    and not receipt.external_effects
                ):
                    self._composition_pending = missing_services
                    await root.dispose()
                    return None, False
                raise RuntimeError(
                    "v3 插件组合拓扑未就绪: "
                    f"required_pending={receipt.required_pending}, "
                    f"missing_services={missing_services}, "
                    f"required_degraded={receipt.required_degraded}, "
                    f"incidents={receipt.incidents}, "
                    f"incident_overflowed={receipt.incident_overflowed}, "
                    f"external_effects={receipt.external_effects}"
                )
            self._composition_pending = ()
            sealing_result = await resolved_root.context.serial(
                SNAPSHOT_SEALING,
                SnapshotSealing(),
            )
            if sealing_result is not None:
                raise CompositionError(
                    "SNAPSHOT_SEALING_BAIL_NOT_ALLOWED",
                    "snapshot.sealing 接入点不接受 Bail",
                )
        except BaseException:
            await root.dispose()
            raise
        return resolved_root, True

    def _candidate_dependency_closure(
        self,
        ordered: tuple[PluginGeneration, ...],
        stable_root: CompositionSnapshotRoot | None,
        candidate_plugin_id: str,
    ) -> frozenset[str]:
        """Find the explicit Service component that must rebuild together."""

        if stable_root is None:
            return frozenset({candidate_plugin_id})
        if not isinstance(stable_root, CompositionRoot):
            raise RuntimeError("candidate 增量验证需要一个正式 stable Root")
        owners_by_name = {
            key.name: owner
            for key, owner in stable_root.plugin_service_owners().items()
        }
        generations = {item.plugin_id: item for item in ordered}
        adjacency = {plugin_id: set() for plugin_id in generations}
        for plugin_id, generation in generations.items():
            plugin = cast(ComposablePlugin, generation.instance)
            dependency_names = {key.name for key in plugin.inject}
            dependency_names.update(
                dependency
                for fiber in stable_root.topology_view(
                    plugin_ids=frozenset({plugin_id})
                ).fibers
                for dependency in fiber.dependencies
            )
            for dependency_name in dependency_names:
                owner = owners_by_name.get(dependency_name)
                if owner is None or owner == plugin_id or owner not in generations:
                    continue
                adjacency[plugin_id].add(owner)
                adjacency[owner].add(plugin_id)
        selected = {candidate_plugin_id}
        pending = [candidate_plugin_id]
        while pending:
            plugin_id = pending.pop()
            for neighbor in adjacency[plugin_id]:
                if neighbor in selected:
                    continue
                selected.add(neighbor)
                pending.append(neighbor)
        return frozenset(selected)

    def _formal_durable_deliveries(self) -> PluginDurableDeliveries:
        """Build one Root-local port over the process-owned delivery ledger."""

        sender = self._durable_delivery_sender
        session_manager = self._session_manager
        projector: DurableProjector | None = None
        if session_manager is not None:

            async def project(request: DurableDeliveryRequest) -> str:
                return await session_manager.append_durable_delivery(
                    session_key=request.projection_session_id,
                    content=request.body,
                    delivery_id=request.logical_delivery_id,
                    control_turn_id=request.accepted_turn.turn_id,
                    metadata=request.metadata,
                )

            projector = project

        service = PluginDurableDeliveries(
            DurableDeliveryStore(
                self._workspace / "runtime" / "deliveries" / "settlements.sqlite"
            ),
            sender,
            projector,
            recover_started=not self._durable_delivery_recovered,
        )
        self._durable_delivery_recovered = True
        return service

    def _preflight_durable_delivery_targets(self, snapshot: RuntimeSnapshot) -> None:
        """Fence forward-completable rows whose target vanished from candidate."""

        store = DurableDeliveryStore(
            self._workspace / "runtime" / "deliveries" / "settlements.sqlite",
            read_only=True,
        )
        forward_targets = store.forward_targets()
        if not forward_targets:
            return
        topology = snapshot.composition_topology
        if topology is None:
            raise RuntimeError("durable delivery candidate 缺少 composition topology")
        missing = tuple(sorted(forward_targets.difference(topology.services)))
        if missing:
            raise RuntimeError(
                "durable delivery forward target service 不可解析: "
                + ", ".join(missing)
            )

    def _composition_service_view(self) -> ServiceView:
        """冻结静态 v3 声明可读取的 Core service 输入。"""

        values: dict[Any, object] = {}
        values[SCOPED_TURNS] = (
            PluginScopedTurns(
                self._conversation_runtime,
                self._programmatic_session_creator,
                self._programmatic_session_reader,
            )
            if self._conversation_runtime is not None
            and self._programmatic_session_creator is not None
            else PluginScopedTurns.candidate_validation()
        )
        values[CONTINUATIONS] = (
            PluginContinuations(self._continuation_publisher)
            if self._continuation_publisher is not None
            else PluginContinuations.candidate_validation()
        )
        values[TIMERS] = PluginTimers(AsyncioOneShotTimer())
        values[DELIVERIES] = (
            PluginDeliveries(self._delivery_sender)
            if self._delivery_sender is not None
            else PluginDeliveries.candidate_validation()
        )
        values[DURABLE_DELIVERIES] = PluginDurableDeliveries.candidate_validation()
        return ServiceView.freeze(values)

    @staticmethod
    def _static_active_generations(
        generations: list[PluginGeneration],
    ) -> list[PluginGeneration]:
        """用 snapshot 相同的 active 合同过滤静态 catalog。"""

        return [
            generation
            for generation in generations
            if cast(ComposablePlugin, generation.instance).static_active
        ]

    async def _mount_generation_composition(
        self,
        root: CompositionRoot,
        generation: PluginGeneration,
    ) -> None:
        """用 generation 自己的正式 runtime 挂载一个 v3 插件。"""

        plugin = cast(ComposablePlugin, generation.instance)
        for name in plugin.workspace_roots:
            _ = resolve_declared_workspace_root(self._workspace, name)
        for name in plugin.workspace_files:
            _ = resolve_declared_workspace_file(self._workspace, name)
        _ = await root._mount_module(  # pyright: ignore[reportPrivateUsage]
            plugin.apply,
            name=generation.plugin_id,
            inject=plugin.inject,
            plugin_module=plugin.module,
            static_active=plugin.static_active,
            runtime=PluginRuntime(
                plugin_id=generation.plugin_id,
                generation_id=generation.generation_id,
                plugin_dir=generation.plugin_dir,
                data_dir=generation.data_dir,
                workspace=self._workspace,
                config=generation.config,
                workspace_roots=plugin.workspace_roots,
                workspace_files=plugin.workspace_files,
            ),
        )

    async def _mount_candidate_composition(
        self,
        root: CompositionRoot,
        selected: tuple[PluginGeneration, ...],
        *,
        candidate_owner: PluginGeneration,
    ) -> None:
        """挂载变更插件与它实际依赖的上游 provider 闭包。"""

        validation_workspace = candidate_owner.validation_workspace
        if validation_workspace is None:
            raise RuntimeError(f"候选缺少隔离 workspace: {candidate_owner.plugin_id}")
        attempt_root = (
            validation_workspace.parent / "composition" / secrets.token_hex(8)
        )
        attempt_workspace = attempt_root / "workspace"
        root._defer_internal_cleanup(  # pyright: ignore[reportPrivateUsage]
            "candidate_attempt_data",
            lambda: _remove_validation_data_dir(attempt_root),
        )
        clones: list[tuple[PluginGeneration, ComposablePlugin, Path, object]] = []
        for generation in selected:
            clone, module_path, data_dir, config = self._clone_candidate_composable(
                generation,
                candidate_owner=candidate_owner,
                attempt_workspace=attempt_workspace,
            )
            root._defer_internal_cleanup(  # pyright: ignore[reportPrivateUsage]
                f"candidate_module:{module_path}",
                lambda module_path=module_path: self._remove_module_tree(module_path),
            )
            original = cast(ComposablePlugin, generation.instance)
            if clone.workspace_roots != original.workspace_roots:
                raise RuntimeError(
                    "candidate workspace_roots 与 generation 冻结声明不一致: "
                    f"{generation.plugin_id}"
                )
            if clone.workspace_files != original.workspace_files:
                raise RuntimeError(
                    "candidate workspace_files 与 generation 冻结声明不一致: "
                    f"{generation.plugin_id}"
                )
            clones.append((generation, clone, data_dir, config))
        self._project_candidate_workspace_roots(
            tuple(item[1] for item in clones),
            attempt_workspace,
        )
        self._project_candidate_workspace_files(
            tuple(item[1] for item in clones),
            attempt_workspace,
        )
        for generation, clone, data_dir, config in clones:
            _ = await root._mount_module(  # pyright: ignore[reportPrivateUsage]
                clone.apply,
                name=generation.plugin_id,
                inject=clone.inject,
                plugin_module=clone.module,
                static_active=clone.static_active,
                runtime=PluginRuntime(
                    plugin_id=generation.plugin_id,
                    generation_id=generation.generation_id,
                    plugin_dir=generation.plugin_dir,
                    data_dir=data_dir,
                    workspace=attempt_workspace,
                    config=config,
                    workspace_roots=clone.workspace_roots,
                    workspace_files=clone.workspace_files,
                ),
            )

    def _project_candidate_workspace_roots(
        self,
        plugins: tuple[ComposablePlugin, ...],
        attempt_workspace: Path,
    ) -> None:
        """把声明式共享目录复制到一次 candidate attempt。"""

        # 1. 全部 generation 由同一个 Manager workspace 发布。
        names: set[str] = set()
        for plugin in plugins:
            names.update(plugin.workspace_roots)

        # 2. 缺失目录保持缺失；已有目录获得独立副本。
        for name in sorted(names):
            source = resolve_declared_workspace_root(self._workspace, name)
            if not source.exists():
                continue
            _ = shutil.copytree(source, attempt_workspace / name)

    def _project_candidate_workspace_files(
        self,
        plugins: tuple[ComposablePlugin, ...],
        attempt_workspace: Path,
    ) -> None:
        """Copy declared product files into the isolated candidate workspace."""

        names = {name for plugin in plugins for name in plugin.workspace_files}
        for name in sorted(names):
            source = resolve_declared_workspace_file(self._workspace, name)
            if not source.exists():
                continue
            target = attempt_workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copy2(source, target)

    def _clone_candidate_composable(
        self,
        generation: PluginGeneration,
        *,
        candidate_owner: PluginGeneration,
        attempt_workspace: Path,
    ) -> tuple[ComposablePlugin, str, Path, object]:
        """重新导入一个 stable v3 插件并绑定 candidate 临时数据。"""

        plugin_dir = generation.plugin_dir
        data_dir = attempt_workspace / "plugin-data" / generation.data_dir.name
        _ = data_dir.parent.mkdir(parents=True, exist_ok=True)
        inventory = _copy_validation_data(
            generation.data_dir,
            data_dir,
            _candidate_data_exclude_paths(generation),
        )
        if generation is candidate_owner:
            generation.validation_data_inventory = inventory
        module_path = (
            f"{generation.module_path}__candidate_"
            f"{candidate_owner.generation_id.replace(':', '_')}_"
            f"{secrets.token_hex(4)}"
        )
        entrypoint = generation.entrypoint
        self._import_plugin(module_path, plugin_dir / entrypoint)
        try:
            module = sys.modules[module_path]
            if generation.static_manifest is not None:
                validate_module_exports(
                    generation.static_manifest,
                    module,
                    plugin_root=plugin_dir,
                )
            clone = ComposablePlugin.from_module(module)
            credential_paths = (
                _static_channel_credential_paths(generation.static_manifest)
                if generation.static_manifest is not None
                else ()
            )
            _validate_channel_credential_schema(
                cast(type[BaseModel] | None, clone.ConfigModel),
                credential_paths=credential_paths,
            )
            config = _validate_plugin_config_projection(
                generation.config_projection,
                cast(type[BaseModel] | None, clone.ConfigModel),
            )
            clone.bind_static_services(self._composition_service_view())
            return clone, module_path, data_dir, config
        except BaseException:
            self._remove_module_tree(module_path)
            raise

    async def _start_composition_generation_runtime(
        self,
        generation: PluginGeneration,
        snapshot: RuntimeSnapshot,
        *,
        mode: Literal["candidate", "formal"],
        expected_mcp_catalog_digests: Mapping[str, str] | None = None,
    ) -> CompositionRuntimeGeneration | None:
        """Start one exact Root runtime and refresh snapshot Tool routes."""

        if generation.reload_tx_id is not None and self._composition_runtime_declared(
            snapshot, generation.plugin_id
        ):
            boot_id = os.environ.get("AKASHIC_BOOT_ID", "").strip()
            if boot_id:
                self._reload_journal.mark_runtime_owner(
                    generation.reload_tx_id,
                    boot_id,
                )
        self._composition_runtime_generations[generation.generation_id] = generation
        try:
            runtime = await self._composition_generation_host.start(
                generation,
                snapshot,
                mode=mode,
                expected_mcp_catalog_digests=expected_mcp_catalog_digests,
            )
        except BaseException:
            if (
                self._composition_generation_host.failure(generation.generation_id)
                is None
            ):
                _ = self._composition_runtime_generations.pop(
                    generation.generation_id,
                    None,
                )
            raise
        if runtime is None:
            _ = self._composition_runtime_generations.pop(
                generation.generation_id,
                None,
            )
        self._refresh_composition_runtime_tools(snapshot)
        return runtime

    async def _start_snapshot_composition_runtimes(
        self,
        snapshot: RuntimeSnapshot,
        *,
        candidate: PluginGeneration | None = None,
        expected_mcp_catalog_digests: Mapping[str, str] | None = None,
    ) -> None:
        """Start every managed runtime owned by one formal Root."""

        started: list[PluginGeneration] = []
        try:
            for item in snapshot.generations.values():
                await self._start_composition_generation_runtime(
                    item,
                    snapshot,
                    mode="formal",
                    expected_mcp_catalog_digests=(
                        expected_mcp_catalog_digests if item is candidate else None
                    ),
                )
                started.append(item)
        except BaseException:
            for item in reversed(started):
                await self._stop_composition_generation_runtime(item)
            raise

    async def _stop_snapshot_composition_runtimes(
        self,
        snapshot: RuntimeSnapshot,
    ) -> None:
        """Stop every managed runtime before its formal Root is disposed."""

        for item in reversed(tuple(snapshot.generations.values())):
            await self._stop_composition_generation_runtime(item)

    async def _stop_composition_generation_runtime(
        self,
        generation: PluginGeneration,
    ) -> None:
        """Stop one generation before its exact Root is disposed."""

        await self._composition_generation_host.stop(generation.generation_id)
        _ = self._composition_runtime_generations.pop(
            generation.generation_id,
            None,
        )

    async def _stop_stable_root(
        self,
        generation: PluginGeneration,
        stable_snapshot: RuntimeSnapshot | None,
    ) -> None:
        """Stop and dispose the old formal Root before mounting its replacement."""

        if stable_snapshot is None or stable_snapshot.composition_root is None:
            return
        previous = self._active_generations.get(generation.plugin_id)
        recovery_anchor = previous or next(
            iter(stable_snapshot.generations.values()),
            None,
        )
        generation.replaced_composition_runtime_generation = recovery_anchor
        generation.formal_root_stopped = True
        generation.formal_root_released = False

        # 1. Lifecycle handlers close and join plugin-owned writers.
        await self._stop_runtime_snapshot(stable_snapshot)

        # 2. Exact managed runtimes and Root effects release every formal owner.
        await self._stop_snapshot_composition_runtimes(stable_snapshot)
        await stable_snapshot.composition_root.dispose()
        generation.formal_root_released = True

    async def _recover_stable_root(
        self,
        generation: PluginGeneration,
        stable_snapshot: RuntimeSnapshot,
    ) -> None:
        """Rebuild a stopped stable generation on a fresh formal Root."""

        previous = generation.replaced_composition_runtime_generation
        if not generation.formal_root_stopped:
            raise RuntimeError("formal Root recovery 缺少 stopped stable Root")

        # 1. Remove any unpublished new owner before rebuilding stable.
        candidate_snapshot = generation.runtime_snapshot
        if candidate_snapshot is not None:
            await self._stop_runtime_snapshot(candidate_snapshot)
            await self._stop_snapshot_composition_runtimes(candidate_snapshot)
        else:
            await self._stop_composition_generation_runtime(generation)
        if not generation.formal_root_released:
            await self._stop_runtime_snapshot(stable_snapshot)
            await self._stop_snapshot_composition_runtimes(stable_snapshot)
            old_root = stable_snapshot.composition_root
            if old_root is not None:
                await old_root.dispose()
            generation.formal_root_released = True
        await self._rebuild_stable_root(previous, stable_snapshot)
        generation.replaced_composition_runtime_generation = None
        generation.formal_root_stopped = False
        generation.formal_root_released = False

    async def _rebuild_stable_root(
        self,
        stable: PluginGeneration | None,
        stable_snapshot: RuntimeSnapshot,
    ) -> None:
        """Replace a terminal stable Root with a fresh formal owner."""

        old_root = stable_snapshot.composition_root
        replacement: RuntimeSnapshot | None = None
        if old_root is not None:
            await old_root.dispose()
        try:
            # 1. Recompile formal generations without reusing the terminal old Root.
            if stable is None:
                composition_root, _ = await self._resolve_composition_root(
                    {},
                    force_fresh=True,
                )
                replacement = self._snapshot_compiler.compile(
                    {},
                    composition_root=composition_root,
                    core_channel_definitions=self._core_channel_definitions,
                    require_composition_ready=True,
                )
                replacement.tool_registry = self._compile_snapshot_tools(
                    {},
                    replacement.plugin_tool_catalog,
                )
            else:
                replacement = await self._compile_generation_snapshot(
                    stable,
                    force_fresh_composition=True,
                )
            await self._start_snapshot_composition_runtimes(
                replacement,
            )
        except BaseException:
            if replacement is not None:
                await self._stop_snapshot_composition_runtimes(replacement)
            if replacement is not None:
                await self._dispose_unreferenced_composition_root(replacement)
            raise

        # 2. Replace the closed stable payload only after fresh STARTED succeeds.
        _replace_snapshot_payload(stable_snapshot, replacement)

    async def _restore_replaced_composition_runtime(
        self,
        generation: PluginGeneration,
    ) -> None:
        """Restore an old stable runtime before reopening its snapshot."""

        previous = generation.replaced_composition_runtime_generation
        if previous is None:
            return
        snapshot = self.current_snapshot
        if (
            snapshot is None
            or snapshot.generations.get(previous.plugin_id) is not previous
        ):
            raise RuntimeError("旧 stable runtime snapshot 身份已失效")
        await self._start_composition_generation_runtime(
            previous,
            snapshot,
            mode="formal",
        )
        generation.replaced_composition_runtime_generation = None

    async def _rollback_composition_runtime_replacement(
        self,
        generation: PluginGeneration,
    ) -> None:
        """Stop an unpublished formal runtime and restore the prior stable owner."""

        await self._stop_composition_generation_runtime(generation)
        await self._restore_replaced_composition_runtime(generation)

    def _record_composition_runtime_failure(
        self,
        generation: PluginGeneration,
        error: BaseException,
        *,
        resource: str = "composition-runtime",
        formal_effects: tuple[str, ...],
        recovery_target: RecoveryTarget | None = None,
    ) -> None:
        """Persist one executable runtime failure without releasing its owner."""

        tx_id = self._ensure_runtime_recovery_transaction(generation)
        failure = self._composition_generation_host.failure(generation.generation_id)
        if failure is None:
            action: RecoveryActionName = (
                "retry_generation_cleanup"
                if resource == "runtime-snapshot-drain"
                else "retry_runtime_recovery"
            )
        else:
            action = failure.action
        phase: ReloadPhase = (
            "degraded" if action == "retry_runtime_recovery" else "cleanup_failed"
        )
        failure_resource = (
            f"{resource}:{generation.generation_id}"
            if failure is None
            else ",".join((*failure.resource_names, resource))
        )
        failure_error = (
            str(error) or type(error).__name__ if failure is None else failure.error
        )
        self._reload_journal.advance(
            tx_id,
            phase,
            error=failure_error,
            resource=failure_resource,
            formal_effects=formal_effects,
            recovery_action=action,
            recovery_target=(
                recovery_target
                if recovery_target is not None
                else self._composition_recovery_target(
                    generation,
                    tx_id=tx_id,
                )
            ),
        )
        ready = self._ready_candidate
        if (
            ready is not None
            and ready.candidate.reload_tx_id == tx_id
            and self._snapshot_store.unpromoted_candidate is ready.snapshot
        ):
            _ = self._snapshot_store.pause_candidate_admission(ready.snapshot)

    def _on_composition_runtime_failure(
        self,
        failure: CompositionRuntimeFailure,
    ) -> None:
        """Persist a watchdog failure for the exact generation owner."""

        generation = self._composition_runtime_generations.get(failure.generation_id)
        if generation is None:
            raise RuntimeError(
                "v3 runtime failure 缺少 Manager generation owner: "
                f"{failure.generation_id}"
            )
        self._record_composition_runtime_failure(
            generation,
            RuntimeError(failure.error),
            formal_effects=("runtime_watchdog_failure",),
        )

    def _ensure_runtime_recovery_transaction(
        self,
        generation: PluginGeneration,
    ) -> str:
        """Create a durable owner when cleanup fails outside an active reload."""

        tx_id = generation.reload_tx_id
        if tx_id is not None:
            phase = self._reload_journal.get(tx_id).phase
            if phase not in {"complete", "aborted", "recovered"}:
                return tx_id

        # 1. A stable failure joins the one in-flight candidate transaction.
        candidate = self._prepared_generations.get(generation.plugin_id)
        ready = self._ready_candidate
        if ready is not None and ready.plugin_id == generation.plugin_id:
            candidate = ready.candidate
        current = self.current_snapshot
        if (
            candidate is not None
            and candidate is not generation
            and current is not None
            and current.generations.get(generation.plugin_id) is generation
            and candidate.reload_tx_id is not None
        ):
            candidate_record = self._reload_journal.get(candidate.reload_tx_id)
            if candidate_record.phase not in {"complete", "aborted", "recovered"}:
                if candidate_record.base_generation_id != generation.generation_id:
                    raise RuntimeError(
                        "runtime recovery candidate base generation 身份不一致"
                    )
                return candidate.reload_tx_id

        # 2. Freeze the exact stable artifact identity before exposing recovery.
        base_snapshot = self.current_snapshot
        base_generation = (
            None
            if base_snapshot is None
            else base_snapshot.generations.get(generation.plugin_id)
        )
        base_pointer: str | None = None
        candidate_pointer: str | None = None
        plugin_base = _installed_artifact_base(generation)
        if plugin_base is not None:
            pointers = read_pointers(plugin_base)
            if pointers is None:
                raise RuntimeError(
                    f"runtime cleanup recovery 缺少 artifact pointer: {plugin_base}"
                )
            base_pointer = pointers.stable.path
            if base_snapshot is not None and base_generation is generation:
                candidate_pointer = generation.plugin_dir.relative_to(
                    plugin_base
                ).as_posix()

        # 3. Persist the process boot owner before returning the cleanup failure.
        tx_id = self._reload_journal.begin(
            plugin_id=generation.plugin_id,
            base_snapshot_id=(
                None if base_snapshot is None else base_snapshot.snapshot_id
            ),
            base_generation_id=(
                None if base_generation is None else base_generation.generation_id
            ),
            generation_id=generation.generation_id,
            source_revision=generation.source_revision,
            config_revision=generation.config_revision,
            base_artifact_pointer=base_pointer,
            candidate_artifact_pointer=candidate_pointer,
        )
        generation.reload_tx_id = tx_id
        boot_id = os.environ.get("AKASHIC_BOOT_ID", "").strip()
        if boot_id:
            self._reload_journal.mark_runtime_owner(tx_id, boot_id)
        return tx_id

    def _record_drained_composition_runtime_failure(
        self,
        snapshot: RuntimeSnapshot,
        generation: PluginGeneration,
        error: BaseException,
    ) -> None:
        """Persist one retained Host owner while allowing Root and module drain."""

        drain_tx_id = self._drain_transactions.get(snapshot.snapshot_id)
        tx_id = drain_tx_id or generation.reload_tx_id
        if tx_id is None:
            return
        record = self._reload_journal.get(tx_id)
        if record.phase in {"complete", "aborted", "recovered"}:
            return
        failure = self._composition_generation_host.failure(generation.generation_id)
        action: RecoveryActionName = (
            "retry_generation_cleanup" if failure is None else failure.action
        )
        phase: ReloadPhase = (
            "degraded" if action == "retry_runtime_recovery" else "cleanup_failed"
        )
        self._reload_journal.advance(
            tx_id,
            phase,
            error=(
                str(error) or type(error).__name__ if failure is None else failure.error
            ),
            resource=(
                f"composition-runtime:{generation.generation_id}"
                if failure is None
                else ",".join(failure.resource_names)
            ),
            formal_effects=(
                (
                    "committed_generation_retained",
                    "old_runtime_cleanup_pending",
                )
                if drain_tx_id is not None
                else (
                    "candidate_pointer_restored",
                    "candidate_runtime_cleanup_pending",
                )
            ),
            recovery_action=action,
            recovery_target=(
                record.recovery_target
                or (
                    "candidate"
                    if drain_tx_id is not None
                    else self._composition_recovery_target(
                        generation,
                        tx_id=tx_id,
                    )
                )
            ),
        )

    def _composition_recovery_target(
        self,
        generation: PluginGeneration,
        *,
        tx_id: str | None = None,
    ) -> RecoveryTarget:
        """Resolve the exact durable artifact selected at failure time."""

        if tx_id is None:
            tx_id = generation.reload_tx_id
        if tx_id is None:
            return "base"
        record = self._reload_journal.get(tx_id)
        if record.generation_id != generation.generation_id:
            if record.base_generation_id == generation.generation_id:
                return "base"
            raise RuntimeError("runtime failure generation 不属于 recovery transaction")
        if (
            record.phase in {"cleanup_failed", "degraded"}
            and record.recovery_target is not None
        ):
            return record.recovery_target
        base_pointer = record.base_artifact_pointer
        candidate_pointer = record.candidate_artifact_pointer
        plugin_base = _installed_artifact_base(generation)
        if plugin_base is None or candidate_pointer is None:
            current = self.current_snapshot
            if (
                current is not None
                and current.generations.get(generation.plugin_id) is generation
            ):
                return "candidate"
            return "base"
        pointers = read_pointers(plugin_base)
        if pointers is None:
            raise RuntimeError(
                f"runtime failure 缺少 durable artifact pointer: {plugin_base}"
            )
        if pointers.stable.path == candidate_pointer:
            return "candidate"
        if pointers.stable.path == base_pointer:
            return "base"
        raise RuntimeError(
            "runtime failure artifact pointer 超出 reload transaction: "
            f"stable={pointers.stable.path} base={base_pointer} "
            f"candidate={candidate_pointer}"
        )

    def _refresh_composition_runtime_tools(
        self,
        snapshot: RuntimeSnapshot,
    ) -> None:
        """Rebuild ToolRegistry and attach every exact live v3 MCP facade."""

        snapshot.tool_registry = self._compile_snapshot_tools(
            dict(snapshot.generations),
            snapshot.plugin_tool_catalog,
        )
        for generation in sorted(
            snapshot.generations.values(),
            key=lambda item: item.plugin_id,
        ):
            runtime = self._composition_generation_host.get(generation.generation_id)
            snapshot.tool_registry = self._composition_generation_host.attach_tools(
                snapshot.tool_registry,
                runtime,
            )

    @staticmethod
    def _composition_runtime_declared(
        snapshot: RuntimeSnapshot,
        plugin_id: str,
    ) -> bool:
        """Return whether one plugin owns runtime declarations in a snapshot."""

        return any(
            binding.descriptor.owner == plugin_id
            for registry in (
                snapshot.managed_process_registry,
                snapshot.mcp_server_registry,
                snapshot.workload_registry,
            )
            if registry is not None
            for binding in registry.values()
        )

    def _compile_snapshot_tools(
        self,
        generations: dict[str, PluginGeneration],
        plugin_tools: PluginToolCatalog | None = None,
    ) -> Any:
        if self._tool_registry is None:
            return None
        registry = self._tool_registry.fork(
            excluded_source_types={"plugin", "mcp"},
        )
        if plugin_tools is not None:
            for binding in plugin_tools.values():
                if registry.has_tool(binding.descriptor.name):
                    raise RuntimeError(f"插件工具名称重复: {binding.descriptor.name}")
                registry.register(
                    _build_v3_plugin_tool(
                        generations,
                        plugin_tools,
                        binding,
                    ),
                    risk=(
                        "write"
                        if binding.descriptor.risk == "read-write"
                        else binding.descriptor.risk
                    ),
                    always_on=binding.descriptor.always_on,
                    preloadable=binding.descriptor.preloadable,
                    requires_turn_search=binding.descriptor.requires_turn_search,
                    search_hint=binding.descriptor.search_hint,
                    source_type="plugin",
                    source_name=binding.plugin_id,
                )
        return registry

    async def _publish_committed_snapshot(
        self,
        snapshot: RuntimeSnapshot,
    ) -> None:
        if self._snapshot_store.current is None:
            registry = snapshot.channel_registry
            catalog = snapshot.channel_catalog
            activity_declared = self._activity_catalog_identity(snapshot) is not None
            if (
                (registry is not None and registry.descriptors)
                or (catalog is not None and catalog.descriptors)
                or activity_declared
            ):
                transaction = self._snapshot_store.begin_publish(snapshot)
                await self._commit_snapshot_with_publication_participants(
                    transaction,
                    old_commands=(),
                    new_commands=(),
                    promote_latest=False,
                )
                return
            self._snapshot_store.install(snapshot)
            return
        transaction = self._snapshot_store.begin_publish(snapshot)
        await self._commit_snapshot_with_publication_participants(
            transaction,
            old_commands=(),
            new_commands=(),
            promote_latest=False,
        )

    def _collect_candidate_contributions(
        self,
        *,
        instance: ComposablePlugin,
        plugin_id: str,
        plugin_dir: Path,
    ) -> PluginContributions:
        return PluginContributions(
            manifest={
                "name": instance.name,
                "version": instance.version,
                "desc": instance.desc,
                "author": instance.author,
            },
            skill_roots=_resolve_declared_roots(
                plugin_dir,
                instance.skill_roots,
            ),
            drift_skill_roots=_resolve_declared_roots(
                plugin_dir,
                instance.drift_skill_roots,
            ),
            dashboard_module=_resolve_dashboard_module(
                plugin_dir,
                instance.dashboard_module,
            ),
            web_module=resolve_web_module(
                plugin_dir,
                instance.web_module,
                requires=instance.web_requires,
                provides=instance.web_provides,
                contract_digests=instance.web_contract_digests,
            ),
        )

    def _validate_candidate(
        self,
        *,
        instance: ComposablePlugin,
        plugin_id: str,
        revision: str,
    ) -> GateResult:
        """Validate the remaining module-level v3 semantic checks."""

        checks = [
            GateCheckResult(
                check_id="api_version",
                status="passed",
                evidence=3,
            ),
            GateCheckResult(
                check_id="lifecycle_api",
                status="passed",
                evidence={"contract": "apply(ctx, config)"},
            ),
        ]
        try:
            semantic_checks = instance.static_semantic_checks()
        except Exception as error:
            checks.append(
                GateCheckResult(
                    check_id="semantic_checks",
                    status="failed",
                    evidence=str(error) or type(error).__name__,
                )
            )
        else:
            invalid_semantic = [
                semantic
                for semantic in semantic_checks
                if not isinstance(semantic, PluginSemanticCheck) or not semantic.passed
            ]
            checks.append(
                GateCheckResult(
                    check_id="semantic_checks",
                    status="failed" if invalid_semantic else "passed",
                    evidence=[
                        getattr(semantic, "evidence", repr(semantic))
                        for semantic in invalid_semantic
                    ],
                )
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
        for imported_name in tuple(sys.modules):
            if imported_name == module_name or imported_name.startswith(
                f"{module_name}."
            ):
                _ = sys.modules.pop(imported_name, None)

    async def terminate_all(self) -> None:
        """完成快照、插件生命周期和作用域资源的全量关闭。"""

        # 1. 先收束正式 Channel owner，再允许对应插件 Root 进入 drain。
        externally_cancelled = False
        channel_runtime = self._active_channel_generation
        if channel_runtime is not None:
            _ = self._snapshot_store.pause_admission()
            channel_runtime.close_admission()
            try:
                _, cancelled = await _complete_critical(channel_runtime.stop())
                externally_cancelled = externally_cancelled or cancelled
            except BaseException as error:
                current = asyncio.current_task()
                externally_cancelled = externally_cancelled or (
                    current is not None and current.cancelling() > 0
                )
                self._cleanup_failures.append(
                    CleanupFailure(
                        resource=f"channel-generation:{channel_runtime.snapshot_id}",
                        error=str(error) or type(error).__name__,
                    )
                )
                raise RuntimeError(
                    "Channel runtime cleanup 未完成，generation owner 已保留"
                ) from error
            else:
                self._active_channel_generation = None
                self._active_channel_catalog_identity = None
        activity_host = self._activity_host
        if activity_host is not None and activity_host.active is not None:
            _ = self._snapshot_store.pause_admission()
            try:
                _, cancelled = await _complete_critical(activity_host.close())
                externally_cancelled = externally_cancelled or cancelled
            except BaseException as error:
                self._cleanup_failures.append(
                    CleanupFailure(
                        resource="activity-host",
                        error=str(error) or type(error).__name__,
                    )
                )
                raise RuntimeError(
                    "Activity runtime cleanup 未完成，generation owner 已保留"
                ) from error

        # 2. 关闭当前 generation admission，再完成快照回收。
        for generation in self._active_generations.values():
            self._retire_generation(generation)
        _, snapshot_cancelled = await _complete_critical(self._snapshot_store.close())
        externally_cancelled = externally_cancelled or snapshot_cancelled
        self._ready_candidate = None
        for plugin_id in tuple(self._prepared_generations):
            _, cancelled = await _complete_critical(self.discard_prepared(plugin_id))
            externally_cancelled = externally_cancelled or cancelled
        # 3. 逐插件关闭 generation scope 并消费全部 cleanup failures。
        for mp in list(self._loaded):
            active_info = self._active_plugins.get(mp)
            scope = self._scopes.pop(mp, None)
            if scope is not None:
                generation = (
                    None
                    if active_info is None
                    else self._active_generations.get(active_info.plugin_id)
                )
                cleanup_failures, cancelled = await _complete_critical(scope.aclose())
                self._cleanup_failures.extend(cleanup_failures)
                externally_cancelled = externally_cancelled or cancelled

            # 4. 注销模块和运行时注册。
            self._remove_module_tree(mp)
            stable_alias = self._stable_aliases.pop(mp, None)
            if stable_alias is not None:
                self._remove_module_tree(stable_alias)
            if active_info is not None:
                generation = self._active_generations.get(active_info.plugin_id)
                if generation is not None and generation.module_path == mp:
                    _ = self._active_generations.pop(active_info.plugin_id)
                    generation.state = "retired"
            _ = self._active_plugins.pop(mp, None)
        self._loaded.clear()
        self._active_plugins.clear()
        self._scopes.clear()
        self._active_generations.clear()
        self._draining_generations.clear()
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


class _StablePluginFailed(Exception):
    """标识一个可以排除后重试的 stable boot 参与者。"""

    def __init__(
        self,
        generation: PluginGeneration,
        phase: str,
        cause: Exception,
    ) -> None:
        super().__init__(str(cause))
        self.generation = generation
        self.phase = phase
        self.cause = cause


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


def _read_plugin_config_projection(
    data_dir: Path,
    *,
    credential_paths: tuple[str, ...] = (),
    credential_alias_groups: tuple[tuple[str, ...], ...] = (),
) -> dict[str, object]:
    """Read plugin config and replace declared secret values with opaque refs."""

    # 1. Core alone reads the formal file before plugin config validation.
    config_path = data_dir / "config.local.toml"
    raw_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise _PluginConfigError(str(e)) from e
    for aliases in credential_alias_groups:
        present = tuple(
            path for path in aliases if _config_path_exists(raw_config, path)
        )
        if len(present) > 1:
            raise _PluginConfigError(
                "同一 channel credential 不得同时声明多个 physical alias: "
                + ", ".join(present)
            )
    projected = cast(dict[str, object], copy.deepcopy(raw_config))
    for path in credential_paths:
        _redact_plugin_config_path(projected, path)
    return projected


def _validate_channel_credential_schema(
    config_model: type[BaseModel] | None,
    *,
    credential_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    """Bind every opaque credential field to its complete physical alias set."""

    # 1. Discover opaque credential fields from the validated Pydantic schema.
    groups = _collect_channel_credential_aliases(config_model)
    schema_paths = tuple(sorted(path for group in groups for path in group))

    # 2. Static admission owns the complete raw-path declaration.
    expected = tuple(sorted(credential_paths))
    if schema_paths != expected:
        raise _PluginConfigError(
            "ConfigModel credential aliases 与静态 manifest 不一致: "
            f"schema={schema_paths} manifest={expected}"
        )
    return groups


def _collect_channel_credential_aliases(
    config_model: type[BaseModel] | None,
    *,
    prefix: tuple[str, ...] = (),
    seen: frozenset[type[BaseModel]] = frozenset(),
) -> tuple[tuple[str, ...], ...]:
    """Collect physical input paths for direct CredentialRef fields."""

    if config_model is None:
        return ()
    if not isinstance(config_model, type) or not issubclass(config_model, BaseModel):
        raise _PluginConfigError("ConfigModel 必须继承 pydantic.BaseModel")
    if config_model in seen:
        return ()

    groups: list[tuple[str, ...]] = []
    next_seen = seen | {config_model}
    validate_by_name = bool(
        config_model.model_config.get("validate_by_name")
        or config_model.model_config.get("populate_by_name")
    )
    validate_by_alias = config_model.model_config.get("validate_by_alias") is not False
    for name, field_info in config_model.model_fields.items():
        aliases = _pydantic_input_aliases(
            name,
            field_info.validation_alias,
            field_info.alias,
            validate_by_name=validate_by_name,
            validate_by_alias=validate_by_alias,
        )
        annotation = field_info.annotation
        if _annotation_contains_credential_ref(annotation):
            if not _is_opaque_credential_annotation(annotation):
                raise _PluginConfigError(
                    f"channel credential 字段只能是 CredentialRef 或 None: {name}"
                )
            groups.append(
                tuple(sorted(".".join((*prefix, *alias)) for alias in aliases))
            )
            continue
        nested_model = _optional_basemodel_type(annotation)
        if nested_model is None:
            continue
        for alias in aliases:
            groups.extend(
                _collect_channel_credential_aliases(
                    nested_model,
                    prefix=(*prefix, *alias),
                    seen=next_seen,
                )
            )

    paths = [path for group in groups for path in group]
    if len(paths) != len(set(paths)):
        raise _PluginConfigError("ConfigModel credential physical alias 重复")
    return tuple(sorted(groups))


def _pydantic_input_aliases(
    field_name: str,
    validation_alias: str | AliasPath | AliasChoices | None,
    alias: str | None,
    *,
    validate_by_name: bool,
    validate_by_alias: bool,
) -> tuple[tuple[str, ...], ...]:
    """Normalize one Pydantic field's accepted mapping paths."""

    configured_alias = validation_alias or alias
    if configured_alias is None:
        choices: tuple[str | AliasPath, ...] = (field_name,)
    elif validate_by_alias:
        choices = (
            tuple(configured_alias.choices)
            if isinstance(configured_alias, AliasChoices)
            else (configured_alias,)
        )
        if validate_by_name:
            choices = (*choices, field_name)
    else:
        choices = (field_name,)
    paths: list[tuple[str, ...]] = []
    for choice in choices:
        raw_path = choice.path if isinstance(choice, AliasPath) else (choice,)
        if not raw_path or any(
            not isinstance(part, str) or not part for part in raw_path
        ):
            raise _PluginConfigError(
                f"channel credential alias 只支持对象字符串路径: {field_name}"
            )
        paths.append(tuple(cast(tuple[str, ...], raw_path)))
    return tuple(sorted(set(paths)))


def _annotation_contains_credential_ref(annotation: object) -> bool:
    if annotation is CredentialRef:
        return True
    return any(
        _annotation_contains_credential_ref(item) for item in get_args(annotation)
    )


def _is_opaque_credential_annotation(annotation: object) -> bool:
    if annotation is CredentialRef:
        return True
    origin = get_origin(annotation)
    return origin in {Union, UnionType} and all(
        item is CredentialRef or item is type(None) for item in get_args(annotation)
    )


def _optional_basemodel_type(annotation: object) -> type[BaseModel] | None:
    candidates = tuple(item for item in get_args(annotation) if item is not type(None))
    value = candidates[0] if len(candidates) == 1 else annotation
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value
    return None


def _config_path_exists(config: Mapping[str, object], path: str) -> bool:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _validate_plugin_config_projection(
    projection: Mapping[str, object],
    config_model: type[BaseModel] | None,
) -> Any:
    """Validate an already redacted config projection through plugin schema."""

    # 1. Each candidate clone receives a fresh value owned by its module class.
    raw_config = cast(dict[str, Any], copy.deepcopy(dict(projection)))
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


def _redact_plugin_config_path(config: dict[str, object], path: str) -> None:
    """Replace one present non-empty config leaf with an opaque credential ref."""

    parts = tuple(path.split("."))
    current: dict[str, object] = config
    for part in parts[:-1]:
        value = current.get(part)
        if value is None:
            return
        if not isinstance(value, dict):
            raise _PluginConfigError(f"channel credential path 不是对象路径: {path}")
        current = cast(dict[str, object], value)
    leaf = parts[-1]
    if leaf not in current:
        return
    value = current[leaf]
    if value is None or value == "":
        return
    current[leaf] = CredentialRef(parts)


def _static_channel_credential_paths(
    manifest: StaticPluginManifest,
) -> tuple[str, ...]:
    """Flatten manifest channel credential declarations without ambiguity."""

    return tuple(
        sorted(
            {path for _channel, paths in manifest.channel_credentials for path in paths}
        )
    )


def _format_validation_error(error: ValidationError) -> str:
    parts: list[str] = []
    for item in error.errors():
        path = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        parts.append(f"{path}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


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
    plugin_dir = generation.plugin_dir
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
    plugin_dir = generation.plugin_dir
    return _installed_candidate_base_from_root(plugin_dir) is not None


def _installed_candidate_base(generation: PluginGeneration) -> Path | None:
    if generation.source_type != "installed":
        return None
    plugin_dir = generation.plugin_dir
    return _installed_candidate_base_from_root(plugin_dir)


def _discard_generation_candidate_pointer(generation: PluginGeneration) -> None:
    plugin_base = _installed_candidate_base(generation)
    if plugin_base is not None:
        _ = discard_latest_pointer(plugin_base)


def _installed_candidate_base_from_root(plugin_dir: Path) -> Path | None:
    """Resolve the candidate pointer owner for an exact installed latest root."""

    # 1. 首次安装与 stable==latest 没有独立 candidate publication。
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


def _preserve_ready_pointer(
    ready: _ReadyPluginCandidate,
    plugin_base: Path,
) -> None:
    """Restore stable while retaining the validated candidate as latest."""

    previous, candidate = _ready_artifact_pointers(ready, plugin_base)
    pointers = read_pointers(plugin_base)
    if pointers is None or (pointers.stable, pointers.latest) not in {
        (previous, candidate),
        (candidate, candidate),
        (previous, previous),
    }:
        raise RuntimeError(f"插件 artifact pointer 已被其他发布改变: {plugin_base}")
    _ = write_pointers(plugin_base, stable=previous, latest=candidate)


def _ready_artifact_pointers(
    ready: _ReadyPluginCandidate,
    plugin_base: Path,
) -> tuple[ArtifactPointer, ArtifactPointer]:
    """解析 ready candidate 事务拥有的前后 artifact pointer。"""

    candidate_root = ready.candidate.plugin_dir
    candidate = relative_artifact_pointer(plugin_base, candidate_root)
    if ready.previous is None:
        return ArtifactPointer(None), candidate
    previous_root = ready.previous.plugin_dir
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
    seen: set[Path] = set()
    for raw_path in declared:
        path = (plugin_dir / raw_path).resolve(strict=False)
        _require_plugin_path(plugin_root, path, "能力目录")
        if not path.is_dir():
            raise RuntimeError(f"插件能力目录不存在: {path}")
        if path in seen:
            raise RuntimeError(f"插件能力目录重复: {path}")
        seen.add(path)
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


def _remove_validation_data_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _candidate_data_exclude_paths(
    generation: PluginGeneration,
) -> tuple[str, ...]:
    """Return candidate-copy exclusions owned by static validation policy."""

    manifest = generation.static_manifest
    if manifest is None:
        return ()
    excluded = set(manifest.exclude_data_paths)
    if manifest.channel_credentials:
        excluded.add("config.local.toml")
    return tuple(sorted(excluded))


def _validate_candidate_formal_snapshot_identity(
    generation: PluginGeneration,
    *,
    candidate: RuntimeSnapshot,
    formal: RuntimeSnapshot,
) -> None:
    """Require the formal Root to preserve the validated candidate identity."""

    if candidate.snapshot_id != formal.snapshot_id:
        raise RuntimeError(
            "候选隔离资源恢复后 snapshot identity 发生变化: "
            f"{candidate.snapshot_id} -> {formal.snapshot_id}"
        )


def _copy_validation_data(
    source: Path,
    target: Path,
    exclude_paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Copy plugin data to a candidate tree while returning copied file paths."""

    validate_workspace_plugin_data_path(source, source.parents[1])
    excluded = tuple(PurePosixPath(item).as_posix() for item in exclude_paths)

    # 1. A new plugin has no formal bytes; candidate starts from an empty tree.
    if not source.exists():
        target.mkdir(parents=True)
        return ()
    source_root = source.resolve(strict=True)

    # 2. Candidate data must never retain an edge back into formal storage.
    for directory, dirnames, filenames in os.walk(source_root, followlinks=False):
        root = Path(directory)
        relative_dir = root.relative_to(source_root)
        dirnames[:] = [
            name
            for name in dirnames
            if not _candidate_data_path_is_excluded(relative_dir / name, excluded)
        ]
        retained_files = [
            name
            for name in filenames
            if not _candidate_data_path_is_excluded(relative_dir / name, excluded)
        ]
        for name in (*dirnames, *retained_files):
            path = root / name
            if path.is_symlink():
                raise RuntimeError(f"candidate plugin-data 不允许复制符号链接: {path}")

    # 3. Copy SQLite through its snapshot API; never race WAL/SHM companion files.
    target.mkdir(parents=True)
    for directory, dirnames, filenames in os.walk(source_root, followlinks=False):
        root = Path(directory)
        relative_dir = root.relative_to(source_root)
        dirnames[:] = [
            name
            for name in dirnames
            if not _candidate_data_path_is_excluded(relative_dir / name, excluded)
        ]
        for name in dirnames:
            relative = relative_dir / name
            (target / relative).mkdir()
        for name in filenames:
            relative = relative_dir / name
            if _candidate_data_path_is_excluded(relative, excluded):
                continue
            if name.endswith(("-wal", "-shm")):
                continue
            source_file = root / name
            target_file = target / relative
            if _is_sqlite_database(source_file):
                _copy_sqlite_snapshot(source_file, target_file)
            else:
                _ = shutil.copy2(source_file, target_file)

    # 4. Freeze a relative file inventory for review and Gate evidence.
    inventory: list[str] = []
    for directory, _dirnames, filenames in os.walk(target):
        root = Path(directory)
        for filename in filenames:
            inventory.append(root.joinpath(filename).relative_to(target).as_posix())
    return tuple(sorted(inventory))


def _is_sqlite_database(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def _copy_sqlite_snapshot(source: Path, target: Path) -> None:
    """Copy one transactionally consistent SQLite snapshot."""

    reader = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    writer = sqlite3.connect(target)
    try:
        reader.backup(writer)
    finally:
        writer.close()
        reader.close()
    _ = shutil.copymode(source, target)


def _candidate_data_path_is_excluded(
    relative_path: Path,
    excluded: tuple[str, ...],
) -> bool:
    relative = relative_path.as_posix()
    return any(relative == item or relative.startswith(item + "/") for item in excluded)


def _replace_snapshot_payload(
    target: RuntimeSnapshot,
    source: RuntimeSnapshot,
) -> None:
    """刷新无 lease 候选载荷，并保留 store 拥有的生命周期字段。"""

    if target.lease_count or target.state not in {"validating", "committed"}:
        raise RuntimeError("只能刷新无 lease 的 candidate snapshot")
    for name in (
        "generations",
        "skill_catalog_generation_id",
        "dashboard_bindings",
        "web_ui_catalog",
        "web_ui_catalog_identity",
        "mobile_ui_registry",
        "mobile_ui_registry_identity",
        "channel_registry",
        "channel_registry_identity",
        "channel_catalog",
        "mcp_server_registry",
        "mcp_server_registry_identity",
        "managed_process_registry",
        "managed_process_registry_identity",
        "workload_registry",
        "workload_registry_identity",
        "tool_registry",
        "plugin_skill_index",
        "command_registry",
        "background_job_catalog",
        "background_job_catalog_identity",
        "plugin_tool_catalog",
        "plugin_tool_catalog_identity",
        "composition_root",
        "composition_topology",
        "composition_active_plugin_ids",
    ):
        setattr(target, name, getattr(source, name))


def _validate_static_manifest_runtime(
    snapshot: RuntimeSnapshot,
    generations: Mapping[str, PluginGeneration],
) -> None:
    """Reconcile static runtime policy with the frozen Root projection."""

    all_manifests = {
        plugin_id: generation.static_manifest
        for plugin_id, generation in generations.items()
        if generation.static_manifest is not None
    }
    if not all_manifests:
        return

    # 1. Install staging owns the interpreter used by every declared Python runtime.
    for _plugin_id, generation in generations.items():
        manifest = generation.static_manifest
        if manifest is None:
            continue
        runtime_commands: list[tuple[str, tuple[str, ...]]] = []
        for runtime in manifest.python:
            _ = staged_python_interpreter(generation.plugin_dir, runtime)
        for kind, declarations in (
            ("mcp", manifest.mcp_servers),
            ("process", manifest.managed_processes),
        ):
            for declaration in declarations:
                runtime_commands.append(
                    (
                        f"{kind}:{declaration.name}",
                        materialize_static_command(
                            generation.plugin_dir,
                            manifest,
                            declaration,
                        ),
                    )
                )
        generation.static_runtime_commands = tuple(
            sorted(runtime_commands, key=lambda item: item[0])
        )

    # 2. Compare every static owner's import-free declarations with the exact
    # Root-frozen descriptors.  Missing, extra, and field drift all fail closed.
    if snapshot.composition_active_plugin_ids is None:
        raise RuntimeError("静态 v3 manifest snapshot 缺少 active plugin projection")
    active_plugin_ids = set(snapshot.composition_active_plugin_ids)
    manifests = {
        plugin_id: manifest
        for plugin_id, manifest in all_manifests.items()
        if plugin_id in active_plugin_ids
    }
    expected: set[tuple[object, ...]] = set()
    for plugin_id, manifest in manifests.items():
        assert manifest is not None
        expected.update(
            (
                plugin_id,
                declaration.name,
                declaration.command,
                declaration.cwd,
                declaration.env,
                declaration.required_tools,
                declaration.candidate_read_only_tools,
                declaration.endpoint_env,
                declaration.workload_env,
                declaration.candidate_env,
            )
            for declaration in manifest.mcp_servers
        )
    registry = snapshot.mcp_server_registry
    actual: set[tuple[object, ...]] = set()
    if registry is not None:
        static_owners = set(all_manifests)
        actual.update(
            (
                descriptor.owner,
                descriptor.name,
                descriptor.command,
                descriptor.cwd,
                descriptor.env,
                descriptor.required_tools,
                descriptor.candidate_read_only_tools,
                tuple(
                    (endpoint.env, endpoint.process)
                    for endpoint in descriptor.endpoint_env
                ),
                tuple(
                    (endpoint.env, endpoint.workload, endpoint.port)
                    for endpoint in descriptor.workload_env
                ),
                descriptor.candidate_env,
            )
            for descriptor in registry.descriptors
            if descriptor.owner in static_owners
        )
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        raise RuntimeError(
            "静态 manifest MCP 声明与 Root frozen registry 不一致: "
            f"missing={missing!r} extra={extra!r}"
        )

    expected_processes: set[tuple[object, ...]] = set()
    for plugin_id, manifest in manifests.items():
        assert manifest is not None
        expected_processes.update(
            (
                plugin_id,
                declaration.name,
                declaration.command,
                declaration.cwd,
                declaration.env,
                declaration.port_env,
                declaration.formal_port,
                declaration.readiness_path,
                declaration.startup_timeout_seconds,
            )
            for declaration in manifest.managed_processes
        )
    process_registry = snapshot.managed_process_registry
    actual_processes: set[tuple[object, ...]] = set()
    if process_registry is not None:
        static_owners = set(all_manifests)
        actual_processes.update(
            (
                descriptor.owner,
                descriptor.name,
                descriptor.command,
                descriptor.cwd,
                descriptor.env,
                descriptor.port_env,
                descriptor.formal_port,
                descriptor.readiness_path,
                descriptor.startup_timeout_seconds,
            )
            for descriptor in process_registry.descriptors
            if descriptor.owner in static_owners
        )
    if actual_processes != expected_processes:
        missing = sorted(expected_processes - actual_processes, key=repr)
        extra = sorted(actual_processes - expected_processes, key=repr)
        raise RuntimeError(
            "静态 manifest managed process 声明与 Root frozen registry 不一致: "
            f"missing={missing!r} extra={extra!r}"
        )

    expected_workloads: set[tuple[object, ...]] = set()
    for plugin_id, manifest in manifests.items():
        assert manifest is not None
        expected_workloads.update(
            (
                plugin_id,
                declaration.name,
                declaration.image,
                declaration.command,
                declaration.ports,
                declaration.loopback_ports,
                declaration.data,
                declaration.health,
                declaration.limits,
                declaration.user_namespaces,
            )
            for declaration in manifest.workloads
        )
    workload_registry = snapshot.workload_registry
    actual_workloads: set[tuple[object, ...]] = set()
    if workload_registry is not None:
        static_owners = set(all_manifests)
        actual_workloads.update(
            (
                descriptor.owner,
                descriptor.name,
                descriptor.image,
                descriptor.command,
                tuple((item.name, item.number) for item in descriptor.ports),
                tuple(
                    (item.name, item.loopback)
                    for item in descriptor.ports
                    if item.loopback is not None
                ),
                tuple(
                    (item.name, item.target, item.writable) for item in descriptor.data
                ),
                (
                    descriptor.health.port,
                    descriptor.health.path,
                    descriptor.health.timeout_seconds,
                ),
                (
                    descriptor.limits.memory_mb,
                    descriptor.limits.cpu_count,
                    descriptor.limits.pids,
                ),
                descriptor.user_namespaces,
            )
            for descriptor in workload_registry.descriptors
            if descriptor.owner in static_owners
        )
    if actual_workloads != expected_workloads:
        missing = sorted(expected_workloads - actual_workloads, key=repr)
        extra = sorted(actual_workloads - expected_workloads, key=repr)
        raise RuntimeError(
            "静态 manifest Workload 声明与 Root frozen registry 不一致: "
            f"missing={missing!r} extra={extra!r}"
        )


def _require_plugin_path(plugin_dir: Path, path: Path, label: str) -> None:
    try:
        _ = path.relative_to(plugin_dir)
    except ValueError as error:
        raise RuntimeError(f"插件 {label} 越界: {path}") from error


def _build_v3_plugin_tool(
    generations: Mapping[str, PluginGeneration],
    catalog: PluginToolCatalog,
    binding: PluginToolBinding,
) -> Any:
    """Build one Tool adapter fenced to its exact committed snapshot catalog."""

    from agent.plugin_composition.tool_catalog import _thaw_json
    from agent.tools.base import (
        Tool as AgentTool,
        ToolExecutionContext,
        ToolResult,
        get_current_tool_context,
    )

    # 1. Resolve and validate the handler at snapshot compilation time.
    generation = generations.get(binding.plugin_id)
    if generation is None or generation.generation_id != binding.generation_id:
        raise RuntimeError(
            "plugin Tool handler 不属于 exact generation: "
            f"{binding.plugin_id}:{binding.generation_id}"
        )
    handler: object = binding.handler
    if handler is None:
        handler = binding.module
        for segment in binding.descriptor.handler_export.replace(":", ".").split("."):
            handler = getattr(handler, segment, None)
            if handler is None:
                break
    if not inspect.iscoroutinefunction(handler):
        raise RuntimeError(
            f"plugin Tool handler 必须是 async function: {binding.descriptor.name}"
        )
    signature = inspect.signature(handler)
    parameters = tuple(signature.parameters.values())
    positional = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if (
        len(parameters) != 2
        or tuple(item.name for item in parameters) != ("context", "arguments")
        or any(item.kind not in positional for item in parameters)
        or any(item.default is not inspect.Parameter.empty for item in parameters)
    ):
        raise RuntimeError(
            "plugin Tool handler 签名必须是 async (context, arguments): "
            f"{binding.descriptor.name}"
        )

    # 2. Execute only through the exact live catalog bound by ToolRegistry.
    async def execute(self: Any, **kwargs: Any) -> str | ToolResult:
        from agent.plugin_composition.diagnostics import plugin_entrypoint

        snapshot = get_current_runtime_snapshot()
        if snapshot is None or snapshot.plugin_tool_catalog is not catalog:
            raise RuntimeError(
                f"plugin Tool 缺少 exact RuntimeSnapshot: {binding.descriptor.name}"
            )
        current = catalog.get(binding.descriptor.name)
        if current is not binding or not binding.is_live():
            raise RuntimeError(f"plugin Tool binding 已失效: {binding.descriptor.name}")
        context = get_current_tool_context()
        if not isinstance(context, ToolExecutionContext):
            raise RuntimeError(
                f"plugin Tool 缺少 ToolExecutionContext: {binding.descriptor.name}"
            )
        with plugin_entrypoint(
            plugin_id=binding.plugin_id,
            generation_id=binding.generation_id,
            fiber=binding.plugin_id,
            operation="tool.call",
            entrypoint=binding.descriptor.name,
        ):
            result = await handler(
                context,
                MappingProxyType(dict(kwargs)),
            )
            if not isinstance(result, (str, ToolResult)):
                raise RuntimeError(
                    f"plugin Tool handler 返回值无效: {binding.descriptor.name}"
                )
        return result

    tool_class = type(
        f"V3PluginTool_{binding.descriptor.name}",
        (AgentTool,),
        {
            "name": binding.descriptor.name,
            "description": binding.descriptor.description,
            "parameters": cast(
                dict[str, Any],
                _thaw_json(binding.descriptor.parameters),
            ),
            "execute": execute,
        },
    )
    return tool_class()


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
    snapshot = generation.runtime_snapshot
    if snapshot is None or snapshot.tool_registry is None:
        return []
    registry = snapshot.mcp_server_registry
    if registry is None:
        return []
    names: set[str] = set()
    for descriptor in registry.descriptors:
        if descriptor.owner == generation.plugin_id:
            names.update(
                snapshot.tool_registry.get_source_tool_names(
                    "mcp",
                    descriptor.name,
                )
            )
    return sorted(names)


def _log_candidate_status(result: dict[str, object]) -> None:
    logger.info(
        "plugin_candidate_status plugin=%s gate=%s active=%s prepared=%s "
        "revision=%s counts=skills:%d,drift_skills:%d,mcp:%d",
        result["plugin_id"],
        result["gate_status"],
        result["active_generation"],
        result["prepared_generation"] or "-",
        str(result["candidate_revision"])[:12],
        len(cast(list[object], result["skills"])),
        len(cast(dict[object, object], result["drift_skill_descriptions"])),
        len(cast(list[object], result["mcp_tools"])),
    )
    logger.debug(
        "plugin_candidate_status_detail %s",
        json.dumps(result, ensure_ascii=False, sort_keys=True),
    )
