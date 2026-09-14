from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import logging
import os
import secrets
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from contextlib import asynccontextmanager
from contextvars import Context as TaskContext
from pathlib import Path, PurePosixPath
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from typing import Any, Literal, TypeVar, cast
from uuid import uuid4


from agent.plugins.archive import PluginArchive, decode_config, encode_config
from agent.plugins._operation import (
    ManagerOperation, OperationBusyError, OperationTimeoutError,
    complete_critical as _complete_critical, current_operation,
    observe_operation, revoke_on_cancel, run_operation,
)
from agent.plugins.python_environment import ENVIRONMENT_FILE, PythonEnvironments, read_environment_refs
from agent.plugins.validation import ValidationHost
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES, PluginUpdates, UpdateStatus
from session.artifact_store import ArtifactStore
from agent.plugin_composition.config_input import CONFIG_INPUT, check_config_format, load_config
from agent.plugin_composition.bindings import BINDINGS, BindingScope, Bindings
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT, ARTIFACT_READ, ArtifactImport, ArtifactRead
from agent.plugin_composition.runtime_catalog import (
    RUNTIME_CATALOG,
    build_runtime_catalog,
)
from agent.plugin_composition.credentials import CREDENTIALS, CredentialClients
from infra.channels.attachment_import import ChannelOutboundAttachmentImporter
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
    MessageWriters, OwnerState, SessionAdmission,
)
from agent.plugin_composition.tasks import TASKS, PluginTasks
from session.log import MessageCatalog, MessageLog
from session.embedding_store import MessageEmbeddings
from agent.plugin_composition.context import RuntimeScope
from agent.restart import RESTART_GATE, RestartGate
from agent.control.frame_book import CONTROL_FRAMES, FrameBook

from agent.plugin_composition import (
    CHANNELS,
    COMMANDS,
    INTERACTION_UNDO,
    CompositionError,
    TIMERS,
    CompositionRoot,
    FiberState,
    PluginChannels,
    InteractionUndoService,
    PluginRuntime,
    PluginTimers,
    ServiceKey,
    RUNTIME_STARTING,
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    SNAPSHOT_SEALING,
    RuntimeStarting,
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
from agent.plugin_composition.processes import PROCESSES, PluginProcesses
from agent.plugin_composition.execution import EXECUTION, WORKLOAD_CONTROLLER
from agent.host_bridge.plugin_execution import CodeOwner, ExecutionAccess, ControllerAccess, cleanup_workloads_for_boot
from agent.plugin_composition.model import (
    resolve_declared_workspace_file,
    resolve_declared_workspace_root,
)
from agent.control.timer import AsyncioOneShotTimer
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from agent.plugins.composable import ComposablePlugin
from agent.plugins.interaction_undo import InteractionUndoCoordinator
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
)
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from session.identities import ChannelIdentities, ChannelIdentityWriteReceipt
from agent.plugins.artifacts import (
    ArtifactSelector,
    read_pointer,
    read_pointers,
    resolve_pointer,
)
from agent.plugins.source_resolver import resolve_plugin_sources
from agent.plugins.selection import PluginSelection, SelectionConflictError, SelectionWriteError
from agent.plugins.scope import CleanupFailure, PluginScope
from agent.plugins.generation import (
    GateCheckResult,
    GateResult,
    PluginGeneration,
)
from agent.plugins.importer import FreshPluginImporter
from agent.plugins.install import PluginInstallResult, install_git_plugin
from agent.plugins.static_manifest import (
    StaticPluginManifest,
    load_static_plugin_manifest,
    command_python_runtime,
    materialize_command,
)
from agent.plugins.reload_journal import (
    RecoveryActionName,
    RecoveryTarget,
    ReloadJournal,
    ReloadPhase,
    ReloadRecoveryAction,
)
from agent.workloads.client import UnixWorkloadController, WorkloadController
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotLease,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    SnapshotTransaction,
    get_current_runtime_snapshot,
)
from bus.event_bus import EventBus

logger = logging.getLogger(__name__)
PLUGIN_ARCHIVE_BINDING_API = 3
U = TypeVar("U")


def _snapshot_command_catalog(
    snapshot: RuntimeSnapshot | None,
) -> tuple[tuple[str, str], ...]:
    """宿主只读取目标 Root 实际选择的命令服务。"""
    if snapshot is None or snapshot.composition_root is None:
        return ()
    commands = snapshot.composition_root.context.get(COMMANDS)
    if commands is None:
        return ()
    return tuple((item.name, item.description) for item in commands.freeze().descriptors)


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


def _reject_retired_owner_recovery(action: ReloadRecoveryAction) -> None:
    """拒绝仍依赖已删除 owner 的旧恢复记录，不伪造恢复完成。"""

    resource = action.failure_resource or ""
    retired = {"activity-publication", "plugin-skill-projection"}.intersection({
        item.strip() for item in resource.split(",") if item.strip()
    })
    if not retired:
        return
    raise RuntimeError(
        "runtime recovery blocked: durable action retains retired "
        f"{sorted(retired)} owner; migrate or resolve it manually before "
        f"recovery (tx={action.tx_id}, plugin={action.plugin_id}, "
        f"resource={resource!r}); journal remains pending"
    )


@dataclass(frozen=True)
class _ReadyPluginCandidate:
    plugin_id: str
    previous: PluginGeneration | None
    snapshot: RuntimeSnapshot

    @property
    def candidate(self) -> PluginGeneration:
        return self.snapshot.generations[self.plugin_id]


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
    old_reopened: bool = False
    new_runtime: ChannelGeneration | None = None


class PluginManager:
    POST_PUBLISH_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: EventBus,
        workspace: Path,
        session_manager: Any = None,
        message_log: MessageLog | None = None,
        channel_identities: ChannelIdentities | None = None,
        installed_cache_root: Path | None = None,
        channel_attachment_store: ChannelAttachmentArtifactStore | None = None,
        disabled_builtin_plugins: frozenset[str] = frozenset(),
        workload_controller: WorkloadController | None = None,
        restart_gate: RestartGate | None = None,
        control_frames: FrameBook | None = None,
    ) -> None:
        self._dirs = plugin_dirs
        self._event_bus = event_bus
        self._workspace = workspace
        self._archive = PluginArchive(workspace / "runtime" / "plugin-archives")
        self._selection = PluginSelection(workspace)
        self._publication: SnapshotTransaction | None = None
        self._python_environments = PythonEnvironments(workspace)
        self._validation_only = False
        self._validation_hosts: dict[str, ValidationHost] = {}
        self._update_publication: tuple[str, asyncio.Task[None]] | None = None
        self._update_watchers: set[asyncio.Event] = set()
        self._session_manager = session_manager
        self._message_log = message_log
        self._artifact_read = None if channel_attachment_store is None else ArtifactRead(channel_attachment_store.acquire)
        self._artifact_import = None if channel_attachment_store is None else ArtifactImport(
            ChannelOutboundAttachmentImporter(channel_attachment_store).import_source
        )
        self._plugin_tasks = PluginTasks()
        self._plugin_processes = PluginProcesses()
        self._interaction_undo = (
            InteractionUndoCoordinator(session_manager)
            if session_manager is not None
            else None
        )
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
        self._cleanup_failures: list[CleanupFailure] = []
        # 仅持有尚未交给 snapshot 的真实 Root，以及它仍需使用的模块和数据 owner。
        self._building_roots: dict[CompositionRoot, tuple[PluginGeneration, ...]] = {}
        self._operation: ManagerOperation | None = None
        self._stopping = False
        self._draining_generations: dict[str, list[PluginGeneration]] = {}
        self._prepared_generations: dict[str, PluginGeneration] = {}
        self._ready_candidate: _ReadyPluginCandidate | None = None
        self._gate_results: dict[str, GateResult] = {}
        self._stable_aliases: dict[str, str] = {}
        self._fresh_importer = FreshPluginImporter()
        if workload_controller is None:
            workload_socket = os.environ.get("AKASHIC_WORKLOAD_SOCKET", "").strip()
            if workload_socket:
                workload_controller = UnixWorkloadController(Path(workload_socket))
        self._workload_controller = workload_controller
        # PluginManager 也可以由嵌入式/测试 host 直接构造；该 host 仍需一
        # 次性的 boot identity，不能退回固定的 unmanaged marker。
        self._host_boot_id = restart_gate.boot_id if restart_gate is not None else uuid4().hex
        self._restart_gate = restart_gate
        self._owns_control_frames = control_frames is None
        self._control_frames = FrameBook() if control_frames is None else control_frames
        self._workload_workspace_id = hashlib.sha256(
            str(workspace.resolve(strict=False)).encode("utf-8")
        ).hexdigest()[:16]
        self._snapshot_compiler = RuntimeSnapshotCompiler()
        self._snapshot_store = RuntimeSnapshotStore(self._on_snapshot_drained)
        self._runtime_started_roots: set[object] = set()
        self._runtime_starting_roots: set[object] = set()
        self._runtime_lifecycle_lock = asyncio.Lock()
        self._runtime_services_enabled = False
        self._reload_journal = ReloadJournal(workspace)
        self._channel_provider_factory_resolver: (
            Callable[
                [RuntimeSnapshot],
                Mapping[str, ProviderClientFactory],
            ]
            | None
        ) = self._default_channel_provider_factories
        self._channel_identities = channel_identities
        self._channel_generation_host = ChannelGenerationHost(
            on_before_start=self._reserve_channel_binding,
            config_revision_checker=self._check_channel_config_revision,
            on_failure=self._on_channel_cleanup_failure,
            boot_id=self._host_boot_id,
            snapshot_lease_acquirer=self._snapshot_store.lease,
            recovery_snapshot_lease_acquirer=self._acquire_channel_recovery_lease,
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
        self._drain_transactions: dict[str, str] = {}
        self._drained_before_commit: set[str] = set()
        self._event_bus.bind_runtime_snapshot_store(self._snapshot_store)

    def _reject_operation_lease(self) -> None:
        """在创建任务前拒绝等待同一 Manager 租约造成的自锁。"""
        from agent.plugins.snapshot import get_current_runtime_lease
        lease = get_current_runtime_lease()
        if lease is not None and lease._store is self._snapshot_store:
            raise RuntimeError("持有本 Manager 的 RuntimeSnapshot lease 时不能同步更新或关闭")

    def _require_operation_idle(self) -> None:
        if self._stopping:
            raise RuntimeError("PluginManager 已停止接纳操作；只能显式重试 terminate")
        if self._operation is not None and not self._operation.task.done():
            raise OperationBusyError("PluginManager busy：原操作和资源尚未退出")
        if any(host.active for host in self._validation_hosts.values()):
            raise OperationBusyError("PluginManager busy：请先退出验证 scope")

    def _check_operation_commit(self) -> ManagerOperation:
        """耐久提交前同步调用；检查到实际同步提交之间不得 await。

        返回同一操作 owner；耐久写入确认成功后将 committed 设为选中的实际 snapshot。
        线程写入结果未定时须保留线程任务并停止接纳，不能先宣称回滚。
        """
        operation = current_operation.get()
        if operation is None or operation is not self._operation or operation.task.done():
            raise RuntimeError("插件提交不属于当前 Manager 操作")
        if operation.task.cancelling():
            operation.revoke(cancel=False)
        if self._stopping or operation.revoked:
            raise asyncio.CancelledError("插件操作提交许可已撤销")
        if asyncio.get_running_loop().time() >= operation.deadline:
            operation.revoke(cancel=False)
            raise OperationTimeoutError(operation)
        return operation

    def _operation_can_continue(self) -> bool:
        """仅查询当前许可；被撤销的操作不能自动恢复或开放接纳。"""
        operation = current_operation.get()
        return (
            operation is not None and operation is self._operation
            and not operation.task.done() and not self._stopping and not operation.revoked
            and not operation.task.cancelling()
            and asyncio.get_running_loop().time() < operation.deadline
        )

    def _operation_finished(self, task: asyncio.Task[object]) -> None:
        """观察迟到错误但保留实际 task，供显式 retry/terminate 查询。"""
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("插件操作结束但失败: %s", error)
        self._notify_updates()

    def _revoke_operation(self, operation: ManagerOperation) -> None:
        """撤销已提交操作也关闭接纳，提交事实本身不回滚。"""
        operation.revoke()
        if (operation is self._operation and operation.committed is not None
                and operation.committed is self.current_snapshot):
            self._snapshot_store.pause_admission()
            if self._active_channel_generation is not None:
                self._active_channel_generation.close_admission()

    def _start_operation(
        self, work: Callable[[], Awaitable[U]], *, background: bool = False,
    ) -> ManagerOperation:
        """一次接纳固定一个任务和截止，后台发布清空调用者的 lease 上下文。"""
        self._require_operation_idle()
        operation = ManagerOperation(asyncio.get_running_loop().time() + self.POST_PUBLISH_TIMEOUT_SECONDS)
        self._operation = operation
        operation.task = asyncio.create_task(
            run_operation(operation, work), name="plugin-manager-operation",
            context=TaskContext() if background else None,
        )
        operation.task.add_done_callback(self._operation_finished)
        timer = asyncio.get_running_loop().call_at(operation.deadline, self._revoke_operation, operation)
        operation.task.add_done_callback(lambda _: timer.cancel())
        return operation

    async def _run_operation(self, work: Callable[[], Awaitable[U]]) -> U:
        """公开入口有限观察同一任务；退出观察不释放仍在工作的 owner。"""
        self._reject_operation_lease()
        operation = self._start_operation(work)
        try:
            return cast(U, await observe_operation(operation, deadline=operation.deadline))
        finally:
            if operation.revoked:
                self._revoke_operation(operation)

    def _acquire_channel_recovery_lease(
        self,
        snapshot_id: str,
    ) -> RuntimeSnapshotLease:
        """Retain the exact current snapshot for internal durable recovery."""

        snapshot = self._snapshot_store.current
        if snapshot is None or snapshot.snapshot_id != snapshot_id:
            raise RuntimeError("Channel durable recovery snapshot owner 不一致")
        if snapshot.accepting_leases:
            return self._snapshot_store.lease(snapshot_id)
        return self._snapshot_store.retain_recovery_target(snapshot)

    async def run_runtime_services(self) -> None:
        """订阅 stable 变化；每次启动都经同一操作 owner，关闭由 terminate 负责。"""
        if self._runtime_services_enabled:
            raise RuntimeError("plugin runtime services 已启动")
        self._runtime_services_enabled = True
        try:
            while not self._stopping:
                try:
                    await self.start_runtime()
                except OperationBusyError:
                    operation = self._operation
                    assert operation is not None
                    if not operation.task.done():
                        await asyncio.wait((operation.task,))
                        continue
                snapshot = self.current_snapshot
                if snapshot is None:
                    raise RuntimeError("plugin runtime services 尚无 stable Root")
                await self._snapshot_store.wait_for_stable_change(snapshot)
        finally:
            self._runtime_services_enabled = False


    async def _start_runtime_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        """Start one exact Root once without retaining an admission lease."""

        async with self._runtime_lifecycle_lock:
            if snapshot is not self.current_snapshot or not snapshot.accepting_leases:
                return
            root = snapshot.composition_root
            if root is None or root.instance_token in self._runtime_started_roots:
                return
            if root.instance_token not in self._runtime_starting_roots:
                raise RuntimeError("Root 尚未完成发布前准备，须经正式发布后启动")
            async def start() -> None:
                async with RuntimeScope(self._snapshot_store.lease(snapshot.snapshot_id)):
                    result = await root.context.serial(RUNTIME_STARTED, RuntimeStarted())
                    if result is not None:
                        raise CompositionError(
                            "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED",
                            "runtime.started 接入点不接受 Bail",
                        )
                    self._runtime_started_roots.add(root.instance_token)

            try:
                _, cancelled = await _complete_critical(start())
            except BaseException as error:
                # 持有生命周期锁完成清理，避免另一次 start 插入半启动状态。
                try:
                    await self._stop_runtime_snapshot_locked(snapshot)
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup("启动与资源清理失败", [error, cleanup_error]) from None
                raise
        if cancelled:
            raise asyncio.CancelledError

    async def start_runtime(self) -> None:
        await self._run_operation(self._start_runtime)

    async def _start_runtime(self) -> None:
        """Start lifecycle only after the exact Root is public and leasable."""
        self._check_operation_commit()
        snapshot = self.current_snapshot
        if snapshot is None:
            return
        if not snapshot.accepting_leases:
            raise RuntimeError("current RuntimeSnapshot 尚未开放")
        await self._start_runtime_snapshot(snapshot)

    async def _stop_runtime_snapshot(
        self, snapshot: RuntimeSnapshot, *, lease: RuntimeSnapshotLease | None = None,
    ) -> None:
        """Settle one started Root once before its effects are disposed."""

        async with self._runtime_lifecycle_lock:
            await self._stop_runtime_snapshot_locked(snapshot, lease=lease)

    async def _stop_runtime_snapshot_locked(
        self, snapshot: RuntimeSnapshot, *, lease: RuntimeSnapshotLease | None = None,
    ) -> None:
        """由持有生命周期锁的调用方完成停止或启动失败清理。"""

        root = snapshot.composition_root
        if root is None or root.instance_token not in self._runtime_started_roots | self._runtime_starting_roots:
            return

        async def stop() -> object:
            if lease is None:
                return await root.context.serial(RUNTIME_STOPPING, RuntimeStopping())
            async with RuntimeScope(lease.fork()):
                return await root.context.serial(RUNTIME_STOPPING, RuntimeStopping())

        result, cancelled = await _complete_critical(stop())
        if result is not None:
            raise CompositionError(
                "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED",
                "runtime.stopping 接入点不接受 Bail",
            )
        self._runtime_started_roots.discard(root.instance_token)
        self._runtime_starting_roots.discard(root.instance_token)
        if cancelled:
            raise asyncio.CancelledError

    @property
    def _active_generations(self) -> Mapping[str, PluginGeneration]:
        """当前实例只由 Store 选中的 snapshot 投影，不另存目录。"""
        snapshot = self.current_snapshot
        return {} if snapshot is None else snapshot.generations

    @property
    def cleanup_failures(self) -> list[CleanupFailure]:
        return list(self._cleanup_failures)

    def generation(self, plugin_id: str) -> PluginGeneration | None:
        return self._active_generations.get(plugin_id)

    def latest_gate(self, plugin_id: str) -> GateResult | None:
        return self._gate_results.get(plugin_id)

    def prepared_generation(self, plugin_id: str) -> PluginGeneration | None:
        return self._prepared_generations.get(plugin_id)

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

    def _resolve_channel_identity(self, channel: str, provider_identity: str) -> str | None:
        if self._channel_identities is None:
            raise RuntimeError("Channel identities 未绑定")
        return self._channel_identities.resolve(channel, provider_identity)

    async def _remember_channel_identity(
        self, channel: str, provider_identity: str, recipient: str,
    ) -> ChannelIdentityWriteReceipt:
        if self._channel_identities is None:
            raise RuntimeError("Channel identities 未绑定")
        return self._channel_identities.remember(channel, provider_identity, recipient)

    async def _rollback_channel_identity(self, receipt: object) -> bool:
        if not isinstance(receipt, ChannelIdentityWriteReceipt):
            raise TypeError("channel identity rollback receipt 类型无效")
        if self._channel_identities is None:
            raise RuntimeError("Channel identities 未绑定")
        return self._channel_identities.rollback(receipt)

    @property
    def channel_generation_host(self) -> ChannelGenerationHost:
        return self._channel_generation_host

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

    def _default_channel_provider_factories(
        self, snapshot: RuntimeSnapshot,
    ) -> Mapping[str, ProviderClientFactory]:
        """Build one formal credential owner for every frozen channel."""

        catalog = snapshot.channel_catalog
        registry = (
            catalog.registry if catalog is not None else snapshot.channel_registry
        )
        if registry is None:
            return {}
        if self._validation_only:
            raise RuntimeError("candidate 验证期禁止读取正式凭据")
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
                generation.data_dir,
                generation.config_projection,
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
            # The candidate still points at validation data here.  Resolve its
            # factories only after the formal payload/root has been installed.
            new_factories={},
            changed=changed,
        )

    def _refresh_channel_publication_factories(
        self,
        state: _ChannelPublicationState,
        *,
        old: bool = False,
        new: bool = False,
    ) -> None:
        """Resolve channel factories from the exact snapshot payload in use."""

        if not state.changed:
            return
        if old:
            state.old_factories = self._channel_provider_factories(state.previous)
        if new:
            state.new_factories = self._channel_provider_factories(state.candidate)

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

    def _close_channel_admission_before_snapshot_drain(
        self,
        state: _ChannelPublicationState | None,
    ) -> None:
        """Cancel long channel followers before waiting for snapshot leases."""

        if (
            state is None
            or not state.changed
            or state.old_runtime is None
            or state.old_stopped
        ):
            return
        state.old_runtime.close_admission()
        state.old_closed = True

    async def _start_channel_publication(
        self,
        state: _ChannelPublicationState,
        *,
        startup_snapshot_lease: RuntimeSnapshotLease | None = None,
    ) -> None:
        """Start the new exact runtime with admission still closed."""

        if not state.changed:
            return
        if state.candidate_identity is not None:
            state.new_runtime = await self._channel_generation_host.start_formal(
                state.candidate,
                state.new_factories,
                startup_snapshot_lease=startup_snapshot_lease,
            )

    def _open_channel_publication(self, state: _ChannelPublicationState) -> None:
        """Publish and open the new exact runtime after the stable pointer moved."""
        # 唯一调用位于 finalize_provisional 的同步提交段；prepare 已检查许可。
        # 这里没有 await，不能在同步 durable 写入之后用第二次 deadline 检查伪造未提交。
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
        *,
        startup_snapshot_lease: RuntimeSnapshotLease | None = None,
    ) -> None:
        """Reconstruct the old runtime after all other owners rolled back."""

        if not state.changed:
            return
        restored = state.old_runtime
        if state.old_stopped and state.previous is not None:
            # The old factory was closed by stop().  Re-resolve it from the
            # rebuilt exact old snapshot instead of reusing the closed object.
            self._refresh_channel_publication_factories(state, old=True)
            try:
                restored = await self._channel_generation_host.start_formal(
                    state.previous,
                    state.old_factories,
                    boot_owner="plugin-manager-rollback",
                    startup_snapshot_lease=startup_snapshot_lease,
                )
            except BaseException:
                self._active_channel_generation = None
                self._active_channel_catalog_identity = None
                raise
            # Keep the exact fresh generation returned by start_formal.  The
            # finish phase must operate on this object, never on the stopped
            # predecessor that occupied the same snapshot/channel key.
            state.old_runtime = restored
        self._active_channel_generation = restored
        self._active_channel_catalog_identity = state.previous_identity

    async def _restore_old_channel_after_failure(
        self,
        state: _ChannelPublicationState,
    ) -> None:
        """Rebuild one old binding while its exact snapshot stays closed.

        Admission and durable recovery are opened by the surrounding
        publication transaction only after snapshot, pointer, endpoint, and
        Root ownership have all been settled.
        """

        snapshot = state.previous
        if snapshot is None:
            return
        startup_lease: RuntimeSnapshotLease | None = None
        try:
            if state.old_stopped:
                if self._snapshot_store.current is not snapshot:
                    raise RuntimeError("旧 Channel recovery snapshot 已不是 current")
                startup_lease = (
                    self._snapshot_store.lease(snapshot.snapshot_id)
                    if snapshot.accepting_leases
                    else self._snapshot_store.retain_recovery_target(snapshot)
                )
                await self._restore_old_channel_publication(
                    state,
                    startup_snapshot_lease=startup_lease,
                )
        finally:
            if startup_lease is not None:
                await startup_lease.release()

    async def _finish_restored_channel_publication(
        self,
        state: _ChannelPublicationState,
    ) -> None:
        """Resume one exact snapshot, then open and recover its old binding."""

        if not state.changed or state.old_reopened:
            return
        snapshot = state.previous
        if snapshot is None:
            state.old_reopened = True
            return
        if self._snapshot_store.current is not snapshot:
            raise RuntimeError(
                "旧 Channel binding 只能在 exact previous snapshot 上开放"
            )
        if state.old_runtime is None:
            if not snapshot.accepting_leases:
                self._check_operation_commit()
                await self._snapshot_store.resume(snapshot)
        else:
            await self._finish_channel_runtime_recovery(
                snapshot,
                state.old_runtime,
                state.previous_identity,
            )
        state.old_reopened = True

    async def _finish_channel_runtime_recovery(
        self,
        snapshot: RuntimeSnapshot,
        runtime: ChannelGeneration,
        catalog_identity: str | None,
    ) -> None:
        """Resume one exact snapshot, then open and recover one channel runtime."""

        if self._snapshot_store.current is not snapshot:
            raise RuntimeError("Channel recovery snapshot 已不是 current")
        try:
            if self._snapshot_store.current is not snapshot:
                raise RuntimeError("Channel recovery snapshot 未能重新开放")
            self._active_channel_generation = runtime
            self._active_channel_catalog_identity = catalog_identity
            self._check_operation_commit()
            runtime.open_admission()
            await self._channel_generation_host.recover_durable_inbounds()
            # The global snapshot admission is the final success gate.  The
            # channel's recovery path uses the exact internal recovery lease
            # while this snapshot is still closed.
            if not snapshot.accepting_leases:
                self._check_operation_commit()
                await self._snapshot_store.resume(snapshot)
            if self._snapshot_store.current is not snapshot or not snapshot.accepting_leases:
                raise RuntimeError("Channel recovery snapshot 未能重新开放")
        except BaseException:
            runtime.close_admission()
            if self._active_channel_generation is runtime:
                self._active_channel_generation = None
                self._active_channel_catalog_identity = None
            if self._snapshot_store.current is snapshot and snapshot.accepting_leases:
                self._snapshot_store.pause_admission()
            raise

    async def _start_channel_with_recovery_lease(
        self,
        snapshot: RuntimeSnapshot,
        factories: Mapping[str, ProviderClientFactory],
        *,
        boot_owner: str,
    ) -> ChannelGeneration:
        """Start a current channel with a lease allowed during closed recovery."""

        if self._snapshot_store.current is not snapshot:
            raise RuntimeError("Channel recovery snapshot 已不是 current")
        startup_lease = (
            self._snapshot_store.lease(snapshot.snapshot_id)
            if snapshot.accepting_leases
            else self._snapshot_store.retain_recovery_target(snapshot)
        )
        try:
            return await self._channel_generation_host.start_formal(
                snapshot,
                factories,
                boot_owner=boot_owner,
                startup_snapshot_lease=startup_lease,
            )
        finally:
            await startup_lease.release()

    def _reopen_restored_channel_publication(
        self,
        state: _ChannelPublicationState,
    ) -> None:
        """Reopen the restored runtime only after the old snapshot is restored."""

        if not state.changed:
            return
        runtime = self._active_channel_generation
        if runtime is not None:
            self._check_operation_commit()
            runtime.open_admission()
        if state.previous is not None:
            self._finish_channel_boot_transactions(state.previous)

    async def _reserve_channel_binding(self, record: ChannelStartRecord) -> None:
        """Persist an exact binding reservation before plugin code can run."""

        if record.plugin_id == "core":
            return
        generation = self._channel_generation(record.plugin_id, record.generation_id)
        tx_id = self._ensure_runtime_recovery_transaction(generation)
        self._reload_journal.annotate(tx_id, {
            "event": "channel_runtime_owner",
            "runtime_generation_id": generation.generation_id,
        })
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
        if generation.config_revision != record.raw_config_revision:
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
        pending = self._snapshot_store.pending_transaction
        for snapshot in (
            self.current_snapshot, self.latest_snapshot,
            None if pending is None else pending.candidate,
        ):
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
            digest.update(_path_metadata(data_dir / CONFIG_INPUT))
        return digest.hexdigest()

    def stable_channel_catalog(self) -> ChannelRegistrySnapshot | None:
        """Return the exact committed stable merged channel declaration catalog."""

        snapshot = self.current_snapshot
        if snapshot is None:
            return None
        catalog = snapshot.channel_catalog
        return catalog.registry if catalog is not None else snapshot.channel_registry

    async def bind_core_channel_definitions(self, definitions: tuple[CoreChannelDefinition, ...]) -> None:
        await self._run_operation(lambda: self._bind_core_channel_definitions(definitions))

    async def _bind_core_channel_definitions(
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
        if self.current_snapshot is None:
            return
        snapshot: RuntimeSnapshot | None = None
        try:
            snapshot = await self._replace_formal_root(
                dict(self._active_generations), expected_ref=self._selection.read(),
            )
        except BaseException:
            if self._publication is None or not self._publication.must_retain:
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
            module_path = source.plugin_root / "plugin.py"
            mods.append(
                {
                    "name": name,
                    "plugin_root": str(source.plugin_root),
                    "module_path": str(module_path),
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
        await self._run_operation(self._load_all)

    async def _load_all(self) -> None:
        """从唯一完整选择启动；null 只允许首次固定安装输入。"""
        if self._validation_only:
            raise RuntimeError("validation Manager 必须显式装配 candidate")
        selection_ref = self._selection.read()
        if self.current_snapshot is not None:
            raise RuntimeError("load_all 不能重复启动正式 Root")
        self._check_operation_commit()
        await cleanup_workloads_for_boot(self._workload_controller, self._workload_workspace_id)
        # 宿主可能延迟返回或吞掉取消；撤销许可后不得继续执行任何 apply。
        self._check_operation_commit()
        self._plugin_tasks.start()
        self._plugin_processes.start()
        recovery = self._reload_journal.pending_recovery()
        receipts = await self._prepare_boot_runtime_recovery(tuple(
            action for action in recovery
            if action.runtime_owner_boot_id is not None
            or action.action in {"retry_generation_cleanup", "retry_runtime_recovery"}
        ))
        for action in recovery:
            self._reload_journal.settle_boot(
                action, committed=self._selection_transition_committed(action.tx_id, selection_ref),
                cleanup_receipt=receipts.get(action.tx_id),
            )
        if selection_ref is not None:
            await self._replace_formal_root(
                self._selection_components(selection_ref), expected_ref=selection_ref,
            )
            return
        enabled = load_plugin_manifest(self.installed_plugins_home)
        selected = tuple(
            mod for mod in self.discover(installed_selector="stable")
            if enabled.get(_resolve_plugin_id(mod), True)
        )
        await self._load_stable_batch(selected)

    def _selection_components(self, ref: str) -> tuple[str, ...]:
        return cast(tuple[str, ...], self._archive.read_descriptor(ref)["components"])

    def _selection_transition_committed(self, tx_id: str, current_ref: str | None) -> bool | None:
        """按前驱和完整 refs 认定同一次转换；旧记录缺证据时明确未知。"""
        intent = self._reload_journal.selection_candidate(tx_id)
        if intent is None:
            return None
        base, components = intent
        ref = current_ref
        while ref is not None and ref != base:
            record = self._archive.read_descriptor(ref)
            if record["previous"] == base and record["components"] == components:
                return True
            ref = cast(str | None, record["previous"])
        return False

    async def _prepare_boot_runtime_recovery(
        self,
        actions: tuple[ReloadRecoveryAction, ...],
    ) -> dict[str, str]:
        """清理真实旧 boot owner；不改变运行选择或安装输入。"""

        if not actions:
            return {}
        for action in actions:
            _reject_retired_owner_recovery(action)
        current_boot_id = os.environ.get("AKASHIC_BOOT_ID", "").strip()
        if os.environ.get("AKASHIC_SUPERVISED") != "1" or not current_boot_id:
            raise RuntimeError("v3 runtime recovery 需要 supervised boot identity")
        from agent.background.boot_guardian import _cleanup_boot_processes

        cleaned_boots: set[str] = set()
        receipts: dict[str, str] = {}
        for action in actions:
            previous_boot_id = action.runtime_owner_boot_id
            if previous_boot_id is not None and (
                not previous_boot_id.strip() or previous_boot_id == current_boot_id
            ):
                raise RuntimeError(
                    "v3 runtime recovery 缺少不同于当前进程的旧 boot identity"
                )
            cleanup = "not-required"
            if previous_boot_id is not None:
                cleanup = "complete"
                if previous_boot_id not in cleaned_boots:
                    await asyncio.to_thread(
                        _cleanup_boot_processes,
                        boot_id=previous_boot_id,
                        gateway_group_id=None,
                    )
                    cleaned_boots.add(previous_boot_id)
            receipts[action.tx_id] = (
                f"boot-reconcile:previous={previous_boot_id}:"
                f"current={current_boot_id}:cleanup={cleanup}:"
                f"target={action.recovery_target}"
            )
        return receipts

    async def _load_stable_batch(self, mods: tuple[dict[str, str], ...]) -> None:
        """固定整组归档后一次构建正式 Root，输入模块不进入运行组合。"""
        inputs: list[PluginGeneration] = []
        failure: BaseException | None = None
        try:
            for mod in mods:
                generation = await self._load_one(mod, stage_stable=True)
                if generation is None:
                    raise RuntimeError(f"完整插件组合加载失败: {_resolve_plugin_id(mod)}")
                inputs.append(generation)
            components = tuple(self._generation_archive_ref(item) for item in sorted(inputs, key=lambda item: item.plugin_id))
            # 临时输入模块的清理同样可能失败，必须先于唯一 durable commit。
            for root, owned in tuple(self._building_roots.items()):
                if owned and all(any(item is source for source in inputs) for item in owned):
                    await self._close_building_root(root)
            await self._replace_formal_root(
                components, expected_ref=None,
            )
        except BaseException as error:
            failure = error
        cleanup_errors: list[BaseException] = []
        for root, owned in tuple(self._building_roots.items()):
            if owned and all(any(item is source for source in inputs) for item in owned):
                if root.root_fiber.state != FiberState.UNLOADING:
                    try:
                        await self._close_building_root(root)
                    except BaseException as error:
                        cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "启动组合或归档输入清理失败",
                cleanup_errors if failure is None else [failure, *cleanup_errors],
            ) from None
        if failure is not None:
            raise failure

    async def prepare_candidate(self, plugin_id: str) -> PluginGeneration | None:
        return await self._run_operation(lambda: self._prepare_candidate(plugin_id))

    async def _prepare_candidate(self, plugin_id: str) -> PluginGeneration | None:
        if self._ready_candidate is not None:
            raise RuntimeError(
                f"已有 latest 等待 promote/discard: {self._ready_candidate.plugin_id}"
            )
        await self._discard_prepared(plugin_id)
        for mod in self.discover(installed_selector="latest"):
            if _resolve_plugin_id(mod) == plugin_id:
                generation = await self._load_one(mod, activate=False)
                return generation
        raise KeyError(f"插件不存在: {plugin_id}")

    async def discard_prepared(
        self, plugin_id: str, *, error: str = "candidate discarded",
    ) -> None:
        await self._run_operation(lambda: self._discard_prepared(
            plugin_id, error=error,
        ))

    async def _discard_prepared(
        self,
        plugin_id: str,
        *,
        error: str = "candidate discarded",
    ) -> None:
        generation = self._prepared_generations.get(plugin_id)
        if generation is None:
            return
        _, cancelled = await _complete_critical(
            self._dispose_generation(generation, state="discarded")
        )
        self._abort_reload(generation, error=error)
        if self._prepared_generations.get(plugin_id) is generation:
            _ = self._prepared_generations.pop(plugin_id)
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
        selection_ref = None if self._validation_only else self._selection.read()
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
            details={"base_selection_ref": selection_ref},
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
        """成功后才解除 owner；失败或取消保留资源供显式关闭重试。"""

        # 1. 调用者可能已移除 active/prepared，先把责任交给既有排空集合。
        tracked = self._draining_generations.setdefault(generation.plugin_id, [])
        if not any(item is generation for item in tracked):
            tracked.append(generation)

        async def close_resources() -> None:
            """后取得的资源关闭成功后，才释放其 Root 和作用域依赖。"""

            if self._snapshot_store.generation_is_referenced_elsewhere(
                generation, excluding_snapshot_id="",
            ):
                raise RuntimeError("snapshot 仍持有 generation，必须由 snapshot owner 完成关闭")
            if self._building_root_uses(generation) and generation.runtime_snapshot is None:
                raise RuntimeError("未交接 Root 仍持有 generation，须显式 terminate 重试")
            if generation.runtime_snapshot is not None:
                root = generation.runtime_snapshot.composition_root
                if root in self._building_roots and root.root_fiber.state == FiberState.UNLOADING:
                    raise RuntimeError("构建 Root 清理已失败，须显式 recovery/terminate 重试")
            if generation.runtime_snapshot is not None:
                await self._dispose_unreferenced_composition_root(generation.runtime_snapshot)
            failures = await generation.scope.aclose()
            self._cleanup_failures.extend(failures)
            if failures:
                raise RuntimeError(
                    "generation scope cleanup 未完成，必须显式 retry: "
                    + "; ".join(f"{item.resource}: {item.error}" for item in failures)
                )

        # 2. 重复取消不能截断清理；任何失败均阻止卸载模块与恢复接纳。
        try:
            _, cancelled = await _complete_critical(close_resources())
        except BaseException as error:
            _ = self._snapshot_store.pause_admission()
            if generation.runtime_snapshot is not None:
                self._record_root_failure(
                    generation,
                    error,
                    formal_effects=("generation_runtime_cleanup_pending",),
                )
            self._cleanup_failures.append(CleanupFailure(
                resource=f"plugin:{generation.plugin_id}:generation:{generation.generation_id}",
                error=str(error) or type(error).__name__,
            ))
            raise

        # 3. 所有资源确认关闭后才移除模块及排空 owner。
        self._remove_module_tree(generation.module_path)
        stable_alias = self._stable_aliases.pop(generation.module_path, None)
        if (
            stable_alias is not None
            and not preserve_stable_alias
            and sys.modules.get(stable_alias) is cast(ComposablePlugin, generation.instance).module
        ):
            self._remove_module_tree(stable_alias)
        generation.state = state
        self._forget_drained_generation(generation)
        if cancelled:
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
        if root in self._building_roots:
            await self._close_building_root(root)
        else:
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
        if any(self._building_root_uses(item) for item in snapshot.generations.values()):
            raise RuntimeError("未交接 Root 仍持有 snapshot 模块和数据，须显式 terminate 重试")
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
        if root_unreferenced:
            assert composition_root is not None
            if self._dashboard_validation_releaser is not None:
                await self._dashboard_validation_releaser(snapshot)
            await composition_root.dispose()
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
            )
            self._forget_drained_generation(generation)
        self._finish_drained_reload(snapshot.snapshot_id)

    async def reconcile_changed(self) -> list[dict[str, object]]:
        return await self._run_operation(self._reconcile_changed)


    async def install_candidate(
        self, *, source: str, marketplace: str, ref_name: str,
        sparse_paths: list[str], update_id: str | None = None,
    ) -> tuple[PluginInstallResult, dict[str, object]]:
        return await self._run_operation(lambda: self._install_candidate(
            source=source, marketplace=marketplace, ref_name=ref_name,
            sparse_paths=sparse_paths, update_id=update_id,
        ))

    async def _install_candidate(
        self,
        *,
        source: str,
        marketplace: str,
        ref_name: str,
        sparse_paths: list[str],
        update_id: str | None = None,
    ) -> tuple[PluginInstallResult, dict[str, object]]:
        """Stage one immutable artifact and publish its latest runtime atomically."""

        # 1. 操作 owner 已取得，写 cache 前拒绝未决候选。
        if update_id is not None:
            try:
                previous_update = self._reload_journal.update(update_id)
            except KeyError:
                previous_update = None
            if previous_update is not None:
                raise RuntimeError("已有更新请求只能查询，不能重跑安装")
        await self._reconcile_changed()
        self._check_operation_commit()
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

        # 2. 线程结束前一直持有操作与目录；取消不能把线程留给下一次更新。
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
                update_id=update_id,
            )
        )
        publication_before = self._publication
        try:
            if install_cancelled:
                raise asyncio.CancelledError
            self._check_operation_commit()
            await self._reconcile_changed()
        except BaseException:
            if self._publication is not publication_before and self._publication is not None:
                if self._publication.must_retain:
                    self._hold_selection_publication(self._publication)
                    raise
            self._reload_journal.rollback_updates(
                self.installed_plugins_home, update_id=result.update_id, error="candidate preparation failed",
            )
            raise
        plugin_id = f"{result.plugin_name}@{result.marketplace}"
        status = self.candidate_status()
        self._check_operation_commit()
        if result.staged_candidate and (
            status["candidate_plugin_id"] != plugin_id
            or status["candidate_state"] != "latest_ready"
        ):
            if self._publication is not publication_before and self._publication is not None:
                if self._publication.must_retain:
                    self._hold_selection_publication(self._publication)
                    raise RuntimeError("运行选择已提交或不确定，安装状态需显式结算")
            self._reload_journal.rollback_updates(
                self.installed_plugins_home, update_id=result.update_id, error="candidate preparation failed",
            )
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
        self._notify_updates()
        return result, status

    def read_update(self, update_id: str) -> UpdateStatus:
        """从原更新 journal 和候选状态生成同一份只读结果。"""
        update = self._reload_journal.update(update_id)
        candidate = self.ready_candidate
        ready = (update.phase == "armed" and candidate is not None and update.reload_tx_id is not None
                 and candidate.reload_tx_id == update.reload_tx_id
                 and self._reload_journal.get(update.reload_tx_id).phase == "latest_ready")
        return UpdateStatus(update_id, update.plugin_id, update.phase, ready,
                            self.update_is_publishing(update_id), update.error)

    def start_update_publication(self, update_id: str) -> None:
        """同步接纳无 lease 的宿主任务；返回只表示已接纳，不表示已晋升。"""
        update = self._reload_journal.update(update_id)
        if update.phase == "committed":
            return
        if update.phase != "armed":
            raise RuntimeError("已回退的更新不能再次发布")
        current = self._update_publication
        if current is not None and not current[1].done():
            if current[0] == update_id:
                return
            raise RuntimeError("另一个插件更新正在发布")
        ready = self._require_ready_candidate(update.plugin_id)
        if ready.candidate.reload_tx_id != update.reload_tx_id:
            raise RuntimeError("更新恢复点与当前候选不匹配")
        retained = [host.identity for host in self._validation_hosts.values()
                    if host.parent_lease.snapshot is ready.snapshot]
        if retained:
            raise RuntimeError(f"验证尚未退出或资源尚未清理: {retained}")
        operation = self._start_operation(
            lambda: self._publish_update(update_id, update.plugin_id), background=True,
        )
        self._update_publication = (update_id, operation.task)
        self._notify_updates()

    def _notify_updates(self) -> None:
        for event in self._update_watchers:
            event.set()

    async def watch_updates(self) -> AsyncGenerator[None]:
        """订阅先登记再读；通知只唤醒，读取仍以现有 journal 为准。"""
        event = asyncio.Event()
        self._update_watchers.add(event)
        event.set()
        try:
            while True:
                _ = await event.wait()
                event.clear()
                yield None
        finally:
            self._update_watchers.remove(event)

    def update_is_publishing(self, update_id: str) -> bool:
        current = self._update_publication
        return current is not None and current[0] == update_id and not current[1].done()

    async def _publish_update(self, update_id: str, plugin_id: str) -> None:
        """沿原发布 owner 排空和切换；失败作为恢复点诊断保留。"""
        try:
            _ = await self._switch_ready(plugin_id, update_id=update_id)
        except asyncio.CancelledError:
            if self._reload_journal.update(update_id).phase == "committed":
                self._reload_journal.record_update_error(update_id, "已提交；运行收尾取消，须检查保留的恢复 owner")
                return
            self._reload_journal.record_update_error(update_id, "publication cancelled")
            raise
        except Exception as error:
            self._reload_journal.record_update_error(update_id, str(error) or type(error).__name__)
            logger.exception("插件更新未完成: update=%s plugin=%s", update_id, plugin_id)
        finally:
            self._notify_updates()

    def annotate_reload(self, tx_id: str, details: dict[str, object]) -> None:
        """Append turn lineage evidence to an existing reload transaction."""

        self._reload_journal.annotate(tx_id, details)

    def require_installed_plugin(self, plugin_id: str) -> None:
        """Fail before registering uninstall when the plugin has no installed owner."""

        manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
        if plugin_id not in manifest:
            raise RuntimeError(f"插件未安装: {plugin_id}")

    async def _reconcile_changed(self) -> list[dict[str, object]]:
        """持有同一操作 owner，处理发现的固定制品输入。"""

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
                continue
            publication = await self._publish_prepared(plugin_id)
            results.append(publication)
            if publication.get("publication_state") == "latest_ready":
                return results
        return results

    async def reconcile_disabled_and_drain(self, plugin_id: str) -> None:
        await self._run_operation(lambda: self._reconcile_disabled_and_drain(plugin_id))

    async def _reconcile_disabled_and_drain(self, plugin_id: str) -> None:
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
        for generation in tuple(draining):
            await self._snapshot_store.wait_for_generation_drained(generation)
            if not generation.scope.closed:
                if self._snapshot_store.generation_is_referenced_elsewhere(
                    generation, excluding_snapshot_id="",
                ):
                    raise RuntimeError(f"插件旧代仍属于快照: {plugin_id}")
                await self._dispose_generation(generation, state="retired")
        _ = self._draining_generations.pop(plugin_id, None)

    async def _deactivate_plugin(self, plugin_id: str) -> dict[str, object]:
        """禁用也重建整个组合，不能保留其他插件的旧模块。"""
        active = self._active_generations[plugin_id]
        snapshot = await self._replace_formal_root({
            key: item for key, item in self._active_generations.items() if key != plugin_id
        }, expected_ref=self._selection.read())
        return {
            "plugin_id": plugin_id, "old_generation": active.generation_id,
            "new_generation": None, "snapshot_id": snapshot.snapshot_id,
            "publication_state": "disabled",
        }

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

    def _prepare_runtime_snapshot(self, lease: RuntimeSnapshotLease) -> None:
        """在 exact closed scope 内同步取得启动资源；失败标记留给 stopping 清理。"""
        from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

        snapshot = lease.snapshot
        root = snapshot.composition_root
        if root is None:
            return
        if snapshot.accepting_leases or root.instance_token in self._runtime_starting_roots:
            raise RuntimeError("Root 启动准备必须位于未准备的 closed snapshot")
        self._runtime_starting_roots.add(root.instance_token)
        token = bind_runtime_snapshot(lease)
        try:
            root.context.emit(RUNTIME_STARTING, RuntimeStarting())
        finally:
            reset_runtime_snapshot(token)

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
        preclosed_channel_state: _ChannelPublicationState | None = None,
    ) -> SnapshotTransaction:
        """关闭目标 lease 内准备临时资源，全部完成后才开放正式接纳。"""

        lease = self._snapshot_store.retain_publication_target(transaction)
        try:
            return await self._commit_snapshot_participants(
                transaction, old_commands=old_commands, new_commands=new_commands,
                promote_latest=promote_latest, force_provisional=force_provisional,
                provisional_started=provisional_started,
                reopen_previous_on_failure=reopen_previous_on_failure,
                before_open=before_open, after_open=after_open,
                startup_snapshot_lease=lease,
                preclosed_channel_state=preclosed_channel_state,
            )
        except BaseException as error:
            if transaction.must_retain:
                self._hold_selection_publication(transaction)
                raise
            try:
                await self._stop_runtime_snapshot(transaction.candidate, lease=lease)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup("发布与启动资源清理失败", [error, cleanup_error]) from None
            raise
        finally:
            await lease.release()

    async def _start_closed_runtime_snapshot(self, lease: RuntimeSnapshotLease) -> None:
        """全部生命周期初始化在 closed exact scope 内完成，之后才允许提交。"""
        async with self._runtime_lifecycle_lock:
            self._check_operation_commit()
            root = lease.snapshot.composition_root
            if root is None:
                return
            self._prepare_runtime_snapshot(lease)
            async with RuntimeScope(lease.fork()):
                result = await root.context.serial(RUNTIME_STARTED, RuntimeStarted())
                if result is not None:
                    raise CompositionError(
                        "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED", "runtime.started 接入点不接受 Bail",
                    )
                self._runtime_started_roots.add(root.instance_token)

    async def _commit_snapshot_participants(
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
        startup_snapshot_lease: RuntimeSnapshotLease | None = None,
        preclosed_channel_state: _ChannelPublicationState | None = None,
    ) -> SnapshotTransaction:
        """Publish one snapshot around a single closed external-participant step."""

        # 1. Snapshots without external participants retain the one-step path.
        endpoints_changed = force_provisional or old_commands != new_commands
        channel_binding_changed = self._channel_binding_changed(
            transaction.previous,
            transaction.candidate,
        )
        if preclosed_channel_state is not None:
            if (
                preclosed_channel_state.previous is not transaction.previous
                or preclosed_channel_state.candidate is not transaction.candidate
            ):
                raise RuntimeError("预关闭 Channel publication 与 snapshot 不匹配")
            channel_binding_changed = True
        if (
            not endpoints_changed
            and not channel_binding_changed
            and not force_provisional
            and not provisional_started
        ):
            if startup_snapshot_lease is not None:
                await self._start_closed_runtime_snapshot(startup_snapshot_lease)
            if promote_latest:
                return await self._snapshot_store.promote_latest(
                    before_open=before_open,
                    after_open=after_open,
                )
            await self._snapshot_store.commit(
                transaction, before_open=before_open, after_open=after_open,
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
        participants_switch_attempted = False
        forward_error: BaseException | None = None
        try:
            channel_state = preclosed_channel_state
            if channel_state is None:
                channel_state = self._prepare_channel_publication(
                    provisional.previous,
                    provisional.candidate,
                )
                await self._close_channel_publication(channel_state)
            elif channel_state.old_runtime is not None and not channel_state.old_stopped:
                raise RuntimeError("预关闭 Channel publication 尚未完成 old stop")
            # A preclosed state is refreshed by its caller after formal Root
            # replacement; a state closed in this method still has the live
            # previous Root and can resolve the old side now.  The candidate
            # side is always resolved from the final snapshot payload here.
            if channel_state is not None:
                if preclosed_channel_state is None:
                    self._refresh_channel_publication_factories(
                        channel_state, old=True
                    )
                self._refresh_channel_publication_factories(
                    channel_state, new=True
                )
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
            await self._start_channel_publication(
                channel_state,
                startup_snapshot_lease=startup_snapshot_lease,
            )
            if startup_snapshot_lease is not None:
                await self._start_closed_runtime_snapshot(startup_snapshot_lease)
            def open_participants() -> None:
                if after_open is not None:
                    after_open()
                assert channel_state is not None
                self._open_channel_publication(channel_state)

            await self._snapshot_store.finalize_provisional(
                provisional,
                before_open=before_open,
                after_open=open_participants,
                schedule_previous_drain=False,
            )
            if (
                channel_state is not None
                and channel_state.old_runtime is not None
                and channel_state.new_runtime is not None
            ):
                # A stopped binding has already handed reserve-only rows back
                # to the Bus.  Recover them only after the new exact catalog
                # is public and its admissions are open.
                await self._channel_generation_host.recover_durable_inbounds()
            if provisional.previous is not None:
                self._snapshot_store.schedule_retired_drain(provisional.previous)
        except BaseException as publication_error:
            revoke_on_cancel(publication_error)
            if provisional.must_retain:
                self._hold_selection_publication(provisional)
                if channel_state is not None and channel_state.new_runtime is not None:
                    self._active_channel_generation = channel_state.new_runtime
                    self._active_channel_catalog_identity = channel_state.candidate_identity
                    channel_state.new_runtime.close_admission()
                raise
            rollback_errors: list[BaseException] = []
            channel_cleanup_failed = False
            endpoint_restore_failed = False
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
            published_rollback = self._snapshot_store.current is provisional.candidate
            if published_rollback:
                # finalize_provisional has already retired the old snapshot;
                # restore its committed state before start_formal validates the
                # old channel catalog.
                await self._snapshot_store.rollback_published(
                    provisional,
                    keep_candidate_latest=promote_latest,
                    reopen_previous=(
                        reopen_previous_on_failure and self._operation_can_continue()
                        and not rollback_errors
                        and (channel_state is None or not channel_state.changed)
                    ),
                )
            if (
                channel_state is not None
                and not channel_cleanup_failed
                and preclosed_channel_state is None
                and self._operation_can_continue()
            ):
                try:
                    await self._restore_old_channel_after_failure(channel_state)
                except BaseException as caught:
                    rollback_errors.append(caught)
                    channel_cleanup_failed = True
            if not published_rollback:
                await self._snapshot_store.rollback_provisional(
                    provisional,
                    keep_candidate_latest=promote_latest,
                    reopen_previous=(
                        reopen_previous_on_failure and self._operation_can_continue()
                        and not rollback_errors
                        and (channel_state is None or not channel_state.changed)
                    ),
                )
            if (
                channel_state is not None
                and not rollback_errors
                # A preclosed state belongs to the outer formal-root
                # transaction.  Its old runtime was stopped before the Root
                # handoff and must be rebuilt from the exact old snapshot
                # before it can be reopened.  Finishing it here would call
                # recovery on that closed generation and hide the original
                # participant failure.
                and preclosed_channel_state is None
                and self._operation_can_continue()
                and channel_state.previous is self.current_snapshot
                and channel_state.previous is not None
            ):
                try:
                    await self._finish_restored_channel_publication(channel_state)
                except BaseException as caught:
                    rollback_errors.append(caught)
                    channel_cleanup_failed = True
            self._abort_channel_boot_transactions(
                provisional.candidate,
                publication_error,
            )
            if rollback_errors:
                resources: list[str] = []
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
        inputs: dict[str, PluginGeneration] | tuple[str, ...],
    ) -> RuntimeSnapshot:
        generations = {} if isinstance(inputs, tuple) else dict(inputs)
        composition_root = await self._resolve_composition_root(
            generations, components=inputs if isinstance(inputs, tuple) else None,
        )
        try:
            snapshot = self._snapshot_compiler.compile(
                generations,
                composition_root=composition_root,
                core_channel_definitions=self._core_channel_definitions,
            )
        except BaseException as error:
            await self._discard_building_root(composition_root, error)
            raise
        for item in generations.values():
            item.runtime_snapshot = snapshot
        return snapshot

    async def publish_prepared(self, plugin_id: str) -> dict[str, object]:
        return await self._run_operation(lambda: self._publish_prepared(plugin_id))


    async def switch_ready(self, plugin_id: str, *, update_id: str | None = None) -> dict[str, object]:
        return await self._run_operation(lambda: self._switch_ready(plugin_id, update_id=update_id))


    async def _switch_ready(self, plugin_id: str, *, update_id: str | None = None) -> dict[str, object]:
        ready = self._require_ready_candidate(plugin_id)
        candidate = ready.candidate
        tx_id = candidate.reload_tx_id
        if tx_id is None or self._reload_journal.get(tx_id).phase != "latest_ready":
            raise RuntimeError("latest candidate 已失去发布授权")
        if update_id is not None:
            update = self._reload_journal.update(update_id)
            if update.phase != "armed" or update.reload_tx_id != tx_id:
                raise RuntimeError("待发布更新与当前候选不匹配")
        record = self._reload_journal.get(tx_id)
        current = self.current_snapshot
        if record.base_snapshot_id != (None if current is None else current.snapshot_id):
            raise RuntimeError("候选基线已变化，必须重新准备完整组合")
        inputs = tuple(self._generation_archive_ref(item) for item in ready.snapshot.generations.values())
        intent = self._reload_journal.selection_candidate(tx_id)
        if intent is None or intent[1] != inputs:
            raise RuntimeError("候选缺少完整 selection 提交证据")
        expected_ref = intent[0]
        if self._selection.read() != expected_ref:
            raise SelectionConflictError("候选 stable 基线已变化")
        from agent.plugins.snapshot import get_current_runtime_lease
        if get_current_runtime_lease() is not None:
            raise RuntimeError("持有 RuntimeSnapshot lease 时不能切换完整 Root")

        # 1. 验证实例先退出，不能把它的模块、目录或身份改作正式实例。
        self._snapshot_store.pause_candidate_admission(ready.snapshot)
        try:
            await self._snapshot_store.wait_for_no_leases(ready.snapshot)
            self._check_operation_commit()
            self._snapshot_store.seal_candidate_validation(ready.snapshot)
        except BaseException:
            if not self._stopping and current_operation.get() is self._operation:
                await self._snapshot_store.resume(ready.snapshot)
            raise
        try:
            _, cancelled = await _complete_critical(self._snapshot_store.discard_latest(ready.snapshot))
        except BaseException as error:
            self._record_root_failure(
                candidate, error, resource="runtime-snapshot-drain",
                formal_effects=("candidate_runtime_cleanup_pending",), recovery_target="base",
            )
            raise
        if cancelled:
            self._abort_reload(candidate, error="候选关闭期间取消")
            self._ready_candidate = None
            raise asyncio.CancelledError
        def before_open() -> None:
            if self._reload_journal.get(tx_id).phase != "latest_ready":
                raise RuntimeError("更新已失去发布授权")
            self._advance_reload(candidate, "promoting")

        try:
            formal = await self._replace_formal_root(
                inputs, expected_ref=expected_ref, before_open=before_open,
                attempt=candidate, validated=ready.snapshot,
            )
        except BaseException:
            publication = self._publication
            if publication is not None and publication.must_retain and (
                plugin_id in publication.candidate.generations
                and publication.candidate.generations[plugin_id].reload_tx_id == tx_id
            ):
                self._ready_candidate = None
                raise
            if self._reload_journal.get(tx_id).phase not in {"cleanup_failed", "degraded"}:
                self._abort_reload(candidate, error="正式组合发布失败")
                self._ready_candidate = None
                if current is not None:
                    self._drain_transactions.pop(current.snapshot_id, None)
                    self._drained_before_commit.discard(current.snapshot_id)
            raise
        generation = formal.generations[plugin_id]
        self._ready_candidate = None
        try:
            self._track_reload_drain(generation, current)
        except BaseException:
            assert self._publication is not None
            self._hold_selection_publication(self._publication)
            raise
        return self._publication_status(
            plugin_id, active=ready.previous, candidate=generation,
            publication_state="promoted",
        )

    async def _replace_formal_root(
        self,
        inputs: dict[str, PluginGeneration] | tuple[str, ...],
        *,
        expected_ref: str | None,
        before_open: Callable[[], None] | None = None,
        attempt: PluginGeneration | None = None,
        validated: RuntimeSnapshot | None = None,
    ) -> RuntimeSnapshot:
        """排空旧组合，真实关闭外部 owner；失败只在释放成功后重建旧输入。"""
        self._check_operation_commit()
        from agent.plugins.snapshot import get_current_runtime_lease
        if get_current_runtime_lease() is not None:
            raise RuntimeError("持有 RuntimeSnapshot lease 时不能切换完整 Root")
        if self._publication is not None and self._publication.must_retain:
            if not self._publication.candidate.accepting_leases:
                raise RuntimeError("持久发布尚在 maintenance，必须显式恢复或关闭")
        if not self._validation_only and self._selection.read() != expected_ref:
            raise SelectionConflictError("完整 Root 基线已变化")
        self._publication = None
        current = self.current_snapshot
        if any(
            root.root_fiber.state == FiberState.UNLOADING
            or any(item.state == FiberState.UNLOADING for item in root.receipt().fibers)
            for root in self._building_roots
        ):
            raise RuntimeError("构建 Root 清理尚未完成，必须显式 recovery/terminate 重试")
        if any(
            identity != (None if current is None else current.snapshot_id)
            for identity in self._snapshot_store.retained_snapshot_ids
        ):
            raise RuntimeError("仍有未释放的 snapshot owner，必须先完成回收")
        previous = self._snapshot_store.pause_admission()
        old_channel = self._active_channel_generation
        try:
            if self._endpoint_quiescer is not None:
                await self._endpoint_quiescer()
            if old_channel is not None:
                old_channel.close_admission()
            if previous is not None:
                await self._snapshot_store.wait_for_no_leases(previous)
            self._check_operation_commit()
        except BaseException:
            # 只有这段尚未开始 release 的路径可以恢复原接纳；shutdown 永远优先。
            if not self._stopping and current_operation.get() is self._operation:
                if old_channel is not None:
                    old_channel.open_admission()
                await self._snapshot_store.resume(previous)
                if not self._stopping and self._endpoint_resumer is not None:
                    await self._endpoint_resumer()
            raise

        try:
            # 释放阶段取消也必须等待实际 owner 返回，随后进入恢复而非继续提交。
            _, cancelled = await _complete_critical(self._close_formal_root(previous))
            if cancelled:
                raise asyncio.CancelledError
            self._check_operation_commit()
            return await self._build_and_publish_root(
                inputs, previous=previous, old_channel=old_channel,
                expected_ref=expected_ref,
                before_open=before_open, attempt=attempt, validated=validated,
            )
        except BaseException as error:
            revoke_on_cancel(error)
            if self._publication is not None and self._publication.must_retain:
                raise
            if not self._operation_can_continue():
                self._snapshot_store.pause_admission()
                owner = attempt or (None if isinstance(inputs, tuple) else next(iter(inputs.values()), None))
                if owner is not None:
                    self._record_root_failure(
                        owner, error, formal_effects=("old_runtime_restore_uncertain",),
                        recovery_target="base",
                    )
                raise
            try:
                self._check_operation_commit()
                # 构建或 teardown 已失败的 owner 只允许显式 recovery/terminate 重试。
                if not self._validation_only and self._selection.read() != expected_ref:
                    raise SelectionConflictError("恢复前 stable 已变化，不能重建旧输入")
                if any(
                    root.root_fiber.state == FiberState.UNLOADING
                    or any(item.state == FiberState.UNLOADING for item in root.receipt().fibers)
                    for root in self._building_roots
                ) or any(
                    identity != (None if previous is None else previous.snapshot_id)
                    for identity in self._snapshot_store.retained_snapshot_ids
                ):
                    raise RuntimeError("新组合仍持有未完成清理的 Root")
                if previous is not None:
                    old_root = previous.composition_root
                    if old_root is not None and old_root.root_fiber.state != FiberState.DISPOSED:
                        raise RuntimeError("旧 Root 释放未完成，必须显式 retry")
                    _, recovery_cancelled = await _complete_critical(
                        self._build_and_publish_root(
                            tuple(self._generation_archive_ref(item) for item in previous.generations.values()),
                            previous=previous, old_channel=old_channel, expected_ref=expected_ref,
                            commit_selection=False,
                        )
                    )
                    if recovery_cancelled and not isinstance(error, asyncio.CancelledError):
                        error = BaseExceptionGroup("发布失败且恢复期间取消", [error, asyncio.CancelledError()])
            except BaseException as recovery_error:
                self._snapshot_store.pause_admission()
                owner = attempt or (None if isinstance(inputs, tuple) else next(iter(inputs.values()), None))
                if owner is not None:
                    self._record_root_failure(
                        owner, recovery_error, formal_effects=("old_runtime_restore_uncertain",),
                        recovery_target=(
                            "candidate"
                            if previous is None and attempt is None and owner.source_type == "builtin"
                            else "base"
                        ),
                    )
                raise BaseExceptionGroup("完整 Root 发布与恢复均失败", [error, recovery_error]) from None
            raise error

    async def _close_formal_root(self, snapshot: RuntimeSnapshot | None) -> None:
        """Channel 和外部 runtime 先确认退出，再关闭其依赖的 Root。"""
        channel = self._active_channel_generation
        if channel is not None:
            channel.close_admission()
            await channel.drain()
            await channel.stop()
            self._active_channel_generation = None
            self._active_channel_catalog_identity = None
        if snapshot is not None:
            await self._stop_runtime_snapshot(snapshot)
            if snapshot.composition_root is not None:
                await snapshot.composition_root.dispose()

    async def _build_and_publish_root(
        self,
        inputs: dict[str, PluginGeneration] | tuple[str, ...],
        *,
        expected_ref: str | None,
        commit_selection: bool = True,
        previous: RuntimeSnapshot | None,
        old_channel: ChannelGeneration | None,
        before_open: Callable[[], None] | None = None,
        attempt: PluginGeneration | None = None,
        validated: RuntimeSnapshot | None = None,
    ) -> RuntimeSnapshot:
        """把新物理 snapshot 交给 Store，再启动并提交它的全部 owner。"""
        operation = self._check_operation_commit()

        def authorize_commit() -> None:
            # 外部启停需要完成清理，但取消不能因此获得 stable 提交授权。
            self._check_operation_commit()
            if before_open is not None:
                before_open()
            if not self._validation_only:
                assert transaction is not None
                components = tuple(self._generation_archive_ref(item) for item in snapshot.generations.values())
                try:
                    if self._selection.read() != expected_ref:
                        raise SelectionConflictError("提交前 stable 基线已变化")
                    if not commit_selection and (
                        expected_ref is None or self._selection_components(expected_ref) != components
                    ):
                        raise RuntimeError("恢复 Root 输入必须等于已提交完整选择")
                    if commit_selection and (
                        expected_ref is None or self._selection_components(expected_ref) != components
                    ):
                        self._check_operation_commit()
                        # 同步调用在返回结果前中断，也不能假定没有提交。
                        transaction.selection_result = SelectionWriteError(
                            operation="commit", target_ref=None, outcome="uncertain",
                            observed_ref=None, observation_error=None,
                        )
                        transaction.selection_result = self._selection.commit(components, expected_ref=expected_ref)
                    else:
                        self._check_operation_commit()
                        transaction.selection_result = expected_ref
                    operation.committed = snapshot
                except SelectionWriteError as error:
                    transaction.selection_result = error
                    raise
                except SelectionConflictError:
                    transaction.selection_result = None
                    raise

        snapshot = await self._compile_topology_snapshot(inputs)
        transaction: SnapshotTransaction | None = None
        try:
            if validated is not None:
                _validate_candidate_formal_snapshot_identity(
                    candidate=validated, formal=snapshot,
                )
            if attempt is not None:
                actual = snapshot.generations[attempt.plugin_id]
                actual.reload_tx_id = attempt.reload_tx_id
                self._reload_journal.annotate(cast(str, attempt.reload_tx_id), {
                    "event": "formal_root_created",
                    "formal_snapshot_id": snapshot.snapshot_id,
                    "runtime_generations": {key: item.generation_id for key, item in snapshot.generations.items()},
                })
            transaction = self._begin_snapshot_publication(snapshot)
            self._publication = transaction
            self._check_operation_commit()
            await self._post_snapshot_invariants(snapshot)
            self._snapshot_store.seal_pending_validation(snapshot)
            channel_state = _ChannelPublicationState(
                previous=previous, candidate=snapshot,
                previous_identity=self._channel_catalog_identity(previous),
                candidate_identity=self._channel_catalog_identity(snapshot),
                old_runtime=old_channel, old_factories={}, new_factories={},
                changed=self._channel_binding_changed(previous, snapshot),
                old_closed=old_channel is not None, old_stopped=old_channel is not None,
            )
            if previous is not None and attempt is not None:
                self._drain_transactions[previous.snapshot_id] = cast(str, attempt.reload_tx_id)
            _, cancelled = await _complete_critical(
                self._commit_snapshot_with_publication_participants(
                    transaction,
                    old_commands=_snapshot_command_catalog(previous),
                    new_commands=_snapshot_command_catalog(snapshot),
                    promote_latest=False, force_provisional=True,
                    reopen_previous_on_failure=False,
                    before_open=authorize_commit,
                    after_open=lambda: self._activate_snapshot(snapshot, previous),
                    preclosed_channel_state=channel_state,
                )
            )
        except BaseException as error:
            if transaction is not None and transaction.must_retain:
                self._hold_selection_publication(transaction)
                raise
            # Participant 的清理已经失败时保留 pending；外层不能隐式重试它。
            if isinstance(error, _PublicationParticipantRestoreError):
                if operation.revoked and isinstance(error, Exception):
                    raise BaseExceptionGroup("发布取消且资源清理失败", [asyncio.CancelledError(), error]) from None
                raise
            if operation.revoked and isinstance(error, Exception):
                error = BaseExceptionGroup("发布取消且资源操作失败", [asyncio.CancelledError(), error])
            cleanup_cancelled = False
            try:
                if transaction is not None and self._snapshot_store.pending_transaction is transaction:
                    _, cleanup_cancelled = await _complete_critical(
                        self._snapshot_store.abort(transaction, reopen_previous=False)
                    )
                elif transaction is None:
                    _, cleanup_cancelled = await _complete_critical(
                        self._dispose_unreferenced_composition_root(snapshot)
                    )
            except BaseException as cleanup_error:
                raise BaseExceptionGroup("新 Root 发布与清理均失败", [error, cleanup_error]) from None
            if cleanup_cancelled and not isinstance(error, asyncio.CancelledError):
                raise BaseExceptionGroup("发布失败且清理期间取消", [error, asyncio.CancelledError()]) from None
            raise error
        try:
            if self._endpoint_resumer is not None:
                await self._endpoint_resumer()
            self._check_operation_commit()
        except BaseException as error:
            assert transaction is not None
            self._hold_selection_publication(transaction)
            owner = (
                snapshot.generations[attempt.plugin_id]
                if attempt is not None else next(iter(snapshot.generations.values()), None)
            )
            if owner is not None:
                self._record_root_failure(
                    owner, error, formal_effects=("committed_runtime_start_failed",),
                    recovery_target="candidate",
                )
            raise
        if cancelled:
            assert transaction is not None
            self._hold_selection_publication(transaction)
            raise asyncio.CancelledError
        return snapshot

    def _hold_selection_publication(self, transaction: SnapshotTransaction) -> None:
        """保留提交事实和新 owner；失败返回后不得后台开放接纳。"""
        if transaction.must_retain:
            self._snapshot_store.hold_failed_publication(transaction)
        else:
            self._snapshot_store.pause_admission()
            transaction.candidate.accepting_leases = False
        if self._active_channel_generation is not None:
            self._active_channel_generation.close_admission()

    def _begin_snapshot_publication(self, snapshot: RuntimeSnapshot) -> SnapshotTransaction:
        transaction = self._snapshot_store.begin_publish(snapshot)
        if snapshot.composition_root is not None:
            self._building_roots.pop(snapshot.composition_root, None)
        return transaction

    def _activate_snapshot(self, snapshot: RuntimeSnapshot, previous: RuntimeSnapshot | None) -> None:
        """提交时激活整个新组合，旧组合整体退役。"""
        old = {} if previous is None else previous.generations
        for generation in snapshot.generations.values():
            self._activate_published_generation(generation, old.get(generation.plugin_id))
            generation.state = "active"
        for generation in old.values():
            self._retire_generation(generation)

    async def retry_runtime_recovery(self, plugin_id: str) -> dict[str, object]:
        return await self._run_operation(lambda: self._retry_runtime_recovery(plugin_id))

    async def _retry_runtime_recovery(self, plugin_id: str) -> dict[str, object]:
        """显式重试原资源句柄，再从当前 stable 的固定输入重建整组实例。"""
        publication = self._publication
        if publication is not None and isinstance(publication.selection_result, SelectionWriteError):
            if publication.selection_result.outcome == "uncertain":
                raise RuntimeError("stable 写入耐久性未确认；保留 owner，需关闭后从磁盘重新启动")
        selection_ref = self._selection.read()
        if selection_ref is None:
            raise RuntimeError("尚无已提交完整选择；关闭后显式重试首次启动")
        actions = tuple(
            action for action in self._reload_journal.pending_recovery()
            if action.plugin_id == plugin_id
            and action.action in {"retry_generation_cleanup", "retry_runtime_recovery"}
        )
        if len(actions) != 1:
            raise RuntimeError("插件没有唯一待执行的 runtime recovery")
        action = actions[0]
        _reject_retired_owner_recovery(action)
        current = self._snapshot_store.pause_admission()
        if self._endpoint_quiescer is not None:
            await self._endpoint_quiescer()
        if self._active_channel_generation is not None:
            self._active_channel_generation.close_admission()
        if current is not None:
            await self._snapshot_store.wait_for_no_leases(current)
        self._check_operation_commit()

        # 1. 实际资源由其 Root Effect 关闭，不能按插件 ID 重放取得操作。
        for token in (action.failure_resource or "").split(","):
            if token.startswith("channel-binding:"):
                await _complete_critical(self._channel_generation_host.retry_generation_cleanup(
                    token.removeprefix("channel-binding:")
                ))
        pending = self._snapshot_store.pending_transaction
        if pending is not None:
            failure = self._channel_generation_host.failure(pending.candidate.snapshot_id)
            if failure is not None:
                for owner in cast(tuple[ChannelCleanupTombstone, ...], failure):
                    await _complete_critical(self._channel_generation_host.retry_generation_cleanup(owner.binding_token))
            await _complete_critical(self._snapshot_store.abort(pending, reopen_previous=False))
        await _complete_critical(self._snapshot_store.retry_drains())
        for root in tuple(self._building_roots):
            await self._close_building_root(root)
        latest = self._snapshot_store.unpromoted_candidate
        if latest is not None:
            self._snapshot_store.pause_candidate_admission(latest)
            await _complete_critical(self._snapshot_store.discard_latest(latest))
        self._ready_candidate = None
        for key in tuple(self._prepared_generations):
            await self._discard_prepared(key)

        # 2. 旧 owner 全部退出后才重建；恢复本身发布真实新 snapshot。
        self._check_operation_commit()
        old_channel = self._active_channel_generation
        await _complete_critical(self._close_formal_root(current))
        replacement = await self._build_and_publish_root(
            self._selection_components(selection_ref),
            previous=current, old_channel=old_channel, expected_ref=selection_ref,
            commit_selection=False,
        )
        self._reload_journal.settle_boot(
            action, committed=self._selection_transition_committed(action.tx_id, selection_ref),
            cleanup_receipt="fresh-stable-root-restored",
        )
        generation = replacement.generations.get(plugin_id)
        return {
            "plugin_id": plugin_id, "publication_state": "recovered",
            "recovery_target": action.recovery_target,
            "generation_id": None if generation is None else generation.generation_id,
            "snapshot_id": replacement.snapshot_id,
            "retry_receipt": "fresh-stable-root-restored",
        }

    async def drop_candidate(self, plugin_id: str) -> dict[str, object]:
        return await self._run_operation(lambda: self._drop_ready(plugin_id))


    async def discard_update(self, update_id: str, *, reason: str = "candidate behavior rejected") -> None:
        await self._run_operation(lambda: self._discard_update(update_id, reason=reason))

    async def _discard_update(self, update_id: str, *, reason: str = "candidate behavior rejected") -> None:
        """持候选锁核对原请求；不能撤销期间已被替换的另一候选。"""
        update = self._reload_journal.update(update_id)
        ready = self._require_ready_candidate(update.plugin_id)
        if update.phase != "armed" or update.reload_tx_id != ready.candidate.reload_tx_id:
            raise RuntimeError("更新与当前待处理候选不匹配")
        if any(host.parent_lease.snapshot is ready.snapshot for host in self._validation_hosts.values()):
            raise RuntimeError("验证尚未退出或资源尚未清理")
        _ = await self._drop_ready(update.plugin_id, error=reason)
        self._notify_updates()

    async def _drop_ready(self, plugin_id: str, *, error: str = "candidate behavior rejected") -> dict[str, object]:
        ready = self._require_ready_candidate(plugin_id)
        tx_id = ready.candidate.reload_tx_id
        if tx_id is None:
            raise RuntimeError("latest candidate 缺少 reload transaction")
        phase = self._reload_journal.get(tx_id).phase
        if phase in {"latest_ready", "promoting"}:
            self._advance_reload(
                ready.candidate,
                "discarding",
                error=error,
            )
        elif phase != "discarding":
            raise RuntimeError(f"latest candidate 不能从 {phase} discard")
        _, cancelled = await _complete_critical(
            self._snapshot_store.discard_latest(ready.snapshot)
        )
        retained = self._reload_journal.get(tx_id)
        if retained.phase in {"cleanup_failed", "degraded"}:
            raise RuntimeError("candidate runtime cleanup 未完成，必须先执行 recovery")
        self._advance_reload(
            ready.candidate,
            "aborted",
            error=error,
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
        """发布完整隔离候选；直接更新也通过同一个 ready → formal 入口。"""
        self._check_operation_commit()
        generation = self._prepared_generations[plugin_id]
        tx_id = generation.reload_tx_id
        if tx_id is None or self._reload_journal.get(tx_id).phase != "prepared":
            raise RuntimeError("插件候选已失去发布授权")
        snapshot = generation.runtime_snapshot
        if snapshot is None:
            raise RuntimeError("插件候选缺少实际 Root snapshot")
        active = self._active_generations.get(plugin_id)
        if not self._validation_only:
            self._reload_journal.annotate(tx_id, {
                "event": "selection_candidate",
                "components": [self._generation_archive_ref(item) for item in snapshot.generations.values()],
            })
        transaction = self._begin_snapshot_publication(snapshot)
        try:
            self._advance_reload(generation, "validating", candidate_snapshot_id=snapshot.snapshot_id)
            if self._dashboard_preparer is not None:
                self._dashboard_preparer(snapshot)
            await self._post_publish_invariants(generation, snapshot)
            def open_candidate() -> None:
                self._check_operation_commit()
                self._advance_reload(generation, "commit_started")

            _, cancelled = await _complete_critical(
                self._snapshot_store.commit_latest(
                    transaction,
                    before_open=open_candidate,
                )
            )
        except BaseException as error:
            try:
                _, cleanup_cancelled = await _complete_critical(
                    self._snapshot_store.abort(transaction, reopen_previous=False)
                )
            except BaseException as cleanup_error:
                self._record_root_failure(
                    generation, cleanup_error, resource="runtime-snapshot-drain",
                    formal_effects=("candidate_runtime_cleanup_pending",), recovery_target="base",
                )
                raise BaseExceptionGroup("候选发布与清理均失败", [error, cleanup_error]) from None
            self._prepared_generations.pop(plugin_id, None)
            self._abort_reload(generation, error=str(error) or type(error).__name__)
            if cleanup_cancelled and not isinstance(error, asyncio.CancelledError):
                raise BaseExceptionGroup("候选发布失败且清理期间取消", [error, asyncio.CancelledError()]) from None
            raise
        for item in snapshot.generations.values():
            item.state = "candidate"
        self._prepared_generations.pop(plugin_id)
        self._ready_candidate = _ReadyPluginCandidate(
            plugin_id=plugin_id, previous=active, snapshot=snapshot,
        )
        self._advance_reload(generation, "latest_ready")
        if cancelled:
            raise asyncio.CancelledError
        if not _installed_generation_is_candidate(generation):
            result = await self._switch_ready(plugin_id)
            result["publication_state"] = "committed"
            return result
        return self._publication_status(
            plugin_id, active=active, candidate=generation, publication_state="latest_ready",
        )

    def _activate_published_generation(
        self,
        generation: PluginGeneration,
        previous: PluginGeneration | None,
    ) -> None:
        published_module = sys.modules[generation.module_path]
        stable_alias = self._stable_aliases.get(generation.module_path)
        if stable_alias is None and previous is not None:
            stable_alias = self._stable_aliases.get(previous.module_path)
        if stable_alias is None:
            stable_alias = "_akashic_stable_" + hashlib.sha256(generation.plugin_id.encode()).hexdigest()[:16]

        # 先完成可能失败的查找，再替换 stable import alias。
        code_dir = generation.code_dir
        self._remove_module_tree(stable_alias)
        self._fresh_importer.register(stable_alias, code_dir)
        sys.modules[stable_alias] = published_module
        if previous is not None:
            _ = self._stable_aliases.pop(previous.module_path, None)
        self._stable_aliases[generation.module_path] = stable_alias

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
        for item in snapshot.generations.values():
            if not item.scope.accepting_resources:
                raise RuntimeError("RuntimeSnapshot 插件作用域已关闭")
            root = snapshot.composition_root
            if root is not None:
                runtime = root.plugin_runtime(item.plugin_id)
                if runtime.generation_id != item.generation_id or runtime.data_dir != item.data_dir:
                    raise RuntimeError("RuntimeSnapshot generation 与实际挂载 runtime 不一致")
                if not any(
                    fiber.runtime is runtime
                    and fiber.plugin_module is cast(ComposablePlugin, item.instance).module
                    for fiber in root.root_fiber.children
                ):
                    raise RuntimeError("RuntimeSnapshot generation 与实际挂载模块不一致")

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
        self._notify_updates()

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

    def _track_reload_drain(
        self,
        generation: PluginGeneration,
        previous_snapshot: RuntimeSnapshot | None,
    ) -> None:
        tx_id = generation.reload_tx_id
        if tx_id is None:
            return
        phase = self._reload_journal.get(tx_id).phase
        if phase in {"cleanup_failed", "degraded", "aborted", "recovered", "complete"}:
            return
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
            source_revision = _source_revision(plugin_dir)
            _, config_revision = load_config(
                _resolve_plugin_data_dir(mod["name"], mod, self._workspace)
            )
            current_prepared = self._prepared_generations.get(plugin_id)
            if force_reprepare and current_prepared is not None:
                await self._discard_prepared(plugin_id)
                current_prepared = None
            matches_active = (
                source_revision == active.source_revision
                and config_revision == active.config_revision
            )
            if matches_active:
                if current_prepared is None:
                    continue
                await self._discard_prepared(plugin_id)
                result = {
                    "plugin_id": plugin_id,
                    "active_generation": active.generation_id,
                    "prepared_generation": None,
                    "preparation_state": "active",
                    "candidate_revision": source_revision,
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
            await self._discard_prepared(plugin_id)
            prepared = await self._load_one(mod, activate=False)
            result: dict[str, object] = {
                "plugin_id": plugin_id,
                "active_generation": active.generation_id,
                "prepared_generation": (
                    prepared.generation_id if prepared is not None else None
                ),
                "preparation_state": "prepared" if prepared is not None else "failed",
                "candidate_revision": (
                    prepared.source_revision if prepared is not None else source_revision
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
        stage_stable: bool = False,
    ) -> PluginGeneration | None:
        """先固定安装输入，再为目标环境构建独立的完整 Root。"""

        base_snapshot = self.current_snapshot
        base_selection_ref = None if self._validation_only else self._selection.read()
        plugin_id = _resolve_plugin_id(mod)
        if activate and plugin_id in self._active_generations:
            return self._active_generations[plugin_id]
        if load_plugin_manifest(_plugins_home(self._installed_cache_root)).get(plugin_id, True) is False:
            return None
        self._gate_results.pop(plugin_id, None)
        plugin_dir = Path(mod["plugin_root"])
        entry = plugin_dir / "plugin.py"
        if Path(mod["module_path"]).absolute() != entry.absolute():
            raise RuntimeError("source discovery module path 与制品 plugin.py 不一致")
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"插件 plugin.py 必须是普通文件: {entry}")
        identity = load_static_plugin_manifest(plugin_dir)
        if mod.get("manifest_digest", "") != identity.identity_digest:
            raise RuntimeError("source discovery identity 已漂移")
        revision = _source_revision(plugin_dir)
        data_dir = _resolve_plugin_data_dir(mod["name"], mod, self._workspace)
        validate_workspace_plugin_data_path(data_dir, self._workspace)
        config, config_revision = load_config(data_dir)
        code_ref = self._archive.save(
            plugin_dir, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE}),
        )
        code_dir = self._archive.open(code_ref)
        if _source_revision(code_dir) != revision or load_static_plugin_manifest(code_dir) != identity:
            raise RuntimeError("插件源码或安装身份在归档前发生变化")
        environments: dict[str, str] = {}
        if identity.python:
            if (plugin_dir / ENVIRONMENT_FILE).exists():
                environments = read_environment_refs(plugin_dir, identity)
            elif mod["source_type"] == "installed":
                raise RuntimeError("插件尚未准备固定 Python 环境；请通过安装流程重建")

        # 1. 固定输入所需的临时导入也有真实 Root owner。
        namespace = secrets.token_hex(12)
        root = CompositionRoot("archive-input:" + namespace)
        self._building_roots[root] = ()
        module_path = f"_akashic_input_{namespace}"
        generation_id = f"{plugin_id}:input:{namespace}"
        root._defer_internal_cleanup(
            f"module:{module_path}", lambda: self._remove_module_tree(module_path),
        )
        scope = PluginScope(plugin_id, generation_id=generation_id)
        root._defer_internal_cleanup(
            f"scope:{module_path}", lambda: self._close_root_scope(scope, module_path),
        )
        try:
            self._import_plugin(module_path, code_dir)
            instance = ComposablePlugin.from_module(sys.modules[module_path], identity)
            ref = self._archive.save_descriptor({
                "version": 4, "code": code_ref, "python_environments": environments,
                "plugin_id": plugin_id, "source_revision": revision,
                "config_revision": config_revision, "config": encode_config(config),
                "source_type": mod["source_type"],
                "data_dir": data_dir.resolve().relative_to(self._workspace.resolve()).as_posix(),
                "runtime": {"python_tag": sys.implementation.cache_tag, "binding_api": PLUGIN_ARCHIVE_BINDING_API},
            })
            source = PluginGeneration(
                plugin_id=plugin_id, generation_id=generation_id, module_path=module_path,
                source_revision=revision, config_revision=config_revision,
                plugin_dir=plugin_dir, data_dir=data_dir, instance=instance, scope=scope,
                config_projection=config, archive_ref=ref, static_manifest=identity,
                source_type=cast(Literal["builtin", "installed"], mod["source_type"]),
                state="prepared",
            )
            self._building_roots[root] = (source,)
            self._stable_aliases[module_path] = mod["import_path"]
            if stage_stable:
                return source
            if activate:
                inputs = dict(self._active_generations)
                inputs[plugin_id] = source
                components = tuple(self._generation_archive_ref(item) for _, item in sorted(inputs.items()))
                await self._close_building_root(root)
                snapshot = await self._replace_formal_root(
                    components, expected_ref=base_selection_ref,
                )
            else:
                snapshot = await self._compile_generation_snapshot(source, candidate_owner=source)
            generation = snapshot.generations[plugin_id]
            if root in self._building_roots:
                await self._close_building_root(root)
            self._stable_aliases.pop(module_path, None)
            if not activate:
                if self.current_snapshot is not base_snapshot or (
                    not self._validation_only and self._selection.read() != base_selection_ref
                ):
                    assert snapshot.composition_root is not None
                    await self._discard_building_root(snapshot.composition_root, SelectionConflictError("候选构建期间基线变化"))
                    raise SelectionConflictError("候选构建期间基线变化")
                generation.reload_tx_id = self._begin_reload_attempt(
                    plugin_id=plugin_id, generation_id=generation.generation_id,
                    source_revision=revision, config_revision=config_revision,
                    plugin_dir=plugin_dir, source_type=mod["source_type"],
                )
                self._advance_reload(
                    generation, "prepared", candidate_snapshot_id=snapshot.snapshot_id,
                )
                self._prepared_generations[plugin_id] = generation
            self._gate_results.pop(plugin_id, None)
            return generation
        except BaseException as error:
            if root in self._building_roots:
                await self._discard_building_root(root, error)
            raise

    async def _compile_generation_snapshot(
        self,
        generation: PluginGeneration,
        *,
        candidate_owner: PluginGeneration | None = None,
    ) -> RuntimeSnapshot:
        generations = dict(self._active_generations)
        generations[generation.plugin_id] = generation
        composition_root = await self._resolve_composition_root(
            generations,
            candidate_owner=candidate_owner,
        )
        try:
            snapshot = self._snapshot_compiler.compile(
                generations,
                catalog_generation=generations[generation.plugin_id],
                composition_root=composition_root,
                core_channel_definitions=self._core_channel_definitions,
                require_composition_ready=True,
            )
            if candidate_owner is not None:
                self._preflight_durable_delivery_targets(snapshot)
        except BaseException as error:
            await self._discard_building_root(composition_root, error)
            if not isinstance(error, Exception):
                raise
            gate = self._record_failed_gate(
                plugin_id=generation.plugin_id,
                revision=generation.source_revision,
                check_id="runtime_snapshot",
                reason=str(error),
            )
            raise _CandidateRejected(gate) from error
        for item in generations.values():
            item.runtime_snapshot = snapshot
        return snapshot

    @asynccontextmanager
    async def open_validation(self, update_id: str) -> AsyncGenerator[BindingScope]:
        """准备和回收使用 Manager owner；调用程序仍拥有验证 scope。"""
        caller = asyncio.current_task()
        assert caller is not None
        host: ValidationHost | None = None
        scope: BindingScope | None = None
        failure: BaseException | None = None
        try:
            host, scope = await self._run_operation(lambda: self._prepare_validation(update_id, caller))
            async with RuntimeScope(host.parent_lease.fork()):
                async with RuntimeScope(await host.manager._snapshot_store.acquire()):
                    yield scope
        except asyncio.CancelledError as error:
            failure = error
            self._reload_journal.record_update_error(update_id, "validation cancelled")
            self._notify_updates()
            raise
        except BaseException as error:
            failure = error
            self._reload_journal.record_update_error(update_id, str(error) or type(error).__name__)
            self._notify_updates()
            raise
        finally:
            if scope is not None:
                scope._expire()  # pyright: ignore[reportPrivateUsage]
            # 观察者可能恰好在准备结束时超时；实际 host 仍在 Manager 内。
            retained = host or next(
                (item for item in self._validation_hosts.values() if item.task is caller), None,
            )
            operation = self._operation
            if retained is not None:
                retained.active = False
                if not self._stopping and (operation is None or operation.task.done()):
                    try:
                        await self._run_operation(lambda: self._retry_validation_cleanup(retained.identity))
                    except BaseException as cleanup_error:
                        if failure is not None:
                            raise BaseExceptionGroup("验证与资源回收均失败", [failure, cleanup_error]) from None
                        raise
                    update = self._reload_journal.update(update_id)
                    self._reload_journal.annotate(cast(str, update.reload_tx_id), {
                        "event": "business_validation_closed", "validation_id": retained.identity,
                    })

    async def _prepare_validation(
        self, update_id: str, caller: asyncio.Task[object],
    ) -> tuple[ValidationHost, BindingScope]:
        """从租约取得到资源交接都属于同一次有限操作，复制线程不脱离 owner。"""
        update = self._reload_journal.update(update_id)
        ready = self._require_ready_candidate(update.plugin_id)
        if update.phase != "armed" or update.reload_tx_id != ready.candidate.reload_tx_id:
            raise RuntimeError("更新恢复点与当前候选不匹配")
        lease = self._snapshot_store.lease(ready.snapshot.snapshot_id)
        host: ValidationHost | None = None
        scope: BindingScope | None = None
        async with RuntimeScope(lease):
            try:
                host = await self._build_validation_host(lease)
                host.task = caller
                self._validation_hosts[host.identity] = host
                await self._copy_validation_bindings(host)
                await self._copy_validation_artifacts(host)
                self._reload_journal.annotate(cast(str, update.reload_tx_id), {
                    "event": "business_validation_opened", "validation_id": host.identity,
                    "candidate_generation": ready.candidate.generation_id,
                    "candidate_snapshot": ready.snapshot.snapshot_id,
                    "workspace": str(host.workspace),
                    "components": [item.archive_ref for item in ready.snapshot.generations.values()],
                    "profile": "explicit_program_with_candidate_resources",
                })
                child = host.manager
                components = tuple(cast(str, item.archive_ref) for item in ready.snapshot.generations.values())
                root = CompositionRoot("validation:" + host.identity)
                host.root = root
                child._building_roots[root] = ()
                generations = child._archived_generations(
                    components, root, workspace=host.workspace,
                    sources=ready.snapshot.generations,
                )
                root._bind_runtime_scope_acquirer(  # pyright: ignore[reportPrivateUsage]
                    lambda: child._snapshot_store.acquire_composition_root(root)
                )
                ordered = tuple(generations[key] for key in sorted(generations))
                await child._provide_composition_services(root, ordered, candidate=False)
                for generation in ordered:
                    await child._mount_generation_composition(root, generation)
                if not root.receipt().ready:
                    raise RuntimeError(f"验证 provider 闭包不完整: {root.receipt().required_pending}")
                result = await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
                if result is not None:
                    raise RuntimeError("验证 snapshot.sealing 不接受 Bail")
                snapshot = RuntimeSnapshotCompiler().compile(generations, composition_root=root)
                for generation in generations.values():
                    generation.runtime_snapshot = snapshot
                self._check_operation_commit()
                child._snapshot_store.install(snapshot)
                child._building_roots.pop(root)
                # 2. apply 已通过候选授权取得资源，不再重复执行启动阶段。
                scope = BindingScope(root)
                self._reload_journal.annotate(cast(str, update.reload_tx_id), {
                    "event": "business_validation_ready", "validation_id": host.identity,
                    "validation_snapshot": snapshot.snapshot_id,
                    "generations": {key: value.generation_id for key, value in generations.items()},
                })
                self._check_operation_commit()
                return host, scope
            except BaseException as error:
                if scope is not None:
                    scope._expire()  # pyright: ignore[reportPrivateUsage]
                if host is not None:
                    host.active = False
                    try:
                        await _complete_critical(self._retry_validation_cleanup(host.identity))
                    except BaseException as cleanup_error:
                        raise BaseExceptionGroup("验证准备与资源关闭均失败", [error, cleanup_error]) from None
                raise

    async def _build_validation_host(
        self, lease: RuntimeSnapshotLease,
    ) -> ValidationHost:
        """先读取排除声明，再复制候选数据，最后固定消息库副本。"""
        if self._message_log is None:
            raise RuntimeError("业务验证缺少正式 MessageLog")
        identity = secrets.token_hex(16)
        workspace = self._workspace / "runtime" / "plugin-update-validation" / identity / "workspace"
        workspace.mkdir(parents=True)
        archive = PluginArchive(workspace / "runtime" / "plugin-archives")
        messages: MessageLog | None = None
        try:
            bindings = self._message_log.read_bindings()
            await self._copy_validation_components(
                lease.snapshot, workspace, archive, bindings,
            )
            # 图副本可能引用复制期间追加的 Message；最后一次 backup 必须覆盖这些引用。
            await _copy_in_thread(self._message_log.backup, workspace / "sessions.db")
            messages = MessageLog(workspace / "sessions.db")
            # 新 binding 可能带来旧凭据排除声明；不开放按较早声明复制的数据。
            # backup 之后追加的 binding 不在副本内，必须再对正式库复查一次。
            if (
                messages.read_bindings() != bindings
                or self._message_log.read_bindings() != bindings
            ):
                raise RuntimeError("候选复制期间 binding 已变化；本次验证副本不可用")
        except BaseException:
            if messages is not None:
                messages.close()
            await _copy_in_thread(_remove_validation_data_dir, workspace.parent)
            raise
        assert messages is not None
        try:
            artifacts = ArtifactStore(workspace / "sessions.db")
        except BaseException:
            messages.close()
            raise
        physical = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=artifacts)
        bus = EventBus()
        try:
            child = PluginManager(
                [], event_bus=bus, workspace=workspace, message_log=messages,
                installed_cache_root=workspace.parent / "plugins" / "cache",
                channel_attachment_store=physical, workload_controller=self._workload_controller,
            )
        except BaseException:
            artifacts.close()
            messages.close()
            raise
        child._python_environments = self._python_environments
        child._validation_only = True
        task = asyncio.current_task()
        assert task is not None
        return ValidationHost(identity, workspace, child, messages, artifacts, bus, task, lease.fork())

    async def _copy_validation_components(
        self,
        snapshot: RuntimeSnapshot,
        workspace: Path,
        archive: PluginArchive,
        bindings: tuple[Mapping[str, object], ...],
    ) -> None:
        """逐项保存声明数据；每个 SQLite 自身一致，不承诺跨文件的共同切点。"""
        binding_exclusions = self._validation_binding_exclusions(bindings)
        for generation in snapshot.generations.values():
            if generation.archive_ref is None:
                raise RuntimeError(f"验证组件缺少归档: {generation.plugin_id}")
            record = self._archive.read_descriptor(generation.archive_ref)
            code = cast(str, record["code"])
            if archive.save(self._archive.open(code)) != code:
                raise RuntimeError("验证代码副本身份不一致")
            if archive.save_descriptor(record) != generation.archive_ref:
                raise RuntimeError("验证组件描述身份不一致")
            data_dir = workspace / cast(str, record["data_dir"])
            validate_workspace_plugin_data_path(data_dir, workspace)
            excluded = set(_candidate_data_exclude_paths(generation.static_manifest))
            excluded.update(binding_exclusions.get(cast(str, record["data_dir"]), ()))
            # 历史 binding 的旧声明必须在 current data 首次复制前生效。
            _ = await _copy_in_thread(
                _copy_validation_tree, generation.data_dir, data_dir,
                tuple(sorted(excluded)),
            )
        plugins = tuple(cast(ComposablePlugin, item.instance) for item in snapshot.generations.values())
        await self._project_candidate_workspace_roots(plugins, workspace)
        await self._project_candidate_workspace_files(plugins, workspace)

    def _validation_binding_exclusions(
        self, bindings: tuple[Mapping[str, object], ...],
    ) -> dict[str, tuple[str, ...]]:
        """只读扫描固定 binding manifest，提前合并候选数据的排除路径。"""
        exclusions: dict[str, set[str]] = {}
        for binding in bindings:
            if binding["version"] != 1:
                raise ValueError("验证副本包含不支持的 binding 版本")
            root = self._archive.read_descriptor(cast(str, binding["root_ref"]))
            for component_ref in cast(tuple[str, ...], root["components"]):
                record = self._archive.read_descriptor(component_ref)
                plugin_dir = self._archive.open(cast(str, record["code"]))
                manifest = load_static_plugin_manifest(plugin_dir)
                exclusions.setdefault(cast(str, record["data_dir"]), set()).update(
                    _candidate_data_exclude_paths(manifest)
                )
        return {data_dir: tuple(sorted(paths)) for data_dir, paths in exclusions.items()}

    async def _copy_validation_bindings(self, host: ValidationHost) -> None:
        """只保存 binding provenance，不导入旧代码、数据或 workspace。"""
        archive = host.manager._archive
        current = {item.archive_ref for item in host.parent_lease.snapshot.generations.values()}
        refs = dict.fromkeys(cast(str, ref) for ref in current)
        for binding in host.messages.read_bindings():
            if binding["version"] != 1:
                raise ValueError("验证副本包含不支持的 binding 版本")
            root_ref = cast(str, binding["root_ref"])
            root = self._archive.read_descriptor(root_ref)
            if archive.save_descriptor(root) != root_ref:
                raise RuntimeError("验证 binding 闭包身份不一致")
            refs.update(dict.fromkeys(cast(tuple[str, ...], root["components"])))

        for ref in refs:
            record = self._archive.read_descriptor(ref)
            if archive.save_descriptor(record) != ref:
                raise RuntimeError("验证历史组件身份不一致")

    async def _copy_validation_artifacts(self, host: ValidationHost) -> None:
        """复制消息副本已引用的不可变文件，读取仍经过正式 Artifact owner 校验。"""
        for record in host.artifacts.list_attachments():
            if self._artifact_read is None:
                raise RuntimeError("验证历史 Artifact 缺少读取能力")
            lease = await self._artifact_read.acquire(record.ref)
            try:
                payload = await lease.read_bytes(max_bytes=record.ref.size_bytes)
                path = host.workspace / record.storage_key
                path.parent.mkdir(parents=True, exist_ok=True)
                def write_payload() -> None:
                    with path.open("xb") as output:
                        _ = output.write(payload)
                await _copy_in_thread(write_payload)
            finally:
                await lease.aclose()

    async def stop_validation_resources(self) -> None:
        """验证宿主也经同一关闭 owner；迟到清理不能绕过 shutdown。"""
        parent = current_operation.get()
        if parent is None or parent is self._operation:
            await self.terminate_all()
            return
        # 父 Manager 的有限观察已经覆盖这个内部关闭；连接不能越过子任务先关闭。
        operation = self._begin_termination()
        await _complete_critical(operation.task)

    async def retry_validation_cleanup(self, identity: str) -> None:
        await self._run_operation(lambda: self._retry_validation_cleanup(identity))

    async def _retry_validation_cleanup(self, identity: str) -> None:
        """只重试现存验证资源的清理；不重跑验证或建立候选。"""
        host = self._validation_hosts[identity]
        if host.active:
            raise RuntimeError("验证程序尚未退出")
        _, cancelled = await _complete_critical(host.close())
        _ = self._validation_hosts.pop(identity, None)
        if cancelled:
            raise asyncio.CancelledError

    def _archived_generations(
        self,
        components: tuple[str, ...],
        root: CompositionRoot,
        *,
        workspace: Path,
        sources: Mapping[str, PluginGeneration],
    ) -> dict[str, PluginGeneration]:
        """从固定输入创建只属于当前 Root 的模块、Scope 和 generation。"""

        records = tuple(self._archive.read_descriptor(ref) for ref in components)
        for record in records:
            if record["version"] != 4 or record["runtime"] != {
                "python_tag": sys.implementation.cache_tag,
                "binding_api": PLUGIN_ARCHIVE_BINDING_API,
            }:
                raise RuntimeError("插件归档运行合同不兼容；保留原归档并使用原 Core 恢复")
        generations: dict[str, PluginGeneration] = {}
        namespace = secrets.token_hex(12)
        for index, (ref, record) in enumerate(zip(components, records, strict=True)):
            code_dir = self._archive.open(cast(str, record["code"]))
            revision = cast(str, record["source_revision"])
            if _source_revision(code_dir) != revision:
                raise RuntimeError("插件归档源码身份不一致")
            plugin_id = cast(str, record["plugin_id"])
            if plugin_id in generations:
                raise ValueError(f"归档重复包含插件: {plugin_id}")
            source = sources.get(plugin_id)
            data_dir = workspace / cast(str, record["data_dir"])
            validate_workspace_plugin_data_path(data_dir, workspace)
            module_path = f"_akashic_archive_{namespace}_{index}"
            generation_id = f"{plugin_id}:{namespace}:{index}"
            # 导入也会取得 sys.modules；必须先把整个 namespace 交给 Root。
            root._defer_internal_cleanup(
                f"module:{module_path}",
                lambda module_path=module_path: self._remove_module_tree(module_path),
            )
            scope = PluginScope(plugin_id, generation_id=generation_id)
            root._defer_internal_cleanup(
                f"scope:{module_path}",
                lambda scope=scope, module_path=module_path: self._close_root_scope(scope, module_path),
            )
            manifest = load_static_plugin_manifest(code_dir)
            self._import_plugin(module_path, code_dir)
            plugin = ComposablePlugin.from_module(sys.modules[module_path], manifest)
            if plugin.name != plugin_id.split("@", 1)[0]:
                raise ValueError("归档插件身份不一致")
            projection = decode_config(record["config"])
            if not isinstance(projection, dict):
                raise ValueError("归档插件配置必须是对象")
            generation = PluginGeneration(
                plugin_id=plugin_id, generation_id=generation_id, module_path=module_path,
                source_revision=revision, config_revision=cast(str, record["config_revision"]),
                plugin_dir=code_dir if source is None else source.plugin_dir,
                data_dir=data_dir, config_projection=cast(dict[str, object], projection),
                instance=plugin, scope=scope,
                static_manifest=manifest,
                source_type=cast(Literal["builtin", "installed"], record["source_type"]),
                archive_ref=ref,
                reload_tx_id=None,
                validation_workspace=workspace if workspace != self._workspace or self._validation_only else None,
                state="prepared",
            )
            generations[plugin_id] = generation
            self._building_roots[root] = tuple(generations.values())
            if source is not None:
                alias = self._stable_aliases.get(source.module_path)
                if alias is not None:
                    self._stable_aliases[module_path] = alias
        return generations

    async def _close_root_scope(self, scope: PluginScope, module_path: str) -> None:
        """Root 只在 Scope 关闭成功后释放句柄，失败时继续持有依赖。"""
        failures = await scope.aclose()
        self._cleanup_failures.extend(failures)
        if failures:
            raise RuntimeError(
                f"Root scope cleanup 未完成，必须显式 retry: {module_path}: "
                + "; ".join(f"{item.resource}: {item.error}" for item in failures)
            )

    async def _resolve_composition_root(
        self,
        generations: dict[str, PluginGeneration],
        *,
        candidate_owner: PluginGeneration | None = None,
        components: tuple[str, ...] | None = None,
    ) -> CompositionRoot:
        """同一归档组合在每个环境重新实例化，返回实际挂载的 generation 表。"""

        sources = dict(sorted(generations.items()))
        if components is None:
            components = tuple(self._generation_archive_ref(item) for item in sources.values())
        root = CompositionRoot(
            "plugins:" + secrets.token_hex(16),
            candidate_incident_limit=1024 if candidate_owner is not None else None,
        )
        self._building_roots[root] = ()
        root._bind_runtime_scope_acquirer(
            lambda: self._snapshot_store.acquire_composition_root(root)
        )
        workspace = self._workspace
        if candidate_owner is not None:
            workspace = self._workspace / "runtime" / "plugin-validation" / secrets.token_hex(16) / "workspace"
            root._defer_internal_cleanup(
                "candidate_data",
                lambda: _copy_in_thread(_remove_validation_data_dir, workspace.parent),
            )
        try:
            actual = self._archived_generations(
                components, root, workspace=workspace, sources=sources,
            )
            generations.clear()
            generations.update(actual)
            ordered = tuple(actual.values())
            if candidate_owner is not None:
                for item in ordered:
                    record = self._archive.read_descriptor(self._generation_archive_ref(item))
                    source_data = self._workspace / cast(str, record["data_dir"])
                    item.data_dir.parent.mkdir(parents=True, exist_ok=True)
                    await _copy_in_thread(
                        _copy_validation_tree, source_data, item.data_dir,
                        _candidate_data_exclude_paths(item.static_manifest),
                    )
                plugins = tuple(cast(ComposablePlugin, item.instance) for item in ordered)
                await self._project_candidate_workspace_roots(plugins, workspace)
                await self._project_candidate_workspace_files(plugins, workspace)
            else:
                for item in ordered:
                    ensure_workspace_plugin_data_dir(item.data_dir, workspace)
            await self._provide_composition_services(
                root, ordered, candidate=candidate_owner is not None,
            )
            for item in ordered:
                await self._mount_generation_composition(root, item)
            receipt = root.receipt()
            if not receipt.ready:
                raise RuntimeError(
                    "v3 插件组合拓扑未就绪: "
                    f"required_pending={receipt.required_pending}, "
                    f"required_degraded={receipt.required_degraded}, "
                    f"incidents={receipt.incidents}"
                )
            result = await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
            if result is not None:
                raise CompositionError(
                    "SNAPSHOT_SEALING_BAIL_NOT_ALLOWED",
                    "snapshot.sealing 接入点不接受 Bail",
                )
        except BaseException as error:
            await self._discard_building_root(root, error)
            raise
        return root

    @staticmethod
    def _generation_archive_ref(generation: PluginGeneration) -> str:
        if generation.archive_ref is None:
            raise RuntimeError(f"插件缺少固定归档: {generation.plugin_id}")
        return generation.archive_ref

    def _building_root_uses(self, generation: PluginGeneration) -> bool:
        return any(
            item is generation
            for generations in self._building_roots.values()
            for item in generations
        )

    async def _close_building_root(self, root: CompositionRoot) -> None:
        """关闭成功才解除构建 owner，调用者取消仍等待真实清理结束。"""

        async def close() -> None:
            await root.dispose()
            for generation in self._building_roots.get(root, ()):
                self._stable_aliases.pop(generation.module_path, None)
            # Root.dispose 自己合并并发关闭；各等待者确认同一成功结果。
            self._building_roots.pop(root, None)

        try:
            _, cancelled = await _complete_critical(close())
        except BaseException as error:
            _ = self._snapshot_store.pause_admission()
            operation = current_operation.get()
            if operation is not None and operation.revoked and isinstance(error, Exception):
                raise BaseExceptionGroup(
                    "Root 清理期间调用者取消且清理失败",
                    [asyncio.CancelledError(), error],
                ) from None
            raise
        if cancelled:
            raise asyncio.CancelledError

    async def _discard_building_root(
        self, root: CompositionRoot, error: BaseException,
    ) -> None:
        """异常回退只尝试一次；Root 已在关闭时保留原句柄供显式重试。"""

        # 1. Fiber 挂载回退或 Root 关闭已经失败时，不重放同一次清理。
        if root.root_fiber.state == FiberState.UNLOADING or any(
            fiber.state == FiberState.UNLOADING for fiber in root.receipt().fibers
        ):
            _ = self._snapshot_store.pause_admission()
            raise error
        # 2. 首次回收保留初始化与清理的两份真实错误。
        try:
            await self._close_building_root(root)
        except BaseException as cleanup_error:
            if root not in self._building_roots and isinstance(cleanup_error, asyncio.CancelledError):
                if isinstance(error, asyncio.CancelledError):
                    raise error
                raise BaseExceptionGroup(
                    "Root 构建失败且回收期间取消", [error, cleanup_error],
                ) from None
            raise BaseExceptionGroup(
                "Root 构建和清理均失败", [error, cleanup_error],
            ) from None

    async def _provide_root_registries(
        self, root: CompositionRoot, mount_order: tuple[PluginGeneration, ...],
    ) -> None:
        """注册表只属于当前 Root，不接入正式执行 owner。"""
        if any(
            CHANNELS in cast(ComposablePlugin, item.instance).inject
            for item in mount_order
        ):
            _ = await root.context.provide(
                CHANNELS,
                PluginChannels(root.instance_token),
            )
    async def _provide_composition_services(
        self,
        root: CompositionRoot,
        mount_order: tuple[PluginGeneration, ...],
        *,
        candidate: bool,
    ) -> None:
        """向当前 stable 或 candidate Root 提供宿主能力。"""

        await self._provide_root_registries(root, mount_order)
        execution = ExecutionAccess(root.instance_token, {
            item.plugin_id: CodeOwner(item.generation_id, item.code_dir,
                lambda command, cwd, item=item: self._resolve_runtime_command(item, command, cwd))
            for item in mount_order
        }, candidate=candidate or self._validation_only)
        await root.context.provide(EXECUTION, execution)
        await root.context.provide(WORKLOAD_CONTROLLER,
            ControllerAccess(execution, self._workload_controller, self._workload_workspace_id))
        requested = {
            key
            for generation in mount_order
            for key in cast(ComposablePlugin, generation.instance).inject
        }
        if RUNTIME_CATALOG in requested:
            def read_runtime_catalog() -> dict[str, object]:
                """Read neutral runtime facts from the exact task-bound snapshot."""

                snapshot = get_current_runtime_snapshot()
                if snapshot is None:
                    raise RuntimeError("读取 runtime catalog 需要当前任务的 runtime scope")
                current = snapshot.composition_root
                if (
                    current is None
                    or current.context.require(RUNTIME_CATALOG) is not read_runtime_catalog
                ):
                    raise RuntimeError("runtime catalog 不属于当前 runtime scope")
                return build_runtime_catalog(snapshot)

            _ = await root.context.provide(RUNTIME_CATALOG, read_runtime_catalog)
        if CREDENTIALS in requested:
            clients = CredentialClients(None if candidate or self._validation_only else {
                generation.plugin_id: CoreProviderClientFactory(
                    generation.data_dir,
                    generation.config_projection, generation.config_revision,
                )
                for generation in mount_order
            })
            _ = await root.context.provide(CREDENTIALS, clients)
            root._defer_internal_cleanup("credential_clients", clients.aclose)  # pyright: ignore[reportPrivateUsage]
        if PLUGIN_UPDATES in requested:
            _ = await root.context.provide(
                PLUGIN_UPDATES, PluginUpdates(None if candidate or self._validation_only else self),
            )
        message_services: set[ServiceKey[object]] = {
            MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION, BINDINGS
        }
        if RESTART_GATE in requested:
            gate = self._restart_gate
            if candidate:
                gate = RestartGate(
                    boot_id="candidate",
                    supervised=gate is not None and gate.supervised,
                    execution_enabled=False,
                )
            elif gate is None:
                # 直接使用 PluginManager 的测试/嵌入式运行没有 Supervisor；仍提供
                # 一个允许正常 work 的 unmanaged gate，不伪造可提交的重启通道。
                gate = RestartGate(boot_id=self._host_boot_id, supervised=False)
                self._restart_gate = gate
            _ = await root.context.provide(RESTART_GATE, gate)
        if CONTROL_FRAMES in requested:
            # 候选与独立验证只读取副本；它们不能取得正式连接的发送 owner。
            isolated = candidate
            frames = FrameBook() if candidate else self._control_frames
            _ = await root.context.provide(CONTROL_FRAMES, frames)
            if isolated:
                root._defer_internal_cleanup("control_frames.close", frames.close)  # pyright: ignore[reportPrivateUsage]
        # 正式能力归宿主所有，不计入历史 provider 的依赖拓扑。
        log = None if candidate else self._message_log
        if requested & message_services and self._message_log is None:
            raise RuntimeError("消息能力需要 bootstrap 提供已迁移的 MessageLog")
        if self._message_log is not None:
            _ = await root.context.provide(MESSAGE_CATALOG, MessageCatalog(log))
            _ = await root.context.provide(MESSAGE_EMBEDDINGS, MessageEmbeddings(log))
            _ = await root.context.provide(MESSAGE_WRITERS, MessageWriters(log))
            _ = await root.context.provide(OWNER_STATE, OwnerState(log))
            _ = await root.context.provide(SESSION_ADMISSION, SessionAdmission(log))
            _ = await root.context.provide(
                BINDINGS, Bindings(log, self._archive, root)
            )
        if TASKS in requested or self._message_log is not None:
            _ = await root.context.provide(
                TASKS, PluginTasks(formal=False) if candidate else self._plugin_tasks
            )
        if PROCESSES in requested:
            _ = await root.context.provide(
                PROCESSES, PluginProcesses(formal=False) if candidate else self._plugin_processes
            )
        if self._artifact_read is not None:
            _ = await root.context.provide(
                ARTIFACT_READ, ArtifactRead(None) if candidate else self._artifact_read
            )
        if ARTIFACT_IMPORT in requested and self._artifact_import is not None:
            _ = await root.context.provide(
                ARTIFACT_IMPORT, ArtifactImport(None) if candidate else self._artifact_import
            )
        if TIMERS in requested:
            timers = (
                PluginTimers(AsyncioOneShotTimer())
                if not candidate else PluginTimers.candidate_validation()
            )
            _ = await root.context.provide(TIMERS, timers)

        # Client UI and message display are neutral snapshot projections.  The
        # host publishes them under stable names; each client request resolves
        # the provider again inside its exact RuntimeSnapshot scope.
        host_ui_requested = {
            key.name
            for key in requested
            if key.name in {
                "core.message_display.v1",
                "core.mobile_ui.v1",
            }
        }
        if "core.message_display.v1" in host_ui_requested:
            from agent.plugins.snapshot import project_message_rows

            async def display_message_page(
                page: object,
                *,
                display_only: bool,
            ) -> list[dict[str, object]]:
                return await project_message_rows(
                    self._snapshot_store,
                    page,
                    display_only=display_only,
                )

            _ = await root.context.provide(
                ServiceKey[object]("core.message_display.v1"),
                display_message_page,
            )
        if "core.mobile_ui.v1" in host_ui_requested:
            from agent.plugins.mobile_ui import PluginMobileUiProvider

            mobile_ui = PluginMobileUiProvider(self)
            _ = await root.context.provide(
                ServiceKey[object]("core.mobile_ui.v1"),
                mobile_ui,
            )
            root._defer_internal_cleanup(  # pyright: ignore[reportPrivateUsage]
                "mobile_ui_provider.close",
                mobile_ui.aclose,
            )
        if any(
            INTERACTION_UNDO in cast(ComposablePlugin, item.instance).inject
            for item in mount_order
        ):
            if not candidate and self._interaction_undo is None:
                raise RuntimeError("INTERACTION_UNDO 需要 Session owner")
            interaction_undo = (
                InteractionUndoService(self._interaction_undo.undo_latest)
                if not candidate and self._interaction_undo is not None
                else InteractionUndoService.candidate_validation()
            )
            _ = await root.context.provide(INTERACTION_UNDO, interaction_undo)

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

    async def _mount_generation_composition(
        self,
        root: CompositionRoot,
        generation: PluginGeneration,
    ) -> None:
        """用 generation 自己的正式 runtime 挂载一个 v3 插件。"""

        plugin = cast(ComposablePlugin, generation.instance)
        workspace = generation.validation_workspace or self._workspace
        for name in plugin.workspace_roots:
            _ = resolve_declared_workspace_root(workspace, name)
        for name in plugin.workspace_files:
            _ = resolve_declared_workspace_file(workspace, name)
        _ = await root._mount_module(  # pyright: ignore[reportPrivateUsage]
            plugin.apply,
            name=generation.plugin_id,
            inject=plugin.inject,
            plugin_module=plugin.module,
            runtime=PluginRuntime(
                plugin_id=generation.plugin_id,
                generation_id=generation.generation_id,
                plugin_dir=generation.code_dir,
                data_dir=generation.data_dir,
                workspace=workspace,
                config=copy.deepcopy(generation.config_projection),
                workspace_roots=plugin.workspace_roots,
                workspace_files=plugin.workspace_files,
            ),
        )


    async def _project_candidate_workspace_roots(
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
            _ = await _copy_in_thread(_copy_validation_tree, source, attempt_workspace / name, ())

    async def _project_candidate_workspace_files(
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
            if _is_sqlite_database(source):
                await _copy_in_thread(_copy_sqlite_snapshot, source, target)
            else:
                _ = await _copy_in_thread(shutil.copy2, source, target)


    def _resolve_runtime_command(
        self,
        generation: PluginGeneration,
        command: tuple[str, ...],
        cwd: str,
    ) -> tuple[str, ...]:
        """仅在实际打开目标前校验其环境，不阻挡同组件的纯读取能力。"""
        manifest = generation.static_manifest
        runtimes = () if manifest is None else manifest.python
        runtime_root = command_python_runtime(generation.code_dir, command, cwd, runtimes)
        environment = None
        if runtime_root is not None:
            if generation.archive_ref is None:
                raise RuntimeError("外部 runtime 缺少代码归档")
            record = self._archive.read_descriptor(generation.archive_ref)
            refs = cast(Mapping[str, str], record["python_environments"])
            if runtime_root not in refs:
                raise RuntimeError("插件命令缺少固定 Python 环境；请通过安装流程准备")
            runtime = next(
                item
                for item in runtimes
                if item.runtime_root == runtime_root
            )
            environment = self._python_environments.open(
                refs[runtime.runtime_root], generation.code_dir, runtime
            )
        return materialize_command(
            generation.code_dir, runtimes, command, cwd, environment_root=environment
        )

    def _record_root_failure(
        self,
        generation: PluginGeneration,
        error: BaseException,
        *,
        resource: str = "root",
        formal_effects: tuple[str, ...],
        recovery_target: RecoveryTarget | None = None,
    ) -> None:
        """Persist one executable runtime failure without releasing its owner."""

        tx_id = self._ensure_runtime_recovery_transaction(generation)
        self._reload_journal.annotate(tx_id, {
            "event": "runtime_failure_owner",
            "runtime_generation_id": generation.generation_id,
        })
        action: RecoveryActionName = (
            "retry_generation_cleanup" if resource == "runtime-snapshot-drain" else "retry_runtime_recovery"
        )
        phase: ReloadPhase = "degraded" if action == "retry_runtime_recovery" else "cleanup_failed"
        failure_resource = f"{resource}:{generation.generation_id}"
        failure_error = str(error) or type(error).__name__
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

        # 2. 运行恢复记录只引用完整 stable，不从归档路径反推安装目录。
        base_snapshot = self.current_snapshot
        base_generation = None if base_snapshot is None else base_snapshot.generations.get(generation.plugin_id)

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
            details={"base_selection_ref": None if self._validation_only else self._selection.read()},
        )
        generation.reload_tx_id = tx_id
        boot_id = os.environ.get("AKASHIC_BOOT_ID", "").strip()
        if boot_id:
            self._reload_journal.mark_runtime_owner(tx_id, boot_id)
        return tx_id

    def _record_drained_root_failure(
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
        self._reload_journal.annotate(tx_id, {
            "event": "drained_runtime_failure_owner",
            "runtime_generation_id": generation.generation_id,
        })
        action: RecoveryActionName = "retry_generation_cleanup"
        self._reload_journal.advance(
            tx_id,
            "cleanup_failed",
            error=str(error) or type(error).__name__,
            resource=f"root:{generation.generation_id}",
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
        self, generation: PluginGeneration, *, tx_id: str | None = None,
    ) -> RecoveryTarget:
        """仅用于故障诊断；恢复输入始终从唯一完整 selection 读取。"""
        publication = self._publication
        if publication is not None and publication.must_retain:
            return "candidate" if publication.candidate.generations.get(generation.plugin_id) is generation else "base"
        ref = None if self._validation_only else self._selection.read()
        if ref is not None and generation.archive_ref in self._selection_components(ref):
            return "candidate"
        return "base"


    def _record_failed_gate(
        self,
        *,
        plugin_id: str,
        revision: str,
        check_id: str,
        reason: str,
    ) -> GateResult:
        result = GateResult(
            gate_id="assembly",
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
        self._gate_results[plugin_id] = result
        return result

    def _import_plugin(self, module_name: str, plugin_root: Path) -> None:
        """只从固定制品根导入普通 plugin.py，不接受入口别名。"""
        path = plugin_root / "plugin.py"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"插件 plugin.py 必须是普通文件: {path}")
        self._fresh_importer.register(module_name, plugin_root)
        spec = self._fresh_importer.root_spec(module_name, path)
        if spec is None or spec.loader is None:
            self._fresh_importer.unregister(module_name)
            raise ImportError(f"无法加载插件文件: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # 调用者已在导入前登记 Root cleanup；异常时不能越过 Scope 卸载模块。
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if module.__file__ is None or Path(module.__file__).resolve(strict=True) != path.resolve(strict=True):
            raise RuntimeError("插件 module 文件与固定制品 plugin.py 不一致")

    def _remove_module_tree(self, module_name: str) -> None:
        self._fresh_importer.unregister(module_name)
        for imported_name in tuple(sys.modules):
            if imported_name == module_name or imported_name.startswith(
                f"{module_name}."
            ):
                _ = sys.modules.pop(imported_name, None)

    async def terminate_all(self) -> None:
        """同步撤销当前操作，再有限等待同一次全量关闭；超时仍保留任务。"""
        deadline = asyncio.get_running_loop().time() + self.POST_PUBLISH_TIMEOUT_SECONDS
        operation = self._begin_termination(deadline=deadline)
        await observe_operation(operation, deadline=deadline, cancel=False)

    def _begin_termination(self, *, deadline: float | None = None) -> ManagerOperation:
        """停止接纳并交接唯一 owner；重复调用返回同一关闭任务。"""
        self._reject_operation_lease()
        previous = self._operation
        if previous is not None and current_operation.get() is previous and not previous.task.done():
            raise RuntimeError("cleanup 不能等待其所属 PluginManager 操作")
        current = asyncio.current_task()
        if any(host.active and host.task is current for host in self._validation_hosts.values()):
            raise RuntimeError("请先退出验证 scope，再关闭其插件宿主")
        if self._stopping and previous is not None and previous.task.done():
            if not previous.task.cancelled() and previous.task.exception() is None:
                return previous
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + self.POST_PUBLISH_TIMEOUT_SECONDS
        if not self._stopping or previous is None or previous.task.done():
            # 停止标记与撤销之间没有 await；旧操作从此不能提交或重开接纳。
            self._stopping = True
            self._snapshot_store.pause_admission()
            if self._active_channel_generation is not None:
                self._active_channel_generation.close_admission()
            if previous is not None:
                previous.revoke()
            operation = ManagerOperation(deadline)
            self._operation = operation
            operation.task = asyncio.create_task(
                run_operation(operation, lambda: self._finish_termination(previous)),
                name="plugin-manager-terminate",
            )
            operation.task.add_done_callback(self._operation_finished)
        else:
            operation = previous
        return operation

    async def _finish_termination(self, previous: ManagerOperation | None) -> None:
        """关闭任务持有原操作，等实际工作结束后才接管资源。"""
        if previous is not None and not previous.task.done():
            # 不再次 cancel；此前的关闭或不可取消线程仍沿原调用栈结束。
            await asyncio.wait((previous.task,))
        await self._terminate_all()

    async def _terminate_all(self) -> None:
        """完成快照、插件生命周期和作用域资源的全量关闭。"""

        _ = self._snapshot_store.pause_admission()
        if self._endpoint_quiescer is not None:
            await self._endpoint_quiescer()
        # 1. 当前操作已退出；关闭任务独占剩余实际资源。
        # 验证必须先归还候选租约，正式 generation 才能退出。
        current = asyncio.current_task()
        validations = tuple(self._validation_hosts.values())
        if any(host.active and host.task is current for host in validations):
            raise RuntimeError("请先退出验证 scope，再关闭其插件宿主")
        running = tuple(host.task for host in validations if host.active and not host.task.done())
        for task in running:
            _ = task.cancel()
        if running:
            await asyncio.wait(running)
        for identity in tuple(self._validation_hosts):
            self._validation_hosts[identity].active = False
            await self._retry_validation_cleanup(identity)

        # 2. 停止接纳和插件生产者，再关闭它们仍需排空的 Task 与资源。
        snapshot = self._snapshot_store.pause_admission()
        externally_cancelled = False
        if snapshot is not None:
            _, externally_cancelled = await _complete_critical(
                self._stop_runtime_snapshot(snapshot)
            )
        _, cancelled = await _complete_critical(self._plugin_tasks.close())
        externally_cancelled = externally_cancelled or cancelled
        _, cancelled = await _complete_critical(self._plugin_processes.close())
        externally_cancelled = externally_cancelled or cancelled
        channel_runtime = self._active_channel_generation
        if channel_runtime is not None:
            _ = self._snapshot_store.pause_admission()
            channel_runtime.close_admission()
            try:
                _, cancelled = await _complete_critical(channel_runtime.stop())
                externally_cancelled = externally_cancelled or cancelled
            except BaseException as error:
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
        # 未交接 Root 先释放；SnapshotStore 和 generation 随后才能移除其依赖。
        for root in tuple(self._building_roots):
            await self._close_building_root(root)
        pending = self._snapshot_store.pending_transaction
        if pending is not None:
            failure = self._channel_generation_host.failure(pending.candidate.snapshot_id)
            if failure is not None:
                for owner in cast(tuple[ChannelCleanupTombstone, ...], failure):
                    await self._channel_generation_host.retry_generation_cleanup(owner.binding_token)
            await self._snapshot_store.abort(pending, reopen_previous=False)
        # 2. 关闭当前 generation admission，再完成快照回收。
        for generation in self._active_generations.values():
            self._retire_generation(generation)
        _, snapshot_cancelled = await _complete_critical(self._snapshot_store.close())
        externally_cancelled = externally_cancelled or snapshot_cancelled
        self._ready_candidate = None
        for plugin_id in tuple(self._prepared_generations):
            _, cancelled = await _complete_critical(self._discard_prepared(plugin_id))
            externally_cancelled = externally_cancelled or cancelled
        # 3. 快照之外的失败 generation 仍由排空集合持有，显式关闭时重试。
        for tracked in tuple(self._draining_generations.values()):
            for generation in tuple(tracked):
                _, cancelled = await _complete_critical(
                    self._dispose_generation(generation, state="retired")
                )
                externally_cancelled = externally_cancelled or cancelled
        # 4. 导入前的 Scope 也由前面的 building Root 关闭，不绕过 Root 扫尾。
        for alias in self._stable_aliases.values():
            self._remove_module_tree(alias)
        self._stable_aliases.clear()
        if self._owns_control_frames:
            self._control_frames.close()
        if externally_cancelled:
            raise asyncio.CancelledError


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


def _installed_generation_is_candidate(generation: PluginGeneration) -> bool:
    """Return whether this installed generation is the explicit latest pointer."""

    if generation.source_type != "installed":
        return False
    plugin_dir = generation.plugin_dir
    return _installed_candidate_base_from_root(plugin_dir) is not None


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


def _installed_artifact_base_from_root(plugin_dir: Path) -> Path:
    return (
        plugin_dir.parent.parent
        if plugin_dir.parent.name == ".artifacts"
        else plugin_dir.parent
    )


def _remove_validation_data_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _candidate_data_exclude_paths(
    manifest: StaticPluginManifest | None,
) -> tuple[str, ...]:
    """验证副本只排除插件显式声明的数据路径。"""
    if manifest is None:
        return ()
    return tuple(sorted(manifest.exclude_data_paths))


def _validate_candidate_formal_snapshot_identity(
    *,
    candidate: RuntimeSnapshot,
    formal: RuntimeSnapshot,
) -> None:
    """比较固定制品与配置输入；运行 snapshot 和 generation 身份必须独立。"""

    candidate_inputs = tuple(
        (key, PluginManager._generation_archive_ref(item))
        for key, item in sorted(candidate.generations.items())
    )
    formal_inputs = tuple(
        (key, PluginManager._generation_archive_ref(item))
        for key, item in sorted(formal.generations.items())
    )
    if candidate_inputs != formal_inputs:
        raise RuntimeError("候选与正式 Root 的固定归档输入不一致")


async def _copy_in_thread(copy_files: Callable[..., U], *args: Any, **kwargs: Any) -> U:
    """复制完成后才传播取消，避免清理目录时后台线程仍在写入。"""
    result, cancelled = await _complete_critical(asyncio.to_thread(copy_files, *args, **kwargs))
    if cancelled:
        raise asyncio.CancelledError
    return result


def _copy_validation_tree(
    source: Path,
    target: Path,
    exclude_paths: tuple[str, ...],
    *, keep_existing: bool = False,
) -> tuple[str, ...]:
    """复制已核对的数据树；历史补全可保留已经固定的候选文件。"""
    if ".plugin-credentials" in source.parts:
        raise RuntimeError("候选不能复制正式私有凭据目录")
    excluded = tuple(PurePosixPath(item).as_posix() for item in exclude_paths)

    # 1. A new plugin has no formal bytes; candidate starts from an empty tree.
    if not source.exists():
        target.mkdir(parents=True, exist_ok=keep_existing)
        return ()
    if source.parent.name == "plugin-data":
        check_config_format(source)
    source_root = source.resolve(strict=True)

    # 2. Candidate data must never retain an edge back into formal storage.
    for directory, dirnames, filenames in os.walk(source_root, followlinks=False):
        root = Path(directory)
        if ".plugin-credentials" in dirnames:
            raise RuntimeError("候选数据不能包含正式私有凭据目录")
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
                raise RuntimeError(f"candidate 数据不允许复制符号链接: {path}")
            if not path.is_file() and not path.is_dir():
                raise RuntimeError(f"candidate 数据只能复制普通文件或目录: {path}")

    # 3. Copy SQLite through its snapshot API; never race WAL/SHM companion files.
    target.mkdir(parents=True, exist_ok=keep_existing)
    for directory, dirnames, filenames in os.walk(source_root, followlinks=False):
        root = Path(directory)
        if ".plugin-credentials" in dirnames:
            raise RuntimeError("候选数据不能包含正式私有凭据目录")
        relative_dir = root.relative_to(source_root)
        dirnames[:] = [
            name
            for name in dirnames
            if not _candidate_data_path_is_excluded(relative_dir / name, excluded)
        ]
        for name in dirnames:
            relative = relative_dir / name
            (target / relative).mkdir(exist_ok=keep_existing)
        retained = [name for name in filenames
                    if not _candidate_data_path_is_excluded(relative_dir / name, excluded)]
        databases = {name for name in retained if _is_sqlite_database(root / name)}
        sidecars = {name + suffix for name in databases for suffix in ("-wal", "-shm")}
        for name in retained:
            if name in sidecars:
                continue
            relative = relative_dir / name
            source_file = root / name
            target_file = target / relative
            if keep_existing and target_file.exists():
                continue
            if name in databases:
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


_SQLITE_BACKUP_LOCK_TIMEOUT_SECONDS = 5.0


def _copy_sqlite_snapshot(source: Path, target: Path) -> None:
    """Copy one transactionally consistent SQLite snapshot."""

    deadline = time.monotonic() + _SQLITE_BACKUP_LOCK_TIMEOUT_SECONDS

    def check_progress(status: int, remaining: int, total: int) -> None:
        # Chromium 等外部进程可持有独占锁；失败由候选 owner 撤销临时副本。
        if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) and time.monotonic() >= deadline:
            raise TimeoutError(f"候选 SQLite 备份等待锁超时: {source}")

    reader = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.0)
    writer = sqlite3.connect(target, timeout=0.0)
    try:
        reader.backup(writer, pages=256, progress=check_progress, sleep=0.05)
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


def _require_plugin_path(plugin_dir: Path, path: Path, label: str) -> None:
    try:
        _ = path.relative_to(plugin_dir)
    except ValueError as error:
        raise RuntimeError(f"插件 {label} 越界: {path}") from error


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
        ENVIRONMENT_FILE,
    }
    for current, directories, filenames in os.walk(plugin_dir, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in excluded)
        current_path = Path(current)
        for name in [*directories, *sorted(filenames)]:
            if name in excluded:
                continue
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
        ENVIRONMENT_FILE,
    }
    for current, directories, filenames in os.walk(plugin_dir, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in excluded)
        current_path = Path(current)
        for name in [*directories, *sorted(filenames)]:
            if name in excluded:
                continue
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


def _log_candidate_status(result: dict[str, object]) -> None:
    logger.info(
        "plugin_candidate_status plugin=%s preparation=%s active=%s prepared=%s "
        "revision=%s",
        result["plugin_id"],
        result["preparation_state"],
        result["active_generation"],
        result["prepared_generation"] or "-",
        str(result["candidate_revision"])[:12],
    )
    logger.debug(
        "plugin_candidate_status_detail %s",
        json.dumps(result, ensure_ascii=False, sort_keys=True),
    )
