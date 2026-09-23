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
import sys
from dataclasses import asdict, dataclass
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import Context as TaskContext
from pathlib import Path
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from typing import Any, Literal, TypeVar, cast
from uuid import uuid4


from agent.plugins.archive import PluginArchive, decode_config
from agent.plugins._operation import (
    ManagerOperation, OperationBusyError, OperationTimeoutError,
    complete_critical as _complete_critical, current_operation,
    observe_operation, revoke_on_cancel, run_operation,
)
from agent.plugins.python_environment import ENVIRONMENT_FILE, PythonEnvironments
from agent.plugins.validation import ValidationHost
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES, PluginUpdates, UpdateStatus
from session.artifact_store import ArtifactStore
from agent.plugin_composition.config_input import CONFIG_INPUT, load_config
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
from session.log import MessageCatalog, MessageLog, MessagePage, read_persisted_messages
from session.message import Message
from session.embedding_store import MessageEmbeddings
from agent.plugin_composition.context import Context, Fiber
from agent.restart import RESTART_GATE, RestartGate
from agent.control.frame_book import CONTROL_FRAMES, FrameBook

from agent.plugin_composition import (
    COMMANDS,
    INTERACTION_UNDO,
    CompositionError,
    TIMERS,
    CompositionRoot,
    FiberState,
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
from agent.plugin_composition.host import HOST_INFO, HostInfo
from agent.plugin_composition.ui import DASHBOARD_ROUTES
from agent.plugin_composition.channel_io import (
    INPUT_CUSTODY, CHANNEL_IDENTITY, CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ,
    InputCustody, ChannelIdentity, ChannelAttachmentImport, ChannelAttachmentRead,
    unavailable, unavailable_input_custody,
)
from agent.plugin_composition.processes import PROCESSES, PluginProcesses
from agent.plugin_composition.execution import EXECUTION, WORKLOAD_CONTROLLER
from agent.host_bridge.plugin_execution import CodeOwner, ExecutionAccess, ControllerAccess, cleanup_workloads_for_boot
from agent.plugin_composition.model import (
    resolve_declared_workspace_file,
    resolve_declared_workspace_root,
)
from agent.control.timer import AsyncioOneShotTimer
from agent.plugins.composable import ComposablePlugin
from agent.plugins.interaction_undo import InteractionUndoCoordinator
from agent.plugins.channel_credentials import CoreProviderClientFactory

from agent.plugins.manifest import (
    ensure_workspace_plugin_data_dir,
    load_plugin_manifest,
    plugins_root,
    validate_workspace_plugin_data_path,
)
from agent.plugins.input_preparation import (
    PLUGIN_ARCHIVE_BINDING_API,
    _resolve_plugin_data_dir,
    _resolve_plugin_id,
    _source_revision,
    prepare_plugin_input,
)
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from session.identities import ChannelIdentities, ChannelIdentityWriteReceipt
from agent.plugins.artifacts import (
    ArtifactSelector,
    read_pointer,
    read_pointers,
    resolve_pointer,
)
from agent.plugins.source_resolver import (
    PluginSourceFailure,
    scan_plugin_sources,
)
from agent.plugins.selection import PluginSelection, SelectionConflictError, SelectionWriteError
from agent.plugins.scope import CleanupFailure, PluginScope
from agent.plugins.generation import PluginGeneration
from agent.plugins.importer import FreshPluginImporter
from agent.plugins.install import (
    PluginInstallResult,
    _split_installed_plugin_id,
    finalize_uninstall_plugin,
    install_git_plugin,
    set_installed_plugin_enabled,
)
from agent.plugins.static_manifest import (
    PluginSourceCompileError,
    PluginSourceContentError,
    load_static_plugin_manifest,
    source_error_details,
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
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from bus.event_bus import EventBus

logger = logging.getLogger(__name__)
U = TypeVar("U")


@asynccontextmanager
async def _snapshot_lease_scope(lease: RuntimeSnapshotLease) -> AsyncGenerator[None]:
    """Bind a legacy snapshot lease to this Task and release its exact claim."""
    token = bind_runtime_snapshot(lease)
    try:
        yield
    finally:
        reset_runtime_snapshot(token)
        await lease.aclose()


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


def _reject_retired_owner_recovery(action: ReloadRecoveryAction) -> None:
    """拒绝仍依赖已删除 owner 的旧恢复记录，不伪造恢复完成。"""

    resource = action.failure_resource or ""
    retired = {"activity-publication", "plugin-skill-projection"}.intersection({
        item.strip() for item in resource.split(",") if item.strip()
    })
    retired.update(
        item.strip() for item in resource.split(",")
        if item.strip().startswith(("channel-binding:", "channel-generation:"))
        or item.strip() == "channel-publication"
    )
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


class PluginManager:
    # 提交预算覆盖候选准备、整组重建与发布；生产组合规模下挂载数十个
    # 归档插件远超秒级，预算只用于截断真正挂起的提交，不能按交互延迟设定。
    POST_PUBLISH_TIMEOUT_SECONDS = 300.0
    # 冷启动还须归档全部安装输入并挂载完整组合，规模随安装数增长。
    BOOT_COMMIT_TIMEOUT_SECONDS = 7200.0

    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: EventBus,
        workspace: Path,
        session_manager: Any = None,
        message_log: MessageLog | None = None,
        channel_identities: ChannelIdentities | None = None,
        input_custody: InputCustody | None = None,
        installed_cache_root: Path | None = None,
        channel_attachment_store: ChannelAttachmentArtifactStore | None = None,
        disabled_builtin_plugins: frozenset[str] = frozenset(),
        source_failures: tuple[PluginSourceFailure, ...] = (),
        workload_controller: WorkloadController | None = None,
        restart_gate: RestartGate | None = None,
        control_frames: FrameBook | None = None,
    ) -> None:
        self._dirs = plugin_dirs
        self._workspace = workspace
        self._archive = PluginArchive(workspace / "runtime" / "plugin-archives")
        self._selection = PluginSelection(workspace)
        self._publication: SnapshotTransaction | None = None
        self._python_environments = PythonEnvironments(workspace)
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
        self._source_failures: dict[str, PluginSourceFailure] = {
            _source_failure_key(failure): failure
            for failure in source_failures
        }
        self._dashboard_routes: tuple[object, ...] | None = None
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
        # The formal local runtime has one Root and one live generation owner map.
        self._live_root: CompositionRoot | None = None
        self._active_generations: dict[str, PluginGeneration] = {}
        self._live_execution_access: ExecutionAccess | None = None
        self._live_credentials: CredentialClients | None = None
        self._ready_candidate: _ReadyPluginCandidate | None = None
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
        self._reload_journal = ReloadJournal(workspace)
        self._channel_identities = channel_identities
        self._input_custody = input_custody
        self._channel_attachment_store = channel_attachment_store
        self._drain_transactions: dict[str, str] = {}
        self._drained_before_commit: set[str] = set()

    def _reject_operation_lease(self, *, allow_stable_lease: bool = False) -> None:
        """在创建任务前拒绝等待同一 Manager 租约造成的自锁。"""
        from agent.plugins.snapshot import get_current_runtime_lease
        lease = get_current_runtime_lease()
        if lease is not None and lease._store is self._snapshot_store:
            if (allow_stable_lease and lease.active
                    and lease.snapshot is self.current_snapshot
                    and lease.snapshot.state == "committed"
                    and lease.snapshot.accepting_leases):
                return
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
        operation = self._operation if self._operation is not None and self._operation.task is task else None
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("插件操作结束但失败: %s", error)
                if operation is not None and operation.accepted is not None and not operation.accepted.done():
                    operation.accepted.set_exception(error)
            elif operation is not None and operation.accepted is not None and not operation.accepted.done():
                operation.accepted.set_exception(RuntimeError("插件更新操作结束但未到达 accepted"))
        elif operation is not None and operation.accepted is not None and not operation.accepted.done():
            operation.accepted.set_exception(RuntimeError("插件更新操作被取消，未到达 accepted"))
        if operation is not None and operation.accepted is not None and operation.accepted.done():
            # Background owner futures must be consumed even when the caller disconnected.
            if not operation.accepted.cancelled():
                _ = operation.accepted.exception()
        self._notify_updates()

    def _revoke_operation(self, operation: ManagerOperation) -> None:
        """截止时结束 accepted 等待，再撤销迟到提交许可。"""
        operation.settle_accepted(OperationTimeoutError(operation))
        operation.revoke()

    def _start_operation(
        self, work: Callable[[], Awaitable[U]], *, background: bool = False,
        commit_timeout: float | None = None, update_id: str | None = None,
        accepted: asyncio.Future[Any] | None = None,
    ) -> ManagerOperation:
        """Start one finite operation whose task retains cleanup ownership."""
        self._require_operation_idle()
        if commit_timeout is None:
            commit_timeout = self.POST_PUBLISH_TIMEOUT_SECONDS
        loop = asyncio.get_running_loop()
        operation = ManagerOperation(
            loop.time() + commit_timeout, update_id=update_id, accepted=accepted,
        )
        self._operation = operation

        async def admitted_work() -> U:
            return await work()

        operation.task = asyncio.create_task(
            run_operation(operation, admitted_work), name="plugin-manager-operation",
            context=TaskContext() if background else None,
        )
        operation.task.add_done_callback(self._operation_finished)
        timer = loop.call_at(operation.deadline, self._revoke_operation, operation)
        operation.task.add_done_callback(lambda _: timer.cancel())
        return operation

    async def _run_operation(
        self, work: Callable[[], Awaitable[U]], *, allow_stable_lease: bool = False,
        commit_timeout: float | None = None,
    ) -> U:
        """公开入口有限观察同一任务；退出观察不释放仍在工作的 owner。"""
        self._reject_operation_lease(allow_stable_lease=allow_stable_lease)
        from agent.plugins.snapshot import get_current_runtime_lease
        lease = get_current_runtime_lease()
        if lease is not None and lease._store is self._snapshot_store:
            async def scoped_work() -> U:
                # 任务取得自己的真实租约；调用者超时退出不能释放仍在工作的 scope。
                self._check_operation_commit()
                async with _snapshot_lease_scope(lease.fork()):
                    return await work()

            operation = self._start_operation(scoped_work, commit_timeout=commit_timeout)
        else:
            operation = self._start_operation(work, commit_timeout=commit_timeout)
        try:
            return cast(U, await observe_operation(operation, deadline=operation.deadline))
        finally:
            if operation.revoked:
                self._revoke_operation(operation)

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
                async with _snapshot_lease_scope(self._snapshot_store.lease(snapshot.snapshot_id)):
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
        await self._run_operation(
            self._start_runtime, commit_timeout=self.BOOT_COMMIT_TIMEOUT_SECONDS,
        )

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
            async with _snapshot_lease_scope(lease.fork()):
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
    def cleanup_failures(self) -> list[CleanupFailure]:
        return list(self._cleanup_failures)

    @property
    def live_root(self) -> CompositionRoot | None:
        """Expose the existing formal Root for read-only recovery routing."""

        return self._live_root

    def generation(self, plugin_id: str) -> PluginGeneration | None:
        return self._active_generations.get(plugin_id)

    def prepared_generation(self, plugin_id: str) -> PluginGeneration | None:
        raise RuntimeError("候选 generation registry 已停用；请读取 live generation")

    def configure_dashboard_routes(self, routes: tuple[object, ...]) -> None:
        """Store the one real host route tuple before the live Root exists."""

        if not isinstance(routes, tuple):
            raise TypeError("Dashboard host routes 必须是 tuple")
        if self._live_root is not None:
            raise RuntimeError("Dashboard host routes 必须在 live Root 前配置")
        if self._dashboard_routes is not None:
            raise RuntimeError("Dashboard host routes 不能重复配置")
        self._dashboard_routes = routes

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
        mods, failures = self._discover_modules(
            installed_selector="latest", record_source_failures=False,
        )
        for failure in failures:
            digest.update(failure.source_type.encode())
            digest.update(str(failure.source_root).encode())
            digest.update(_source_metadata_revision(failure.source_root))
        for mod in mods:
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

    # 扫描所有 plugin_dirs，返回可加载的插件描述列表
    def discover(
        self,
        *,
        installed_selector: ArtifactSelector = "stable",
        record_source_failures: bool = True,
    ) -> list[dict[str, str]]:
        mods, _failures = self._discover_modules(
            installed_selector=installed_selector,
            record_source_failures=record_source_failures,
        )
        return mods

    def _discover_modules(
        self,
        *,
        installed_selector: ArtifactSelector,
        record_source_failures: bool,
    ) -> tuple[list[dict[str, str]], tuple[PluginSourceFailure, ...]]:
        mods: list[dict[str, str]] = []
        seen_names: set[str] = set()
        scan = scan_plugin_sources(
            self._dirs,
            installed_cache_root=self._installed_cache_root,
            installed_selector=installed_selector,
        )
        if record_source_failures:
            self._remember_source_failures(scan.failures)
        for source in scan.sources:
            name = source.plugin_name
            if not name:
                raise RuntimeError(
                    f"source scan 未提供已验证 plugin id: {source.plugin_root}"
                )
            if (
                source.source_type == "builtin"
                and name in self._disabled_builtin_plugins
            ):
                continue
            if name in seen_names and source.source_type == "builtin":
                logger.warning("插件名重复，跳过: %s (%s)", name, source.plugin_root)
                continue
            seen_names.add(name)
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
                    "marketplace": source.marketplace,
                    "source_type": source.source_type,
                }
            )
        return mods, scan.failures

    def _remember_source_failures(
        self, failures: tuple[PluginSourceFailure, ...],
    ) -> None:
        for failure in failures:
            self._source_failures[_source_failure_key(failure)] = failure

    def _clear_source_failure(self, mod: Mapping[str, str]) -> None:
        source_root = Path(mod["plugin_root"]).resolve(strict=False)
        source_type = cast(Literal["builtin", "installed"], mod["source_type"])
        self._source_failures.pop(
            _source_failure_key_for_root(source_root, source_type), None,
        )

    def _source_failure_for_error(
        self,
        mod: Mapping[str, str],
        error: BaseException,
        *,
        phase: str,
        plugin_id: str | None,
    ) -> PluginSourceFailure:
        if isinstance(error, PluginSourceContentError):
            error_type, error_text = source_error_details(error)
        else:
            error_type, error_text = type(error).__name__, str(error) or type(error).__name__
        return PluginSourceFailure(
            source_root=Path(mod["plugin_root"]).resolve(strict=False),
            source_type=cast(Literal["builtin", "installed"], mod["source_type"]),
            phase=phase,
            error_type=error_type,
            error_text=error_text,
            plugin_id=plugin_id,
        )

    async def _load_one_with_source_diagnostics(
        self,
        mod: dict[str, str],
        *,
        activate: bool = True,
        stage_stable: bool = False,
    ) -> PluginGeneration | None:
        try:
            generation = await self._load_one(
                mod, activate=activate, stage_stable=stage_stable,
            )
        except PluginSourceCompileError as error:
            self._remember_source_failures((self._source_failure_for_error(
                mod, error, phase="compile", plugin_id=_resolve_plugin_id(mod),
            ),))
            return None
        except PluginSourceContentError as error:
            self._remember_source_failures((self._source_failure_for_error(
                mod, error, phase="identity", plugin_id=None,
            ),))
            return None
        if generation is not None:
            self._clear_source_failure(mod)
        return generation

    def _source_root_hint(
        self, plugin_id: str, active: PluginGeneration | None,
    ) -> Path | None:
        if active is None:
            return None
        source_root = active.plugin_dir.resolve(strict=False)
        code_root = active.code_dir.resolve(strict=False)
        if source_root == code_root:
            return None
        return source_root

    def _remember_missing_source(
        self, plugin_id: str, active: PluginGeneration | None,
    ) -> None:
        source_root = self._source_root_hint(plugin_id, active)
        if source_root is None:
            return
        source_type = (
            active.source_type if active is not None else
            ("installed" if "@" in plugin_id else "builtin")
        )
        failure = PluginSourceFailure(
            source_root=source_root,
            source_type=cast(Literal["builtin", "installed"], source_type),
            phase="source",
            error_type="SourceUnavailable",
            error_text="插件源码目录暂时不可用；selection 未改变",
            plugin_id=plugin_id,
        )
        key = _source_failure_key(failure)
        if key not in self._source_failures:
            self._source_failures[key] = failure

    async def load_all(self) -> None:
        await self._run_operation(
            self._load_all, commit_timeout=self.BOOT_COMMIT_TIMEOUT_SECONDS,
        )

    async def _load_all(self) -> None:
        """从唯一完整选择启动；null 只允许首次固定安装输入。"""
        selection_ref = self._selection.read()
        if self.current_snapshot is not None or self._live_root is not None:
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
            components = self._selection_components(selection_ref)
            await self._load_live_initial(components, expected_ref=selection_ref)
            return
        enabled = load_plugin_manifest(self.installed_plugins_home)
        selected = tuple(
            mod for mod in self.discover(installed_selector="stable")
            if enabled.get(_resolve_plugin_id(mod), True)
        )
        inputs: list[PluginGeneration] = []
        try:
            for mod in selected:
                generation = await self._load_one_with_source_diagnostics(
                    mod, activate=False, stage_stable=True,
                )
                if generation is None:
                    if _source_failure_key_for_mod(mod) in self._source_failures:
                        continue
                    raise RuntimeError(f"完整插件组合加载失败: {_resolve_plugin_id(mod)}")
                inputs.append(generation)
            components = tuple(
                self._generation_archive_ref(item)
                for item in sorted(inputs, key=lambda item: item.plugin_id)
            )
            await self._load_live_initial(components, expected_ref=None, prepared=tuple(inputs))
        except BaseException:
            for generation in tuple(inputs):
                if generation.state != "active":
                    try:
                        await self._dispose_generation(generation, state="discarded")
                    except BaseException:
                        logger.exception("首次启动插件输入清理失败: %s", generation.plugin_id)
            raise

    def _selection_components(self, ref: str) -> tuple[str, ...]:
        return cast(tuple[str, ...], self._archive.read_descriptor(ref)["components"])

    def _selection_for_plugin(
        self,
        expected_ref: str | None,
        plugin_id: str,
        replacement: str | None,
    ) -> tuple[str, ...]:
        """Replace one persisted input while preserving every other input ref."""
        components = () if expected_ref is None else self._selection_components(expected_ref)
        result: list[str] = []
        replaced = False
        for ref in components:
            record = self._archive.read_descriptor(ref)
            current = record.get("plugin_id")
            if current != plugin_id:
                result.append(ref)
            elif replacement is not None and not replaced:
                result.append(replacement)
                replaced = True
        if replacement is not None and not replaced:
            result.append(replacement)
        return tuple(result)

    def _generation_for_context(self, context: object) -> PluginGeneration:
        """Resolve a real current Context by its Root and runtime generation."""
        root = self._live_root
        if root is None:
            raise RuntimeError("正式 live Root 尚未建立")
        if not isinstance(context, type(root.context)):
            raise TypeError("generation lookup 需要同一 CompositionRoot 的 Context")
        if context.root_instance_token is not root.instance_token:
            raise ValueError("Context 不属于当前 live Root")
        runtime = context.runtime
        generation = self._active_generations.get(runtime.plugin_id)
        if generation is None or generation.generation_id != runtime.generation_id:
            raise ValueError("Context 与当前 generation 不匹配")
        if generation.code_dir.resolve() != runtime.plugin_dir.resolve():
            raise ValueError("Context 与当前固定代码制品不匹配")
        # The Context may be a child provider/contributor.  Identity is the
        # Root's registered Context plus runtime tuple, not the top-level Fiber.
        if not any(
            fiber.context is context
            and fiber.runtime is not None
            and fiber.runtime.plugin_id == runtime.plugin_id
            and fiber.runtime.generation_id == runtime.generation_id
            for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
        ):
            raise ValueError("Context 不属于当前 Root 登记的 generation Context")
        return generation

    async def _load_live_initial(
        self,
        components: tuple[str, ...],
        *,
        expected_ref: str | None,
        prepared: tuple[PluginGeneration, ...] = (),
    ) -> None:
        """Build the single formal Root without compiling a snapshot or fallback Root."""
        root = CompositionRoot("plugins-live:" + secrets.token_hex(16))
        generations: tuple[PluginGeneration, ...] = prepared
        cleanup_generations: tuple[PluginGeneration, ...] = generations
        runnable: tuple[PluginGeneration, ...] = ()
        self._live_root = root
        self._building_roots[root] = ()
        try:
            if not prepared:
                loaded = self._archived_generations(
                    components, root, workspace=self._workspace, sources={},
                )
                generations = tuple(loaded.values())
                cleanup_generations = generations
            else:
                for generation in generations:
                    current = self._active_generations.get(generation.plugin_id)
                    if current is not None and current is not generation:
                        raise RuntimeError(
                            f"live generation owner 已存在: {generation.plugin_id}"
                        )
                    self._active_generations[generation.plugin_id] = generation
                self._building_roots[root] = generations
            # Selection is the durable input boundary.  No import or apply is
            # allowed before this commit when booting a fresh selection.
            if expected_ref is None:
                operation = self._check_operation_commit()
                committed = self._selection.commit(components, expected_ref=None)
                operation.committed = committed
            runnable_items: list[PluginGeneration] = []
            for generation in generations:
                try:
                    await self._load_live_generation(generation)
                except Exception:
                    await self._retain_pre_fiber_failure(generation)
                    continue
                current = self._active_generations.get(generation.plugin_id)
                if current is not None and current is not generation:
                    raise RuntimeError(f"live generation owner 已存在: {generation.plugin_id}")
                self._active_generations[generation.plugin_id] = generation
                runnable_items.append(generation)
                ensure_workspace_plugin_data_dir(generation.data_dir, self._workspace)
            runnable = tuple(runnable_items)
            self._building_roots[root] = runnable
            await self._provide_composition_services(root, runnable, candidate=False)
            self._check_live_host_dependencies(runnable)
            for generation in runnable:
                await self._mount_generation_composition(root, generation)
            # Receipt readiness is diagnostic; each Fiber owns its local failure state.
            for generation in runnable:
                generation.state = "active"
            self._building_roots.pop(root, None)
        except BaseException:
            cleanup_errors: list[BaseException] = []
            tracked = self._building_roots.get(root, ())
            cleanup: list[PluginGeneration] = list(cleanup_generations)
            for generation in tracked:
                if not any(item is generation for item in cleanup):
                    cleanup.append(generation)
            for generation in reversed(tuple(cleanup)):
                if generation.state not in {"discarded", "retired"}:
                    try:
                        await self._dispose_generation(generation, state="discarded")
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
            try:
                await root.dispose()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            else:
                self._building_roots.pop(root, None)
                if self._live_root is root:
                    self._live_root = None
            if cleanup_errors:
                raise BaseExceptionGroup("live Root 失败且清理未完成", cleanup_errors)
            raise

    async def _load_selected_generation(self, generation: PluginGeneration) -> None:
        """Load one already-selected archived module before its real mount."""
        if generation.fiber is not None:
            raise RuntimeError("generation 已有 Fiber，不能重新执行 pre-Fiber load")
        if generation.instance is None:
            self._import_plugin(generation.module_path, generation.code_dir)
            manifest = generation.static_manifest
            if manifest is None:
                manifest = load_static_plugin_manifest(generation.code_dir)
                generation.static_manifest = manifest
            generation.instance = ComposablePlugin.from_module(
                sys.modules[generation.module_path], manifest,
            )
            if generation.instance.name != generation.plugin_id.split("@", 1)[0]:
                raise ValueError("归档插件身份不一致")
        generation.state = "loading"

    async def _load_live_generation(self, generation: PluginGeneration) -> None:
        """Record only live pre-Fiber load failures before normal retention."""
        try:
            await self._load_selected_generation(generation)
        except Exception as error:
            if generation.fiber is not None:
                raise RuntimeError(
                    "pre-Fiber load 失败时 generation 已有 Fiber"
                ) from error
            if generation.load_error is not None:
                raise RuntimeError(
                    "generation load_error 不得被第二次覆盖"
                ) from error
            generation.load_error = error
            generation.state = "failed"
            raise

    async def _retain_pre_fiber_failure(self, generation: PluginGeneration) -> None:
        """Retain one selected pre-Fiber failure after its physical cleanup attempt."""
        current = self._active_generations.get(generation.plugin_id)
        selection_ref = self._selection.read()
        if (
            generation.load_error is None
            or generation.fiber is not None
            or generation.state != "failed"
            or (current is not None and current is not generation)
            or generation.archive_ref is None
            or selection_ref is None
            or generation.archive_ref not in self._selection_components(selection_ref)
        ):
            raise RuntimeError("不是可 retained 的 pre-Fiber generation failure")
        if current is None:
            self._active_generations[generation.plugin_id] = generation
        try:
            await self._dispose_generation(
                generation,
                state="failed",
                retain_selected_failed=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "selected pre-Fiber generation cleanup 未完成: %s",
                generation.plugin_id,
            )

    def _check_live_host_dependencies(
        self, generations: tuple[PluginGeneration, ...],
    ) -> None:
        """Report known host gaps after the selected code has been loaded."""
        root = self._live_root
        if root is None:
            raise RuntimeError("正式 live Root 尚未建立")
        host_keys: set[ServiceKey[object]] = {
            HOST_INFO, DASHBOARD_ROUTES, INPUT_CUSTODY, CHANNEL_IDENTITY,
            CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ,
            EXECUTION, WORKLOAD_CONTROLLER, RUNTIME_CATALOG, CREDENTIALS,
            PLUGIN_UPDATES, RESTART_GATE, CONTROL_FRAMES,
            MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, MESSAGE_WRITERS,
            OWNER_STATE, SESSION_ADMISSION, BINDINGS, TASKS, PROCESSES,
            ARTIFACT_READ, ARTIFACT_IMPORT, TIMERS, INTERACTION_UNDO,
            ServiceKey[object]("core.message_display.v1"),
            ServiceKey[object]("core.mobile_ui.v1"),
        }
        for generation in generations:
            plugin = cast(ComposablePlugin, generation.instance)
            for key in plugin.inject:
                # Unknown keys may be provided by another plugin Fiber; let the kernel
                # report PENDING. Only known host-owned capabilities are a migration gate.
                if key not in host_keys or root.context.get(key) is not None:
                    continue
                raise RuntimeError(
                    f"宿主能力尚未迁移，阻止启用 {generation.plugin_id}: {key.name}"
                )

    def _generation_fibers(self, generation: PluginGeneration) -> tuple[Fiber, ...]:
        """Return the live Root Fibers owned by one exact generation."""
        root = self._live_root
        if root is None:
            return ()
        return tuple(
            fiber
            for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.runtime is not None
            and fiber.runtime.plugin_id == generation.plugin_id
            and fiber.runtime.generation_id == generation.generation_id
        )

    def _capture_local_readiness(
        self, generations: tuple[PluginGeneration, ...],
    ) -> tuple[Fiber, ...]:
        """Capture actual downstream Fibers before an old provider edge disappears."""
        root = self._live_root
        if root is None:
            return ()
        owned = {
            fiber
            for generation in generations
            for fiber in self._generation_fibers(generation)
        }
        affected = set(owned)
        changed = True
        while changed:
            changed = False
            for candidate in tuple(root._fibers.values()):  # pyright: ignore[reportPrivateUsage]
                if candidate in affected or candidate.state == FiberState.DISPOSED:
                    continue
                owned_child = candidate.parent in affected
                provider_consumer = any(
                    provider.owner in affected
                    for provider in candidate.dependency_store.values()
                )
                if owned_child or provider_consumer:
                    affected.add(candidate)
                    changed = True
        return tuple(affected - owned)

    def _current_local_consumers(
        self, generation: PluginGeneration,
    ) -> tuple[Fiber, ...]:
        """Find current hard consumers from provider identity and declared keys."""
        root = self._live_root
        if root is None:
            return ()
        target = set(self._generation_fibers(generation))
        owners = set(target)
        consumers: set[Fiber] = set()
        changed = True
        while changed:
            changed = False
            for candidate in tuple(root._fibers.values()):  # pyright: ignore[reportPrivateUsage]
                if candidate in owners or candidate.state == FiberState.DISPOSED:
                    continue
                owned_child = candidate.parent in owners
                provider_consumer = any(
                    (provider := root._providers.get(key)) is not None
                    and provider.owner in owners
                    for key in candidate.dependencies
                )
                if owned_child or provider_consumer:
                    owners.add(candidate)
                    consumers.add(candidate)
                    changed = True
        return tuple(consumers)

    def _require_local_generation_ready(
        self,
        generation: PluginGeneration,
        *,
        affected: tuple[Fiber, ...] = (),
    ) -> None:
        """Check the target and captured downstream Fibers without a whole-Root gate."""
        root = self._live_root
        fibers = self._generation_fibers(generation)
        if root is None or not fibers:
            raise RuntimeError(f"目标 generation 未建立 Fiber: {generation.plugin_id}")
        registered = set(root._fibers.values())  # pyright: ignore[reportPrivateUsage]
        current_consumers = self._current_local_consumers(generation)
        seen: set[int] = set()
        for candidate in (
            *fibers,
            *(fiber for fiber in affected if fiber in registered),
            *current_consumers,
        ):
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if candidate.required_for_readiness and candidate.state != FiberState.ACTIVE:
                raise RuntimeError(
                    f"目标依赖未 ACTIVE: {candidate.name} state={candidate.state}"
                )
            if candidate.error is not None and candidate.required_for_readiness:
                raise RuntimeError(f"目标依赖启动失败: {candidate.name}") from candidate.error
            degraded = tuple(
                entry.name
                for entry in root._health_entries.values()  # pyright: ignore[reportPrivateUsage]
                if entry.owner is candidate
                and entry.required
                and entry.reason is not None
            )
            if degraded:
                raise RuntimeError(
                    f"目标依赖 required health 失败: {candidate.name}:{','.join(degraded)}"
                )

    async def _start_local_generation(
        self,
        generation: PluginGeneration,
        *,
        affected: tuple[Fiber, ...] = (),
    ) -> None:
        """Load and mount one selected generation on the existing live Root."""
        root = self._live_root
        if root is None:
            raise RuntimeError("正式 live Root 尚未建立")
        self._check_operation_commit()
        try:
            await self._load_live_generation(generation)
        except Exception:
            await self._retain_pre_fiber_failure(generation)
            raise
        ensure_workspace_plugin_data_dir(generation.data_dir, self._workspace)
        self._check_live_host_dependencies((generation,))
        self._active_generations[generation.plugin_id] = generation
        await self._attach_generation_hosts(generation)
        await self._mount_generation_composition(root, generation)
        self._check_operation_commit()
        self._require_local_generation_ready(generation, affected=affected)
        generation.state = "active"

    async def _attach_generation_hosts(self, generation: PluginGeneration) -> None:
        """Add this exact generation to stable host facades already provided by the Root."""
        root = self._live_root
        if root is None:
            raise RuntimeError("正式 live Root 尚未建立")
        execution = self._live_execution_access
        if execution is None:
            raise RuntimeError("live Root 缺少稳定 ExecutionAccess")
        execution.add_owner(
            generation.plugin_id, generation.generation_id,
            CodeOwner(
                generation.generation_id, generation.code_dir,
                lambda command, cwd, generation=generation: self._resolve_runtime_command(
                    generation, command, cwd,
                ),
            ),
        )
        clients = self._live_credentials
        if clients is None:
            raise RuntimeError("live Root 缺少稳定 CredentialClients")
        clients.add_factory(
            generation.plugin_id, generation.generation_id,
            CoreProviderClientFactory(
                generation.data_dir, generation.config_projection, generation.config_revision,
            ),
        )

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
        """Prepare a stable discovery batch for the single live Root."""
        inputs: list[PluginGeneration] = []
        for mod in mods:
            generation = await self._load_one_with_source_diagnostics(
                mod, activate=False, stage_stable=True,
            )
            if generation is None:
                raise RuntimeError(f"完整插件组合加载失败: {_resolve_plugin_id(mod)}")
            inputs.append(generation)
        components = tuple(
            self._generation_archive_ref(item)
            for item in sorted(inputs, key=lambda item: item.plugin_id)
        )
        await self._load_live_initial(components, expected_ref=None, prepared=tuple(inputs))

    async def prepare_candidate(self, plugin_id: str) -> PluginGeneration | None:
        raise RuntimeError("候选发布入口已停用；local Loader 只允许 live Root 更新")

    async def discard_prepared(
        self, plugin_id: str, *, error: str = "candidate discarded",
    ) -> None:
        raise RuntimeError("候选 generation registry 已停用；无第二张运行图")

    async def _discard_prepared(
        self,
        plugin_id: str,
        *,
        error: str = "candidate discarded",
    ) -> None:
        raise RuntimeError("候选 generation registry 已停用；无第二张运行图")

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
        selection_ref = self._selection.read()
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
        retain_selected_failed: bool = False,
    ) -> None:
        """成功后才解除 owner；失败或取消保留资源供显式关闭重试。"""

        # 1. 调用者可能已移除 active/prepared，先把责任交给既有排空集合。
        tracked = self._draining_generations.setdefault(generation.plugin_id, [])
        if not any(item is generation for item in tracked):
            tracked.append(generation)

        async def close_resources() -> None:
            """Close Fiber and generation-owned resources before module removal."""

            # The local loader owns a Fiber in the stable Root, never a snapshot Root.
            if generation.runtime_snapshot is None and (
                generation.fiber is not None or self._live_root is not None
            ):
                if generation.fiber is not None:
                    await generation.fiber.dispose()
                if self._live_credentials is not None:
                    await self._live_credentials.remove_factory(
                        generation.plugin_id, generation.generation_id,
                    )
                failures = await generation.scope.aclose()
                self._cleanup_failures.extend(failures)
                if failures:
                    raise RuntimeError(
                        "generation scope cleanup 未完成，必须显式 retry: "
                        + "; ".join(f"{item.resource}: {item.error}" for item in failures)
                    )
                if self._live_execution_access is not None:
                    self._live_execution_access.remove_owner(
                        generation.plugin_id, generation.generation_id,
                    )
                return

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
        generation.state = state
        if retain_selected_failed:
            generation.instance = None
            generation.fiber = None
        elif self._active_generations.get(generation.plugin_id) is generation:
            self._active_generations.pop(generation.plugin_id, None)
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
        """回收这张 snapshot 独占的 Root 和 generations。"""
        if any(
            self._active_generations.get(plugin_id) is generation
            for plugin_id, generation in snapshot.generations.items()
        ):
            raise RuntimeError("legacy snapshot 不能回收 live Root 的 generation owner")
        if any(self._building_root_uses(item) for item in snapshot.generations.values()):
            raise RuntimeError("未交接 Root 仍持有 snapshot 模块和数据，须显式 terminate 重试")
        composition_root = snapshot.composition_root
        # Store 在接纳时已拒绝物理 Root/generation 共享，回收不再解释共享图。
        if composition_root is not None:
            await self._stop_runtime_snapshot(snapshot)
            await composition_root.dispose()
        state = "aborted" if snapshot.state == "aborted" else "retired"
        for generation in snapshot.generations.values():
            await self._dispose_generation(
                generation,
                state=state,
            )
            self._forget_drained_generation(generation)
        self._finish_drained_reload(snapshot.snapshot_id)

    async def reconcile_changed(self) -> list[dict[str, object]]:
        return await self._run_operation(self._reconcile_changed)


    async def install_candidate(
        self, *, source: str, marketplace: str, ref_name: str,
        sparse_paths: list[str], update_id: str | None = None,
    ) -> tuple[PluginInstallResult, dict[str, object]]:
        raise RuntimeError("候选安装/发布入口已停用；T05 consumer migration pending")

    async def install(
        self, *, source: str, marketplace: str, ref_name: str,
        sparse_paths: list[str], update_id: str,
    ) -> UpdateStatus:
        """Install one fixed artifact and return when its selection CAS is accepted."""
        if not isinstance(update_id, str) or not update_id or update_id.strip() != update_id:
            raise ValueError("插件更新 ID 必须是非空且无首尾空白的字符串")
        try:
            self._reload_journal.update(update_id)
        except KeyError:
            pass
        else:
            raise RuntimeError("已有插件更新请求只能查询，不能重跑安装")
        accepted: asyncio.Future[UpdateStatus] = asyncio.get_running_loop().create_future()
        operation = self._start_operation(
            lambda: self._install_public(
                source=source, marketplace=marketplace, ref_name=ref_name,
                sparse_paths=sparse_paths, update_id=update_id, accepted=accepted,
            ),
            background=True, update_id=update_id, accepted=accepted,
        )
        try:
            return await asyncio.shield(accepted)
        except asyncio.CancelledError:
            # The owner task remains responsible for the install and drain after CAS.
            raise

    async def uninstall(self, plugin_id: str) -> dict[str, object]:
        """Return after selection removal; one Manager task owns physical cleanup."""
        if not isinstance(plugin_id, str) or not plugin_id or plugin_id.strip() != plugin_id:
            raise ValueError("插件 ID 必须是非空且无首尾空白的字符串")
        _ = _split_installed_plugin_id(plugin_id)
        accepted: asyncio.Future[dict[str, object]] = asyncio.get_running_loop().create_future()
        _ = self._start_operation(
            lambda: self._uninstall_public(plugin_id, accepted=accepted),
            background=True,
            accepted=accepted,
        )
        try:
            return cast(dict[str, object], await asyncio.shield(accepted))
        except asyncio.CancelledError:
            # Caller cancellation only abandons this wait; the Manager task keeps ownership.
            raise

    async def _uninstall_public(
        self, plugin_id: str, *, accepted: asyncio.Future[dict[str, object]],
    ) -> dict[str, object]:
        """Disable, CAS-remove, drain the target owners, then finalize its install."""
        self._check_operation_commit()
        self.require_installed_plugin(plugin_id)
        _ = set_installed_plugin_enabled(
            plugin_id,
            enabled=False,
            plugins_home=self.installed_plugins_home,
        )
        expected_ref = self._selection.read()
        result = await self._deactivate_plugin(
            plugin_id,
            expected_ref=expected_ref,
            accepted=accepted,
        )
        for generation in tuple(self._draining_generations.get(plugin_id, ())):
            await self._dispose_generation(generation, state="retired")
        self._check_operation_commit()
        finalize_result, finalize_cancelled = await _complete_critical(
            asyncio.to_thread(
                finalize_uninstall_plugin,
                plugin_id,
                workspace=self._workspace,
                plugins_home=self.installed_plugins_home,
            )
        )
        if finalize_cancelled:
            raise asyncio.CancelledError
        cache_path, data_path = finalize_result
        self._notify_updates()
        return {
            "plugin_id": plugin_id,
            "state": "removed",
            "selection_ref": result["selection_ref"],
            "cache_path": str(cache_path),
            "data_path": str(data_path),
        }

    async def _install_public(
        self, *, source: str, marketplace: str, ref_name: str,
        sparse_paths: list[str], update_id: str,
        accepted: asyncio.Future[UpdateStatus],
    ) -> UpdateStatus:
        """Run install, input binding, selection CAS, and same-Root activation."""
        installation_started = False
        try:
            self._check_operation_commit()
            try:
                self._reload_journal.update(update_id)
            except KeyError:
                pass
            else:
                # The public pre-check closes the normal path; this second check
                # closes the race where another owner persisted the ID meanwhile.
                raise RuntimeError("已有插件更新请求只能查询，不能重跑安装")
            installation_started = True
            result, install_cancelled = await _complete_critical(
                asyncio.to_thread(
                    install_git_plugin,
                    workspace=self._workspace,
                    source=source,
                    marketplace=marketplace,
                    ref_name=ref_name,
                    sparse_paths=sparse_paths,
                    plugins_home=self.installed_plugins_home,
                    stage_candidate=False,
                    update_id=update_id,
                )
            )
            if install_cancelled:
                raise asyncio.CancelledError
            self._check_operation_commit()
            mod = {
                "name": result.plugin_name,
                "plugin_root": str(result.installed_path),
                "module_path": str(result.installed_path / "plugin.py"),
                "manifest_digest": load_static_plugin_manifest(result.installed_path).identity_digest,
                "marketplace": result.marketplace,
                "source_type": "installed",
            }
            generation = await self._load_one_with_source_diagnostics(
                mod, activate=False, stage_stable=True,
            )
            if generation is None:
                raise RuntimeError(f"安装目标未进入 live generation: {result.plugin_name}@{result.marketplace}")
            if generation.archive_ref is None:
                raise RuntimeError("安装目标没有归档引用")
            self._reload_journal.set_input_ref(result.update_id, generation.archive_ref)
            expected_ref = self._selection.read()
            previous = self._active_generations.get(generation.plugin_id)
            await self._update_live_generation(
                generation, previous, expected_ref=expected_ref,
                update_id=result.update_id, accepted=accepted,
            )
            return self.read_update(result.update_id)
        except BaseException as error:
            if installation_started:
                try:
                    current = self._reload_journal.update(update_id)
                except KeyError:
                    current = None
                if current is not None and not current.error:
                    self._reload_journal.record_update_error(
                        update_id, str(error) or type(error).__name__,
                    )
            if not accepted.done():
                accepted.set_exception(error if isinstance(error, Exception) else RuntimeError(str(error)))
            raise

    def read_update(self, update_id: str) -> UpdateStatus:
        """Project durable input, exact selection, and the live generation state."""
        update = self._reload_journal.update(update_id)
        input_ref = update.input_ref
        selection_ref = self._selection.read()
        components = () if selection_ref is None else self._selection_components(selection_ref)
        selected = input_ref is not None and input_ref in components
        generation = self._active_generations.get(update.plugin_id)
        fiber_state = None if generation is None or generation.fiber is None else generation.fiber.state.value
        generation_id = None if generation is None else generation.generation_id
        archive_ref = None if generation is None else generation.archive_ref
        state: Literal["accepted", "active", "failed", "unknown"] = "unknown"
        readiness_error: str | None = None
        exact_generation = (
            input_ref is not None
            and selected
            and generation is not None
            and generation.archive_ref == input_ref
        )
        if exact_generation and generation is not None and generation.load_error is not None:
            readiness_error = (
                str(generation.load_error)
                or type(generation.load_error).__name__
            )
            state = "failed"
        elif (
            exact_generation
            and generation is not None
            and generation.state == "active"
            and generation.fiber is not None
            and generation.fiber.state == FiberState.ACTIVE
        ):
            try:
                self._require_local_generation_ready(generation)
            except RuntimeError as error:
                readiness_error = str(error) or type(error).__name__
            else:
                state = "active"
        if state == "unknown":
            operation = self._operation
            if (
                operation is not None
                and operation.update_id == update_id
                and not operation.task.done()
                and not operation.revoked
                and selected
            ):
                state = "accepted"
            elif readiness_error is not None or update.error:
                state = "failed"
        selection = (
            "unknown" if input_ref is None or selection_ref is None
            else "selected" if selected else "not_selected"
        )
        return UpdateStatus(
            update_id=update_id,
            plugin_id=update.plugin_id,
            input_ref=input_ref,
            selection=selection,
            generation_id=generation_id,
            archive_ref=archive_ref,
            fiber_state=fiber_state,
            state=state,
            error=readiness_error or update.error,
        )

    def read_validation_messages(self, update_id: str, session_id: str) -> tuple[Message, ...]:
        """只读原调用消息；关闭后的证据沿原 Message 解码，不初始化或重放程序。"""
        if session_id != "plugin-validation:" + update_id:
            raise PermissionError("会话不属于该更新调用")
        update = self._reload_journal.update(update_id)
        if update.reload_tx_id is None:
            return ()
        snapshot_id = self._reload_journal.get(update.reload_tx_id).candidate_snapshot_id
        for host in self._validation_hosts.values():
            if host.parent_lease.snapshot.snapshot_id == snapshot_id:
                return host.messages.reader(session_id).snapshot()
        opened = [event for event in self._reload_journal.events(update.reload_tx_id)
                  if event.details.get("event") == "business_validation_opened"]
        if not opened:
            return ()
        if len(opened) != 1:
            raise RuntimeError("更新调用证据不唯一")
        details = opened[0].details
        workspace = self._workspace / "runtime" / "plugin-update-validation" / cast(str, details["validation_id"]) / "workspace"
        if (details["candidate_snapshot"] != snapshot_id
                or Path(cast(str, details["workspace"])).resolve() != workspace.resolve()):
            raise RuntimeError("调用证据与固定候选不一致")
        database = workspace / "sessions.db"
        evidence_root = (self._workspace / "runtime" / "plugin-update-validation").resolve()
        if not database.resolve().is_relative_to(evidence_root):
            raise RuntimeError("调用证据越过隔离目录")
        return read_persisted_messages(database, session_id)

    def start_update_publication(self, update_id: str) -> None:
        """同步接纳无 lease 的宿主任务；返回只表示已接纳，不表示已晋升。"""
        raise RuntimeError("候选发布入口已停用；local Loader 不等待旧 snapshot lease")
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
        if self._reload_journal.get(cast(str, update.reload_tx_id)).phase != "latest_ready":
            raise RuntimeError("候选已失去提交授权")
        if update.error:
            raise RuntimeError("候选调用失败或结果未知，不能发布")
        retained = [host.identity for host in self._validation_hosts.values()
                    if host.parent_lease.snapshot is ready.snapshot]
        if retained:
            raise RuntimeError(f"验证尚未退出或资源尚未清理: {retained}")
        try:
            operation = self._start_operation(
                lambda: self._publish_update(update_id, update.plugin_id), background=True,
            )
        except OperationBusyError as error:
            self._reload_journal.record_update_error(update_id, f"发布未开始：{error}")
            self._notify_updates()
            raise
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
            self._check_operation_commit()
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
        """Reconcile changed inputs in the one formal Root and owner map."""
        if self._live_root is None:
            raise RuntimeError("local Loader 尚未建立 live Root；已知宿主迁移缺口阻止更新")
        if any(self._draining_generations.values()):
            raise OperationBusyError("上一次更新仍有资源 owner；必须显式 retry/terminate")
        results: list[dict[str, object]] = []
        ready = self._ready_candidate
        if ready is not None:
            raise RuntimeError("旧候选入口仍有待处理状态；T05 migration 完成前禁止 local update")
        discovered_mods, _failures = self._discover_modules(
            installed_selector="latest", record_source_failures=True,
        )
        discovered = {
            _resolve_plugin_id(mod): mod for mod in discovered_mods
        }
        manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
        desired = {
            plugin_id
            for plugin_id, mod in discovered.items()
            if manifest.get(plugin_id, True)
        }
        selection_ref = self._selection.read()
        selected_ids = set()
        selected_components: tuple[str, ...] = ()
        if selection_ref is not None:
            selected_components = self._selection_components(selection_ref)
            selected_ids = {
                cast(str, self._archive.read_descriptor(ref)["plugin_id"])
                for ref in selected_components
            }
        explicitly_disabled = {
            plugin_id
            for plugin_id in selected_ids
            if manifest.get(plugin_id, True) is False
            or plugin_id in self._disabled_builtin_plugins
        }
        for plugin_id in sorted((set(self._active_generations) | selected_ids) - desired):
            if plugin_id not in explicitly_disabled:
                active = self._active_generations.get(plugin_id)
                source_root = self._source_root_hint(plugin_id, active)
                self._remember_missing_source(plugin_id, active)
                results.append({
                    "plugin_id": plugin_id,
                    "publication_state": "source_unavailable",
                    "selection_ref": selection_ref,
                    "source_root": None if source_root is None else str(source_root),
                })
                continue
            result = await self._deactivate_plugin(plugin_id, expected_ref=selection_ref)
            selection_ref = cast(str, result["selection_ref"])
            results.append(result)
        for plugin_id in sorted(desired):
            if plugin_id not in selected_ids:
                results.append({
                    "plugin_id": plugin_id,
                    "publication_state": "unselected_source",
                    "selection_ref": selection_ref,
                })
                continue
            active = self._active_generations.get(plugin_id)
            mod = discovered[plugin_id]
            if (
                active is not None
                and active.load_error is not None
                and active.fiber is None
                and active.state == "failed"
                and active.archive_ref in selected_components
            ):
                results.append({
                    "plugin_id": plugin_id,
                    "publication_state": "failed_selected",
                    "error": (
                        str(active.load_error)
                        or type(active.load_error).__name__
                    ),
                })
                continue
            revision = _source_revision(Path(mod["plugin_root"]))
            _config, config_revision = load_config(
                _resolve_plugin_data_dir(mod["name"], mod, self._workspace),
            )
            had_source_failure = (
                _source_failure_key_for_mod(mod) in self._source_failures
            )
            if (
                active is not None
                and active.state == "active"
                and active.fiber is not None
                and active.fiber.state == FiberState.ACTIVE
                and active.source_revision == revision
                and active.config_revision == config_revision
                and not had_source_failure
            ):
                continue
            generation = await self._load_one_with_source_diagnostics(
                mod, activate=False, stage_stable=True,
            )
            if generation is None:
                results.append({
                    "plugin_id": plugin_id,
                    "publication_state": "source_unavailable",
                    "selection_ref": selection_ref,
                })
                continue
            if (
                active is not None
                and active.state == "active"
                and active.fiber is not None
                and active.fiber.state == FiberState.ACTIVE
                and active.source_revision == revision
                and active.config_revision == config_revision
            ):
                # A disappeared source can be revalidated without replacing an
                # unrelated live generation merely to clear its old diagnostic.
                await self._dispose_generation(generation, state="discarded")
                continue
            result = await self._update_live_generation(
                generation, active, expected_ref=selection_ref,
            )
            selection_ref = cast(str, result["selection_ref"])
            results.append(result)
        return results

    async def _update_live_generation(
        self,
        generation: PluginGeneration,
        previous: PluginGeneration | None,
        *,
        expected_ref: str | None,
        update_id: str | None = None,
        accepted: asyncio.Future[UpdateStatus] | None = None,
    ) -> dict[str, object]:
        """Commit B, drain A, and mount B on the same live Root."""
        root = self._live_root
        if root is None:
            raise RuntimeError("正式 live Root 尚未建立")
        affected = self._capture_local_readiness(
            () if previous is None else (previous,)
        )
        try:
            replacement_ref = self._generation_archive_ref(generation)
            components = self._selection_for_plugin(
                expected_ref, generation.plugin_id, replacement_ref,
            )
            if (
                previous is not None
                and previous.archive_ref == replacement_ref
                and self._selection_contains(expected_ref, replacement_ref)
                and self._generation_is_locally_ready(previous)
            ):
                self._check_operation_commit()
                await self._dispose_generation(generation, state="discarded")
                if update_id is not None and accepted is not None and not accepted.done():
                    accepted.set_result(self.read_update(update_id))
                return {
                    "plugin_id": generation.plugin_id,
                    "old_generation": previous.generation_id,
                    "new_generation": previous.generation_id,
                    "selection_ref": cast(str, expected_ref),
                    "publication_state": "active",
                }
            operation = self._check_operation_commit()
            selection_ref = self._selection.commit(
                components,
                expected_ref=expected_ref,
            )
            operation.committed = selection_ref
            if update_id is not None and accepted is not None and not accepted.done():
                accepted.set_result(self.read_update(update_id))
            if not self._operation_can_continue():
                raise OperationTimeoutError(operation)
            if previous is not None:
                await self._dispose_generation(previous, state="retired")
            if not self._operation_can_continue():
                raise OperationTimeoutError(operation)
            await self._start_local_generation(generation, affected=affected)
            return {
                "plugin_id": generation.plugin_id,
                "old_generation": None if previous is None else previous.generation_id,
                "new_generation": generation.generation_id,
                "selection_ref": selection_ref,
                "publication_state": "active",
            }
        except BaseException:
            retained_pre_fiber_failure = (
                generation.load_error is not None
                and generation.fiber is None
                and generation.state == "failed"
                and self._active_generations.get(generation.plugin_id) is generation
            )
            selected_ref = self._selection.read()
            retained_failed_fiber = (
                generation.fiber is not None
                and generation.fiber.state == FiberState.FAILED
                and generation.fiber.error is not None
                and generation.archive_ref is not None
                and self._selection_contains(selected_ref, generation.archive_ref)
                and self._active_generations.get(generation.plugin_id) is generation
            )
            if retained_failed_fiber:
                generation.load_error = generation.fiber.error
                generation.state = "failed"
                await self._dispose_generation(
                    generation, state="failed", retain_selected_failed=True,
                )
            elif not retained_pre_fiber_failure and (
                generation.state != "active"
                or self._active_generations.get(generation.plugin_id) is generation
            ):
                try:
                    await self._dispose_generation(generation, state="discarded")
                except BaseException:
                    # Selection B remains committed; the draining owner is retained for explicit retry.
                    raise
            raise

    def _selection_contains(self, selection_ref: str | None, input_ref: str) -> bool:
        """Check the exact persisted selection without accepting a missing one."""
        return selection_ref is not None and input_ref in self._selection_components(selection_ref)

    def _generation_is_locally_ready(self, generation: PluginGeneration) -> bool:
        """Return readiness for an already-owned local generation."""
        if (
            generation.state != "active"
            or generation.fiber is None
            or generation.fiber.state != FiberState.ACTIVE
        ):
            return False
        try:
            self._require_local_generation_ready(generation)
        except RuntimeError:
            return False
        return True

    async def reconcile_disabled_and_drain(self, plugin_id: str) -> None:
        await self._run_operation(lambda: self._reconcile_disabled_and_drain(plugin_id))

    async def _reconcile_disabled_and_drain(self, plugin_id: str) -> None:
        manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
        if manifest.get(plugin_id, False):
            raise RuntimeError(f"插件尚未禁用: {plugin_id}")
        for generation in tuple(self._draining_generations.get(plugin_id, ())):
            await self._dispose_generation(generation, state="retired")
        selection_ref = self._selection.read()
        selected = False
        if selection_ref is not None:
            selected = any(
                self._archive.read_descriptor(ref).get("plugin_id") == plugin_id
                for ref in self._selection_components(selection_ref)
            )
        if plugin_id in self._active_generations or selected:
            _ = await self._deactivate_plugin(plugin_id, expected_ref=selection_ref)
        for generation in tuple(self._draining_generations.get(plugin_id, ())):
            await self._dispose_generation(generation, state="retired")
        if self._draining_generations.get(plugin_id):
            raise RuntimeError(f"插件仍有未关闭的资源 owner，须先 recovery/terminate: {plugin_id}")

    async def _deactivate_plugin(
        self, plugin_id: str, *, expected_ref: str | None = None,
        accepted: asyncio.Future[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """Remove one generation from the selection and the same live Root."""
        active = self._active_generations.get(plugin_id)
        if expected_ref is None:
            expected_ref = self._selection.read()
        components = self._selection_for_plugin(expected_ref, plugin_id, None)
        operation = self._check_operation_commit()
        selection_ref = self._selection.commit(components, expected_ref=expected_ref)
        operation.committed = selection_ref
        if accepted is not None and not accepted.done():
            accepted.set_result({
                "plugin_id": plugin_id,
                "state": "accepted",
                "selection_ref": selection_ref,
            })
        if active is not None:
            await self._dispose_generation(active, state="retired")
        return {
            "plugin_id": plugin_id, "old_generation": None if active is None else active.generation_id,
            "new_generation": None, "selection_ref": selection_ref,
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

    async def _prepare_runtime_snapshot(self, lease: RuntimeSnapshotLease) -> None:
        """在 exact closed scope 内等待启动资源；失败标记留给 stopping 清理。"""
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
            result = await root.context.serial(RUNTIME_STARTING, RuntimeStarting())
            if result is not None:
                raise CompositionError("RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED", "runtime.starting 不接受 Bail")
        finally:
            reset_runtime_snapshot(token)

    async def _commit_snapshot_with_publication_participants(
        self,
        transaction: SnapshotTransaction,
        *,
        old_commands: tuple[tuple[str, str], ...],
        new_commands: tuple[tuple[str, str], ...],
        before_open: Callable[[], None] | None = None,
        after_open: Callable[[], None] | None = None,
    ) -> SnapshotTransaction:
        """关闭目标 lease 内准备临时资源，全部完成后才开放正式接纳。"""

        lease = self._snapshot_store.retain_publication_target(transaction)
        try:
            return await self._commit_snapshot_participants(
                transaction, old_commands=old_commands, new_commands=new_commands,
                before_open=before_open, after_open=after_open,
                startup_snapshot_lease=lease,
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
            await self._prepare_runtime_snapshot(lease)
        async with _snapshot_lease_scope(lease.fork()):
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
        before_open: Callable[[], None] | None = None,
        after_open: Callable[[], None] | None = None,
        startup_snapshot_lease: RuntimeSnapshotLease,
    ) -> SnapshotTransaction:
        """Publish one snapshot around a single closed external-participant step."""

        # 1. 旧 Root 已退出；所有新资源在 closed scope 内由 provider 初始化。
        provisional = transaction
        await self._snapshot_store.commit_provisional(provisional)

        participants_switch_attempted = False
        forward_error: BaseException | None = None
        try:
            participants_switch_attempted = True
            try:
                await self._switch_plugin_endpoints(old_commands, new_commands)
            except BaseException as error:
                forward_error = error
                raise
            await self._start_closed_runtime_snapshot(startup_snapshot_lease)
            await self._snapshot_store.finalize_provisional(
                provisional,
                before_open=before_open,
                after_open=after_open,
                schedule_previous_drain=False,
            )
            if provisional.previous is not None:
                self._snapshot_store.schedule_retired_drain(provisional.previous)
        except BaseException as publication_error:
            revoke_on_cancel(publication_error)
            if provisional.must_retain:
                self._hold_selection_publication(provisional)
                raise
            rollback_errors: list[BaseException] = []
            endpoint_restore_failed = False
            if participants_switch_attempted:
                try:
                    await self._switch_plugin_endpoints(new_commands, old_commands)
                except BaseException as caught:
                    rollback_errors.append(caught)
                    endpoint_restore_failed = True
            published_rollback = self._snapshot_store.current is provisional.candidate
            if published_rollback:
                await self._snapshot_store.rollback_published(
                    provisional,
                    keep_candidate_latest=False,
                    reopen_previous=False,
                )
            if not published_rollback:
                await self._snapshot_store.rollback_provisional(
                    provisional,
                    keep_candidate_latest=False,
                    reopen_previous=False,
                )
            if rollback_errors:
                resources: list[str] = []
                if endpoint_restore_failed:
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
            )
        except BaseException as error:
            await self._discard_building_root(composition_root, error)
            raise
        for item in generations.values():
            item.runtime_snapshot = snapshot
        return snapshot

    async def publish_prepared(self, plugin_id: str) -> dict[str, object]:
        raise RuntimeError("候选发布入口已停用；local Loader 只允许 live Root 更新")


    async def switch_ready(self, plugin_id: str, *, update_id: str | None = None) -> dict[str, object]:
        raise RuntimeError("候选发布入口已停用；local Loader 只允许 live Root 更新")


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
            if self._reload_journal.get(tx_id).phase == "aborted":
                self._reload_journal.annotate(tx_id, {"cleanup_receipt": "candidate-root-closed"})
            self._ready_candidate = None
            raise asyncio.CancelledError
        def before_open() -> None:
            if self._reload_journal.get(tx_id).phase != "latest_ready":
                raise RuntimeError("更新已失去发布授权")
            if update_id is not None:
                update = self._reload_journal.update(update_id)
                if update.phase != "armed" or update.reload_tx_id != tx_id or update.error:
                    raise RuntimeError("更新请求已撤销或失去精确候选授权")
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
                if self._reload_journal.get(tx_id).phase == "aborted":
                    self._reload_journal.annotate(tx_id, {"cleanup_receipt": "candidate-root-closed"})
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
        if self._selection.read() != expected_ref:
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
        try:
            if self._endpoint_quiescer is not None:
                await self._endpoint_quiescer()
            if previous is not None:
                await self._snapshot_store.wait_for_no_leases(previous)
            self._check_operation_commit()
        except BaseException:
            # 只有这段尚未开始 release 的路径可以恢复原接纳；shutdown 永远优先。
            if not self._stopping and current_operation.get() is self._operation:
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
                inputs, previous=previous,
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
                if self._selection.read() != expected_ref:
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
                            previous=previous, expected_ref=expected_ref,
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
        """关闭实际 Root，让依赖顺序决定资源退出。"""
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
            if previous is not None and attempt is not None:
                self._drain_transactions[previous.snapshot_id] = cast(str, attempt.reload_tx_id)
            _, cancelled = await _complete_critical(
                self._commit_snapshot_with_publication_participants(
                    transaction,
                    old_commands=_snapshot_command_catalog(previous),
                    new_commands=_snapshot_command_catalog(snapshot),
                    before_open=authorize_commit,
                    after_open=lambda: self._activate_snapshot(snapshot, previous),
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

    def _begin_snapshot_publication(self, snapshot: RuntimeSnapshot) -> SnapshotTransaction:
        transaction = self._snapshot_store.begin_publish(snapshot)
        if snapshot.composition_root is not None:
            self._building_roots.pop(snapshot.composition_root, None)
        return transaction

    def _activate_snapshot(self, snapshot: RuntimeSnapshot, previous: RuntimeSnapshot | None) -> None:
        """提交时激活整个新组合，旧组合整体退役。"""
        old = {} if previous is None else previous.generations
        for generation in snapshot.generations.values():
            generation.state = "active"
        for generation in old.values():
            self._retire_generation(generation)

    async def retry_runtime_recovery(self, plugin_id: str) -> dict[str, object]:
        return await self._run_operation(lambda: self._retry_runtime_recovery(plugin_id))

    async def _retry_runtime_recovery(self, plugin_id: str) -> dict[str, object]:
        """Retry the selected archive after closing every retained local owner."""
        root = self._live_root
        selection_ref = self._selection.read()
        if root is None or selection_ref is None:
            raise RuntimeError("没有该插件的局部 live retry selection")
        selected = tuple(
            ref for ref in self._selection_components(selection_ref)
            if self._archive.read_descriptor(ref)["plugin_id"] == plugin_id
        )
        if len(selected) != 1:
            raise RuntimeError("selection 未提供该插件的唯一局部 retry archive")
        owners: list[PluginGeneration] = []
        active = self._active_generations.get(plugin_id)
        if active is not None:
            owners.append(active)
        owners.extend(self._draining_generations.get(plugin_id, ()))
        unique_owners = tuple(
            owner for index, owner in enumerate(owners)
            if not any(owner is prior for prior in owners[:index])
        )
        affected = self._capture_local_readiness(unique_owners)
        for owner in unique_owners:
            await self._dispose_generation(owner, state="retired")
        self._check_operation_commit()
        fresh = self._archived_generations(
            selected, root, workspace=self._workspace, sources={}, register_live=False,
        )[plugin_id]
        try:
            await self._start_local_generation(fresh, affected=affected)
        except BaseException:
            retained_pre_fiber_failure = (
                fresh.load_error is not None
                and fresh.fiber is None
                and fresh.state == "failed"
                and self._active_generations.get(fresh.plugin_id) is fresh
            )
            if not retained_pre_fiber_failure:
                try:
                    await self._dispose_generation(fresh, state="discarded")
                except BaseException:
                    raise
            raise
        return {
            "plugin_id": plugin_id,
            "publication_state": "recovered",
            "generation_id": fresh.generation_id,
            "retry_receipt": "selected-local-generation-retried",
        }

    async def drop_candidate(self, plugin_id: str) -> dict[str, object]:
        """安装候选交回所属更新结算，普通候选只关闭运行资源。"""
        self._reject_operation_lease(allow_stable_lease=True)
        ready = self._require_ready_candidate(plugin_id)
        tx_id = ready.candidate.reload_tx_id
        if tx_id is None:
            raise RuntimeError("latest candidate 缺少 reload transaction")
        update = self._reload_journal.update_for_reload(tx_id)
        if update is not None:
            # 更新入口先撤销提交权，再取得 operation；不能在 operation 内嵌套调用。
            await self.discard_update(update.update_id)
            return self._publication_status(
                plugin_id, active=ready.previous, candidate=ready.candidate,
                publication_state="discarded",
            )
        return await self._run_operation(
            lambda: self._drop_ready(plugin_id), allow_stable_lease=True,
        )


    async def discard_update(self, update_id: str, *, reason: str = "candidate behavior rejected") -> None:
        """先同步撤销提交权，再等待原调用和资源退出；提交后明确拒绝假回滚。"""
        deadline = asyncio.get_running_loop().time() + self.POST_PUBLISH_TIMEOUT_SECONDS
        self._reject_operation_lease(allow_stable_lease=True)
        update = self._reload_journal.update(update_id)
        if update.phase == "rolled_back":
            return
        if update.phase == "committed" or (
            self._update_publication is not None and self._update_publication[0] == update_id
            and self._operation is not None and self._operation.committed is not None
        ):
            raise RuntimeError("更新已提交；revert 不能撤销已提交选择或插件数据")
        if self._ready_candidate is None:
            async with asyncio.timeout_at(deadline):
                await self._run_operation(
                    lambda: self._discard_update(update_id, reason=reason), allow_stable_lease=True,
                )
            return
        ready = self._require_ready_candidate(update.plugin_id)
        if update.phase != "armed" or update.reload_tx_id != ready.candidate.reload_tx_id:
            raise RuntimeError("更新与当前待处理候选不匹配")
        publication = self._publication
        if (publication is not None and publication.must_retain
                and publication.candidate.generations.get(update.plugin_id) is not None
                and publication.candidate.generations[update.plugin_id].reload_tx_id == update.reload_tx_id):
            raise RuntimeError("提交结果已确认或不确定；保留恢复 owner，不能伪称 revert 成功")
        phase = self._reload_journal.get(cast(str, update.reload_tx_id)).phase
        if phase in {"latest_ready", "promoting"}:
            self._advance_reload(ready.candidate, "discarding", error=reason)
        elif phase != "discarding":
            raise RuntimeError(f"候选不能从 {phase} revert；请沿原恢复 owner 处理")
        self._reload_journal.record_update_error(update_id, reason)
        self._notify_updates()
        current = asyncio.current_task()
        running = tuple(host.task for host in self._validation_hosts.values()
                        if host.parent_lease.snapshot is ready.snapshot and host.active)
        if current in running:
            raise RuntimeError("提交权已撤销；请先退出自己的候选调用再清理")
        for task in running:
            task.cancel()
        operation = self._operation
        if (self.update_is_publishing(update_id) and operation is not None
                and operation.task is self._update_publication[1]):
            operation.revoke()
            async with asyncio.timeout_at(deadline):
                await asyncio.wait((operation.task,))
        if running:
            async with asyncio.timeout_at(deadline):
                await asyncio.wait(running)
        async with asyncio.timeout_at(deadline):
            await self._run_operation(
                lambda: self._discard_update(update_id, reason=reason), allow_stable_lease=True,
            )

    async def _discard_update(self, update_id: str, *, reason: str = "candidate behavior rejected") -> None:
        """持候选锁核对原请求；不能撤销期间已被替换的另一候选。"""
        update = self._reload_journal.update(update_id)
        if update.phase == "rolled_back":
            return
        if update.phase != "armed" or update.reload_tx_id is None:
            raise RuntimeError("更新不是等待丢弃的候选")
        record = self._reload_journal.get(update.reload_tx_id)
        if (record.plugin_id != update.plugin_id
                or record.candidate_artifact_pointer != update.candidate.path
                or record.candidate_snapshot_id is None):
            raise RuntimeError("更新缺少精确候选资源记录")
        if self._ready_candidate is not None:
            ready = self._require_ready_candidate(update.plugin_id)
            if update.phase != "armed" or update.reload_tx_id != ready.candidate.reload_tx_id:
                raise RuntimeError("更新与当前待处理候选不匹配")
        # 原程序已退出才重试原 host；保留失败 owner，不扫描清理其他候选。
        retained = tuple(
            host.identity for host in self._validation_hosts.values()
            if host.parent_lease.snapshot.snapshot_id == record.candidate_snapshot_id
            and not host.active
        )
        for identity in retained:
            await self._retry_validation_cleanup(identity)
        if self._ready_candidate is not None:
            ready = self._require_ready_candidate(update.plugin_id)
            if any(host.parent_lease.snapshot is ready.snapshot for host in self._validation_hosts.values()):
                raise RuntimeError("验证尚未退出或资源尚未清理")
            _ = await self._drop_ready(update.plugin_id, error=reason)
        self._check_discarded_update(update_id)
        self._check_operation_commit()
        # runtime 清理不回写安装状态；由持有本次请求的安装入口结算恢复点。
        self._reload_journal.rollback_updates(
            self.installed_plugins_home, update_id=update_id, error=reason,
        )
        self._notify_updates()

    def _check_discarded_update(self, update_id: str) -> None:
        """只允许原候选已关闭且原 stable 未改变的安装恢复点再次结算。"""
        update = self._reload_journal.update(update_id)
        if update.phase != "armed" or update.reload_tx_id is None:
            raise RuntimeError("更新不是等待结算的已关闭候选")
        record = self._reload_journal.get(update.reload_tx_id)
        events = self._reload_journal.events(record.tx_id)
        if (record.plugin_id != update.plugin_id
                or record.candidate_artifact_pointer != update.candidate.path
                or record.phase != "aborted" or record.candidate_snapshot_id is None
                or not events or events[-1].phase != "aborted"
                or events[-1].details.get("cleanup_receipt") != "candidate-root-closed"):
            raise RuntimeError("候选缺少实际关闭回执；请沿原资源 owner 恢复")
        if (self._ready_candidate is not None
                or self._snapshot_store.unpromoted_candidate is not None
                or self._snapshot_store.pending_transaction is not None
                or record.candidate_snapshot_id in self._snapshot_store.retained_snapshot_ids
                or self._building_roots
                or any(host.parent_lease.snapshot.snapshot_id == record.candidate_snapshot_id
                       for host in self._validation_hosts.values())):
            raise RuntimeError("候选或资源 owner 仍在，不能结算旧安装更新")
        if self._publication is not None and self._publication.must_retain:
            raise RuntimeError("提交结果已确认或不确定；保留原恢复 owner")
        intent = self._reload_journal.selection_candidate(record.tx_id)
        if intent is None or self._selection.read() != intent[0]:
            raise RuntimeError("原 stable 选择已改变或证据未知，不能回退安装更新")

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
            details={"cleanup_receipt": "candidate-root-closed"},
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

    def plugin_status(self) -> dict[str, object]:
        """Project manifest, selection, owner generations, and the current operation."""
        manifest = load_plugin_manifest(self.installed_plugins_home)
        selection_ref = self._selection.read()
        selected_refs: dict[str, str] = {}
        if selection_ref is not None:
            for archive_ref in self._selection_components(selection_ref):
                descriptor = self._archive.read_descriptor(archive_ref)
                plugin_id = descriptor.get("plugin_id")
                if not isinstance(plugin_id, str):
                    raise TypeError("selection archive descriptor 缺少 plugin_id")
                if plugin_id in selected_refs:
                    raise RuntimeError(f"selection 重复包含插件: {plugin_id}")
                selected_refs[plugin_id] = archive_ref

        def cleanup_pending(generation: PluginGeneration) -> bool:
            return any(
                item is generation
                for item in self._draining_generations.get(generation.plugin_id, ())
            )

        def error_text(generation: PluginGeneration) -> str | None:
            if generation.load_error is None:
                return None
            return str(generation.load_error) or type(generation.load_error).__name__

        def generation_status(generation: PluginGeneration) -> dict[str, object]:
            return {
                "generation_id": generation.generation_id,
                "archive_ref": generation.archive_ref,
                "state": generation.state,
                "load_error": error_text(generation),
                "cleanup_pending": cleanup_pending(generation),
                "fiber_state": (
                    None if generation.fiber is None else generation.fiber.state.value
                ),
            }

        operation = self._operation
        operation_view: dict[str, object] | None = None
        operation_kind = (
            "install" if operation is not None and operation.update_id is not None
            else "unknown"
        )
        operation_plugin_id: str | None = None
        if operation is not None:
            if operation.task.cancelled():
                task_state = "cancelled"
                task_error = None
            elif not operation.task.done():
                task_state = "running"
                task_error = None
            else:
                task_error_value = operation.task.exception()
                task_state = "error" if task_error_value is not None else "done"
                task_error = (
                    None
                    if task_error_value is None
                    else str(task_error_value) or type(task_error_value).__name__
                )

            accepted_view: dict[str, object] | None = None
            if operation.accepted is not None:
                if operation.accepted.cancelled():
                    accepted_view = {"state": "cancelled"}
                elif not operation.accepted.done():
                    accepted_view = {"state": "pending"}
                else:
                    try:
                        accepted_result = operation.accepted.result()
                    except asyncio.CancelledError:
                        accepted_view = {"state": "cancelled"}
                    except Exception as error:
                        accepted_view = {
                            "state": "error",
                            "error": str(error) or type(error).__name__,
                        }
                    else:
                        if isinstance(accepted_result, UpdateStatus):
                            operation_kind = "install"
                            operation_plugin_id = accepted_result.plugin_id
                            accepted_view = {
                                "kind": "install",
                                "state": "accepted",
                                "result": asdict(accepted_result),
                            }
                        elif isinstance(accepted_result, dict):
                            accepted_plugin_id = accepted_result.get("plugin_id")
                            accepted_state = accepted_result.get("state")
                            if not isinstance(accepted_plugin_id, str):
                                raise TypeError(
                                    "ManagerOperation.accepted uninstall 结果缺少 plugin_id"
                                )
                            if not isinstance(accepted_state, str):
                                raise TypeError(
                                    "ManagerOperation.accepted uninstall 结果缺少 state"
                                )
                            operation_kind = "uninstall"
                            operation_plugin_id = accepted_plugin_id
                            accepted_view = {
                                "kind": "uninstall",
                                "state": accepted_state,
                                "result": dict(accepted_result),
                            }
                        else:
                            raise TypeError(
                                "ManagerOperation.accepted 结果类型不受支持: "
                                f"{type(accepted_result).__name__}"
                            )
            operation_view = {
                "kind": operation_kind,
                "plugin_id": operation_plugin_id,
                "update_id": operation.update_id,
                "state": task_state,
                "error": task_error,
                "revoked": operation.revoked,
                "committed": operation.committed is not None,
                "accepted": accepted_view,
            }

        plugin_ids = (
            set(manifest)
            | set(selected_refs)
            | set(self._active_generations)
            | set(self._draining_generations)
        )
        if operation_plugin_id is not None:
            plugin_ids.add(operation_plugin_id)
        plugins: list[dict[str, object]] = []
        for plugin_id in sorted(plugin_ids):
            active = self._active_generations.get(plugin_id)
            draining = tuple(self._draining_generations.get(plugin_id, ()))
            if "@" not in plugin_id:
                cache_exists = False
            else:
                plugin_name, marketplace = _split_installed_plugin_id(plugin_id)
                cache_exists = (
                    self.installed_plugins_home / "cache" / marketplace / plugin_name
                ).exists()
            plugins.append({
                "plugin_id": plugin_id,
                "installed": plugin_id in manifest,
                "enabled": manifest.get(plugin_id),
                "selected_ref": selected_refs.get(plugin_id),
                "cache_exists": cache_exists,
                "draining_generations": [generation_status(item) for item in draining],
                # Keep the original flat projection for existing status consumers.
                "generation_id": None if active is None else active.generation_id,
                "archive_ref": None if active is None else active.archive_ref,
                "state": None if active is None else active.state,
                "load_error": None if active is None else error_text(active),
                "cleanup_pending": (
                    False if active is None else cleanup_pending(active)
                ),
                "fiber_state": (
                    None if active is None or active.fiber is None
                    else active.fiber.state.value
                ),
            })
        source_failures = [
            {
                "source_root": str(failure.source_root),
                "source_type": failure.source_type,
                "phase": failure.phase,
                "error_type": failure.error_type,
                "error_text": failure.error_text,
                "plugin_id": failure.plugin_id,
            }
            for failure in sorted(
                self._source_failures.values(),
                key=lambda item: (str(item.source_root), item.source_type),
            )
        ]
        return {
            "selection_ref": selection_ref,
            "selection_components": list(selected_refs.values()),
            "plugins": plugins,
            "source_failures": source_failures,
            "operation": operation_view,
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

    async def _load_one(
        self,
        mod: dict[str, str],
        *,
        activate: bool = True,
        stage_stable: bool = False,
    ) -> PluginGeneration | None:
        """Prepare one fixed archive input without importing or mounting plugin code."""
        plugin_id = _resolve_plugin_id(mod)
        if activate and plugin_id in self._active_generations:
            return self._active_generations[plugin_id]
        if load_plugin_manifest(_plugins_home(self._installed_cache_root)).get(plugin_id, True) is False:
            return None
        prepared = prepare_plugin_input(
            mod, workspace=self._workspace, archive=self._archive,
        )
        plugin_id = prepared.plugin_id
        # 1. Preparation owns only the returned generation; no candidate Root is built.
        namespace = secrets.token_hex(12)
        module_path = f"_akashic_input_{namespace}"
        generation_id = f"{plugin_id}:input:{namespace}"
        scope = PluginScope(plugin_id, generation_id=generation_id)
        source = PluginGeneration(
            plugin_id=plugin_id, generation_id=generation_id, module_path=module_path,
            source_revision=prepared.source_revision, config_revision=prepared.config_revision,
            plugin_dir=prepared.plugin_dir, data_dir=prepared.data_dir, instance=None, scope=scope,
            config_projection=prepared.config, archive_ref=prepared.archive_ref,
            static_manifest=prepared.static_manifest,
            code_dir_path=prepared.code_dir, source_type=prepared.source_type,
            state="prepared",
        )
        if stage_stable:
            return source
        if activate:
            await self._dispose_generation(source, state="discarded")
            raise RuntimeError("旧 local Loader activation 入口已停用；请经 reconcile_changed")
        # Candidate publication is intentionally blocked until T05 consumers migrate.
        await self._dispose_generation(source, state="discarded")
        raise RuntimeError("候选发布入口已停用；T05 consumer migration pending")

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
                composition_root=composition_root,
            )
        except BaseException as error:
            await self._discard_building_root(composition_root, error)
            raise
        for item in generations.values():
            item.runtime_snapshot = snapshot
        return snapshot

    def open_validation(self, update_id: str) -> AbstractAsyncContextManager[BindingScope]:
        """Reject the retired candidate validation owner."""
        raise RuntimeError("候选验证入口已停用；T05 consumer migration pending")

    async def _prepare_validation(
        self, update_id: str, caller: asyncio.Task[object],
    ) -> tuple[ValidationHost, BindingScope]:
        """从租约取得到资源交接都属于同一次有限操作，复制线程不脱离 owner。"""
        update = self._reload_journal.update(update_id)
        ready = self._require_ready_candidate(update.plugin_id)
        if update.phase != "armed" or update.reload_tx_id != ready.candidate.reload_tx_id:
            raise RuntimeError("更新恢复点与当前候选不匹配")
        if self._reload_journal.get(cast(str, update.reload_tx_id)).phase != "latest_ready" or update.error:
            raise RuntimeError("候选调用已失败或提交权已撤销")
        if any(event.details.get("event") == "business_validation_opened"
               for event in self._reload_journal.events(cast(str, update.reload_tx_id))):
            raise RuntimeError("已有候选调用只能查询，未知结果不得重跑")
        lease = self._snapshot_store.lease(ready.snapshot.snapshot_id)
        host: ValidationHost | None = None
        scope: BindingScope | None = None
        async with _snapshot_lease_scope(lease):
            try:
                host = await self._build_validation_host(lease)
                host.task = caller
                self._validation_hosts[host.identity] = host
                self._reload_journal.annotate(cast(str, update.reload_tx_id), {
                    "event": "business_validation_opened", "validation_id": host.identity,
                    "candidate_generation": ready.candidate.generation_id,
                    "candidate_snapshot": ready.snapshot.snapshot_id,
                    "workspace": str(host.workspace),
                    "components": [item.archive_ref for item in ready.snapshot.generations.values()],
                    "profile": "explicit_program_with_candidate_resources",
                })
                components = tuple(cast(str, item.archive_ref) for item in ready.snapshot.generations.values())
                generations = dict(ready.snapshot.generations)
                root = await self._resolve_composition_root(
                    generations, components=components, validation_host=host,
                )
                snapshot = RuntimeSnapshotCompiler().compile(generations, composition_root=root)
                for generation in generations.values():
                    generation.runtime_snapshot = snapshot
                self._check_operation_commit()
                host.snapshot_store.install(snapshot)
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
        """建立独立消息 owner 与固定代码；不扫描或复制插件业务数据。"""
        identity = secrets.token_hex(16)
        workspace = self._workspace / "runtime" / "plugin-update-validation" / identity / "workspace"
        workspace.mkdir(parents=True)
        archive = PluginArchive(workspace / "runtime" / "plugin-archives")
        for generation in lease.snapshot.generations.values():
            ref = self._generation_archive_ref(generation)
            record = self._archive.read_descriptor(ref)
            code = cast(str, record["code"])
            if archive.save(self._archive.open(code)) != code or archive.save_descriptor(record) != ref:
                raise RuntimeError("隔离候选固定制品身份不一致")
        from contextlib import ExitStack
        from bus.queue import MessageBus
        from session.admissions import SessionAdmissions
        from session.inbound_store import InboundHandoffStore

        # 1. 只创建验证库的连接；构造尚未取得异步资源时可直接关闭。
        with ExitStack() as cleanup:
            messages = MessageLog(workspace / "sessions.db")
            cleanup.callback(messages.close)
            artifacts = ArtifactStore(workspace / "sessions.db")
            cleanup.callback(artifacts.close)
            admissions = SessionAdmissions(workspace / "sessions.db")
            cleanup.callback(admissions.close)
            identities = ChannelIdentities(workspace / "sessions.db")
            cleanup.callback(identities.close)
            inbound_store = InboundHandoffStore(workspace / "sessions.db")
            cleanup.callback(inbound_store.close)
            physical = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=artifacts)
            event_bus = EventBus()
            message_bus = MessageBus()
            message_bus.bind_session_admission_owner(admissions)
            message_bus.bind_durable_inbound_store(inbound_store)
            task = asyncio.current_task()
            assert task is not None
            # 2. 宿主只持有实际资源；没有第二份安装、更新或 stable 控制面。
            host = ValidationHost(
                identity, workspace, messages, artifacts, event_bus, task, lease.fork(),
                message_bus=message_bus, admissions=admissions,
                identities=identities, inbound_store=inbound_store,
                archive=archive, attachments=physical,
            )
            message_bus.bind_durable_inbound_recoverer(host.recover_input)
            cleanup.pop_all()
            return host


    async def retry_validation_cleanup(self, identity: str) -> None:
        await self._run_operation(
            lambda: self._retry_validation_cleanup(identity), allow_stable_lease=True,
        )

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
        validation_host: ValidationHost | None = None,
        register_live: bool = True,
    ) -> dict[str, PluginGeneration]:
        """从固定输入创建只属于当前 Root 的模块、Scope 和 generation。"""

        archive = self._archive if validation_host is None else validation_host.archive
        records = tuple(archive.read_descriptor(ref) for ref in components)
        for record in records:
            if record["version"] != 4 or record["runtime"] != {
                "python_tag": sys.implementation.cache_tag,
                "binding_api": PLUGIN_ARCHIVE_BINDING_API,
            }:
                raise RuntimeError("插件归档运行合同不兼容；保留原归档并使用原 Core 恢复")
        generations: dict[str, PluginGeneration] = {}
        namespace = secrets.token_hex(12)
        for index, (ref, record) in enumerate(zip(components, records, strict=True)):
            code_dir = archive.open(cast(str, record["code"]))
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
            scope = PluginScope(plugin_id, generation_id=generation_id)
            manifest = load_static_plugin_manifest(code_dir)
            projection = decode_config(record["config"])
            if not isinstance(projection, dict):
                raise ValueError("归档插件配置必须是对象")
            generation = PluginGeneration(
                plugin_id=plugin_id, generation_id=generation_id, module_path=module_path,
                source_revision=revision, config_revision=cast(str, record["config_revision"]),
                plugin_dir=code_dir if source is None else source.plugin_dir,
                data_dir=data_dir, config_projection=cast(dict[str, object], projection),
                instance=None, scope=scope,
                static_manifest=manifest,
                source_type=cast(Literal["builtin", "installed"], record["source_type"]),
                archive_ref=ref,
                code_dir_path=code_dir,
                reload_tx_id=None,
                validation_workspace=workspace if workspace != self._workspace else None,
                state="prepared",
            )
            generations[plugin_id] = generation
            if validation_host is None and register_live:
                existing = self._active_generations.get(plugin_id)
                if existing is not None and existing is not generation:
                    raise RuntimeError(f"live generation owner 已存在: {plugin_id}")
                self._active_generations[plugin_id] = generation
                self._building_roots[root] = tuple(generations.values())
            elif validation_host is not None:
                validation_host.generations = tuple(generations.values())
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
        validation_host: ValidationHost | None = None,
    ) -> CompositionRoot:
        """同一归档组合在每个环境重新实例化，返回实际挂载的 generation 表。"""

        sources = dict(sorted(generations.items()))
        if components is None:
            components = tuple(self._generation_archive_ref(item) for item in sources.values())
        root = CompositionRoot(
            "plugins:" + secrets.token_hex(16),
            candidate_incident_limit=1024 if candidate_owner is not None else None,
        )
        if validation_host is None:
            self._building_roots[root] = ()
        else:
            validation_host.root = root
        store = self._snapshot_store if validation_host is None else validation_host.snapshot_store
        root._bind_runtime_scope_acquirer(
            lambda: store.acquire_composition_root(root)
        )
        workspace = self._workspace if validation_host is None else validation_host.workspace
        if candidate_owner is not None:
            workspace = self._workspace / "runtime" / "plugin-validation" / secrets.token_hex(16) / "workspace"
        try:
            actual = self._archived_generations(
                components, root, workspace=workspace, sources=sources, validation_host=validation_host,
            )
            generations.clear()
            generations.update(actual)
            ordered = tuple(actual.values())
            for item in ordered:
                await self._load_selected_generation(item)
            # 数据初始化由实际插件完成；每个实例只获得自己环境的数据目录。
            for item in ordered:
                ensure_workspace_plugin_data_dir(item.data_dir, workspace)
            await self._provide_composition_services(
                root, ordered, candidate=candidate_owner is not None, validation_host=validation_host,
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
            # 验证构建从分配 Root 起就归 host；由外层同一次失败路径清理。
            if validation_host is None:
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

    async def _provide_composition_services(
        self,
        root: CompositionRoot,
        mount_order: tuple[PluginGeneration, ...],
        *,
        candidate: bool,
        validation_host: ValidationHost | None = None,
    ) -> None:
        """向当前 stable 或 candidate Root 提供宿主能力。"""

        host = validation_host
        isolated = host is not None
        store = self._snapshot_store if host is None else host.snapshot_store
        boot_id = self._host_boot_id if host is None else host.identity
        archive = self._archive if host is None else host.archive
        message_log = self._message_log if host is None else host.messages
        tasks = self._plugin_tasks if host is None else host.tasks
        processes = self._plugin_processes if host is None else host.processes
        channel_store = self._channel_attachment_store if host is None else host.attachments
        artifact_read = self._artifact_read if host is None else host.artifact_read
        artifact_import = self._artifact_import if host is None else host.artifact_import
        restart_gate = self._restart_gate if host is None else host.restart_gate
        control_frames = self._control_frames if host is None else host.control_frames
        workspace_id = self._workload_workspace_id if host is None else host.workspace_id
        interaction_owner = self._interaction_undo if host is None else None

        await root.context.provide(
            HOST_INFO,
            HostInfo(
                boot_id=boot_id,
                validation=candidate or isolated,
            ),
        )
        await root.context.provide(
            DASHBOARD_ROUTES,
            () if self._dashboard_routes is None else self._dashboard_routes,
        )
        # 隔离装配保留能力形状，但所有 I/O 都显式拒绝，不取得正式对象。
        custody = None if candidate else (self._input_custody if host is None else host.input_custody)
        await root.context.provide(INPUT_CUSTODY,
            unavailable_input_custody() if custody is None else custody)
        if candidate:
            identity = ChannelIdentity(unavailable, unavailable, unavailable)
        elif host is not None:
            identity = host.channel_identity
        elif self._channel_identities is None:
            identity = ChannelIdentity(unavailable, unavailable, unavailable)
        else:
            identity = ChannelIdentity(
                self._resolve_channel_identity, self._remember_channel_identity,
                self._rollback_channel_identity,
            )
        await root.context.provide(CHANNEL_IDENTITY, identity)
        attachments = None if candidate else channel_store
        await root.context.provide(CHANNEL_ATTACHMENT_IMPORT, ChannelAttachmentImport(
            unavailable if attachments is None else attachments.import_bytes,
        ))
        await root.context.provide(CHANNEL_ATTACHMENT_READ, ChannelAttachmentRead(
            unavailable if attachments is None else attachments.resolve_refs,
            unavailable if attachments is None else attachments.acquire,
        ))
        execution = ExecutionAccess(root.instance_token, {
            (item.plugin_id, item.generation_id): CodeOwner(item.generation_id, item.code_dir,
                lambda command, cwd, item=item: self._resolve_runtime_command(item, command, cwd, archive=archive))
            for item in mount_order
        }, candidate=candidate or isolated)
        await root.context.provide(EXECUTION, execution)
        if not candidate and not isolated and root is self._live_root:
            self._live_execution_access = execution
        await root.context.provide(WORKLOAD_CONTROLLER,
            ControllerAccess(execution, self._workload_controller, workspace_id))
        requested = {
            key
            for generation in mount_order
            for key in cast(ComposablePlugin, generation.instance).inject
        }
        if not candidate and not isolated:
            # Stable host facades are installed once for the live Root so a
            # later local generation is not rejected merely because boot had
            # no consumer for a lazy capability.
            requested.update({
                RUNTIME_CATALOG, PLUGIN_UPDATES, RESTART_GATE,
                CONTROL_FRAMES, PROCESSES, TIMERS,
                ServiceKey[object]("core.message_display.v1"),
                ServiceKey[object]("core.mobile_ui.v1"),
            })
            if artifact_import is not None:
                requested.add(ARTIFACT_IMPORT)
            if interaction_owner is not None:
                requested.add(INTERACTION_UNDO)
        if RUNTIME_CATALOG in requested:
            if candidate or isolated or root is not self._live_root:
                raise RuntimeError(
                    "runtime catalog 只在正式 live Root 提供，不为 candidate 或 isolated Root 提供"
                )

            def read_runtime_catalog(context: Context) -> dict[str, object]:
                """Read live runtime facts only from the exact owner scope."""

                if context.root_instance_token is not root.instance_token:
                    raise RuntimeError("runtime catalog 不属于当前 live Root")
                if RUNTIME_CATALOG not in context._declared_dependencies():
                    raise CompositionError(
                        "UNDECLARED_SERVICE",
                        "当前 Fiber 未声明 runtime catalog 依赖",
                    )
                context.require_runtime_owner(RUNTIME_CATALOG, read_runtime_catalog)
                return build_runtime_catalog(
                    root,
                    self._active_generations,
                    self._draining_generations,
                )

            _ = await root.context.provide(RUNTIME_CATALOG, read_runtime_catalog)
        if CREDENTIALS in requested or (not candidate and not isolated):
            clients = CredentialClients(None if candidate or isolated else {
                (generation.plugin_id, generation.generation_id): CoreProviderClientFactory(
                    generation.data_dir,
                    generation.config_projection, generation.config_revision,
                )
                for generation in mount_order
            })
            _ = await root.context.provide(CREDENTIALS, clients)
            root._defer_internal_cleanup("credential_clients", clients.aclose)  # pyright: ignore[reportPrivateUsage]
            if not candidate and not isolated and root is self._live_root:
                self._live_credentials = clients
        if PLUGIN_UPDATES in requested:
            _ = await root.context.provide(
                PLUGIN_UPDATES, PluginUpdates(None if candidate or isolated else self),
            )
        message_services: set[ServiceKey[object]] = {
            MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION, BINDINGS
        }
        if RESTART_GATE in requested:
            gate = restart_gate
            if candidate:
                gate = RestartGate(
                    boot_id="candidate",
                    supervised=gate is not None and gate.supervised,
                    execution_enabled=False,
                )
            elif gate is None:
                # 直接使用 PluginManager 的测试/嵌入式运行没有 Supervisor；仍提供
                # 一个允许正常 work 的 unmanaged gate，不伪造可提交的重启通道。
                gate = RestartGate(boot_id=boot_id, supervised=False)
                self._restart_gate = gate
            _ = await root.context.provide(RESTART_GATE, gate)
        if CONTROL_FRAMES in requested:
            # 候选与独立验证只读取副本；它们不能取得正式连接的发送 owner。
            frames = FrameBook() if candidate else control_frames
            _ = await root.context.provide(CONTROL_FRAMES, frames)
            if candidate:
                root._defer_internal_cleanup("control_frames.close", frames.close)  # pyright: ignore[reportPrivateUsage]
        # 正式能力归宿主所有，不计入历史 provider 的依赖拓扑。
        log = None if candidate else message_log
        if requested & message_services and message_log is None:
            raise RuntimeError("消息能力需要 bootstrap 提供已迁移的 MessageLog")
        if message_log is not None:
            _ = await root.context.provide(MESSAGE_CATALOG, MessageCatalog(log))
            _ = await root.context.provide(MESSAGE_EMBEDDINGS, MessageEmbeddings(log))
            _ = await root.context.provide(MESSAGE_WRITERS, MessageWriters(log))
            _ = await root.context.provide(OWNER_STATE, OwnerState(log))
            _ = await root.context.provide(SESSION_ADMISSION, SessionAdmission(log))
            _ = await root.context.provide(
                BINDINGS, Bindings(log, archive, root, self._generation_for_context)
            )
        if TASKS in requested or message_log is not None:
            _ = await root.context.provide(
                TASKS, PluginTasks(formal=False) if candidate else tasks
            )
        if PROCESSES in requested:
            _ = await root.context.provide(
                PROCESSES, PluginProcesses(formal=False) if candidate else processes
            )
        if artifact_read is not None:
            _ = await root.context.provide(
                ARTIFACT_READ, ArtifactRead(None) if candidate else artifact_read
            )
        if ARTIFACT_IMPORT in requested and artifact_import is not None:
            _ = await root.context.provide(
                ARTIFACT_IMPORT, ArtifactImport(None) if candidate else artifact_import
            )
        if TIMERS in requested:
            timers = (
                PluginTimers(AsyncioOneShotTimer())
                if not candidate else PluginTimers.candidate_validation()
            )
            _ = await root.context.provide(TIMERS, timers)

        # Client UI and message display are neutral projections.  The host
        # publishes stable names; each display request opens only its provider
        # Context scope while retaining the same live Root.
        host_ui_requested = {
            key.name
            for key in requested
            if key.name in {
                "core.message_display.v1",
                "core.mobile_ui.v1",
            }
        }
        if "core.message_display.v1" in host_ui_requested:
            from agent.plugin_composition.message_view import project_message_rows

            async def display_message_page(
                page: MessagePage,
                *,
                display_only: bool,
            ) -> list[dict[str, object]]:
                return await project_message_rows(
                    root,
                    page,
                    display_only=display_only,
                )

            _ = await root.context.provide(
                ServiceKey[object]("core.message_display.v1"),
                display_message_page,
            )
        if "core.mobile_ui.v1" in host_ui_requested:
            from agent.plugins.mobile_ui import PluginMobileUiProvider

            mobile_ui = PluginMobileUiProvider(root)
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
            if candidate or isolated:
                interaction_undo = InteractionUndoService.candidate_validation()
            else:
                if interaction_owner is None:
                    raise RuntimeError("INTERACTION_UNDO 需要 Session owner")
                interaction_undo = InteractionUndoService(interaction_owner.undo_latest)
            _ = await root.context.provide(INTERACTION_UNDO, interaction_undo)

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
        try:
            generation.fiber = await root._mount_module(  # pyright: ignore[reportPrivateUsage]
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
        except BaseException:
            # Keep the exact runtime-identified Fiber if the kernel retained it.
            generation.fiber = next(
                (
                    fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
                    if fiber.runtime is not None
                    and fiber.runtime.plugin_id == generation.plugin_id
                    and fiber.runtime.generation_id == generation.generation_id
                ),
                None,
            )
            raise




    def _resolve_runtime_command(
        self,
        generation: PluginGeneration,
        command: tuple[str, ...],
        cwd: str,
        *,
        archive: PluginArchive | None = None,
    ) -> tuple[str, ...]:
        """仅在实际打开目标前校验其环境，不阻挡同组件的纯读取能力。"""
        manifest = generation.static_manifest
        runtimes = () if manifest is None else manifest.python
        runtime_root = command_python_runtime(generation.code_dir, command, cwd, runtimes)
        environment = None
        if runtime_root is not None:
            if generation.archive_ref is None:
                raise RuntimeError("外部 runtime 缺少代码归档")
            record = (self._archive if archive is None else archive).read_descriptor(generation.archive_ref)
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
        candidate = None
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
            details={"base_selection_ref": self._selection.read()},
        )
        generation.reload_tx_id = tx_id
        boot_id = os.environ.get("AKASHIC_BOOT_ID", "").strip()
        if boot_id:
            self._reload_journal.mark_runtime_owner(tx_id, boot_id)
        return tx_id

    def _composition_recovery_target(
        self, generation: PluginGeneration, *, tx_id: str | None = None,
    ) -> RecoveryTarget:
        """仅用于故障诊断；恢复输入始终从唯一完整 selection 读取。"""
        publication = self._publication
        if publication is not None and publication.must_retain:
            return "candidate" if publication.candidate.generations.get(generation.plugin_id) is generation else "base"
        ref = self._selection.read()
        if ref is not None and generation.archive_ref in self._selection_components(ref):
            return "candidate"
        return "base"


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
        _, cancelled = await _complete_critical(self._plugin_processes.close())
        externally_cancelled = externally_cancelled or cancelled
        # The local path drains child Fibers first, then closes the one formal Root.
        live_root = self._live_root
        for generation in tuple(reversed(tuple(self._active_generations.values()))):
            _, cancelled = await _complete_critical(
                self._dispose_generation(generation, state="retired")
            )
            externally_cancelled = externally_cancelled or cancelled
        for generations in tuple(self._draining_generations.values()):
            for generation in tuple(generations):
                _, cancelled = await _complete_critical(
                    self._dispose_generation(generation, state="retired")
                )
                externally_cancelled = externally_cancelled or cancelled
        _, cancelled = await _complete_critical(self._plugin_tasks.close())
        externally_cancelled = externally_cancelled or cancelled
        if live_root is not None:
            _, cancelled = await _complete_critical(live_root.dispose())
            externally_cancelled = externally_cancelled or cancelled
            self._live_root = None
            self._live_execution_access = None
            self._live_credentials = None
        # 未交接 Root 先释放；SnapshotStore 和 generation 随后才能移除其依赖。
        for root in tuple(self._building_roots):
            await self._close_building_root(root)
        pending = self._snapshot_store.pending_transaction
        if pending is not None:
            await self._snapshot_store.abort(pending, reopen_previous=False)
        # 2. 关闭当前 generation admission，再完成快照回收。
        for generation in self._active_generations.values():
            self._retire_generation(generation)
        _, snapshot_cancelled = await _complete_critical(self._snapshot_store.close())
        externally_cancelled = externally_cancelled or snapshot_cancelled
        self._ready_candidate = None
        # 3. 快照之外的失败 generation 仍由排空集合持有，显式关闭时重试。
        for tracked in tuple(self._draining_generations.values()):
            for generation in tuple(tracked):
                _, cancelled = await _complete_critical(
                    self._dispose_generation(generation, state="retired")
                )
                externally_cancelled = externally_cancelled or cancelled
        # 4. 导入前的 Scope 也由前面的 building Root 关闭，不绕过 Root 扫尾。
        if self._owns_control_frames:
            self._control_frames.close()
        if externally_cancelled:
            raise asyncio.CancelledError


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




def _source_failure_key(failure: PluginSourceFailure) -> str:
    """Key one source diagnostic by normalized root and source kind."""
    return _source_failure_key_for_root(failure.source_root, failure.source_type)


def _source_failure_key_for_root(
    source_root: Path,
    source_type: Literal["builtin", "installed"],
) -> str:
    """Build a diagnostic key without fabricating a failure record."""
    return f"{source_type}:{source_root.resolve(strict=False)}"


def _source_failure_key_for_mod(mod: Mapping[str, str]) -> str:
    """Build the diagnostic key used by a discovered source module."""
    return _source_failure_key_for_root(
        Path(mod["plugin_root"]).resolve(strict=False),
        cast(Literal["builtin", "installed"], mod["source_type"]),
    )


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
