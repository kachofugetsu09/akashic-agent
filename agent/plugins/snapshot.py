from __future__ import annotations

import asyncio
import hashlib
from contextlib import aclosing, asynccontextmanager, suppress
from contextvars import ContextVar, Token
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, cast

from agent.control.scoped_turn import TurnAdmissionRetiredError
from agent.plugin_composition.effect import _join_cleanup

from agent.plugins.generation import PluginGeneration
from agent.plugins.selection import SelectionWriteError
from agent.plugin_composition import (
    CHANNELS,
    CompositionRoot,
    CompositionError,
    TopologyView,
)

if TYPE_CHECKING:
    from session.log import MessagePage
from agent.plugin_composition.channels import (
    ChannelFactoryFreezeInput,
    ChannelRegistrySnapshot,
    CommittedChannelCatalog,
    CoreChannelDefinition,
    _freeze_plugin_channels,
    channel_config_revision,
)

SnapshotState = Literal[
    "compiled",
    "validating",
    "committed",
    "aborted",
    "retired",
]
RuntimeSelector = Literal["stable", "latest"]


class _ReplyStatusReader(Protocol):
    """窄读取合同；回复插件实现仍由当前 composition root 提供。"""

    def follow(
        self,
        session_id: str,
    ) -> AsyncGenerator[tuple[dict[str, object], ...], None]: ...


@dataclass
class RuntimeSnapshot:
    snapshot_id: str
    generations: Mapping[str, PluginGeneration]
    channel_registry: ChannelRegistrySnapshot | None = None
    channel_registry_identity: str | None = None
    channel_catalog: CommittedChannelCatalog | None = None
    composition_root: CompositionRoot | None = None
    composition_topology: TopologyView | None = None
    composition_active_plugin_ids: frozenset[str] | None = None
    state: SnapshotState = "compiled"
    lease_count: int = 0
    accepting_leases: bool = True
    _store_token: object | None = field(default=None, repr=False)

    def active_generations(self) -> tuple[PluginGeneration, ...]:
        if self.generations and self.composition_active_plugin_ids is None:
            raise RuntimeError("RuntimeSnapshot 缺少 Root active plugin projection")
        active_plugin_ids = self.composition_active_plugin_ids or frozenset()
        return tuple(
            generation
            for generation in self.generations.values()
            if generation.plugin_id in active_plugin_ids
        )

    def claim(self, store_token: object) -> None:
        if (
            self.state != "compiled"
            or self.lease_count
            or self._store_token is not None
        ):
            raise RuntimeError("RuntimeSnapshot 不是可发布的全新 compiled 快照")
        self._store_token = store_token


@dataclass
class SnapshotTransaction:
    previous: RuntimeSnapshot | None
    candidate: RuntimeSnapshot
    selection_result: str | SelectionWriteError | None = None

    @property
    def must_retain(self) -> bool:
        """持久提交成功或结果不确定时，必须保留真实的新 owner。"""
        result = self.selection_result
        return isinstance(result, str) or (
            isinstance(result, SelectionWriteError) and result.outcome == "uncertain"
        )


class RuntimeSnapshotCompiler:
    def compile(
        self,
        generations: Mapping[str, PluginGeneration],
        *,
        catalog_generation: PluginGeneration | None = None,
        snapshot_revision: str = "",
        composition_root: CompositionRoot | None = None,
        core_channel_definitions: tuple[CoreChannelDefinition, ...] = (),
        require_composition_ready: bool = True,
    ) -> RuntimeSnapshot:
        ordered = [generations[key] for key in sorted(generations)]
        if any(generation.plugin_id != key for key, generation in generations.items()):
            raise RuntimeError("RuntimeSnapshot generation key 与 plugin_id 不一致")
        composition_topology: TopologyView | None = None
        composition_active_plugin_ids: frozenset[str] | None = None
        channel_registry: ChannelRegistrySnapshot | None = None
        channel_catalog: CommittedChannelCatalog | None = None
        if composition_root is not None:
            catalog_root_token = composition_root.instance_token
            catalog_context = composition_root.context
            receipt = composition_root.receipt()
            if require_composition_ready and not receipt.ready:
                raise RuntimeError(
                    "RuntimeSnapshot 插件组合拓扑未就绪: "
                    f"required_pending={receipt.required_pending}, "
                    f"required_degraded={receipt.required_degraded}, "
                    f"incident_overflowed={receipt.incident_overflowed}, "
                    f"external_effects={receipt.external_effects}"
                )
            composition_topology = composition_root.topology_view()
            composition_active_plugin_ids = composition_root.active_plugin_ids()
            channel_declarations = catalog_context.get(CHANNELS)
            if channel_declarations is not None:
                channel_registry = _freeze_plugin_channels(
                    channel_declarations,
                    catalog_root_token,
                    factory_provenance_by_owner={
                        generation.plugin_id: ChannelFactoryFreezeInput(
                            generation_id=generation.generation_id,
                            source_revision=generation.source_revision,
                            config_revision=channel_config_revision(
                                generation.config_projection
                            ),
                        )
                        for generation in ordered
                    },
                )
            assert composition_active_plugin_ids is not None
            self._validate_channel_registry(
                channel_registry,
                generations,
            )
        if core_channel_definitions:
            channel_catalog = CommittedChannelCatalog(
                plugin_registry=channel_registry,
                core_definitions=tuple(core_channel_definitions),
                root_instance_token=(
                    None
                    if composition_root is None
                    else composition_root.instance_token
                ),
            )
        canonical_identity = "|".join(
            (
                *(
                    f"{item.plugin_id}:{item.generation_id}:{item.source_revision}:{item.config_revision}"
                    for item in ordered
                ),
                f"snapshot:{snapshot_revision}",
                "root:" + (
                    "" if composition_root is None
                    else f"{composition_root.generation_id}:{id(composition_root.instance_token)}"
                ),
                "composition:"
                + (
                    ""
                    if composition_topology is None
                    else composition_topology.identity
                ),
                "channels:"
                + ("" if channel_registry is None else channel_registry.identity),
                "channel-catalog:"
                + ("" if channel_catalog is None else channel_catalog.identity),
            )
        )
        snapshot_id = hashlib.sha256(canonical_identity.encode()).hexdigest()[:16]
        snapshot = RuntimeSnapshot(
            snapshot_id=snapshot_id,
            generations=MappingProxyType(dict(generations)),
            channel_registry=channel_registry,
            channel_registry_identity=(
                None if channel_registry is None else channel_registry.identity
            ),
            channel_catalog=channel_catalog,
            composition_root=composition_root,
            composition_topology=composition_topology,
            composition_active_plugin_ids=composition_active_plugin_ids,
        )
        if composition_root is not None:
            composition_root.freeze()
        return snapshot

    @staticmethod
    def _validate_channel_registry(
        registry: ChannelRegistrySnapshot | None,
        generations: Mapping[str, PluginGeneration],
    ) -> None:
        """确认每个渠道归属于当前组合中的插件。"""

        for descriptor in () if registry is None else registry.descriptors:
            generation = generations.get(descriptor.owner)
            if generation is None:
                raise RuntimeError(
                    "RuntimeSnapshot channel owner 不属于 generations: "
                    f"{descriptor.owner}"
                )

# 插件生命周期边界：一个 turn、job、event 或 proactive tick 必须始终使用同一
# snapshot；旧 generation 只有在全部 lease 释放后才能 retire 和清理。
class RuntimeSnapshotLease:
    def __init__(
        self,
        store: RuntimeSnapshotStore,
        snapshot: RuntimeSnapshot,
        validation_candidate_plugin_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._store = store
        self.snapshot = snapshot
        self.validation_candidate_plugin_ids = validation_candidate_plugin_ids
        self._released = False

    @property
    def active(self) -> bool:
        return not self._released

    def fork(self) -> RuntimeSnapshotLease:
        return self._store.fork_lease(self)

    async def __aenter__(self) -> RuntimeSnapshot:
        return self.snapshot

    async def __aexit__(self, *exc_info: object) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._store.release_lease(self.snapshot)


@dataclass(frozen=True)
class _RuntimeSnapshotBinding:
    lease: RuntimeSnapshotLease
    owner_task: asyncio.Task[object] | None


_current_runtime_binding: ContextVar[_RuntimeSnapshotBinding | None] = ContextVar(
    "current_runtime_binding",
    default=None,
)


def bind_runtime_snapshot(
    lease: RuntimeSnapshotLease,
) -> Token[_RuntimeSnapshotBinding | None]:
    return _current_runtime_binding.set(
        _RuntimeSnapshotBinding(
            lease=lease,
            owner_task=asyncio.current_task(),
        )
    )


def reset_runtime_snapshot(token: Token[_RuntimeSnapshotBinding | None]) -> None:
    _current_runtime_binding.reset(token)


def get_current_runtime_snapshot() -> RuntimeSnapshot | None:
    binding = _current_runtime_binding.get()
    if (
        binding is None
        or not binding.lease.active
        or binding.owner_task is not asyncio.current_task()
    ):
        return None
    return binding.lease.snapshot


def get_lifecycle_runtime_snapshot() -> RuntimeSnapshot | None:
    """Resolve a lifecycle snapshot while rejecting inherited or stale bindings."""

    binding = _current_runtime_binding.get()
    if binding is None:
        return None
    if binding.owner_task is not asyncio.current_task():
        raise CompositionError(
            "RUNTIME_SNAPSHOT_BINDING_MISMATCH",
            "lifecycle 必须在绑定 RuntimeSnapshot lease 的 owner task 中运行",
        )
    if not binding.lease.active:
        raise CompositionError(
            "RUNTIME_SNAPSHOT_BINDING_INACTIVE",
            "lifecycle 不能使用已释放的 RuntimeSnapshot lease",
        )
    return binding.lease.snapshot


def lease_current_runtime_snapshot() -> RuntimeSnapshotLease | None:
    lease = get_current_runtime_lease()
    return lease.fork() if lease is not None else None


@asynccontextmanager
async def lease_runtime_snapshot(
    store: "RuntimeSnapshotStore",
) -> AsyncIterator[RuntimeSnapshot]:
    """Lease and bind one runtime snapshot for the current task."""

    lease = lease_current_runtime_snapshot()
    token: Token[_RuntimeSnapshotBinding | None] | None = None
    if lease is None:
        lease = await store.acquire()
        token = bind_runtime_snapshot(lease)
    try:
        yield lease.snapshot
    finally:
        if token is not None:
            reset_runtime_snapshot(token)
        await lease.release()


async def project_message_rows(
    store: "RuntimeSnapshotStore",
    page: object,
    *,
    display_only: bool,
) -> list[dict[str, object]]:
    """Project one message page through the exact current snapshot lease."""

    from agent.plugin_composition.message_view import (
        MessageDisplayProviders,
        PartDisplayProvider,
        message_rows,
    )
    from agent.plugin_composition.model import ServiceKey
    from session.log import MessagePage

    if not isinstance(page, MessagePage):
        raise TypeError("消息展示需要 MessagePage")
    async with lease_runtime_snapshot(store) as snapshot:
        root = snapshot.composition_root
        if root is None:
            raise RuntimeError("消息展示需要已发布的插件 Root")
        providers = root.provided_services()
        renderers = {
            key.name.removeprefix("message.display:"): cast(PartDisplayProvider, value)
            for key, value in providers.items()
            if key.name.startswith("message.display:")
            and callable(value)
        }
        tool_name = root.context.get(
            ServiceKey[Callable[[str], str]]("tools.display-name.v1")
        )
        return message_rows(
            page,
            display_only=display_only,
            providers=MessageDisplayProviders(
                tool_name=tool_name,
                part_display=renderers,
            ),
        )


async def follow_reply_status(
    store: "RuntimeSnapshotStore",
    session_id: str,
) -> AsyncGenerator[dict[str, object], None]:
    """Resolve the reply reader briefly; a long follow must not pin a generation."""

    from agent.plugin_composition.model import ServiceKey

    while True:
        # 在精确租约内解析 provider，等待通知前释放租约。
        # generation 停止时 provider 会关闭只读 follower，唤醒循环解析新代。
        async with lease_runtime_snapshot(store) as snapshot:
            root = snapshot.composition_root
            if root is None:
                raise RuntimeError("回复状态需要已发布的插件 Root")
            read = root.context.get(ServiceKey[object]("reply.status.v2"))
            base: dict[str, object] = {
                "version": 2,
                "session_id": session_id,
                "snapshot_id": snapshot.snapshot_id,
            }

        if read is None:
            changed = asyncio.create_task(store.wait_for_stable_change(snapshot))
            try:
                yield {**base, "available": False, "items": []}
                await changed
            finally:
                if not changed.done():
                    changed.cancel()
                with suppress(asyncio.CancelledError):
                    await changed
            continue

        follow = getattr(read, "follow", None)
        if not callable(follow):
            raise TypeError("reply.status.v2 provider 缺少 follow(session_id)")
        follower = cast(_ReplyStatusReader, read).follow(session_id)
        changed = asyncio.create_task(store.wait_for_stable_change(snapshot))
        pending: asyncio.Task[tuple[dict[str, object], ...]] | None = None
        try:
            while store.current is snapshot:
                pending = asyncio.create_task(anext(follower))
                done, _ = await asyncio.wait(
                    (pending, changed),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if changed in done:
                    _ = changed.result()
                    break
                try:
                    items = pending.result()
                except StopAsyncIteration:
                    yield {**base, "available": False, "items": []}
                    await changed
                    break
                yield {**base, "available": True, "items": list(items)}
                pending = None
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await pending
            if not changed.done():
                changed.cancel()
            with suppress(asyncio.CancelledError):
                await changed
            async with aclosing(follower):
                pass


def get_current_runtime_lease() -> RuntimeSnapshotLease | None:
    binding = _current_runtime_binding.get()
    if (
        binding is None
        or not binding.lease.active
        or binding.owner_task is not asyncio.current_task()
    ):
        return None
    return binding.lease


class RuntimeSnapshotStore:
    def __init__(
        self,
        on_drained: Callable[[RuntimeSnapshot], Awaitable[None]] | None = None,
    ) -> None:
        self._current: RuntimeSnapshot | None = None
        self._latest: RuntimeSnapshot | None = None
        self._snapshots: dict[str, RuntimeSnapshot] = {}
        self._pending: SnapshotTransaction | None = None
        self._provisional: SnapshotTransaction | None = None
        self._on_drained = on_drained
        self._token = object()
        self._condition = asyncio.Condition()
        self._drain_tasks: dict[str, asyncio.Task[None]] = {}
        self._drain_failures: dict[str, BaseException] = {}

    @property
    def current(self) -> RuntimeSnapshot | None:
        return self._current

    async def wait_for_stable_change(
        self,
        current: RuntimeSnapshot,
    ) -> RuntimeSnapshot:
        """Wait until another committed snapshot becomes the stable owner."""

        async with self._condition:
            await self._condition.wait_for(lambda: self._current is not current)
            if self._current is None:
                raise RuntimeError("RuntimeSnapshot stable owner 不可为空")
            return self._current

    async def wait_for_snapshot_drained(self, snapshot: RuntimeSnapshot) -> None:
        """Wait until one retired snapshot finishes its exact drain callback."""

        async with self._condition:
            await self._condition.wait_for(
                lambda: (
                    snapshot.snapshot_id not in self._snapshots
                    or snapshot.snapshot_id in self._drain_failures
                )
            )
            failure = self._drain_failures.get(snapshot.snapshot_id)
            if failure is not None:
                raise failure

    @property
    def latest(self) -> RuntimeSnapshot | None:
        return self._latest or self._current

    @property
    def unpromoted_candidate(self) -> RuntimeSnapshot | None:
        latest = self.latest
        return latest if latest is not self._current else None

    @property
    def pending_candidate(self) -> RuntimeSnapshot | None:
        if self._pending is None:
            return None
        return self._pending.candidate

    @property
    def pending_transaction(self) -> SnapshotTransaction | None:
        """Expose the exact pending owner for its caller's failure cleanup."""

        return self._pending

    @property
    def retained_snapshot_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshots))

    def generation_is_referenced_elsewhere(
        self,
        generation: PluginGeneration,
        *,
        excluding_snapshot_id: str,
    ) -> bool:
        return any(
            snapshot.snapshot_id != excluding_snapshot_id
            and (
                snapshot.state in {"validating", "committed"}
                or snapshot.lease_count > 0
            )
            and any(item is generation for item in snapshot.generations.values())
            for snapshot in self._snapshots.values()
        )

    def composition_is_referenced_elsewhere(
        self,
        root: CompositionRoot,
        *,
        excluding_snapshot_id: str,
    ) -> bool:
        return any(
            snapshot.snapshot_id != excluding_snapshot_id
            and (
                snapshot.state in {"validating", "committed"}
                or snapshot.lease_count > 0
            )
            and snapshot.composition_root is root
            for snapshot in self._snapshots.values()
        )

    def install(self, snapshot: RuntimeSnapshot) -> None:
        if (
            self._current is not None
            or self._pending is not None
            or self._provisional is not None
        ):
            raise RuntimeError("RuntimeSnapshotStore 已安装初始快照")
        self._validate_composition(snapshot)
        self._adopt(snapshot)
        snapshot.state = "committed"
        self._current = snapshot
        self._latest = snapshot
        self._snapshots[snapshot.snapshot_id] = snapshot

    def begin_publish(
        self,
        candidate: RuntimeSnapshot,
    ) -> SnapshotTransaction:
        if self._pending is not None or self._provisional is not None:
            raise RuntimeError("已有 RuntimeSnapshot 发布事务")
        if self.unpromoted_candidate is not None:
            raise RuntimeError("已有 RuntimeSnapshot 候选等待 promote/discard")
        if candidate.snapshot_id in self._snapshots:
            raise RuntimeError(f"RuntimeSnapshot 已存在: {candidate.snapshot_id}")
        self._validate_composition(candidate)
        self._adopt(candidate)
        transaction = SnapshotTransaction(previous=self._current, candidate=candidate)
        candidate.state = "validating"
        candidate.accepting_leases = False
        self._snapshots[candidate.snapshot_id] = candidate
        self._pending = transaction
        return transaction

    async def commit(
        self,
        transaction: SnapshotTransaction,
        *,
        before_open: Callable[[], None] | None = None,
        after_open: Callable[[], None] | None = None,
    ) -> None:
        self._require_pending(transaction)
        self._validate_composition(transaction.candidate)
        try:
            if before_open is not None:
                before_open()
        except BaseException:
            if transaction.must_retain:
                self.hold_failed_publication(transaction)
            raise
        transaction.candidate.state = "committed"
        self._current = transaction.candidate
        self._latest = transaction.candidate
        self._pending = None
        previous = transaction.previous
        try:
            if after_open is not None:
                after_open()
        except BaseException:
            if transaction.must_retain:
                self.hold_failed_publication(transaction)
            raise
        transaction.candidate.accepting_leases = True
        if previous is not None:
            previous.state = "retired"
            self._schedule_drain(previous)
        async with self._condition:
            self._condition.notify_all()

    async def commit_latest(
        self,
        transaction: SnapshotTransaction,
        *,
        before_open: Callable[[], None] | None = None,
    ) -> None:
        """Publish a validation candidate without changing the stable pointer."""

        # 1. Open only the explicitly selected candidate.
        self._require_pending(transaction)
        self._validate_composition(transaction.candidate)
        if before_open is not None:
            before_open()
        transaction.candidate.state = "committed"
        transaction.candidate.accepting_leases = True
        self._latest = transaction.candidate
        self._pending = None

        # 2. Wake latest waiters while stable readers stay on the previous snapshot.
        async with self._condition:
            self._condition.notify_all()

    async def commit_provisional(
        self,
        transaction: SnapshotTransaction,
    ) -> None:
        """Stage a closed candidate without exposing it as the published stable."""

        # 1. Validate and close both sides before the external publication step.
        self._require_pending(transaction)
        self._validate_composition(transaction.candidate)
        transaction.candidate.state = "committed"
        transaction.candidate.accepting_leases = False
        if transaction.previous is not None:
            transaction.previous.accepting_leases = False

        # 2. Keep discovery pinned to the old stable while retaining the target.
        self._latest = transaction.candidate
        self._pending = None
        self._provisional = transaction
        async with self._condition:
            self._condition.notify_all()

    async def finalize_provisional(
        self,
        transaction: SnapshotTransaction,
        *,
        before_open: Callable[[], None] | None = None,
        after_open: Callable[[], None] | None = None,
        schedule_previous_drain: bool = True,
    ) -> None:
        """Open a provisional stable and retire its rollback snapshot."""

        # 1. Complete fallible projection work while the old stable stays visible.
        self._require_provisional(transaction)
        self._validate_composition(transaction.candidate)
        try:
            if before_open is not None:
                before_open()
        except BaseException:
            if transaction.must_retain:
                self.hold_failed_publication(transaction)
            raise

        # 2. Switch the stable pointer synchronously around the owner callback.
        transaction.candidate.state = "committed"
        previous = transaction.previous
        self._current = transaction.candidate
        try:
            if after_open is not None:
                after_open()
        except BaseException:
            if transaction.must_retain:
                self.hold_failed_publication(transaction)
            else:
                self._current = previous
            raise

        # 3. Open the new stable only after all publication work succeeded.
        transaction.candidate.accepting_leases = True
        if previous is not None:
            previous.state = "retired"
            previous.accepting_leases = False
        self._provisional = None
        if previous is not None and schedule_previous_drain:
            self._schedule_drain(previous)
        async with self._condition:
            self._condition.notify_all()

    def schedule_retired_drain(self, snapshot: RuntimeSnapshot) -> None:
        """Schedule a retired snapshot only after post-publication work succeeds."""

        if self._current is snapshot or snapshot.state != "retired":
            raise RuntimeError("只能排空已退役且不再为 stable 的 RuntimeSnapshot")
        self._schedule_drain(snapshot)

    def hold_failed_publication(self, transaction: SnapshotTransaction) -> None:
        """保留新物理 owner 供显式关闭；closed current 不声称磁盘写入已确认。"""
        if not transaction.must_retain:
            raise RuntimeError("没有需要保留的持久写入结果")
        if self._snapshots.get(transaction.candidate.snapshot_id) is not transaction.candidate:
            raise RuntimeError("发布 owner 已丢失")
        self._current = self._latest = transaction.candidate
        transaction.candidate.accepting_leases = False
        transaction.candidate.state = "committed"
        if self._pending is transaction:
            self._pending = None
        if self._provisional is transaction:
            self._provisional = None
        if transaction.previous is not None:
            transaction.previous.state = "retired"
            transaction.previous.accepting_leases = False

        # 失败路径只保留 owner；显式 close/retry_drains 再尝试旧资源清理。

    async def rollback_provisional(
        self,
        transaction: SnapshotTransaction,
        *,
        keep_candidate_latest: bool,
        reopen_previous: bool = True,
    ) -> None:
        """Restore the previous stable before disposing or retrying the candidate."""

        # 1. Reopen the old pointer; the candidate was never publicly current.
        self._require_provisional(transaction)
        if transaction.must_retain:
            raise RuntimeError("stable 已提交或结果不确定，不能恢复旧 snapshot")
        candidate = transaction.candidate
        previous = transaction.previous
        if self._current is not previous:
            raise RuntimeError("RuntimeSnapshot provisional stable 指针已漂移")
        if previous is not None:
            previous.state = "committed"
            previous.accepting_leases = reopen_previous
        self._provisional = None

        # 2. Either retain latest for normal discard or restore the pending transaction.
        candidate.accepting_leases = False
        if keep_candidate_latest:
            candidate.state = "committed"
            self._latest = candidate
        else:
            candidate.state = "validating"
            self._latest = previous
            self._pending = transaction
        async with self._condition:
            self._condition.notify_all()

    async def rollback_published(
        self,
        transaction: SnapshotTransaction,
        *,
        keep_candidate_latest: bool,
        reopen_previous: bool,
    ) -> None:
        """Rollback a publication whose post-open participant failed.

        ``finalize_provisional`` clears the provisional marker before returning;
        a participant that runs immediately after it therefore needs the same
        pointer restoration without pretending the transaction is still
        provisional.
        """

        if transaction.must_retain:
            raise RuntimeError("stable 已提交或结果不确定，不能回滚发布")
        if self._provisional is not None:
            raise RuntimeError("RuntimeSnapshot 已仍处于 provisional 发布阶段")
        if self._current is not transaction.candidate:
            raise RuntimeError("RuntimeSnapshot 已不是待回滚的 published candidate")
        previous = transaction.previous
        self._current = previous
        if previous is not None:
            previous.state = "committed"
            previous.accepting_leases = reopen_previous
        candidate = transaction.candidate
        candidate.accepting_leases = False
        if keep_candidate_latest:
            candidate.state = "committed"
            self._latest = candidate
        else:
            candidate.state = "validating"
            self._latest = previous
            self._pending = transaction
        async with self._condition:
            self._condition.notify_all()

    async def discard_latest(
        self,
        expected: RuntimeSnapshot | None = None,
    ) -> RuntimeSnapshot:
        """Discard the ready latest snapshot without changing stable."""

        # 1. Remove candidate admission once; retries resume its failed drain.
        if self._provisional is not None:
            raise RuntimeError("RuntimeSnapshot provisional 发布事务尚未结束")
        candidate = self.unpromoted_candidate
        if candidate is None:
            if expected is None or expected.state != "aborted":
                raise RuntimeError("没有等待 discard 的 RuntimeSnapshot 候选")
            candidate = expected
            if candidate.snapshot_id not in self._snapshots:
                return candidate
        elif expected is not None and candidate is not expected:
            raise RuntimeError("等待 discard 的 RuntimeSnapshot 候选不一致")
        if candidate.state != "aborted":
            candidate.state = "aborted"
            candidate.accepting_leases = False
            self._latest = self._current
        await self.wait_for_no_leases(candidate)
        self._schedule_drain(candidate)

        # 2. Wait for validation leases and candidate-owned resources to drain.
        await self._await_drain_tasks((candidate.snapshot_id,))
        self._raise_drain_failures((candidate.snapshot_id,))
        async with self._condition:
            self._condition.notify_all()
        return candidate

    async def abort(
        self,
        transaction: SnapshotTransaction,
        *,
        reopen_previous: bool = True,
    ) -> None:
        self._require_pending(transaction)
        if transaction.must_retain:
            raise RuntimeError("stable 已提交或结果不确定，不能丢弃 owner")
        transaction.candidate.state = "aborted"
        transaction.candidate.accepting_leases = False
        if self._current is transaction.previous and transaction.previous is not None:
            transaction.previous.accepting_leases = reopen_previous
        self._pending = None
        self._schedule_drain(transaction.candidate)
        await self._await_drain_tasks((transaction.candidate.snapshot_id,))
        self._raise_drain_failures((transaction.candidate.snapshot_id,))
        async with self._condition:
            self._condition.notify_all()

    async def quiesce_current(self) -> RuntimeSnapshot | None:
        snapshot = self.pause_admission()
        if snapshot is None:
            return None
        try:
            await self.wait_for_no_leases(snapshot)
        except BaseException:
            await self.resume(snapshot)
            raise
        return snapshot

    def pause_admission(self) -> RuntimeSnapshot | None:
        snapshot = self._current
        if snapshot is not None:
            snapshot.accepting_leases = False
        return snapshot

    def pause_candidate_admission(
        self,
        expected: RuntimeSnapshot,
    ) -> RuntimeSnapshot:
        """Atomically seal the exact unpromoted candidate against new leases."""

        candidate = self.unpromoted_candidate
        if candidate is None or candidate is not expected:
            raise RuntimeError("等待 promote 的 RuntimeSnapshot 候选不一致")
        candidate.accepting_leases = False
        return candidate

    def seal_candidate_validation(self, expected: RuntimeSnapshot) -> None:
        """Seal the Core-observed receipt after validation leases have drained."""

        candidate = self.unpromoted_candidate
        if candidate is None or candidate is not expected:
            raise RuntimeError("等待封存验证回执的 RuntimeSnapshot 候选不一致")
        if candidate.accepting_leases or candidate.lease_count:
            raise RuntimeError("封存验证回执前必须暂停并排空 candidate lease")
        self._seal_composition_validation(candidate)

    def seal_pending_validation(self, expected: RuntimeSnapshot) -> None:
        """封存尚未公开的 direct candidate 组合验证事实。"""

        candidate = self.pending_candidate
        if candidate is None or candidate is not expected:
            raise RuntimeError("等待封存验证回执的 pending candidate 不一致")
        self._seal_composition_validation(candidate)

    def _seal_composition_validation(self, candidate: RuntimeSnapshot) -> None:
        """检查已停止接纳且无 lease 的实际隔离 Root。"""

        if candidate.accepting_leases or candidate.lease_count:
            raise RuntimeError("封存验证回执前必须暂停并排空 candidate lease")
        self._validate_composition(candidate)

    async def wait_for_no_leases(self, snapshot: RuntimeSnapshot) -> None:
        async with self._condition:
            while snapshot.lease_count:
                await self._condition.wait()

    async def resume(self, snapshot: RuntimeSnapshot | None) -> None:
        if snapshot is None:
            return
        if snapshot.state == "committed" and (
            self._current is snapshot or self.unpromoted_candidate is snapshot
        ):
            snapshot.accepting_leases = True
        async with self._condition:
            self._condition.notify_all()

    async def acquire(
        self,
        snapshot_id: str | None = None,
        *,
        selector: RuntimeSelector = "stable",
    ) -> RuntimeSnapshotLease:
        async with self._condition:
            while True:
                snapshot = (
                    self._selected(selector)
                    if snapshot_id is None
                    else self._snapshots.get(snapshot_id)
                )
                if snapshot is None:
                    raise RuntimeError("RuntimeSnapshot 不可用")
                if snapshot.state != "committed":
                    raise RuntimeError(f"RuntimeSnapshot 不可租用: {snapshot.state}")
                if snapshot.accepting_leases:
                    return self._claim_lease(snapshot)
                await self._condition.wait()

    async def acquire_composition_root(
        self,
        root: CompositionRoot,
    ) -> RuntimeSnapshotLease:
        """Lease the committed snapshot that owns one exact composition Root."""

        async with self._condition:
            while True:
                snapshot = next(
                    (
                        item
                        for item in self._snapshots.values()
                        if item.composition_root is root
                        and item.state in {"validating", "committed"}
                    ),
                    None,
                )
                if snapshot is None:
                    raise TurnAdmissionRetiredError(
                        "composition Root 已退役，Turn 尚未进入 admission"
                    )
                if (
                    snapshot is self._current
                    and snapshot.state == "committed"
                    and snapshot.accepting_leases
                ):
                    return self._claim_lease(snapshot)
                await self._condition.wait()

    async def close(self) -> None:
        if self._pending is not None or self._provisional is not None:
            raise RuntimeError("RuntimeSnapshot 发布事务尚未结束")
        leased = [
            snapshot.snapshot_id
            for snapshot in self._snapshots.values()
            if snapshot.lease_count
        ]
        if leased:
            raise RuntimeError(
                f"RuntimeSnapshot 仍有 lease: {', '.join(sorted(leased))}"
            )
        await self.retry_drains()
        latest = self.unpromoted_candidate
        self._latest = self._current
        if latest is not None:
            latest.state = "aborted"
            latest.accepting_leases = False
            self._schedule_drain(latest)
        current = self._current
        self._current = None
        self._latest = None
        if current is not None:
            current.state = "retired"
            self._schedule_drain(current)
            await self.retry_drains()

    def lease(
        self,
        snapshot_id: str | None = None,
        *,
        selector: RuntimeSelector = "stable",
    ) -> RuntimeSnapshotLease:
        snapshot = (
            self._selected(selector)
            if snapshot_id is None
            else self._snapshots.get(snapshot_id)
        )
        if snapshot is None:
            raise RuntimeError("RuntimeSnapshot 不可用")
        if snapshot.state != "committed":
            raise RuntimeError(f"RuntimeSnapshot 不可租用: {snapshot.state}")
        if not snapshot.accepting_leases:
            raise RuntimeError("RuntimeSnapshot 暂停接收新 lease")
        return self._claim_lease(snapshot)

    def retain_publication_target(
        self,
        transaction: SnapshotTransaction,
    ) -> RuntimeSnapshotLease:
        """Retain the closed exact target for one Core publication participant."""

        candidate = transaction.candidate
        if self._pending is not transaction and self._provisional is not transaction:
            raise RuntimeError("RuntimeSnapshot publication target 已失效")
        if self._snapshots.get(candidate.snapshot_id) is not candidate:
            raise RuntimeError("RuntimeSnapshot publication target 未被 Store 持有")
        return self._claim_lease(candidate)

    def retain_recovery_target(self, snapshot: RuntimeSnapshot) -> RuntimeSnapshotLease:
        """仅允许 Core 为已关闭并排空的当前快照准备重建后的资源。"""
        if (snapshot is not self._current or snapshot.state != "committed"
                or snapshot.accepting_leases or snapshot.lease_count):
            raise RuntimeError("RuntimeSnapshot recovery target 必须是已关闭并排空的 current")
        return self._claim_lease(snapshot)

    def _claim_lease(self, snapshot: RuntimeSnapshot) -> RuntimeSnapshotLease:
        snapshot.lease_count += 1
        for generation in snapshot.generations.values():
            generation.lease_count += 1
        return RuntimeSnapshotLease(
            self,
            snapshot,
            self._validation_candidate_plugin_ids(snapshot),
        )

    def fork_lease(self, source: RuntimeSnapshotLease) -> RuntimeSnapshotLease:
        snapshot = source.snapshot
        if (
            not source.active
            or self._snapshots.get(snapshot.snapshot_id) is not snapshot
        ):
            raise RuntimeError("RuntimeSnapshot lease 不可复制")
        snapshot.lease_count += 1
        for generation in snapshot.generations.values():
            generation.lease_count += 1
        return RuntimeSnapshotLease(
            self,
            snapshot,
            source.validation_candidate_plugin_ids,
        )

    def _validation_candidate_plugin_ids(
        self,
        snapshot: RuntimeSnapshot,
    ) -> frozenset[str]:
        stable = self._current
        if stable is None or snapshot is not self.unpromoted_candidate:
            return frozenset()
        return frozenset(
            plugin_id
            for plugin_id, generation in snapshot.generations.items()
            if stable.generations.get(plugin_id) is not generation
        )

    async def release_lease(self, snapshot: RuntimeSnapshot) -> None:
        if snapshot.lease_count <= 0:
            raise RuntimeError(
                f"RuntimeSnapshot lease 计数失衡: {snapshot.snapshot_id}"
            )
        snapshot.lease_count -= 1
        for generation in snapshot.generations.values():
            generation.lease_count -= 1
        self._schedule_drain(snapshot)
        async with self._condition:
            self._condition.notify_all()

    async def wait_for_generation_drained(
        self,
        generation: PluginGeneration,
    ) -> None:
        async with self._condition:
            while generation.lease_count:
                await self._condition.wait()
        await self.retry_drains()

    def _schedule_drain(self, snapshot: RuntimeSnapshot) -> None:
        if (
            self._snapshots.get(snapshot.snapshot_id) is not snapshot
            or snapshot.state not in {"retired", "aborted"}
            or snapshot.lease_count
        ):
            return
        existing = self._drain_tasks.get(snapshot.snapshot_id)
        if existing is not None and not existing.done():
            return
        _ = self._drain_failures.pop(snapshot.snapshot_id, None)
        self._drain_tasks[snapshot.snapshot_id] = asyncio.create_task(
            self._run_drain(snapshot),
            name=f"runtime_snapshot_drain:{snapshot.snapshot_id}",
        )

    async def _run_drain(self, snapshot: RuntimeSnapshot) -> None:
        try:
            if self._on_drained is not None:
                await self._on_drained(snapshot)
        except BaseException as error:
            self._drain_failures[snapshot.snapshot_id] = error
        else:
            _ = self._snapshots.pop(snapshot.snapshot_id, None)
        finally:
            _ = self._drain_tasks.pop(snapshot.snapshot_id, None)
            async with self._condition:
                self._condition.notify_all()

    async def retry_drains(self) -> None:
        """等待已有关闭；每个尚未关闭的 snapshot 本次最多尝试一次。"""

        running = tuple(self._drain_tasks)
        await self._await_drain_tasks(running)
        for snapshot in tuple(self._snapshots.values()):
            if snapshot.snapshot_id not in running:
                self._schedule_drain(snapshot)
        attempted = tuple(self._drain_tasks)
        await self._await_drain_tasks(attempted)
        self._raise_drain_failures((*running, *attempted))

    async def _await_drain_tasks(self, snapshot_ids: tuple[str, ...]) -> None:
        tasks = [
            task
            for snapshot_id in snapshot_ids
            if (task := self._drain_tasks.get(snapshot_id)) is not None
        ]
        if tasks:
            async def join() -> None:
                await asyncio.gather(*tasks)

            await _join_cleanup(asyncio.create_task(join(), name="snapshot-drain-join"))

    def _raise_drain_failures(self, snapshot_ids: tuple[str, ...]) -> None:
        failures = [
            (snapshot_id, self._drain_failures[snapshot_id])
            for snapshot_id in snapshot_ids
            if snapshot_id in self._drain_failures
        ]
        if not failures:
            return
        snapshot_id, error = failures[0]
        raise RuntimeError(f"RuntimeSnapshot drain 失败: {snapshot_id}") from error

    def _require_pending(self, transaction: SnapshotTransaction) -> None:
        if self._pending is not transaction:
            raise RuntimeError("RuntimeSnapshot 发布事务已失效")

    def _require_provisional(self, transaction: SnapshotTransaction) -> None:
        if self._provisional is not transaction:
            raise RuntimeError("RuntimeSnapshot provisional 发布事务已失效")

    def _adopt(self, snapshot: RuntimeSnapshot) -> None:
        # 同一物理 Root 或 generation 不能通过另一张 snapshot 再次发布。
        for owned in self._snapshots.values():
            if snapshot.composition_root is not None and snapshot.composition_root is owned.composition_root:
                raise RuntimeError("不同 RuntimeSnapshot 不能共享 Root")
            if any(
                generation is owned.generations.get(plugin_id)
                for plugin_id, generation in snapshot.generations.items()
            ):
                raise RuntimeError("不同 RuntimeSnapshot 不能共享 PluginGeneration")
        snapshot.claim(self._token)

    @staticmethod
    def _validate_composition(
        snapshot: RuntimeSnapshot,
    ) -> None:
        root = snapshot.composition_root
        if root is None:
            if (
                snapshot.composition_topology is not None
                or snapshot.channel_registry is not None
                or snapshot.channel_registry_identity is not None
                or snapshot.channel_catalog is not None
            ):
                raise RuntimeError(
                    "RuntimeSnapshot composition identity 缺少 Root Context"
                )
            return
        if snapshot.channel_registry_identity != (
            None
            if snapshot.channel_registry is None
            else snapshot.channel_registry.identity
        ):
            raise RuntimeError("RuntimeSnapshot channel descriptor 在编译后发生变化")
        if (
            snapshot.channel_registry is not None
            and snapshot.channel_registry.root_instance_token is not root.instance_token
        ):
            raise RuntimeError("RuntimeSnapshot channel registry 不属于 exact Root")
        if (
            snapshot.channel_catalog is not None
            and snapshot.channel_catalog.root_instance_token is not root.instance_token
        ):
            raise RuntimeError("RuntimeSnapshot channel catalog 不属于 exact Root")
        topology = snapshot.composition_topology
        if topology is None:
            raise RuntimeError("RuntimeSnapshot composition Root 缺少 TopologyView")
        receipt = root.receipt()
        if receipt.incident_overflowed or receipt.external_effects:
            raise RuntimeError(
                "RuntimeSnapshot 插件组合拓扑未就绪: "
                f"required_pending={receipt.required_pending}, "
                f"required_degraded={receipt.required_degraded}, "
                f"incident_overflowed={receipt.incident_overflowed}, "
                f"external_effects={receipt.external_effects}"
            )
        if not receipt.ready:
            raise RuntimeError(
                "RuntimeSnapshot 插件组合拓扑未就绪: "
                f"required_pending={receipt.required_pending}, "
                f"required_degraded={receipt.required_degraded}, "
                f"incident_overflowed={receipt.incident_overflowed}, "
                f"external_effects={receipt.external_effects}"
            )
        if root.topology_identity() != topology.identity:
            raise RuntimeError("RuntimeSnapshot 插件组合拓扑在编译后发生变化")
        if root.composition_revision != topology.composition_revision:
            raise RuntimeError("RuntimeSnapshot 插件组合拓扑在编译后发生过结构变化")

    def _selected(self, selector: RuntimeSelector) -> RuntimeSnapshot | None:
        if selector == "stable":
            return self._current
        if selector == "latest":
            return self.latest
        raise ValueError(f"未知 RuntimeSnapshot selector: {selector}")
