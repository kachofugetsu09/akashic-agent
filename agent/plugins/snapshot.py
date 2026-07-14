from __future__ import annotations

import asyncio
import hashlib
from contextvars import ContextVar, Token
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from agent.lifecycle.phase import topo_sort_modules
from agent.mcp.generation import WorkspaceMcpGeneration
from agent.plugins.generation import PluginGeneration
from agent.plugins.jobs import RegisteredPluginJob, plugin_job_key
from agent.plugins.specs import RegisteredProactiveSource, proactive_source_key
from agent.tools.registry import ToolRegistry
from agent.tool_hooks import ToolHook
from agent.skills import SkillIndex
from bus.event_bus import Handler
from infra.channels.contract import Channel

SnapshotState = Literal[
    "compiled",
    "published_pending",
    "committed",
    "aborted",
    "retired",
]

@dataclass
class RuntimeSnapshot:
    snapshot_id: str
    generations: Mapping[str, PluginGeneration]
    before_turn_modules: tuple[object, ...]
    before_reasoning_modules: tuple[object, ...]
    prompt_render_modules: tuple[object, ...]
    before_step_modules: tuple[object, ...]
    after_step_modules: tuple[object, ...]
    after_reasoning_modules: tuple[object, ...]
    after_turn_modules: tuple[object, ...]
    jobs: Mapping[str, RegisteredPluginJob]
    proactive_sources: Mapping[str, RegisteredProactiveSource]
    proactive_modules: tuple[object, ...]
    proactive_lifecycles: tuple[object, ...]
    proactive_module_factories: tuple[object, ...]
    proactive_runtime_factories: tuple[object, ...]
    tool_hooks: tuple[ToolHook, ...]
    channels: Mapping[str, Channel]
    skill_catalog_generation_id: str | None
    mcp_catalog_generation_ids: Mapping[str, str]
    workspace_mcp_generation: WorkspaceMcpGeneration | None = None
    managed_services: Mapping[str, Mapping[str, object]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    dashboard_bindings: tuple[object, ...] = ()
    tool_registry: ToolRegistry | None = None
    plugin_skill_index: SkillIndex | None = None
    event_handlers: Mapping[type[object], tuple[Handler[object], ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    state: SnapshotState = "compiled"
    lease_count: int = 0
    accepting_leases: bool = True
    _store_token: object | None = field(default=None, repr=False)

    def active_generations(self) -> tuple[PluginGeneration, ...]:
        return tuple(
            generation
            for generation in self.generations.values()
            if plugin_is_active(generation.instance, plugin_id=generation.plugin_id)
        )

    def claim(self, store_token: object) -> None:
        if self.state != "compiled" or self.lease_count or self._store_token is not None:
            raise RuntimeError("RuntimeSnapshot 不是可发布的全新 compiled 快照")
        self._store_token = store_token


def plugin_is_active(instance: object, *, plugin_id: str) -> bool:
    checker = getattr(instance, "is_active", None)
    if not callable(checker):
        return True
    try:
        return bool(checker())
    except Exception as error:
        raise RuntimeError(f"插件 active 状态检查失败: {plugin_id}") from error


@dataclass(frozen=True)
class SnapshotTransaction:
    previous: RuntimeSnapshot | None
    candidate: RuntimeSnapshot


@dataclass(frozen=True)
class _SnapshotModuleOrder:
    module: object
    slot: str
    requires: tuple[str, ...]


class RuntimeSnapshotCompiler:
    _PHASE_FIELDS = (
        "before_turn_modules",
        "before_reasoning_modules",
        "prompt_render_modules",
        "before_step_modules",
        "after_step_modules",
        "after_reasoning_modules",
        "after_turn_modules",
    )

    def compile(
        self,
        generations: Mapping[str, PluginGeneration],
        *,
        catalog_generation: PluginGeneration | None = None,
        snapshot_revision: str = "",
        workspace_mcp_generation: WorkspaceMcpGeneration | None = None,
    ) -> RuntimeSnapshot:
        ordered = [generations[key] for key in sorted(generations)]
        if any(generation.plugin_id != key for key, generation in generations.items()):
            raise RuntimeError("RuntimeSnapshot generation key 与 plugin_id 不一致")
        phases: dict[str, tuple[object, ...]] = {}
        for field_name in self._PHASE_FIELDS:
            modules = tuple(
                module
                for generation in ordered
                for module in getattr(generation.contributions, field_name)
            )
            phases[field_name] = self.order_plugin_modules(modules)
        jobs = self._compile_jobs(ordered)
        sources = self._compile_sources(ordered)
        proactive_modules = tuple(
            module
            for generation in ordered
            for module in generation.contributions.proactive_modules
        )
        proactive_lifecycles = tuple(
            lifecycle
            for generation in ordered
            for lifecycle in generation.contributions.proactive_lifecycles
        )
        proactive_module_factories = tuple(
            factory
            for generation in ordered
            for factory in generation.contributions.proactive_module_factories
        )
        proactive_runtime_factories = tuple(
            factory
            for generation in ordered
            for factory in generation.contributions.proactive_runtime_factories
        )
        channels: dict[str, Channel] = {}
        for generation in ordered:
            for channel in generation.contributions.channels:
                name = str(channel.name).strip()
                if not name or name in channels:
                    raise RuntimeError(f"RuntimeSnapshot Channel 名称冲突: {name}")
                channels[name] = channel
        catalog_owner = catalog_generation or next(
            (generation for generation in reversed(ordered) if generation.skill_catalog),
            None,
        )
        if catalog_owner is not None and generations.get(catalog_owner.plugin_id) is not catalog_owner:
            raise RuntimeError("RuntimeSnapshot catalog owner 不属于 generations")
        mcp_catalogs = {
            generation.plugin_id: generation.mcp_catalog.generation_id
            for generation in ordered
            if generation.mcp_catalog is not None
        }
        managed_services = {
            generation.plugin_id: MappingProxyType(
                dict(generation.contributions.managed_services)
            )
            for generation in ordered
            if generation.contributions.managed_services
        }
        identity = "|".join(
            f"{generation.plugin_id}:{generation.generation_id}:"
            f"{generation.source_revision}:{generation.config_revision}"
            for generation in ordered
        )
        identity += "|skill:" + (
            catalog_owner.skill_catalog.generation_id
            if catalog_owner is not None and catalog_owner.skill_catalog is not None
            else ""
        )
        identity += "|mcp:" + "|".join(
            f"{plugin_id}:{generation_id}"
            for plugin_id, generation_id in sorted(mcp_catalogs.items())
        )
        identity += "|workspace-mcp:" + (
            workspace_mcp_generation.generation_id
            if workspace_mcp_generation is not None
            else ""
        )
        identity += f"|snapshot:{snapshot_revision}"
        snapshot_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        return RuntimeSnapshot(
            snapshot_id=snapshot_id,
            generations=MappingProxyType(dict(generations)),
            jobs=MappingProxyType(jobs),
            proactive_sources=MappingProxyType(sources),
            proactive_modules=proactive_modules,
            proactive_lifecycles=proactive_lifecycles,
            proactive_module_factories=proactive_module_factories,
            proactive_runtime_factories=proactive_runtime_factories,
            tool_hooks=(),
            channels=MappingProxyType(channels),
            skill_catalog_generation_id=(
                catalog_owner.skill_catalog.generation_id
                if catalog_owner is not None and catalog_owner.skill_catalog is not None
                else None
            ),
            mcp_catalog_generation_ids=MappingProxyType(mcp_catalogs),
            workspace_mcp_generation=workspace_mcp_generation,
            managed_services=MappingProxyType(managed_services),
            plugin_skill_index=(
                catalog_owner.skill_catalog.normal_plugins
                if catalog_owner is not None and catalog_owner.skill_catalog is not None
                else None
            ),
            before_turn_modules=phases["before_turn_modules"],
            before_reasoning_modules=phases["before_reasoning_modules"],
            prompt_render_modules=phases["prompt_render_modules"],
            before_step_modules=phases["before_step_modules"],
            after_step_modules=phases["after_step_modules"],
            after_reasoning_modules=phases["after_reasoning_modules"],
            after_turn_modules=phases["after_turn_modules"],
        )

    @staticmethod
    def order_plugin_modules(modules: tuple[object, ...]) -> tuple[object, ...]:
        slots = {
            str(slot)
            for slot in (getattr(module, "slot", None) for module in modules)
            if isinstance(slot, str) and slot
        }
        bindings = [
            _SnapshotModuleOrder(
                module=module,
                slot=str(getattr(module, "slot", "")),
                requires=tuple(
                    str(required)
                    for required in getattr(module, "requires", ())
                    if str(required) in slots
                ),
            )
            for module in modules
        ]
        ordered = cast(list[_SnapshotModuleOrder], topo_sort_modules(bindings))
        return tuple(binding.module for binding in ordered)

    @staticmethod
    def _compile_jobs(
        generations: list[PluginGeneration],
    ) -> dict[str, RegisteredPluginJob]:
        jobs: dict[str, RegisteredPluginJob] = {}
        for generation in generations:
            catalog = generation.job_catalog
            if catalog is None:
                continue
            for key, job in catalog.jobs.items():
                if key in jobs or key != plugin_job_key(job):
                    raise RuntimeError(f"RuntimeSnapshot Job 稳定键冲突: {key}")
                jobs[key] = job
        return jobs

    @staticmethod
    def _compile_sources(
        generations: list[PluginGeneration],
    ) -> dict[str, RegisteredProactiveSource]:
        sources: dict[str, RegisteredProactiveSource] = {}
        for generation in generations:
            catalog = generation.proactive_catalog
            if catalog is None:
                continue
            for key, source in catalog.sources.items():
                if key in sources or key != proactive_source_key(source):
                    raise RuntimeError(f"RuntimeSnapshot proactive 稳定键冲突: {key}")
                sources[key] = source
        return sources


class RuntimeSnapshotLease:
    def __init__(self, store: RuntimeSnapshotStore, snapshot: RuntimeSnapshot) -> None:
        self._store = store
        self.snapshot = snapshot
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


def lease_current_runtime_snapshot() -> RuntimeSnapshotLease | None:
    lease = get_current_runtime_lease()
    return lease.fork() if lease is not None else None


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
        self._snapshots: dict[str, RuntimeSnapshot] = {}
        self._pending: SnapshotTransaction | None = None
        self._on_drained = on_drained
        self._token = object()
        self._condition = asyncio.Condition()

    @property
    def current(self) -> RuntimeSnapshot | None:
        return self._current

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
                snapshot.state in {"published_pending", "committed"}
                or snapshot.lease_count > 0
            )
            and any(item is generation for item in snapshot.generations.values())
            for snapshot in self._snapshots.values()
        )

    def workspace_mcp_is_referenced_elsewhere(
        self,
        generation: WorkspaceMcpGeneration,
        *,
        excluding_snapshot_id: str,
    ) -> bool:
        return any(
            snapshot.snapshot_id != excluding_snapshot_id
            and (
                snapshot.state in {"published_pending", "committed"}
                or snapshot.lease_count > 0
            )
            and snapshot.workspace_mcp_generation is generation
            for snapshot in self._snapshots.values()
        )

    def install(self, snapshot: RuntimeSnapshot) -> None:
        if self._current is not None or self._pending is not None:
            raise RuntimeError("RuntimeSnapshotStore 已安装初始快照")
        self._adopt(snapshot)
        snapshot.state = "committed"
        self._current = snapshot
        self._snapshots[snapshot.snapshot_id] = snapshot

    def begin_publish(
        self,
        candidate: RuntimeSnapshot,
        *,
        admission_gated: bool = False,
    ) -> SnapshotTransaction:
        if self._pending is not None:
            raise RuntimeError("已有 RuntimeSnapshot 发布事务")
        if candidate.snapshot_id in self._snapshots:
            raise RuntimeError(f"RuntimeSnapshot 已存在: {candidate.snapshot_id}")
        self._adopt(candidate)
        transaction = SnapshotTransaction(previous=self._current, candidate=candidate)
        candidate.state = "published_pending"
        candidate.accepting_leases = not admission_gated
        self._snapshots[candidate.snapshot_id] = candidate
        self._current = candidate
        self._pending = transaction
        return transaction

    async def commit(self, transaction: SnapshotTransaction) -> None:
        self._require_pending(transaction)
        transaction.candidate.state = "committed"
        transaction.candidate.accepting_leases = True
        self._pending = None
        previous = transaction.previous
        if previous is not None:
            previous.state = "retired"
            await self._drain_if_ready(previous)
        async with self._condition:
            self._condition.notify_all()

    async def abort(self, transaction: SnapshotTransaction) -> None:
        self._require_pending(transaction)
        transaction.candidate.state = "aborted"
        transaction.candidate.accepting_leases = False
        self._current = transaction.previous
        if transaction.previous is not None:
            transaction.previous.accepting_leases = True
        self._pending = None
        await self._drain_if_ready(transaction.candidate)
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

    async def wait_for_no_leases(self, snapshot: RuntimeSnapshot) -> None:
        async with self._condition:
            while snapshot.lease_count:
                await self._condition.wait()

    async def resume(self, snapshot: RuntimeSnapshot | None) -> None:
        if snapshot is None:
            return
        if self._current is snapshot and snapshot.state == "committed":
            snapshot.accepting_leases = True
        async with self._condition:
            self._condition.notify_all()

    async def acquire(
        self,
        snapshot_id: str | None = None,
    ) -> RuntimeSnapshotLease:
        async with self._condition:
            while True:
                snapshot = (
                    self._current
                    if snapshot_id is None
                    else self._snapshots.get(snapshot_id)
                )
                if snapshot is None:
                    raise RuntimeError("RuntimeSnapshot 不可用")
                if snapshot.state not in {"published_pending", "committed"}:
                    raise RuntimeError(f"RuntimeSnapshot 不可租用: {snapshot.state}")
                if snapshot.accepting_leases:
                    return self._claim_lease(snapshot)
                await self._condition.wait()

    async def close(self) -> None:
        if self._pending is not None:
            raise RuntimeError("RuntimeSnapshot 发布事务尚未结束")
        leased = [
            snapshot.snapshot_id
            for snapshot in self._snapshots.values()
            if snapshot.lease_count
        ]
        if leased:
            raise RuntimeError(f"RuntimeSnapshot 仍有 lease: {', '.join(sorted(leased))}")
        await self.retry_drains()
        current = self._current
        self._current = None
        if current is not None:
            current.state = "retired"
            await self._drain_if_ready(current)

    def lease(self, snapshot_id: str | None = None) -> RuntimeSnapshotLease:
        snapshot = (
            self._current
            if snapshot_id is None
            else self._snapshots.get(snapshot_id)
        )
        if snapshot is None:
            raise RuntimeError("RuntimeSnapshot 不可用")
        if snapshot.state not in {"published_pending", "committed"}:
            raise RuntimeError(f"RuntimeSnapshot 不可租用: {snapshot.state}")
        if not snapshot.accepting_leases:
            raise RuntimeError("RuntimeSnapshot 暂停接收新 lease")
        return self._claim_lease(snapshot)

    def _claim_lease(self, snapshot: RuntimeSnapshot) -> RuntimeSnapshotLease:
        snapshot.lease_count += 1
        for generation in snapshot.generations.values():
            generation.lease_count += 1
        if snapshot.workspace_mcp_generation is not None:
            snapshot.workspace_mcp_generation.lease_count += 1
        return RuntimeSnapshotLease(self, snapshot)

    def fork_lease(self, source: RuntimeSnapshotLease) -> RuntimeSnapshotLease:
        snapshot = source.snapshot
        if not source.active or self._snapshots.get(snapshot.snapshot_id) is not snapshot:
            raise RuntimeError("RuntimeSnapshot lease 不可复制")
        snapshot.lease_count += 1
        for generation in snapshot.generations.values():
            generation.lease_count += 1
        if snapshot.workspace_mcp_generation is not None:
            snapshot.workspace_mcp_generation.lease_count += 1
        return RuntimeSnapshotLease(self, snapshot)

    async def release_lease(self, snapshot: RuntimeSnapshot) -> None:
        if snapshot.lease_count <= 0:
            raise RuntimeError(f"RuntimeSnapshot lease 计数失衡: {snapshot.snapshot_id}")
        snapshot.lease_count -= 1
        for generation in snapshot.generations.values():
            generation.lease_count -= 1
        if snapshot.workspace_mcp_generation is not None:
            snapshot.workspace_mcp_generation.lease_count -= 1
        await self._drain_if_ready(snapshot)
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

    async def _drain_if_ready(self, snapshot: RuntimeSnapshot) -> None:
        if snapshot.state not in {"retired", "aborted"} or snapshot.lease_count:
            return
        if self._on_drained is not None:
            await self._on_drained(snapshot)
        _ = self._snapshots.pop(snapshot.snapshot_id, None)

    async def retry_drains(self) -> None:
        for snapshot in tuple(self._snapshots.values()):
            await self._drain_if_ready(snapshot)

    def _require_pending(self, transaction: SnapshotTransaction) -> None:
        if self._pending is not transaction or self._current is not transaction.candidate:
            raise RuntimeError("RuntimeSnapshot 发布事务已失效")

    def _adopt(self, snapshot: RuntimeSnapshot) -> None:
        snapshot.claim(self._token)
