from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from agent.plugin_composition.workload_slots import (
    WorkloadBinding,
    WorkloadDescriptor,
)
from agent.workloads.client import WorkloadController, WorkloadEffectUnknown
from agent.workloads.model import (
    WorkloadLease,
    WorkloadStartRequest,
    WorkloadStartReceipt,
    workload_spec_digest,
)

WorkloadMode = Literal["candidate", "formal"]
HealthCallback = Callable[[str, str, bool, str], Awaitable[None] | None]
IncidentCallback = Callable[[str, str, str, str], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class WorkloadCleanupTombstone:
    generation_id: str
    state: Literal["cleanup_failed", "degraded"]
    action: Literal["retry_generation_cleanup", "retry_runtime_recovery"]
    resource_names: tuple[str, ...]
    error: str
    attempt_count: int


FailureCallback = Callable[[WorkloadCleanupTombstone], None]
HealthProbe = Callable[[str, float], Awaitable[tuple[bool, str]]]


@dataclass(slots=True)
class _Entry:
    binding: WorkloadBinding
    receipt: WorkloadStartReceipt
    endpoints: Mapping[str, str]
    watcher: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _Generation:
    generation_id: str
    plugin_id: str
    mode: WorkloadMode
    entries: dict[str, _Entry]
    pending: dict[str, tuple[WorkloadBinding, WorkloadStartRequest]] = field(
        default_factory=dict
    )
    state: Literal["starting", "ready", "stopping", "degraded"] = "starting"
    cleanup_attempts: int = 0


class WorkloadGeneration:
    """Read-only endpoints owned by one exact plugin generation."""

    def __init__(self, generation: _Generation) -> None:
        self.generation_id = generation.generation_id
        self.plugin_id = generation.plugin_id
        self.mode = generation.mode
        self.endpoints = {
            (name, port): url
            for name, entry in generation.entries.items()
            for port, url in entry.endpoints.items()
        }


class WorkloadGenerationHost:
    """Own Controller leases until each workload is strongly stopped."""

    def __init__(
        self,
        controller: WorkloadController,
        *,
        workspace_id: str,
        health_probe: HealthProbe | None = None,
        watch_interval_seconds: float = 5.0,
        on_health: HealthCallback | None = None,
        on_incident: IncidentCallback | None = None,
        on_failure: FailureCallback | None = None,
    ) -> None:
        if not workspace_id:
            raise ValueError("Workload workspace_id 不能为空")
        if watch_interval_seconds <= 0:
            raise ValueError("Workload watch interval 必须大于零")
        self._controller = controller
        self._workspace_id = workspace_id
        self._health_probe = health_probe or _http_health
        self._watch_interval_seconds = watch_interval_seconds
        self._on_health = on_health
        self._on_incident = on_incident
        self._on_failure = on_failure
        self._generations: dict[str, _Generation] = {}
        self._tombstones: dict[str, WorkloadCleanupTombstone] = {}
        self._lock = asyncio.Lock()

    async def start_generation(
        self,
        generation_id: str,
        plugin_id: str,
        bindings: Mapping[str, WorkloadBinding],
        *,
        mode: WorkloadMode,
    ) -> WorkloadGeneration:
        """Start or adopt every workload and publish only after health."""

        async with self._lock:
            return await self._start_generation(
                generation_id, plugin_id, bindings, mode=mode
            )

    async def _start_generation(
        self,
        generation_id: str,
        plugin_id: str,
        bindings: Mapping[str, WorkloadBinding],
        *,
        mode: WorkloadMode,
    ) -> WorkloadGeneration:
        """Run one serialized generation start transaction."""

        if not generation_id or not plugin_id:
            raise ValueError("Workload generation identity 不能为空")
        if mode not in {"candidate", "formal"}:
            raise ValueError(f"Workload mode 无效: {mode}")
        if generation_id in self._generations or generation_id in self._tombstones:
            raise RuntimeError(f"Workload generation 已存在: {generation_id}")
        owned = self._validate_bindings(plugin_id, bindings)
        generation = _Generation(generation_id, plugin_id, mode, {})
        self._generations[generation_id] = generation
        try:
            # 1. Controller returns an exact lease before Core trusts endpoints.
            for name, binding in owned.items():
                request = _start_request(
                    self._workspace_id,
                    generation_id,
                    plugin_id,
                    mode,
                    binding.descriptor,
                )
                generation.pending[name] = (binding, request)
                try:
                    receipt = await self._controller.start(request)
                except (asyncio.CancelledError, WorkloadEffectUnknown):
                    raise
                except BaseException:
                    generation.pending.pop(name, None)
                    raise
                endpoints = _check_start_receipt(request, receipt)
                entry = _Entry(binding, receipt, endpoints)
                generation.entries[name] = entry
                generation.pending.pop(name, None)
                health_url = endpoints[binding.descriptor.health.port]
                healthy, reason = await self._health_probe(
                    health_url + binding.descriptor.health.path,
                    binding.descriptor.health.timeout_seconds,
                )
                if not healthy:
                    raise RuntimeError(f"Workload health 失败: {name}: {reason}")
            # 2. The Root health owner becomes ready only after all entries settle.
            generation.state = "ready"
            for name, entry in generation.entries.items():
                await self._emit_health(generation_id, name, True, "ready")
                entry.watcher = asyncio.create_task(
                    self._watch(generation, name, entry),
                    name=f"workload-watch:{generation_id}:{name}",
                )
            return WorkloadGeneration(generation)
        except BaseException as start_error:
            cleanup_task = asyncio.create_task(self._cleanup(generation))
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError:
                if cleanup_task.done() and cleanup_task.exception() is None:
                    self._generations.pop(generation_id, None)
                else:
                    self._retain(generation, start_error, state="cleanup_failed")
                raise
            except BaseException as cleanup_error:
                self._retain(generation, cleanup_error, state="cleanup_failed")
                raise RuntimeError(
                    "Workload start 失败且 cleanup 未完成"
                ) from cleanup_error
            self._generations.pop(generation_id, None)
            raise start_error

    async def stop_generation(self, generation_id: str) -> None:
        """Strongly stop one generation or retain its cleanup owner."""

        async with self._lock:
            await self._stop_generation(generation_id)

    async def _stop_generation(self, generation_id: str) -> None:
        """Run one serialized generation stop transaction."""

        generation = self._generations.get(generation_id)
        if generation is None:
            if generation_id in self._tombstones:
                raise RuntimeError(
                    f"Workload generation cleanup 未完成: {generation_id}"
                )
            return
        cleanup_task = asyncio.create_task(self._cleanup(generation))
        try:
            await _await_task_after_cancellation(cleanup_task)
        except asyncio.CancelledError:
            if cleanup_task.done() and cleanup_task.exception() is None:
                self._generations.pop(generation_id, None)
                self._tombstones.pop(generation_id, None)
            else:
                self._retain(
                    generation, asyncio.CancelledError(), state="cleanup_failed"
                )
            raise
        except BaseException as error:
            self._retain(generation, error, state="cleanup_failed")
            raise
        self._generations.pop(generation_id, None)
        self._tombstones.pop(generation_id, None)

    def get(self, generation_id: str) -> WorkloadGeneration | None:
        generation = self._generations.get(generation_id)
        return None if generation is None else WorkloadGeneration(generation)

    def tombstone(self, generation_id: str) -> WorkloadCleanupTombstone | None:
        return self._tombstones.get(generation_id)

    async def retry_generation_cleanup(self, generation_id: str) -> None:
        async with self._lock:
            tombstone = self._require_tombstone(generation_id, "cleanup_failed")
            _ = tombstone
            await self._retry_cleanup(generation_id)

    async def retry_runtime_recovery(self, generation_id: str) -> None:
        async with self._lock:
            tombstone = self._require_tombstone(generation_id, "degraded")
            _ = tombstone
            await self._retry_cleanup(generation_id)

    async def _retry_cleanup(self, generation_id: str) -> None:
        generation = self._generations[generation_id]
        try:
            await self._cleanup(generation)
        except BaseException as error:
            state = self._tombstones[generation_id].state
            self._retain(generation, error, state=state)
            raise
        self._generations.pop(generation_id, None)
        self._tombstones.pop(generation_id, None)

    async def _cleanup(self, generation: _Generation) -> None:
        generation.state = "stopping"
        errors: list[BaseException] = []
        for name, (binding, request) in tuple(generation.pending.items()):
            try:
                receipt = await self._controller.start(request)
                endpoints = _check_start_receipt(request, receipt)
                generation.entries[name] = _Entry(
                    binding,
                    receipt,
                    endpoints,
                )
                generation.pending.pop(name)
            except BaseException as error:
                errors.append(error)
        for name, entry in reversed(tuple(generation.entries.items())):
            watcher = entry.watcher
            if watcher is not None and watcher is not asyncio.current_task():
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                entry.watcher = None
            try:
                receipt = await self._controller.stop(entry.receipt.lease)
                _check_stop_receipt(entry.receipt.lease, receipt)
                await self._emit_health(
                    generation.generation_id, name, False, "stopped"
                )
                generation.entries.pop(name)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("Workload cleanup 失败", errors)

    async def _watch(
        self,
        generation: _Generation,
        name: str,
        entry: _Entry,
    ) -> None:
        health = entry.binding.descriptor.health
        url = entry.endpoints[health.port] + health.path
        try:
            while True:
                await asyncio.sleep(self._watch_interval_seconds)
                healthy, reason = await self._health_probe(
                    url, self._watch_interval_seconds
                )
                if (
                    generation.state != "ready"
                    or self._generations.get(generation.generation_id) is not generation
                    or generation.entries.get(name) is not entry
                ):
                    return
                if healthy:
                    continue
                generation.state = "degraded"
                await self._emit_health(generation.generation_id, name, False, reason)
                await self._emit_incident(
                    generation.generation_id,
                    name,
                    "workload_unhealthy",
                    reason,
                )
                self._retain(generation, RuntimeError(reason), state="degraded")
                return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if generation.state != "ready":
                return
            generation.state = "degraded"
            await self._emit_health(
                generation.generation_id, name, False, _error_text(error)
            )
            await self._emit_incident(
                generation.generation_id,
                name,
                "workload_watch_failed",
                _error_text(error),
            )
            self._retain(generation, error, state="degraded")

    def _retain(
        self,
        generation: _Generation,
        error: BaseException,
        *,
        state: Literal["cleanup_failed", "degraded"],
    ) -> None:
        generation.cleanup_attempts += 1
        tombstone = WorkloadCleanupTombstone(
            generation_id=generation.generation_id,
            state=state,
            action=(
                "retry_runtime_recovery"
                if state == "degraded"
                else "retry_generation_cleanup"
            ),
            resource_names=tuple(
                dict.fromkeys((*generation.entries, *generation.pending))
            ),
            error=_error_text(error),
            attempt_count=generation.cleanup_attempts,
        )
        self._tombstones[generation.generation_id] = tombstone
        if self._on_failure is not None:
            self._on_failure(tombstone)

    @staticmethod
    def _validate_bindings(
        plugin_id: str,
        bindings: Mapping[str, WorkloadBinding],
    ) -> dict[str, WorkloadBinding]:
        result: dict[str, WorkloadBinding] = {}
        for name, binding in bindings.items():
            if (
                name != binding.descriptor.name
                or binding.descriptor.owner != plugin_id
                or not binding.owner_fiber.activation_token is binding.activation_token
            ):
                raise RuntimeError(f"Workload binding owner 无效: {plugin_id}:{name}")
            result[name] = binding
        return result

    def _require_tombstone(
        self,
        generation_id: str,
        state: Literal["cleanup_failed", "degraded"],
    ) -> WorkloadCleanupTombstone:
        tombstone = self._tombstones.get(generation_id)
        if tombstone is None or tombstone.state != state:
            raise RuntimeError(
                f"Workload generation 没有 {state} owner: {generation_id}"
            )
        return tombstone

    async def _emit_health(
        self,
        generation_id: str,
        name: str,
        healthy: bool,
        reason: str,
    ) -> None:
        if self._on_health is None:
            return
        result = self._on_health(generation_id, name, healthy, reason)
        if inspect.isawaitable(result):
            await result

    async def _emit_incident(
        self,
        generation_id: str,
        name: str,
        kind: str,
        message: str,
    ) -> None:
        if self._on_incident is None:
            return
        result = self._on_incident(generation_id, name, kind, message)
        if inspect.isawaitable(result):
            await result


def _start_request(
    workspace_id: str,
    generation_id: str,
    plugin_id: str,
    mode: WorkloadMode,
    descriptor: WorkloadDescriptor,
) -> WorkloadStartRequest:
    ports = tuple((item.name, item.number) for item in descriptor.ports)
    loopback_ports = tuple(
        (item.name, item.loopback)
        for item in descriptor.ports
        if item.loopback is not None
    )
    data = tuple((item.name, item.target, item.writable) for item in descriptor.data)
    health = (
        descriptor.health.port,
        descriptor.health.path,
        descriptor.health.timeout_seconds,
    )
    limits = (
        descriptor.limits.memory_mb,
        descriptor.limits.cpu_count,
        descriptor.limits.pids,
    )
    digest = workload_spec_digest(
        plugin_id=plugin_id,
        workload=descriptor.name,
        image=descriptor.image,
        command=descriptor.command,
        ports=ports,
        data=data,
        health=health,
        limits=limits,
        loopback_ports=loopback_ports,
        user_namespaces=descriptor.user_namespaces,
    )
    return WorkloadStartRequest(
        workspace_id=workspace_id,
        plugin_id=plugin_id,
        workload=descriptor.name,
        mode=mode,
        transaction_id=generation_id,
        generation_id=generation_id,
        spec_digest=digest,
        image=descriptor.image,
        command=descriptor.command,
        ports=ports,
        data=data,
        health=health,
        limits=limits,
        loopback_ports=loopback_ports,
        user_namespaces=descriptor.user_namespaces,
    )


def _check_start_receipt(
    request: WorkloadStartRequest,
    receipt: WorkloadStartReceipt,
) -> Mapping[str, str]:
    lease = receipt.lease
    expected = (
        request.workspace_id,
        request.plugin_id,
        request.workload,
        request.mode,
        request.transaction_id,
        request.generation_id,
        request.spec_digest,
    )
    actual = (
        lease.workspace_id,
        lease.plugin_id,
        lease.workload,
        lease.mode,
        lease.transaction_id,
        lease.generation_id,
        lease.spec_digest,
    )
    if actual != expected or not lease.container_id:
        raise RuntimeError("Workload Controller start receipt identity 不匹配")
    endpoints = {item.name: item.url for item in receipt.endpoints}
    if set(endpoints) != {name for name, _ in request.ports}:
        raise RuntimeError("Workload Controller endpoint receipt 不完整")
    if any(not url.startswith("http://") for url in endpoints.values()):
        raise RuntimeError("Workload Controller endpoint 只接受 http URL")
    return endpoints


def _check_stop_receipt(
    lease: WorkloadLease,
    receipt: object,
) -> None:
    from agent.workloads.model import WorkloadStopReceipt

    if not isinstance(receipt, WorkloadStopReceipt) or receipt.lease != lease:
        raise RuntimeError("Workload Controller stop receipt identity 不匹配")
    if not (receipt.container_absent and receipt.mounts_released):
        raise RuntimeError("Workload Controller stop receipt 未证明资源释放")


async def _http_health(url: str, timeout_seconds: float) -> tuple[bool, str]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last = "health deadline exceeded"
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(url, timeout=min(2.0, timeout_seconds))
                if 200 <= response.status_code < 300:
                    return True, f"HTTP {response.status_code}"
                last = f"HTTP {response.status_code}"
            except httpx.HTTPError as error:
                last = _error_text(error)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False, last
            await asyncio.sleep(min(0.2, remaining))


async def _await_task_after_cancellation(task: asyncio.Task[Any]) -> Any:
    """Finish cleanup before restoring caller cancellation."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
            cancelled = True
    try:
        result = task.result()
    except asyncio.CancelledError:
        result = None
    if cancelled:
        raise asyncio.CancelledError
    return result


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
