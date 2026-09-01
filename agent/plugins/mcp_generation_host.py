"""Core-owned generation host for declaration-backed stdio MCP servers.

The plugin composition layer owns only frozen declarations.  This module owns
the materialized :class:`McpClient`, its generation fence and the route facade;
plugins never receive a client, process, or mutable tool wrapper.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from agent.mcp.client import McpClient, McpToolExecutionError
from agent.plugin_composition.mcp_slots import (
    McpServerBinding,
    McpServerDefinition,
    McpServerDescriptor,
    McpServerRegistry,
)

logger = logging.getLogger(__name__)

McpMode = Literal["candidate", "formal"]
McpGenerationState = Literal[
    "starting",
    "ready",
    "degraded",
    "stopping",
    "cleanup_failed",
]

HealthCallback = Callable[[str, str, bool, str], Awaitable[None] | None]
IncidentCallback = Callable[[str, str, str, str], Awaitable[None] | None]

_POLL_SECONDS = 0.02
_STOP_TIMEOUT_SECONDS = 5.0
_READINESS_TIMEOUT_SECONDS = 8.0
_MAX_LOG_LINES = 8


class HealthReporter(Protocol):
    def __call__(
        self,
        generation_id: str,
        server_name: str,
        healthy: bool,
        reason: str,
    ) -> Any: ...


class IncidentReporter(Protocol):
    def __call__(
        self,
        generation_id: str,
        server_name: str,
        kind: str,
        message: str,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class McpMaterializedCommand:
    """Core-resolved command, cwd and base environment for one server."""

    command: tuple[str, ...]
    cwd: str
    env: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class McpToolView:
    """Immutable tool metadata exposed by a generation facade."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class McpCallResult:
    """The only successful route result states: success or tool_error."""

    status: Literal["success", "tool_error"]
    output: str

    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def tool_error(self) -> bool:
        return self.status == "tool_error"


@dataclass(frozen=True, slots=True)
class McpLogView:
    """Bounded protocol stdout/stderr diagnostics owned by the client."""

    stdout: tuple[str, ...]
    stderr: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class McpCleanupTombstone:
    """Retained generation ownership after runtime or cleanup failure."""

    generation_id: str
    state: Literal["cleanup_failed", "degraded"]
    action: Literal["retry_generation_cleanup", "retry_runtime_recovery"]
    resource_names: tuple[str, ...]
    error: str
    attempt_count: int


FailureCallback = Callable[[McpCleanupTombstone], None]


@dataclass
class _McpEntry:
    generation_id: str
    name: str
    binding: McpServerBinding
    materialized: McpMaterializedCommand
    client: McpClient
    mode: McpMode
    allowed_tools: frozenset[str]
    tools: Mapping[str, McpToolView]
    catalog_tools: tuple[McpToolView, ...]
    catalog_digest: str
    epoch: int
    process_identity: object | None
    watcher: asyncio.Task[None] | None = None
    stopping: bool = False


@dataclass
class _Generation:
    generation_id: str
    mode: McpMode
    registry: McpServerRegistry
    entries: dict[str, _McpEntry]
    token: object = field(default_factory=object)
    state: McpGenerationState = "starting"
    cleanup_attempts: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class McpRoute:
    """Single-server, generation-bound route with an enforced tool allowlist."""

    def __init__(
        self,
        host: McpGenerationHost,
        generation_id: str,
        generation_token: object,
        entry: _McpEntry,
    ) -> None:
        self._host = host
        self.generation_id = generation_id
        self.server_name = entry.name
        self._generation_token = generation_token
        self._entry = entry
        self._closed = False

    async def call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> McpCallResult:
        """Call one exact generation tool and preserve transport errors."""

        self._assert_open()
        entry = self._host._resolve_route(
            self.generation_id,
            self._generation_token,
            self._entry,
        )
        if tool_name not in entry.allowed_tools:
            raise PermissionError(
                f"MCP candidate tool 未获 allowlist 授权: {entry.name}:{tool_name}"
            )
        if not isinstance(arguments, Mapping):
            raise TypeError("MCP tool arguments must be a mapping")
        try:
            output = await entry.client.call(
                tool_name,
                dict(arguments),
                timeout=timeout,
            )
        except McpToolExecutionError as error:
            return McpCallResult(status="tool_error", output=str(error))
        return McpCallResult(status="success", output=output)

    async def aclose(self) -> None:
        """Close this facade without changing generation ownership."""

        self._closed = True

    async def __aenter__(self) -> McpRoute:
        self._assert_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("MCP route 已关闭")


class McpServerView:
    """Read-only server facade; no client or process handle is exposed."""

    def __init__(
        self,
        host: McpGenerationHost,
        generation_id: str,
        generation_token: object,
        entry: _McpEntry,
    ) -> None:
        self._host = host
        self.generation_id = generation_id
        self.name = entry.name
        self._generation_token = generation_token
        self._entry = entry

    @property
    def epoch(self) -> int:
        self._host._resolve_route(
            self.generation_id,
            self._generation_token,
            self._entry,
        )
        return self._entry.epoch

    @property
    def tools(self) -> Mapping[str, McpToolView]:
        self._host._resolve_route(
            self.generation_id,
            self._generation_token,
            self._entry,
        )
        return self._entry.tools

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self.tools)

    def route(self) -> McpRoute:
        self._host._resolve_route(
            self.generation_id,
            self._generation_token,
            self._entry,
        )
        return McpRoute(
            self._host,
            self.generation_id,
            self._generation_token,
            self._entry,
        )

    def logs(self) -> McpLogView:
        return self._host.logs(self.generation_id, self.name)


class McpGeneration(Mapping[str, McpServerView]):
    """Read-only generation facade returned after handshake readiness."""

    def __init__(self, host: McpGenerationHost, generation: _Generation) -> None:
        self._host = host
        self.generation_id = generation.generation_id
        self._token = generation.token
        self._servers = MappingProxyType(
            {
                name: McpServerView(
                    host, generation.generation_id, generation.token, entry
                )
                for name, entry in generation.entries.items()
            }
        )

    @property
    def state(self) -> McpGenerationState:
        return self._host.generation_state(self.generation_id, self._token)

    def catalog_digest(self, server_name: str) -> str:
        return self._host.catalog_digest(
            self.generation_id,
            server_name,
            self._token,
        )

    def server(self, server_name: str) -> McpServerView:
        return self._servers[server_name]

    def route(self, server_name: str) -> McpRoute:
        return self.server(server_name).route()

    def assert_healthy(self) -> None:
        self._host.assert_healthy(self.generation_id, self._token)

    def logs(self, server_name: str) -> McpLogView:
        return self._host.logs(self.generation_id, server_name, self._token)

    def __getitem__(self, server_name: str) -> McpServerView:
        return self._servers[server_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._servers)

    def __len__(self) -> int:
        return len(self._servers)


class McpGenerationHost:
    """Own MCP clients by generation while delegating protocol recovery to them."""

    def __init__(
        self,
        *,
        on_health: HealthReporter | None = None,
        on_incident: IncidentReporter | None = None,
        on_failure: FailureCallback | None = None,
        stop_timeout_seconds: float = _STOP_TIMEOUT_SECONDS,
        readiness_timeout_seconds: float = _READINESS_TIMEOUT_SECONDS,
    ) -> None:
        if stop_timeout_seconds <= 0:
            raise ValueError("stop_timeout_seconds must be positive")
        if readiness_timeout_seconds <= 0:
            raise ValueError("readiness_timeout_seconds must be positive")
        self._on_health = on_health
        self._on_incident = on_incident
        self._on_failure = on_failure
        self._stop_timeout_seconds = stop_timeout_seconds
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._generations: dict[str, _Generation] = {}
        self._tombstones: dict[str, McpCleanupTombstone] = {}
        self._next_epoch = 0

    async def start_generation(
        self,
        generation_id: str,
        registry: McpServerRegistry,
        materialized_commands: Mapping[str, McpMaterializedCommand],
        *,
        mode: McpMode = "candidate",
        endpoint_ports: Mapping[str, int] | None = None,
        workload_endpoints: Mapping[tuple[str, str], str] | None = None,
        expected_catalog_digests: Mapping[str, str] | None = None,
    ) -> McpGeneration:
        """Start one exact Root registry and publish it only after MCP readiness."""

        self._validate_generation_id(generation_id)
        if mode not in {"candidate", "formal"}:
            raise ValueError(f"unknown MCP generation mode: {mode!r}")
        if generation_id in self._generations or generation_id in self._tombstones:
            raise RuntimeError(f"MCP generation already exists: {generation_id}")
        bindings = self._validate_registry(registry)
        expected_digests = self._validate_expected_catalog_digests(
            bindings,
            mode,
            expected_catalog_digests,
        )
        commands = self._validate_materialized_commands(
            bindings,
            materialized_commands,
            endpoint_ports or {},
            workload_endpoints or {},
        )
        generation = _Generation(
            generation_id=generation_id,
            mode=mode,
            registry=registry,
            entries={},
        )
        self._generations[generation_id] = generation
        try:
            # 1. Build each Core-owned client and complete its handshake/tools-list.
            for name, binding in bindings.items():
                entry = await self._start_entry(
                    generation,
                    binding,
                    commands[name],
                    endpoint_ports or {},
                    workload_endpoints or {},
                    expected_digests.get(name),
                )
                generation.entries[name] = entry
            generation.state = "ready"
            for entry in generation.entries.values():
                entry.watcher = asyncio.create_task(
                    self._watch_entry(generation, entry, entry.epoch),
                    name=f"mcp_generation_watch:{generation_id}:{entry.name}",
                )
                await self._emit_health(generation_id, entry.name, True, "ready")
            return McpGeneration(self, generation)
        except BaseException as error:
            cleanup_task = asyncio.create_task(
                self._cleanup_generation(generation),
                name=f"mcp_generation_cleanup:{generation_id}",
            )
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError:
                if not cleanup_task.done() or cleanup_task.exception() is not None:
                    self._retain_tombstone(generation, _task_error(cleanup_task, error))
                else:
                    _ = self._generations.pop(generation_id, None)
                raise
            except BaseException as cleanup_error:
                self._retain_tombstone(generation, cleanup_error)
            else:
                _ = self._generations.pop(generation_id, None)
            raise

    async def stop_generation(self, generation_id: str) -> None:
        """Stop all clients, drain watchers and retain failed ownership."""

        generation = self._generations.get(generation_id)
        if generation is None:
            tombstone = self._tombstones.get(generation_id)
            if tombstone is not None:
                raise RuntimeError(f"MCP generation cleanup 未完成: {tombstone.error}")
            return
        async with generation.lock:
            generation.state = "stopping"
            cleanup_task = asyncio.create_task(
                self._cleanup_generation(generation),
                name=f"mcp_generation_stop:{generation_id}",
            )
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError:
                if cleanup_task.done() and cleanup_task.exception() is None:
                    _ = self._generations.pop(generation_id, None)
                    _ = self._tombstones.pop(generation_id, None)
                else:
                    self._retain_tombstone(
                        generation,
                        _task_error(cleanup_task, asyncio.CancelledError()),
                    )
                raise
            except BaseException as error:
                self._retain_tombstone(generation, error)
                raise
            _ = self._generations.pop(generation_id, None)
            _ = self._tombstones.pop(generation_id, None)

    async def retry_generation_cleanup(self, generation_id: str) -> None:
        """Retry retained cleanup ownership and remove its tombstone on success."""

        generation = self._require_generation(generation_id)
        async with generation.lock:
            generation.state = "stopping"
            cleanup_task = asyncio.create_task(
                self._cleanup_generation(generation),
                name=f"mcp_generation_retry:{generation_id}",
            )
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError:
                if cleanup_task.done() and cleanup_task.exception() is None:
                    _ = self._generations.pop(generation_id, None)
                    _ = self._tombstones.pop(generation_id, None)
                else:
                    self._retain_tombstone(
                        generation,
                        _task_error(cleanup_task, asyncio.CancelledError()),
                    )
                raise
            except BaseException as error:
                self._retain_tombstone(generation, error)
                raise
            _ = self._generations.pop(generation_id, None)
            _ = self._tombstones.pop(generation_id, None)

    async def retry_runtime_recovery(self, generation_id: str) -> None:
        """Retry a degraded runtime owner, then finish generation cleanup."""

        tombstone = self._tombstones.get(generation_id)
        if tombstone is None or tombstone.state != "degraded":
            raise RuntimeError(
                f"MCP generation has no degraded runtime owner: {generation_id}"
            )
        await self.retry_generation_cleanup(generation_id)

    def get(self, generation_id: str) -> McpGeneration | None:
        """Return a read-only facade while Core still owns the generation."""

        generation = self._generations.get(generation_id)
        return McpGeneration(self, generation) if generation is not None else None

    def tombstone(self, generation_id: str) -> McpCleanupTombstone | None:
        return self._tombstones.get(generation_id)

    def generation_state(
        self,
        generation_id: str,
        token: object | None = None,
    ) -> McpGenerationState:
        generation = self._generations.get(generation_id)
        if generation is not None:
            self._assert_token(generation, token)
            return generation.state
        tombstone = self._tombstones.get(generation_id)
        if tombstone is not None:
            return cast(McpGenerationState, tombstone.state)
        raise RuntimeError(
            f"MCP generation belongs to a stale or unavailable host: {generation_id}"
        )

    def health(
        self,
        generation_id: str,
        server_name: str,
        token: object | None = None,
    ) -> bool:
        generation = self._require_generation(generation_id)
        self._assert_token(generation, token)
        entry = generation.entries.get(server_name)
        if entry is None:
            raise KeyError(f"unknown MCP server: {generation_id}:{server_name}")
        return (
            not entry.stopping
            and entry.client.connected
            and entry.client._recovery_task is None
            and not entry.client._recovering
            and entry.client._fatal_failure is None
            and generation.state == "ready"
        )

    def logs(
        self,
        generation_id: str,
        server_name: str,
        token: object | None = None,
    ) -> McpLogView:
        generation = self._require_generation(generation_id)
        self._assert_token(generation, token)
        entry = generation.entries.get(server_name)
        if entry is None:
            raise KeyError(f"unknown MCP server: {generation_id}:{server_name}")
        # McpClient owns protocol stream readers and keeps these deques bounded.
        return McpLogView(
            stdout=tuple(entry.client._recent_stdout)[-_MAX_LOG_LINES:],
            stderr=tuple(entry.client._recent_stderr)[-_MAX_LOG_LINES:],
        )

    def catalog_digest(
        self,
        generation_id: str,
        server_name: str,
        token: object | None = None,
    ) -> str:
        generation = self._require_generation(generation_id)
        self._assert_token(generation, token)
        entry = generation.entries.get(server_name)
        if entry is None:
            raise KeyError(f"unknown MCP server: {generation_id}:{server_name}")
        if generation.state != "ready" or entry.stopping:
            raise RuntimeError(f"MCP generation {generation_id!r} 当前不可检查 catalog")
        return entry.catalog_digest

    def route_for(self, generation_id: str, server_name: str) -> McpRoute:
        """Create a route bound to the exact current generation entry."""

        generation = self._require_generation(generation_id)
        entry = generation.entries.get(server_name)
        if entry is None:
            raise KeyError(f"unknown MCP server: {generation_id}:{server_name}")
        self._resolve_route(generation_id, generation.token, entry)
        return McpRoute(self, generation_id, generation.token, entry)

    def assert_healthy(self, generation_id: str, token: object | None = None) -> None:
        """Reject promotion while any client is recovering, failed or disconnected."""

        generation = self._require_generation(generation_id)
        self._assert_token(generation, token)
        if generation.state != "ready":
            raise RuntimeError(
                f"MCP generation {generation_id!r} 当前状态不能晋升: {generation.state}"
            )
        for entry in generation.entries.values():
            if not self.health(generation_id, entry.name, generation.token):
                if entry.client._fatal_failure is not None:
                    raise entry.client._fatal_failure
                if entry.client._recovering or entry.client._recovery_task is not None:
                    raise RuntimeError(
                        f"MCP server {entry.client.name!r} 正在恢复，不能晋升"
                    )
                raise RuntimeError(
                    f"MCP server {entry.client.name!r} 当前无可用 process epoch"
                )

    async def close(self) -> None:
        """Drain every owned generation and preserve cleanup failures."""

        errors: list[BaseException] = []
        cancelled = False
        for generation_id in tuple(self._generations):
            try:
                await self.stop_generation(generation_id)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as error:
                errors.append(error)
        if cancelled:
            raise asyncio.CancelledError
        if errors:
            raise RuntimeError(
                "MCP generation host cleanup failed: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

    async def _start_entry(
        self,
        generation: _Generation,
        binding: McpServerBinding,
        materialized: McpMaterializedCommand,
        endpoint_ports: Mapping[str, int],
        workload_endpoints: Mapping[tuple[str, str], str],
        expected_catalog_digest: str | None,
    ) -> _McpEntry:
        definition = binding.definition
        environment = self._materialize_env(
            binding.descriptor,
            materialized,
            generation.mode,
            endpoint_ports,
            workload_endpoints,
        )
        allowed_tools = frozenset(
            definition.candidate_read_only_tools
            if generation.mode == "candidate"
            else ()
        )
        client = McpClient(
            name=f"{definition.name}@{generation.generation_id}",
            command=list(materialized.command),
            env=environment,
            cwd=materialized.cwd,
            env_scrub_keys=frozenset(
                key for key, _ in binding.descriptor.candidate_env
            ),
        )
        self._next_epoch += 1
        entry = _McpEntry(
            generation_id=generation.generation_id,
            name=definition.name,
            binding=binding,
            materialized=materialized,
            client=client,
            mode=generation.mode,
            allowed_tools=frozenset(),
            tools=MappingProxyType({}),
            catalog_tools=(),
            catalog_digest="",
            epoch=self._next_epoch,
            process_identity=None,
        )
        # Register the client before any await so cancellation or handshake
        # failure leaves a Core-owned cleanup handle behind.
        generation.entries[definition.name] = entry
        await self._emit_health(
            generation.generation_id,
            definition.name,
            False,
            "starting",
        )
        try:
            infos = await asyncio.wait_for(
                client.connect(),
                timeout=self._readiness_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            await self._emit_incident(
                generation.generation_id,
                definition.name,
                "handshake_failed",
                _error_text(error),
            )
            raise
        catalog_tools = tuple(
            _tool_view(info) for info in sorted(infos, key=lambda item: item.name)
        )
        catalog_digest = _catalog_digest(catalog_tools)
        if (
            expected_catalog_digest is not None
            and catalog_digest != expected_catalog_digest
        ):
            await _disconnect_after_readiness_failure(client)
            raise RuntimeError(
                f"MCP server {definition.name!r} tools/list catalog drift: "
                f"expected={expected_catalog_digest} actual={catalog_digest}"
            )
        actual = {info.name for info in infos}
        missing_required = sorted(set(definition.required_tools) - actual)
        if missing_required:
            await _disconnect_after_readiness_failure(client)
            raise RuntimeError(
                f"MCP server {definition.name!r} required tool 缺失: "
                + ", ".join(missing_required)
            )
        missing_candidate = sorted(allowed_tools - actual)
        if missing_candidate:
            await _disconnect_after_readiness_failure(client)
            raise RuntimeError(
                f"MCP server {definition.name!r} candidate read-only tool 缺失: "
                + ", ".join(missing_candidate)
            )
        visible_tools = actual if generation.mode == "formal" else allowed_tools
        entry.allowed_tools = (
            frozenset(actual) if generation.mode == "formal" else allowed_tools
        )
        entry.catalog_tools = catalog_tools
        entry.catalog_digest = catalog_digest
        entry.tools = MappingProxyType(
            {tool.name: tool for tool in catalog_tools if tool.name in visible_tools}
        )
        entry.process_identity = client._process
        return entry

    async def _watch_entry(
        self,
        generation: _Generation,
        entry: _McpEntry,
        observed_epoch: int,
    ) -> None:
        """Observe client-owned epochs without allowing stale recovery callbacks."""

        process_identity = entry.process_identity
        try:
            while True:
                await asyncio.sleep(_POLL_SECONDS)
                if entry.stopping or generation.state in {"stopping", "cleanup_failed"}:
                    return
                if self._generations.get(generation.generation_id) is not generation:
                    return
                if generation.entries.get(entry.name) is not entry:
                    return
                current_process = entry.client._process
                if (
                    current_process is not None
                    and current_process is not process_identity
                ):
                    process_identity = current_process
                    entry.process_identity = current_process
                    self._next_epoch += 1
                    entry.epoch = self._next_epoch
                    observed_epoch = entry.epoch
                    try:
                        if entry.client.connected and not entry.client._recovering:
                            await self._emit_health(
                                generation.generation_id,
                                entry.name,
                                True,
                                "recovered",
                            )
                        await self._emit_incident(
                            generation.generation_id,
                            entry.name,
                            "process_epoch",
                            f"MCP process epoch advanced to {entry.epoch}",
                        )
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        generation.state = "degraded"
                        self._retain_tombstone(generation, error, state="degraded")
                        return
                if entry.client._fatal_failure is not None:
                    if generation.state != "degraded":
                        generation.state = "degraded"
                        failure = RuntimeError(_error_text(entry.client._fatal_failure))
                        try:
                            await self._emit_health(
                                generation.generation_id,
                                entry.name,
                                False,
                                "recovery_exhausted",
                            )
                            await self._emit_incident(
                                generation.generation_id,
                                entry.name,
                                "recovery_exhausted",
                                _error_text(entry.client._fatal_failure),
                            )
                        except asyncio.CancelledError:
                            raise
                        except BaseException as callback_error:
                            failure = callback_error
                        self._retain_tombstone(generation, failure, state="degraded")
                    return
                if entry.epoch != observed_epoch:
                    return
        except asyncio.CancelledError as error:
            if self._watch_stop_requested(generation, entry):
                raise
            failure = RuntimeError(
                f"MCP {entry.name!r} runtime bridge cancelled: {_error_text(error)}"
            )
            self._retain_tombstone(generation, failure, state="degraded")
            logger.error("[mcp] %s", failure)
            return

    async def _cleanup_generation(self, generation: _Generation) -> None:
        """Stop every client in reverse order while retaining all failures."""

        errors: list[BaseException] = []
        for entry in reversed(tuple(generation.entries.values())):
            try:
                await self._cleanup_entry(entry)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                f"MCP generation cleanup failed: {generation.generation_id}: "
                + "; ".join(_error_text(error) for error in errors)
            ) from errors[0]

    async def _cleanup_entry(self, entry: _McpEntry) -> None:
        entry.stopping = True
        watcher = entry.watcher
        if watcher is not None and watcher is not asyncio.current_task():
            if not watcher.done():
                _ = watcher.cancel()
            await _await_task_after_cancellation(watcher)
            entry.watcher = None
        try:
            await asyncio.wait_for(
                entry.client.disconnect(),
                timeout=self._stop_timeout_seconds,
            )
        except TimeoutError as error:
            raise RuntimeError(
                f"MCP server {entry.name!r} stop 超时: {self._stop_timeout_seconds}s"
            ) from error
        try:
            await self._emit_health(entry.generation_id, entry.name, False, "stopped")
        except (asyncio.CancelledError, Exception) as error:
            logger.error(
                "[mcp] observation callback failed for %s:%s: %s",
                entry.generation_id,
                entry.name,
                _error_text(error),
            )

    def _retain_tombstone(
        self,
        generation: _Generation,
        error: BaseException,
        *,
        state: Literal["cleanup_failed", "degraded"] = "cleanup_failed",
    ) -> None:
        generation.cleanup_attempts += 1
        generation.state = state
        action: Literal["retry_generation_cleanup", "retry_runtime_recovery"] = (
            "retry_runtime_recovery"
            if state == "degraded"
            else "retry_generation_cleanup"
        )
        tombstone = McpCleanupTombstone(
            generation_id=generation.generation_id,
            state=state,
            action=action,
            resource_names=tuple(generation.entries),
            error=_error_text(error),
            attempt_count=generation.cleanup_attempts,
        )
        self._tombstones[generation.generation_id] = tombstone
        if self._on_failure is not None:
            self._on_failure(tombstone)

    def _watch_stop_requested(
        self,
        generation: _Generation,
        entry: _McpEntry,
    ) -> bool:
        return (
            entry.stopping
            or generation.state in {"stopping", "cleanup_failed"}
            or self._generations.get(generation.generation_id) is not generation
            or generation.entries.get(entry.name) is not entry
        )

    async def _emit_health(
        self,
        generation_id: str,
        server_name: str,
        healthy: bool,
        reason: str,
    ) -> None:
        await _emit_callback(
            self._on_health, generation_id, server_name, healthy, reason
        )

    async def _emit_incident(
        self,
        generation_id: str,
        server_name: str,
        kind: str,
        message: str,
    ) -> None:
        await _emit_callback(
            self._on_incident, generation_id, server_name, kind, message
        )

    def _resolve_route(
        self,
        generation_id: str,
        token: object,
        entry: _McpEntry,
    ) -> _McpEntry:
        generation = self._require_generation(generation_id)
        self._assert_token(generation, token)
        if generation.state != "ready":
            raise RuntimeError(
                f"MCP generation {generation_id!r} 当前不可调用: {generation.state}"
            )
        if generation.entries.get(entry.name) is not entry or entry.stopping:
            raise RuntimeError("MCP route 属于 stale generation epoch")
        return entry

    @staticmethod
    def _validate_generation_id(generation_id: str) -> None:
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise ValueError("MCP generation_id must be non-empty")

    @staticmethod
    def _validate_registry(
        registry: McpServerRegistry,
    ) -> dict[str, McpServerBinding]:
        if type(registry) is not McpServerRegistry:
            raise TypeError("MCP host requires an exact frozen McpServerRegistry")
        bindings: dict[str, McpServerBinding] = {}
        for name, binding in registry.items():
            if type(binding) is not McpServerBinding:
                raise TypeError("MCP registry binding type invalid")
            if name != binding.definition.name or name != binding.descriptor.name:
                raise ValueError(f"MCP registry key/name mismatch: {name}")
            if type(binding.definition) is not McpServerDefinition:
                raise TypeError("MCP definition type invalid")
            if type(binding.descriptor) is not McpServerDescriptor:
                raise TypeError("MCP descriptor type invalid")
            if not binding.is_owned():
                raise RuntimeError(f"MCP declaration owner is not live: {name}")
            if _descriptor_fields(binding.descriptor) != _definition_fields(
                binding.definition
            ):
                raise ValueError(f"MCP definition/descriptor drift: {name}")
            declared_env = {key for key, _ in binding.descriptor.env}
            candidate_env = {key for key, _ in binding.descriptor.candidate_env}
            overlap = sorted(declared_env & candidate_env)
            if overlap:
                raise ValueError(
                    f"MCP env 与 candidate_env 不得重叠: {name}: " + ", ".join(overlap)
                )
            bindings[name] = binding
        return dict(sorted(bindings.items()))

    @staticmethod
    def _validate_expected_catalog_digests(
        bindings: Mapping[str, McpServerBinding],
        mode: McpMode,
        expected: Mapping[str, str] | None,
    ) -> dict[str, str]:
        if expected is None:
            return {}
        if mode == "candidate":
            raise ValueError(
                "MCP candidate generation cannot receive formal catalog expectations"
            )
        if not isinstance(expected, Mapping):
            raise TypeError("MCP expected catalog digests must be a mapping")
        if set(expected) != set(bindings):
            raise ValueError("MCP expected catalog digests must exactly match registry")
        result: dict[str, str] = {}
        for name, digest in expected.items():
            if not isinstance(digest, str) or not digest:
                raise TypeError(f"MCP expected catalog digest invalid: {name}")
            result[name] = digest
        return result

    @staticmethod
    def _validate_materialized_commands(
        bindings: Mapping[str, McpServerBinding],
        commands: Mapping[str, McpMaterializedCommand],
        endpoint_ports: Mapping[str, int],
        workload_endpoints: Mapping[tuple[str, str], str],
    ) -> dict[str, McpMaterializedCommand]:
        if not isinstance(commands, Mapping):
            raise TypeError("MCP materialized commands must be a mapping")
        if set(commands) != set(bindings):
            raise ValueError("MCP materialized commands must exactly match registry")
        for process_name, port in endpoint_ports.items():
            if (
                not isinstance(process_name, str)
                or not isinstance(port, int)
                or isinstance(port, bool)
            ):
                raise TypeError("MCP endpoint ports must be process-name/integer pairs")
            if not 1 <= port <= 65535:
                raise ValueError(f"MCP endpoint port invalid: {process_name}={port}")
        for name, binding in bindings.items():
            required_processes = {
                endpoint.process for endpoint in binding.descriptor.endpoint_env
            }
            missing_processes = required_processes - set(endpoint_ports)
            if missing_processes:
                raise ValueError(
                    f"MCP endpoint materialization missing for {name}: "
                    + ", ".join(sorted(missing_processes))
                )
            required_workloads = {
                (endpoint.workload, endpoint.port)
                for endpoint in binding.descriptor.workload_env
            }
            missing_workloads = required_workloads - set(workload_endpoints)
            if missing_workloads:
                raise ValueError(
                    f"MCP workload endpoint materialization missing for {name}: "
                    + ", ".join(
                        f"{workload}:{port}"
                        for workload, port in sorted(missing_workloads)
                    )
                )
        for key, url in workload_endpoints.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(not isinstance(item, str) or not item for item in key)
                or not isinstance(url, str)
                or not url.startswith("http://")
            ):
                raise TypeError("MCP workload endpoints must be name/port HTTP pairs")
        result: dict[str, McpMaterializedCommand] = {}
        for name, materialized in commands.items():
            if type(materialized) is not McpMaterializedCommand:
                raise TypeError("MCP command must be Core-owned McpMaterializedCommand")
            if (
                not isinstance(materialized.command, tuple)
                or not materialized.command
                or any(
                    not isinstance(item, str) or not item
                    for item in materialized.command
                )
                or not isinstance(materialized.cwd, str)
                or not materialized.cwd
                or not materialized.cwd.startswith("/")
            ):
                raise ValueError(f"MCP materialized command invalid: {name}")
            argv0 = Path(materialized.command[0])
            if (
                not argv0.is_absolute()
                or not argv0.is_file()
                or not os.access(argv0, os.X_OK)
            ):
                raise ValueError(
                    f"MCP materialized argv[0] must be an absolute executable: {name}"
                )
            if not isinstance(materialized.env, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in materialized.env.items()
            ):
                raise TypeError(f"MCP materialized environment invalid: {name}")
            result[name] = McpMaterializedCommand(
                command=tuple(materialized.command),
                cwd=materialized.cwd,
                env=MappingProxyType(dict(materialized.env)),
            )
        return result

    @staticmethod
    def _materialize_env(
        descriptor: McpServerDescriptor,
        materialized: McpMaterializedCommand,
        mode: McpMode,
        endpoint_ports: Mapping[str, int],
        workload_endpoints: Mapping[tuple[str, str], str],
    ) -> dict[str, str]:
        environment = dict(materialized.env)
        candidate_keys = {key for key, _ in descriptor.candidate_env}
        materialized_candidate_keys = sorted(candidate_keys & set(environment))
        if materialized_candidate_keys:
            raise ValueError(
                f"MCP materialized env 不得提供 candidate-only key: "
                f"{descriptor.name}: " + ", ".join(materialized_candidate_keys)
            )
        for key, value in descriptor.env:
            existing = environment.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"MCP materialized env drift: {descriptor.name}:{key}")
            environment[key] = value
        if mode == "candidate":
            for key, value in descriptor.candidate_env:
                existing = environment.get(key)
                if existing is not None and existing != value:
                    raise ValueError(
                        f"MCP candidate env drift: {descriptor.name}:{key}"
                    )
                environment[key] = value
        for endpoint in descriptor.endpoint_env:
            if endpoint.process not in endpoint_ports:
                raise ValueError(
                    f"MCP endpoint process 未 materialize: {descriptor.name}:{endpoint.process}"
                )
            if endpoint.env in environment:
                raise ValueError(
                    f"MCP endpoint env 已被占用: {descriptor.name}:{endpoint.env}"
                )
            environment[endpoint.env] = str(endpoint_ports[endpoint.process])
        for endpoint in descriptor.workload_env:
            key = (endpoint.workload, endpoint.port)
            if key not in workload_endpoints:
                raise ValueError(
                    "MCP workload endpoint 未 materialize: "
                    f"{descriptor.name}:{endpoint.workload}:{endpoint.port}"
                )
            if endpoint.env in environment:
                raise ValueError(
                    f"MCP workload env 已被占用: {descriptor.name}:{endpoint.env}"
                )
            environment[endpoint.env] = workload_endpoints[key]
        return environment

    def _require_generation(self, generation_id: str) -> _Generation:
        generation = self._generations.get(generation_id)
        if generation is None:
            raise RuntimeError(
                f"MCP generation belongs to a stale or unavailable host: {generation_id}"
            )
        return generation

    @staticmethod
    def _assert_token(generation: _Generation, token: object | None) -> None:
        if token is not None and token is not generation.token:
            raise RuntimeError("MCP generation belongs to another host Root")


def _descriptor_fields(descriptor: McpServerDescriptor) -> tuple[object, ...]:
    return (
        descriptor.name,
        descriptor.command,
        descriptor.cwd,
        descriptor.env,
        descriptor.required_tools,
        descriptor.candidate_read_only_tools,
        descriptor.endpoint_env,
        descriptor.workload_env,
        descriptor.candidate_env,
    )


def _definition_fields(definition: McpServerDefinition) -> tuple[object, ...]:
    return (
        definition.name,
        definition.command,
        definition.cwd,
        tuple(sorted(definition.env.items())),
        definition.required_tools,
        definition.candidate_read_only_tools,
        definition.endpoint_env,
        definition.workload_env,
        tuple(sorted(definition.candidate_env.items())),
    )


def _tool_view(info: Any) -> McpToolView:
    return McpToolView(
        name=info.name,
        description=info.description,
        input_schema=_freeze_schema(info.input_schema),
    )


def _catalog_digest(tools: tuple[McpToolView, ...]) -> str:
    contract = [
        (
            tool.name,
            tool.description,
            _schema_json(tool.input_schema),
        )
        for tool in tools
    ]
    payload = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _schema_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _schema_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_schema_json(child) for child in value]
    return value


def _freeze_schema(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze a tool schema before handing it to the facade."""

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {str(key): freeze(child) for key, child in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return cast(Mapping[str, Any], freeze(value))


async def _disconnect_after_readiness_failure(client: McpClient) -> None:
    try:
        await client.disconnect()
    except BaseException as error:
        logger.error("MCP readiness cleanup failed: %s", _error_text(error))
        raise


async def _emit_callback(
    callback: Callable[..., Any] | None,
    *arguments: object,
) -> None:
    if callback is None:
        return
    try:
        result = callback(*arguments)
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception:
        # A bridge exception is part of the readiness contract.  The caller
        # must fail and drain the generation instead of publishing "ready".
        raise


async def _await_task_after_cancellation(task: asyncio.Task[Any]) -> Any:
    """Finish a cleanup task before restoring caller cancellation."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
            cancelled = True
            continue
    try:
        result = task.result()
    except asyncio.CancelledError:
        # A cleanup child is commonly cancelled intentionally (for example a
        # watcher).  Its cancellation is not a cleanup failure.
        result = None
    if cancelled:
        raise asyncio.CancelledError
    return result


def _task_error(task: asyncio.Task[Any], fallback: BaseException) -> BaseException:
    if not task.done():
        return fallback
    try:
        error = task.exception()
    except asyncio.CancelledError:
        return fallback
    return error if error is not None else fallback


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
