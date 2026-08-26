"""Materialize frozen v3 process and MCP declarations for one generation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from agent.mcp.client import McpToolExecutionError
from agent.plugin_composition import FiberState
from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.mcp_slots import (
    McpServerBinding,
    McpServerRegistry,
)
from agent.plugin_composition.process_slots import (
    ManagedProcessBinding,
    ManagedProcessDefinition,
)
from agent.plugins.generation import PluginGeneration
from agent.plugins.managed_process_host import (
    GenerationCleanupTombstone,
    ManagedProcessGeneration,
    ManagedProcessGenerationHost,
)
from agent.plugins.mcp_generation_host import (
    McpGeneration,
    McpGenerationHost,
    McpCleanupTombstone,
    McpMaterializedCommand,
    McpRoute,
    McpServerView,
    McpToolView,
)
from agent.plugins.snapshot import RuntimeSnapshot
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry

RuntimeMode = Literal["candidate", "formal"]
_PYTHON_COMMAND = re.compile(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", re.IGNORECASE)
_CORE_DATA_ENV = ("AKA_PLUGIN_DATA_DIR", "AKASHIC_PLUGIN_DATA_DIR")


@dataclass(frozen=True, slots=True)
class CompositionRuntimeGeneration:
    """Expose only immutable runtime facades for one exact generation."""

    plugin_id: str
    generation_id: str
    mode: RuntimeMode
    processes: ManagedProcessGeneration | None
    mcp: McpGeneration | None

    @property
    def mcp_catalog_digests(self) -> Mapping[str, str]:
        if self.mcp is None:
            return MappingProxyType({})
        return MappingProxyType(
            {
                name: self.mcp.catalog_digest(name)
                for name in self.mcp
            }
        )


@dataclass(frozen=True, slots=True)
class CompositionRuntimeFailure:
    """Describe retained Core runtime ownership after cleanup or recovery failed."""

    generation_id: str
    state: Literal["cleanup_failed", "degraded"]
    action: Literal["retry_generation_cleanup", "retry_runtime_recovery"]
    resource_names: tuple[str, ...]
    error: str
    attempt_count: int


@dataclass(slots=True)
class _RootBridge:
    root_instance_token: object
    process_bindings: dict[str, ManagedProcessBinding]
    mcp_bindings: dict[str, McpServerBinding]


@dataclass(slots=True)
class _RuntimeOwner:
    generation: CompositionRuntimeGeneration
    bridge: _RootBridge


class CompositionGenerationHost:
    """Own v3 managed-process and MCP runtime state until generation drain."""

    def __init__(
        self,
        *,
        on_failure: Callable[[CompositionRuntimeFailure], None] | None = None,
    ) -> None:
        self._on_failure = on_failure
        self._bridges: dict[str, _RootBridge] = {}
        self._detached_observers: set[str] = set()
        self._owners: dict[str, _RuntimeOwner] = {}
        self._process_host = ManagedProcessGenerationHost(
            on_health=self._on_process_health,
            on_incident=self._on_process_incident,
            on_failure=self._on_runtime_failure,
        )
        self._mcp_host = McpGenerationHost(
            on_health=self._on_mcp_health,
            on_incident=self._on_mcp_incident,
            on_failure=self._on_runtime_failure,
        )

    async def start(
        self,
        generation: PluginGeneration,
        snapshot: RuntimeSnapshot,
        *,
        mode: RuntimeMode,
        expected_mcp_catalog_digests: Mapping[str, str] | None = None,
    ) -> CompositionRuntimeGeneration | None:
        """Start exact Root declarations and return their read-only facades."""

        if generation.generation_id in self._owners:
            raise RuntimeError(
                f"v3 runtime generation 已存在: {generation.generation_id}"
            )
        self._detached_observers.discard(generation.generation_id)
        root = snapshot.composition_root
        if root is None:
            return None

        # 1. Bind only this plugin's declarations to the exact compiled Root.
        process_bindings = _owned_process_bindings(snapshot, generation.plugin_id)
        mcp_bindings = _owned_mcp_bindings(snapshot, generation.plugin_id)
        if not process_bindings and not mcp_bindings:
            return None
        _assert_root_token(snapshot, root.instance_token)
        bridge = _RootBridge(
            root_instance_token=root.instance_token,
            process_bindings=process_bindings,
            mcp_bindings=mcp_bindings,
        )
        self._bridges[generation.generation_id] = bridge
        if not generation.composition_runtime_cleanup_registered:
            generation.scope.defer(
                "composition_runtime_generation",
                lambda: self.stop(generation.generation_id),
            )
            generation.composition_runtime_cleanup_registered = True

        # 2. Start loopback processes before MCP endpoint materialization.
        processes: ManagedProcessGeneration | None = None
        mcp: McpGeneration | None = None
        try:
            if process_bindings:
                definitions = _materialized_process_definitions(
                    generation,
                    process_bindings,
                )
                processes = await self._process_host.start_generation(
                    generation.generation_id,
                    definitions,
                    mode=mode,
                )
            if mcp_bindings:
                registry = McpServerRegistry(
                    mcp_bindings,
                    root_instance_token=root.instance_token,
                )
                commands = _materialized_mcp_commands(
                    generation,
                    mcp_bindings,
                )
                ports = (
                    {
                        name: endpoint.port
                        for name, endpoint in processes.endpoints.items()
                    }
                    if processes is not None
                    else {}
                )
                mcp = await self._mcp_host.start_generation(
                    generation.generation_id,
                    registry,
                    commands,
                    mode=mode,
                    endpoint_ports=ports,
                    expected_catalog_digests=expected_mcp_catalog_digests,
                )
        except BaseException as start_error:
            try:
                await self._stop_partial(
                    generation.generation_id,
                    mcp=mcp,
                    processes=processes,
                )
            except BaseException as cleanup_error:
                runtime = CompositionRuntimeGeneration(
                    plugin_id=generation.plugin_id,
                    generation_id=generation.generation_id,
                    mode=mode,
                    processes=processes,
                    mcp=mcp,
                )
                self._owners[generation.generation_id] = _RuntimeOwner(
                    runtime,
                    bridge,
                )
                raise RuntimeError(
                    "v3 runtime start 失败且 cleanup 未完成"
                ) from cleanup_error
            _ = self._bridges.pop(generation.generation_id, None)
            raise start_error

        # 3. Publish the Core owner only after every readiness/handshake succeeds.
        runtime = CompositionRuntimeGeneration(
            plugin_id=generation.plugin_id,
            generation_id=generation.generation_id,
            mode=mode,
            processes=processes,
            mcp=mcp,
        )
        self._owners[generation.generation_id] = _RuntimeOwner(runtime, bridge)
        return runtime

    def attach_tools(
        self,
        registry: ToolRegistry | None,
        runtime: CompositionRuntimeGeneration | None,
    ) -> ToolRegistry | None:
        """Attach exact generation MCP routes to one snapshot ToolRegistry."""

        if registry is None or runtime is None or runtime.mcp is None:
            return registry
        for server in runtime.mcp.values():
            for tool in server.tools.values():
                wrapper = _McpRouteTool(runtime.plugin_id, server, tool)
                if registry.has_tool(wrapper.name):
                    raise RuntimeError(f"MCP 工具名称重复: {wrapper.name}")
                registry.register(
                    wrapper,
                    risk=(
                        "read-only"
                        if runtime.mode == "candidate"
                        else "external-side-effect"
                    ),
                    source_type="mcp",
                    source_name=server.name,
                )
        return registry

    async def stop(self, generation_id: str) -> None:
        """Stop MCP then managed processes and retain failed Host ownership."""

        self._detached_observers.add(generation_id)
        _ = self._bridges.pop(generation_id, None)
        owner = self._owners.get(generation_id)
        mcp = None if owner is None else owner.generation.mcp
        processes = None if owner is None else owner.generation.processes
        await self._stop_partial(generation_id, mcp=mcp, processes=processes)
        _ = self._owners.pop(generation_id, None)
        self._detached_observers.discard(generation_id)

    def get(self, generation_id: str) -> CompositionRuntimeGeneration | None:
        owner = self._owners.get(generation_id)
        return None if owner is None else owner.generation

    def route_for(self, generation_id: str, server_name: str) -> McpRoute:
        """Return one exact formal MCP route for a generation-bound Core consumer."""

        runtime = self.get(generation_id)
        if runtime is None or runtime.mcp is None:
            raise RuntimeError(
                f"v3 runtime MCP generation 不存在: {generation_id}:{server_name}"
            )
        return runtime.mcp.route(server_name)

    def failure(self, generation_id: str) -> CompositionRuntimeFailure | None:
        """Return one aggregated failure receipt for both runtime hosts."""

        tombstones = tuple(
            (kind, tombstone)
            for kind, tombstone in (
                ("mcp", self._mcp_host.tombstone(generation_id)),
                ("process", self._process_host.tombstone(generation_id)),
            )
            if tombstone is not None
        )
        if not tombstones:
            return None
        degraded = any(item.state == "degraded" for _, item in tombstones)
        resources = tuple(
            f"{kind}:{name}"
            for kind, item in tombstones
            for name in item.resource_names
        )
        return CompositionRuntimeFailure(
            generation_id=generation_id,
            state="degraded" if degraded else "cleanup_failed",
            action=(
                "retry_runtime_recovery"
                if degraded
                else "retry_generation_cleanup"
            ),
            resource_names=resources,
            error="; ".join(
                f"{kind}: {item.error}" for kind, item in tombstones
            ),
            attempt_count=max(item.attempt_count for _, item in tombstones),
        )

    async def retry_generation_cleanup(self, generation_id: str) -> str:
        """Retry every retained cleanup owner and return durable evidence."""

        await self._retry_retained_runtime(generation_id, recover_degraded=False)
        return f"composition-runtime:{generation_id}:cleanup-complete"

    async def retry_runtime_recovery(self, generation_id: str) -> str:
        """Resolve degraded owners, finish cleanup, and return durable evidence."""

        await self._retry_retained_runtime(generation_id, recover_degraded=True)
        return f"composition-runtime:{generation_id}:recovery-complete"

    async def _retry_retained_runtime(
        self,
        generation_id: str,
        *,
        recover_degraded: bool,
    ) -> None:
        """Retry both protocol hosts without invoking unloaded plugin code."""

        if self.failure(generation_id) is None:
            raise RuntimeError(
                f"v3 runtime generation 没有 retained failure: {generation_id}"
            )
        for host in (self._mcp_host, self._process_host):
            tombstone = host.tombstone(generation_id)
            if tombstone is None:
                if host.get(generation_id) is not None:
                    await host.stop_generation(generation_id)
                continue
            if tombstone.state == "degraded":
                if not recover_degraded:
                    raise RuntimeError(
                        f"v3 runtime generation 需要 recovery: {generation_id}"
                    )
                await host.retry_runtime_recovery(generation_id)
            else:
                await host.retry_generation_cleanup(generation_id)
        if self.failure(generation_id) is not None:
            raise RuntimeError(f"v3 runtime generation retry 未清空: {generation_id}")
        _ = self._owners.pop(generation_id, None)
        _ = self._bridges.pop(generation_id, None)
        self._detached_observers.discard(generation_id)

    async def _stop_partial(
        self,
        generation_id: str,
        *,
        mcp: McpGeneration | None,
        processes: ManagedProcessGeneration | None,
    ) -> None:
        errors: list[BaseException] = []
        cancelled = False
        mcp_tombstone = self._mcp_host.tombstone(generation_id)
        if mcp_tombstone is not None:
            errors.append(
                RuntimeError(
                    f"MCP generation cleanup 未完成: {mcp_tombstone.error}"
                )
            )
        elif mcp is not None or self._mcp_host.get(generation_id) is not None:
            try:
                await self._mcp_host.stop_generation(generation_id)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as error:
                errors.append(error)
        process_tombstone = self._process_host.tombstone(generation_id)
        if process_tombstone is not None:
            errors.append(
                RuntimeError(
                    "managed process generation cleanup 未完成: "
                    f"{process_tombstone.error}"
                )
            )
        elif processes is not None or self._process_host.get(generation_id) is not None:
            try:
                await self._process_host.stop_generation(generation_id)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                "v3 runtime generation cleanup failed: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]
        if cancelled:
            raise asyncio.CancelledError

    def _binding(
        self,
        generation_id: str,
        name: str,
        *,
        mcp: bool,
    ) -> ManagedProcessBinding | McpServerBinding:
        bridge = self._bridges.get(generation_id)
        if bridge is None:
            raise RuntimeError(f"v3 runtime Root bridge 已释放: {generation_id}")
        bindings = bridge.mcp_bindings if mcp else bridge.process_bindings
        binding = bindings.get(name)
        if binding is None:
            raise RuntimeError(f"v3 runtime declaration owner 不存在: {generation_id}:{name}")
        if (
            binding.owner_fiber.state is not FiberState.ACTIVE
            or binding.owner_fiber.activation_token is not binding.activation_token
        ):
            raise RuntimeError(f"v3 runtime declaration 已失效: {generation_id}:{name}")
        return binding

    def _on_process_health(
        self,
        generation_id: str,
        process_name: str,
        healthy: bool,
        reason: str,
    ) -> None:
        if generation_id in self._detached_observers:
            return
        binding = cast(
            ManagedProcessBinding,
            self._binding(generation_id, process_name, mcp=False),
        )
        binding.health.recover() if healthy else binding.health.degrade(reason)

    def _on_process_incident(
        self,
        generation_id: str,
        process_name: str,
        kind: str,
        message: str,
    ) -> None:
        if generation_id in self._detached_observers:
            return
        binding = cast(
            ManagedProcessBinding,
            self._binding(generation_id, process_name, mcp=False),
        )
        _ = binding.incident_reporter(kind, message)

    def _on_mcp_health(
        self,
        generation_id: str,
        server_name: str,
        healthy: bool,
        reason: str,
    ) -> None:
        if generation_id in self._detached_observers:
            return
        binding = cast(
            McpServerBinding,
            self._binding(generation_id, server_name, mcp=True),
        )
        binding.health.recover() if healthy else binding.health.degrade(reason)

    def _on_mcp_incident(
        self,
        generation_id: str,
        server_name: str,
        kind: str,
        message: str,
    ) -> None:
        if generation_id in self._detached_observers:
            return
        binding = cast(
            McpServerBinding,
            self._binding(generation_id, server_name, mcp=True),
        )
        _ = binding.incident_reporter(kind, message)

    def _on_runtime_failure(
        self,
        tombstone: GenerationCleanupTombstone | McpCleanupTombstone,
    ) -> None:
        """Forward one aggregated retained owner to the Core publication owner."""

        failure = self.failure(tombstone.generation_id)
        if failure is not None and self._on_failure is not None:
            self._on_failure(failure)


class _McpRouteTool(Tool):
    """Expose one exact generation MCP route as a standard Tool."""

    def __init__(
        self,
        plugin_id: str,
        server: McpServerView,
        tool: McpToolView,
    ) -> None:
        self._plugin_id = plugin_id
        self._server = server
        self._tool = tool

    @property
    def name(self) -> str:
        return f"mcp_{self._server.name}__{self._tool.name}"

    @property
    def description(self) -> str:
        return f"[MCP:{self._server.name}] {self._tool.description}"

    @property
    def parameters(self) -> dict[str, Any]:
        return cast(dict[str, Any], _thaw(self._tool.input_schema))

    async def execute(self, **kwargs: Any) -> str:
        with plugin_entrypoint(
            plugin_id=self._plugin_id,
            generation_id=self._server.generation_id,
            fiber=self._plugin_id,
            operation="mcp.tool",
            entrypoint=f"{self._server.name}.{self._tool.name}",
        ):
            async with self._server.route() as route:
                result = await route.call(self._tool.name, kwargs)
            if result.tool_error:
                raise McpToolExecutionError(result.output)
        return result.output


def _owned_process_bindings(
    snapshot: RuntimeSnapshot,
    plugin_id: str,
) -> dict[str, ManagedProcessBinding]:
    registry = snapshot.managed_process_registry
    if registry is None:
        return {}
    return {
        name: binding
        for name, binding in registry.items()
        if binding.descriptor.owner == plugin_id
    }


def _owned_mcp_bindings(
    snapshot: RuntimeSnapshot,
    plugin_id: str,
) -> dict[str, McpServerBinding]:
    registry = snapshot.mcp_server_registry
    if registry is None:
        return {}
    return {
        name: binding
        for name, binding in registry.items()
        if binding.descriptor.owner == plugin_id
    }


def _assert_root_token(snapshot: RuntimeSnapshot, root_token: object) -> None:
    process_registry = snapshot.managed_process_registry
    mcp_registry = snapshot.mcp_server_registry
    for label, registry in (
        ("managed process", process_registry),
        ("MCP", mcp_registry),
    ):
        if registry is not None and registry.root_instance_token is not root_token:
            raise RuntimeError(f"{label} registry 不属于 snapshot exact Root")


def _materialized_process_definitions(
    generation: PluginGeneration,
    bindings: Mapping[str, ManagedProcessBinding],
) -> dict[str, ManagedProcessDefinition]:
    commands = dict(generation.static_runtime_commands)
    result: dict[str, ManagedProcessDefinition] = {}
    for name, binding in bindings.items():
        command = _runtime_command(
            generation,
            commands,
            key=f"process:{name}",
            declared=binding.definition.command,
        )
        result[name] = replace(
            binding.definition,
            command=command,
            cwd=str(
                _runtime_cwd(
                    binding.runtime_plugin_dir,
                    binding.definition.cwd,
                )
            ),
            env=MappingProxyType(
                {
                    **dict(binding.definition.env),
                    **_core_environment(
                        binding.runtime_data_dir,
                        binding.runtime_workspace,
                    ),
                }
            ),
        )
    return result


def _materialized_mcp_commands(
    generation: PluginGeneration,
    bindings: Mapping[str, McpServerBinding],
) -> dict[str, McpMaterializedCommand]:
    commands = dict(generation.static_runtime_commands)
    return {
        name: McpMaterializedCommand(
            command=_runtime_command(
                generation,
                commands,
                key=f"mcp:{name}",
                declared=binding.definition.command,
            ),
            cwd=str(
                _runtime_cwd(
                    binding.runtime_plugin_dir,
                    binding.definition.cwd,
                )
            ),
            env=MappingProxyType(
                _core_environment(
                    binding.runtime_data_dir,
                    binding.runtime_workspace,
                )
            ),
        )
        for name, binding in bindings.items()
    }


def _runtime_command(
    generation: PluginGeneration,
    commands: Mapping[str, tuple[str, ...]],
    *,
    key: str,
    declared: tuple[str, ...],
) -> tuple[str, ...]:
    command = commands.get(key)
    if command is None:
        head = Path(declared[0])
        if _PYTHON_COMMAND.fullmatch(head.name) is not None:
            raise RuntimeError(f"v3 Python runtime 缺少 staged command: {key}")
        if not head.is_absolute():
            raise RuntimeError(f"v3 runtime command 未固定为绝对路径: {key}")
        command = declared
    executable = Path(command[0])
    if not executable.is_absolute() or not executable.is_file():
        raise RuntimeError(f"v3 runtime executable 无效: {key}: {executable}")
    return command


def _runtime_cwd(plugin_dir: Path, declared: str) -> Path:
    root = plugin_dir.resolve(strict=True)
    path = Path(declared)
    resolved = (
        path.resolve(strict=True)
        if path.is_absolute()
        else (root / path).resolve(strict=True)
    )
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise RuntimeError(f"v3 runtime cwd 越出 immutable artifact: {declared}")
    return resolved


def _core_environment(data_dir: Path, workspace: Path) -> dict[str, str]:
    return {
        _CORE_DATA_ENV[0]: str(data_dir),
        _CORE_DATA_ENV[1]: str(data_dir),
        "AKASHIC_WORKSPACE": str(workspace),
    }


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
