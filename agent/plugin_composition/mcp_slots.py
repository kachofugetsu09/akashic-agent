from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from agent.plugin_composition.context import Context, FiberHandle, HealthHandle
from agent.plugin_composition.model import (
    CompositionError,
    FiberState,
    IncidentView,
    ServiceKey,
)

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_RESERVED_ENV = frozenset(
    {
        "AKA_PLUGIN_DATA_DIR",
        "AKASHIC_PLUGIN_DATA_DIR",
        "AKASHIC_WORKSPACE",
    }
)


@dataclass(frozen=True, slots=True)
class EndpointEnv:
    env: str
    process: str


@dataclass(frozen=True, slots=True)
class WorkloadEnv:
    env: str
    workload: str
    port: str


@dataclass(frozen=True, slots=True)
class McpServerDefinition:
    name: str
    command: tuple[str, ...]
    cwd: str = "."
    env: Mapping[str, str] = field(default_factory=dict)
    required_tools: tuple[str, ...] = ()
    candidate_read_only_tools: tuple[str, ...] = ()
    endpoint_env: tuple[EndpointEnv, ...] = ()
    workload_env: tuple[WorkloadEnv, ...] = ()
    candidate_env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class McpServerDescriptor:
    owner: str
    name: str
    command: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    required_tools: tuple[str, ...]
    candidate_read_only_tools: tuple[str, ...]
    endpoint_env: tuple[EndpointEnv, ...]
    workload_env: tuple[WorkloadEnv, ...]
    candidate_env: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class McpServerBinding:
    descriptor: McpServerDescriptor
    definition: McpServerDefinition
    health: HealthHandle
    owner_fiber: FiberHandle
    activation_token: object
    runtime_plugin_dir: Path = field(repr=False, compare=False)
    runtime_data_dir: Path = field(repr=False, compare=False)
    runtime_workspace: Path = field(repr=False, compare=False)
    incident_reporter: Callable[[str, str], IncidentView] = field(
        repr=False,
        compare=False,
    )

    def is_owned(self) -> bool:
        """Return whether the declaration still belongs to its Fiber activation."""

        return (
            self.owner_fiber.state is FiberState.ACTIVE
            and self.owner_fiber.activation_token is self.activation_token
        )

    def is_live(self) -> bool:
        """Return whether the declaration is owned and currently healthy."""

        return self.is_owned() and self.health.healthy


class McpServerRegistry(Mapping[str, McpServerBinding]):
    """Expose one immutable Root-local MCP declaration catalog."""

    def __init__(
        self,
        bindings: Mapping[str, McpServerBinding],
        *,
        root_instance_token: object,
    ) -> None:
        self._root_instance_token = root_instance_token
        self._bindings = MappingProxyType(
            {name: bindings[name] for name in sorted(bindings)}
        )
        self._descriptors = tuple(
            binding.descriptor for binding in self._bindings.values()
        )
        payload = [
            {
                "owner": item.owner,
                "name": item.name,
                "command": list(item.command),
                "cwd": item.cwd,
                "env": list(item.env),
                "required_tools": list(item.required_tools),
                "candidate_read_only_tools": list(item.candidate_read_only_tools),
                "endpoint_env": [
                    {"env": endpoint.env, "process": endpoint.process}
                    for endpoint in item.endpoint_env
                ],
                "workload_env": [
                    {
                        "env": endpoint.env,
                        "workload": endpoint.workload,
                        "port": endpoint.port,
                    }
                    for endpoint in item.workload_env
                ],
                "candidate_env": list(item.candidate_env),
            }
            for item in self._descriptors
        ]
        self._identity = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def descriptors(self) -> tuple[McpServerDescriptor, ...]:
        return self._descriptors

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def root_instance_token(self) -> object:
        return self._root_instance_token

    def __getitem__(self, name: str) -> McpServerBinding:
        return self._bindings[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)


MCP_SERVERS = ServiceKey["PluginMcpServers"]("core.mcp_servers")


@dataclass(slots=True)
class _McpRegistration:
    token: int
    owner: str
    definition: McpServerDefinition
    descriptor: McpServerDescriptor
    owner_fiber: FiberHandle
    activation_token: object
    runtime_plugin_dir: Path
    runtime_data_dir: Path
    runtime_workspace: Path
    incident_reporter: Callable[[str, str], IncidentView]
    health: HealthHandle | None = None


class _McpServerDeclarations:
    """Own one Root-local mutable declaration set for Core."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _McpRegistration] = {}
        self._names: dict[str, int] = {}
        self._frozen: McpServerRegistry | None = None

    async def register(
        self,
        ctx: Context,
        definition: McpServerDefinition,
    ) -> None:
        """Validate and register one declaration as Fiber-owned effects."""

        # 1. Freeze source-relative inputs before any Root state changes.
        normalized = _normalize_definition(ctx.runtime.plugin_dir, definition)
        owner_fiber = ctx.fiber
        activation_token = owner_fiber.activation_token
        if activation_token is None:
            raise CompositionError(
                "INACTIVE_FIBER",
                f"{ctx.runtime.plugin_id} 当前 Fiber 没有 active activation",
            )
        registration: _McpRegistration | None = None

        def setup() -> Callable[[], None]:
            nonlocal registration
            registration, cleanup = self._register(
                ctx.runtime.plugin_id,
                normalized,
                owner_fiber,
                activation_token,
                ctx.runtime.plugin_dir,
                ctx.runtime.data_dir,
                ctx.runtime.workspace,
                ctx.report_incident,
            )
            return cleanup

        # 2. Registration and required health either both settle or both roll back.
        registration_effect = await ctx.effect(
            setup,
            label=f"mcp-server:{normalized.name}",
        )
        try:
            health = await ctx.health(f"mcp:{normalized.name}", required=True)
        except BaseException:
            await registration_effect.aclose()
            raise
        assert registration is not None
        registration.health = health

    def freeze(self, root_instance_token: object) -> McpServerRegistry:
        """Freeze declarations into an immutable snapshot registry."""

        if self._frozen is not None:
            if self._frozen.root_instance_token is not root_instance_token:
                raise RuntimeError("MCP declaration registry 属于另一棵 Root")
            return self._frozen
        bindings: dict[str, McpServerBinding] = {}
        for registration in sorted(
            self._registrations.values(), key=lambda item: item.token
        ):
            if registration.health is None:
                raise RuntimeError("MCP declaration 缺少 required Health")
            bindings[registration.definition.name] = McpServerBinding(
                descriptor=registration.descriptor,
                definition=registration.definition,
                health=registration.health,
                owner_fiber=registration.owner_fiber,
                activation_token=registration.activation_token,
                runtime_plugin_dir=registration.runtime_plugin_dir,
                runtime_data_dir=registration.runtime_data_dir,
                runtime_workspace=registration.runtime_workspace,
                incident_reporter=registration.incident_reporter,
            )
        self._frozen = McpServerRegistry(
            bindings,
            root_instance_token=root_instance_token,
        )
        return self._frozen

    def _register(
        self,
        owner: str,
        definition: McpServerDefinition,
        owner_fiber: FiberHandle,
        activation_token: object,
        runtime_plugin_dir: Path,
        runtime_data_dir: Path,
        runtime_workspace: Path,
        incident_reporter: Callable[[str, str], IncidentView],
    ) -> tuple[_McpRegistration, Callable[[], None]]:
        """Add one normalized declaration and return its exact inverse."""

        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_MCP_SERVERS_FROZEN",
                "插件 MCP 声明已冻结，不能在 snapshot 发布后新增",
            )
        if definition.name in self._names:
            raise CompositionError(
                "DUPLICATE_PLUGIN_MCP_SERVER",
                f"插件 MCP server 名称重复: {definition.name}",
            )
        token = self._next_token
        self._next_token += 1
        registration = _McpRegistration(
            token=token,
            owner=owner,
            definition=definition,
            descriptor=_descriptor(owner, definition),
            owner_fiber=owner_fiber,
            activation_token=activation_token,
            runtime_plugin_dir=runtime_plugin_dir,
            runtime_data_dir=runtime_data_dir,
            runtime_workspace=runtime_workspace,
            incident_reporter=incident_reporter,
        )
        self._registrations[token] = registration
        self._names[definition.name] = token

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)
            if self._names.get(definition.name) == token:
                _ = self._names.pop(definition.name)

        return registration, cleanup


class PluginMcpServers:
    """Expose only Fiber-owned MCP registration to plugins."""

    def __init__(self, root_instance_token: object) -> None:
        self._root_instance_token = root_instance_token
        self._declarations = _McpServerDeclarations()

    async def register(
        self,
        ctx: Context,
        definition: McpServerDefinition,
    ) -> None:
        """Register one declaration through the Core-owned collector."""

        if (
            ctx._root_instance_token() is not self._root_instance_token
            or ctx.require(MCP_SERVERS) is not self
        ):
            raise CompositionError(
                "MCP_SERVICE_ROOT_MISMATCH",
                "插件 MCP 声明 Service 不属于当前 Root",
            )
        await self._declarations.register(ctx, definition)


def _freeze_plugin_mcp_servers(
    value: object,
    root_instance_token: object,
) -> McpServerRegistry:
    """Freeze the exact Core-created MCP registration facade."""

    if not isinstance(value, PluginMcpServers):
        raise RuntimeError("RuntimeSnapshot MCP Service 类型无效")
    if value._root_instance_token is not root_instance_token:
        raise RuntimeError("RuntimeSnapshot MCP Service 不属于 exact Root")
    return value._declarations.freeze(root_instance_token)


def _normalize_definition(
    plugin_dir: Path,
    definition: McpServerDefinition,
) -> McpServerDefinition:
    """Validate and detach one plugin-owned MCP declaration."""

    if not isinstance(definition, McpServerDefinition):
        raise TypeError("PluginMcpServers.register 只接受 McpServerDefinition")
    if not isinstance(definition.name, str) or not _NAME.fullmatch(definition.name):
        raise ValueError(f"MCP server name 无效: {definition.name}")
    command = _string_tuple(definition.command, "command", allow_empty=False)
    cwd = _relative_path(plugin_dir, definition.cwd, kind="cwd", directory=True)
    for item in command:
        if Path(item).is_absolute():
            raise ValueError("MCP command 不得声明绝对 artifact 路径")
        if item.startswith("-"):
            continue
        if "/" in item or "\\" in item or item.startswith(".") or item.endswith(".py"):
            _ = _relative_path(plugin_dir, item, kind="command", directory=False)
    env = _environment(definition.env, field_name="env")
    candidate_env = _environment(
        definition.candidate_env,
        field_name="candidate_env",
    )
    required_tools = _string_tuple(definition.required_tools, "required_tools")
    candidate_tools = _string_tuple(
        definition.candidate_read_only_tools,
        "candidate_read_only_tools",
    )
    endpoints = _endpoint_tuple(definition.endpoint_env)
    workload_endpoints = _workload_endpoint_tuple(definition.workload_env)
    occupied = set(env) | set(candidate_env)
    endpoint_names = [endpoint.env for endpoint in endpoints]
    endpoint_names.extend(endpoint.env for endpoint in workload_endpoints)
    if occupied.intersection(endpoint_names) or len(endpoint_names) != len(
        set(endpoint_names)
    ):
        raise ValueError(f"MCP endpoint env 与声明 env 冲突: {definition.name}")
    return McpServerDefinition(
        name=definition.name,
        command=command,
        cwd=cwd,
        env=MappingProxyType(env),
        required_tools=required_tools,
        candidate_read_only_tools=candidate_tools,
        endpoint_env=endpoints,
        workload_env=workload_endpoints,
        candidate_env=MappingProxyType(candidate_env),
    )


def _descriptor(owner: str, definition: McpServerDefinition) -> McpServerDescriptor:
    return McpServerDescriptor(
        owner=owner,
        name=definition.name,
        command=definition.command,
        cwd=definition.cwd,
        env=tuple(sorted(definition.env.items())),
        required_tools=definition.required_tools,
        candidate_read_only_tools=definition.candidate_read_only_tools,
        endpoint_env=definition.endpoint_env,
        workload_env=definition.workload_env,
        candidate_env=tuple(sorted(definition.candidate_env.items())),
    )


def _environment(value: Mapping[str, str], *, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"MCP {field_name} 必须是字符串 mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not _ENV_NAME.fullmatch(key)
            or key in _RESERVED_ENV
            or not isinstance(item, str)
        ):
            raise ValueError(f"MCP {field_name} 无效: {key}")
        result[key] = item
    return result


def _endpoint_tuple(value: tuple[EndpointEnv, ...]) -> tuple[EndpointEnv, ...]:
    if not isinstance(value, tuple):
        raise TypeError("MCP endpoint_env 必须是 tuple")
    result: list[EndpointEnv] = []
    seen: set[str] = set()
    for endpoint in value:
        if (
            not isinstance(endpoint, EndpointEnv)
            or not _ENV_NAME.fullmatch(endpoint.env)
            or endpoint.env in _RESERVED_ENV
            or not _NAME.fullmatch(endpoint.process)
            or endpoint.env in seen
        ):
            raise ValueError(f"MCP endpoint env 无效: {endpoint!r}")
        seen.add(endpoint.env)
        result.append(EndpointEnv(endpoint.env, endpoint.process))
    return tuple(result)


def _workload_endpoint_tuple(
    value: tuple[WorkloadEnv, ...],
) -> tuple[WorkloadEnv, ...]:
    if not isinstance(value, tuple):
        raise TypeError("MCP workload_env 必须是 tuple")
    result: list[WorkloadEnv] = []
    seen: set[str] = set()
    for endpoint in value:
        if (
            not isinstance(endpoint, WorkloadEnv)
            or not _ENV_NAME.fullmatch(endpoint.env)
            or endpoint.env in _RESERVED_ENV
            or not _NAME.fullmatch(endpoint.workload)
            or not _NAME.fullmatch(endpoint.port)
            or endpoint.env in seen
        ):
            raise ValueError(f"MCP workload env 无效: {endpoint!r}")
        seen.add(endpoint.env)
        result.append(WorkloadEnv(endpoint.env, endpoint.workload, endpoint.port))
    return tuple(result)


def _string_tuple(
    value: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(
            f"MCP {field_name} 必须是非空 tuple"
            if not allow_empty
            else f"MCP {field_name} 必须是 tuple"
        )
    if any(
        not isinstance(item, str) or not item or item != item.strip() for item in value
    ):
        raise ValueError(f"MCP {field_name} 包含无效字符串")
    if len(set(value)) != len(value):
        raise ValueError(f"MCP {field_name} 包含重复项")
    return tuple(value)


def _relative_path(
    plugin_dir: Path,
    raw: str,
    *,
    kind: str,
    directory: bool,
) -> str:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or Path(raw).is_absolute()
    ):
        raise ValueError(f"MCP {kind} 必须是 artifact 内相对路径")
    root = plugin_dir.resolve(strict=True)
    try:
        resolved = (root / raw).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"MCP {kind} 不存在: {raw}") from error
    valid_type = resolved.is_dir() if directory else resolved.is_file()
    if not resolved.is_relative_to(root) or not valid_type:
        raise ValueError(f"MCP {kind} 越过 immutable artifact: {raw}")
    return raw
