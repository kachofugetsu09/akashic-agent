from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from agent.plugin_composition.context import FiberHandle, HealthHandle
from agent.plugin_composition.model import FiberState, IncidentView
from agent.plugin_composition.mcp_slots import McpServerDefinition, EndpointEnv, WorkloadEnv

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_RESERVED_ENV = frozenset(
    {
        "AKA_PLUGIN_DATA_DIR",
        "AKASHIC_PLUGIN_DATA_DIR",
        "AKASHIC_WORKSPACE",
        "AKASHIC_BOOT_ID",
        "AKASHIC_SUPERVISED",
        "AKASHIC_MCP_SCOPE_ID",
    }
)


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


def _normalize_definition(
    plugin_dir: Path,
    definition: McpServerDefinition,
) -> McpServerDefinition:
    """Validate and detach one plugin-owned MCP declaration."""

    if not isinstance(definition, McpServerDefinition):
        raise TypeError("McpServers.register 只接受 McpServerDefinition")
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
    endpoints = definition.endpoint_env
    workload_endpoints = definition.workload_env
    if not isinstance(endpoints, tuple) or any(not isinstance(item, EndpointEnv) for item in endpoints):
        raise TypeError("MCP endpoint_env 必须是实际进程句柄引用的 tuple")
    if not isinstance(workload_endpoints, tuple) or any(not isinstance(item, WorkloadEnv) for item in workload_endpoints):
        raise TypeError("MCP workload_env 必须是实际 Workload 句柄引用的 tuple")
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
