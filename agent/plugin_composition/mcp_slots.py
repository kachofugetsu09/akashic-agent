from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from agent.plugin_composition.workload_slots import WorkloadHandle
from agent.plugin_composition.process_slots import ManagedProcessHandle
from typing import Any, Literal, Protocol
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class EndpointEnv:
    env: str
    process: ManagedProcessHandle


@dataclass(frozen=True, slots=True)
class WorkloadEnv:
    env: str
    workload: WorkloadHandle
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


class McpRoute(Protocol):
    async def call(self, tool_name: str, arguments: Mapping[str, Any], *, timeout: float | None = None) -> McpCallResult: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self) -> McpRoute: ...
    async def __aexit__(self, *args: object) -> None: ...


class McpServer(Protocol):
    @property
    def tools(self) -> Mapping[str, McpToolView]: ...
    @property
    def tool_names(self) -> tuple[str, ...]: ...
    def route(self) -> McpRoute: ...
    def logs(self) -> McpLogView: ...


@dataclass(frozen=True, slots=True)
class McpSessionFailure:
    identity: str
    server: str
    error: str


class McpServers(Protocol):
    @property
    def root_instance_token(self) -> object: ...
    async def register(self, ctx: Context, definition: McpServerDefinition) -> None: ...
    def open(self, ctx: Context, name: str, *, expected_catalog_digest: str | None = None) -> AbstractAsyncContextManager[McpServer]: ...
    def catalog(self) -> list[dict[str, object]]: ...
    def failures(self) -> tuple[McpSessionFailure, ...]: ...
    async def retry_cleanup(self, ctx: Context, identity: str) -> None: ...


MCP_SERVERS = ServiceKey[McpServers]("core.mcp_servers")
