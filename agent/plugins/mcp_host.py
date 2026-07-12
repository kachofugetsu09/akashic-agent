from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any

from agent.mcp.client import McpClient
from agent.mcp.tool import McpToolWrapper
from agent.plugins.scope import PluginScope
from agent.plugins.specs import RegisteredProactiveSource


@dataclass(frozen=True)
class PreparedMcpServer:
    name: str
    client: McpClient
    tools: tuple[McpToolWrapper, ...]

    @property
    def remote_tool_names(self) -> tuple[str, ...]:
        return tuple(info.name for info in self.client.tool_infos)


@dataclass(frozen=True)
class PreparedMcpCatalog:
    generation_id: str
    servers: Mapping[str, PreparedMcpServer]

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(tool.name for server in self.servers.values() for tool in server.tools)
        )


class PluginMcpHost:
    def __init__(self) -> None:
        self._catalogs: dict[str, PreparedMcpCatalog] = {}

    async def prepare(
        self,
        generation_id: str,
        *,
        server_specs: dict[str, dict[str, Any]],
        proactive_sources: tuple[RegisteredProactiveSource, ...],
        scope: PluginScope,
    ) -> PreparedMcpCatalog:
        servers: dict[str, PreparedMcpServer] = {}
        for server_name, spec in sorted(server_specs.items()):
            client = McpClient(
                name=f"{server_name}@{generation_id}",
                command=list(spec["command"]),
                env=dict(spec.get("env") or {}),
                cwd=str(spec.get("cwd") or "") or None,
            )
            scope.defer(f"mcp_client:{server_name}", client.disconnect)
            infos = await client.connect()
            remote_names = [info.name for info in infos]
            if len(remote_names) != len(set(remote_names)):
                raise RuntimeError(f"MCP server 工具名重复: {server_name}")
            servers[server_name] = PreparedMcpServer(
                name=server_name,
                client=client,
                tools=tuple(
                    McpToolWrapper(client, info, server_name=server_name)
                    for info in infos
                ),
            )
        self._validate_sources(servers, proactive_sources)
        catalog = PreparedMcpCatalog(
            generation_id=generation_id,
            servers=MappingProxyType(servers),
        )
        self._catalogs[generation_id] = catalog
        return catalog

    async def close(self, generation_id: str) -> None:
        catalog = self._catalogs.get(generation_id)
        if catalog is None:
            return
        failures: list[Exception] = []
        try:
            for server in catalog.servers.values():
                try:
                    await server.client.disconnect()
                except Exception as error:
                    failures.append(error)
        finally:
            _ = self._catalogs.pop(generation_id, None)
        if failures:
            raise RuntimeError(
                "MCP catalog 清理失败: "
                + "; ".join(str(error) for error in failures)
            )

    def get(self, generation_id: str) -> PreparedMcpCatalog | None:
        return self._catalogs.get(generation_id)

    @staticmethod
    def _validate_sources(
        servers: Mapping[str, PreparedMcpServer],
        sources: tuple[RegisteredProactiveSource, ...],
    ) -> None:
        missing: list[str] = []
        for source in sources:
            server = servers.get(source.spec.server)
            if server is None:
                missing.append(f"{source.spec.id}:server:{source.spec.server}")
                continue
            available = set(server.remote_tool_names)
            for role, tool_name in (
                ("fetch", source.spec.fetch_tool),
                ("ack", source.spec.ack_tool),
            ):
                if tool_name and tool_name not in available:
                    missing.append(f"{source.spec.id}:{role}:{tool_name}")
        if missing:
            raise RuntimeError(f"proactive source MCP tool 缺失: {', '.join(missing)}")
