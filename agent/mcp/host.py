from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from agent.mcp.client import McpClient
from agent.mcp.tool import McpToolWrapper
from agent.plugins.scope import PluginScope


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


class McpGenerationHost:
    """准备并按代际持有 MCP catalog。"""

    def __init__(self) -> None:
        self._catalogs: dict[str, PreparedMcpCatalog] = {}

    async def prepare(
        self,
        generation_id: str,
        *,
        server_specs: Mapping[str, Mapping[str, Any]],
        required_tools: Mapping[str, tuple[str, ...]],
        scope: PluginScope,
    ) -> PreparedMcpCatalog:
        """连接候选 MCP，并在完整校验后登记 catalog。"""

        # 1. 连接全部候选 server，并立即把客户端纳入作用域清理
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

        # 2. 验证上层声明依赖的远端工具，再发布不可变 catalog
        self._validate_required_tools(servers, required_tools)
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
                "MCP catalog 清理失败: " + "; ".join(str(error) for error in failures)
            )

    def get(self, generation_id: str) -> PreparedMcpCatalog | None:
        return self._catalogs.get(generation_id)

    @staticmethod
    def _validate_required_tools(
        servers: Mapping[str, PreparedMcpServer],
        required_tools: Mapping[str, tuple[str, ...]],
    ) -> None:
        missing: list[str] = []
        for server_name, tool_names in required_tools.items():
            server = servers.get(server_name)
            if server is None:
                missing.append(f"server:{server_name}")
                continue
            available = set(server.remote_tool_names)
            missing.extend(
                f"{server_name}:{tool_name}"
                for tool_name in tool_names
                if tool_name not in available
            )
        if missing:
            raise RuntimeError(f"MCP 依赖工具缺失: {', '.join(missing)}")
