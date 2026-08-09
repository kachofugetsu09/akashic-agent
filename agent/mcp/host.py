from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

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
        self._states: dict[
            str, Literal["candidate", "active", "draining", "stopping"]
        ] = {}
        self._failures: dict[str, RuntimeError] = {}
        self._failure_watchers: dict[str, list[asyncio.Task[None]]] = {}
        self._active_generation_id: str | None = None
        self._active_fatal: RuntimeError | None = None
        self._active_fatal_event = asyncio.Event()

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
        self._states[generation_id] = "candidate"
        self._failure_watchers[generation_id] = [
            asyncio.create_task(
                self._watch_client_failure(generation_id, server.client),
                name=f"mcp_fatal:{generation_id}:{server.name}",
            )
            for server in servers.values()
        ]
        return catalog

    async def close(self, generation_id: str) -> None:
        catalog = self._catalogs.get(generation_id)
        if catalog is None:
            return
        self.mark_stopping(generation_id)
        watchers = self._failure_watchers.pop(generation_id, [])
        for watcher in watchers:
            if not watcher.done():
                _ = watcher.cancel()
        if watchers:
            _ = await asyncio.gather(*watchers, return_exceptions=True)
        failures: list[Exception] = []
        try:
            for server in catalog.servers.values():
                try:
                    await server.client.disconnect()
                except Exception as error:
                    failures.append(error)
        finally:
            _ = self._catalogs.pop(generation_id, None)
            _ = self._states.pop(generation_id, None)
            _ = self._failures.pop(generation_id, None)
            if self._active_generation_id == generation_id:
                self._active_generation_id = None
        if failures:
            raise RuntimeError(
                "MCP catalog 清理失败: " + "; ".join(str(error) for error in failures)
            )

    def get(self, generation_id: str) -> PreparedMcpCatalog | None:
        return self._catalogs.get(generation_id)

    def mark_active(self, generation_id: str) -> None:
        """把已发布 generation 设为唯一 active fatal owner。"""
        if generation_id not in self._catalogs:
            raise KeyError(f"未知 MCP generation: {generation_id}")
        self._states[generation_id] = "active"
        self._active_generation_id = generation_id
        self._active_fatal = self._failures.get(generation_id)
        _ = self._active_fatal_event.clear()
        if self._active_fatal is not None:
            _ = self._active_fatal_event.set()

    def mark_draining(self, generation_id: str) -> None:
        """标记旧 generation 正在排空，不改变新 active ownership。"""
        if generation_id not in self._catalogs:
            raise KeyError(f"未知 MCP generation: {generation_id}")
        self._states[generation_id] = "draining"
        if self._active_generation_id == generation_id:
            self._active_generation_id = None
            self._active_fatal = None
            _ = self._active_fatal_event.clear()

    def mark_stopping(self, generation_id: str) -> None:
        """禁止指定 generation 的后续 fatal escalation。"""
        if generation_id not in self._catalogs:
            raise KeyError(f"未知 MCP generation: {generation_id}")
        self._states[generation_id] = "stopping"
        if self._active_generation_id == generation_id:
            self._active_generation_id = None
            self._active_fatal = None
            _ = self._active_fatal_event.clear()

    def assert_healthy(self, generation_id: str) -> None:
        """候选 gate 和 active probe 都在这里暴露恢复预算耗尽。"""
        catalog = self._catalogs.get(generation_id)
        if catalog is None:
            raise KeyError(f"未知 MCP generation: {generation_id}")
        failure = self._failures.get(generation_id)
        if failure is not None:
            raise failure
        for server in catalog.servers.values():
            server.client.assert_healthy()

    async def wait_fatal_failure(self) -> None:
        """只将当前 active generation 的不可恢复失败升级给 Core。"""
        while True:
            _ = await self._active_fatal_event.wait()
            failure = self._active_fatal
            if failure is not None:
                raise failure

    def state(
        self, generation_id: str
    ) -> Literal["candidate", "active", "draining", "stopping"] | None:
        return self._states.get(generation_id)

    async def _watch_client_failure(
        self,
        generation_id: str,
        client: McpClient,
    ) -> None:
        """记录 generation failure，并仅升级当时仍 active 的故障。"""
        failure = await client.wait_fatal_failure()
        if self._states.get(generation_id) == "stopping":
            return
        self._failures[generation_id] = failure
        if (
            self._states.get(generation_id) == "active"
            and self._active_generation_id == generation_id
        ):
            self._active_fatal = failure
            _ = self._active_fatal_event.set()

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
