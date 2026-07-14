"""McpServerRegistry: 管理多个 MCP server 连接，持久化到 mcp_servers.json。"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from agent.mcp.client import McpClient
from agent.mcp.tool import McpToolWrapper
from agent.tools.registry import ToolRegistry
from infra.persistence.json_store import atomic_save_json, load_json

logger = logging.getLogger(__name__)


class McpServerRegistry:
    """管理 MCP server 连接生命周期，并将工具同步进 ToolRegistry。

    持久化格式（mcp_servers.json）：
    {
      "servers": {
        "calendar": {
          "command": ["python", "/path/to/run_server.py"],
          "env": {"GOOGLE_CLIENT_ID": "..."}
        }
      }
    }
    """

    def __init__(self, config_path: Path, tool_registry: ToolRegistry) -> None:
        self._config_path = config_path
        self._tool_registry = tool_registry
        self._clients: dict[str, McpClient] = {}
        self._server_tools: dict[str, list[str]] = (
            {}
        )  # server_name -> 已注册的工具名列表
        self._connect_task: asyncio.Task[None] | None = None

    async def load_and_connect_all(self) -> None:
        """启动时读取持久化配置，重连所有 server。"""
        async def connect_one(name: str, cfg: dict[str, Any]) -> None:
            try:
                await self._connect(name, cfg["command"], cfg.get("env"), cfg.get("cwd"))
            except Exception as e:
                logger.error("[mcp] 重连 %r 失败: %s", name, e)

        await asyncio.gather(
            *(
                connect_one(name, cfg)
                for name, cfg in self._load_raw_configs().items()
            )
        )

    def start_connect_all_background(self) -> None:
        """后台重连所有 server，不阻塞主服务启动。"""
        if self._connect_task is None or self._connect_task.done():
            self._connect_task = asyncio.create_task(
                self.load_and_connect_all(),
                name="mcp_connect_all",
            )

    async def shutdown(self) -> None:
        if self._connect_task is not None and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
        clients = list(self._clients.values())
        self._clients.clear()
        self._server_tools.clear()
        await asyncio.gather(
            *(client.disconnect() for client in clients),
            return_exceptions=True,
        )

    async def add(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        if name in self._clients:
            return f"MCP server {name!r} 已存在。如需更新，请先 mcp_remove 再重新添加。"
        try:
            tool_names = await self._connect(name, command, env, cwd)
        except Exception as e:
            return f"连接 MCP server {name!r} 失败：{e}"
        self._save()
        return (
            f"已连接 MCP server {name!r}，注册了 {len(tool_names)} 个工具：\n"
            + "\n".join(f"- {n}" for n in tool_names)
        )

    async def remove(self, name: str) -> str:
        if name not in self._clients:
            return f"MCP server {name!r} 不存在，当前已注册：{list(self._clients.keys()) or '无'}"
        await self._disconnect_server(name)
        self._save()
        return f"已注销 MCP server {name!r}。"

    def list_servers(self) -> str:
        if not self._clients:
            return "当前没有已注册的 MCP server。"
        lines = []
        for name in self._clients:
            tools = self._server_tools.get(name, [])
            lines.append(f"- {name}（{len(tools)} 个工具）：{', '.join(tools) or '无'}")
        return "\n".join(lines)

    def connected_server_names(self) -> set[str]:
        return set(self._clients)

    async def _connect(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None,
        cwd: str | None = None,
    ) -> list[str]:
        client = McpClient(name=name, command=command, env=env, cwd=cwd)
        tool_infos = await client.connect()
        tool_names = []
        for info in tool_infos:
            wrapper = McpToolWrapper(client, info)
            self._tool_registry.register(
                wrapper,
                risk="external-side-effect",
                source_type="mcp",
                source_name=name,
            )
            tool_names.append(wrapper.name)
        self._clients[name] = client
        self._server_tools[name] = tool_names
        return tool_names

    def _load_raw_configs(self) -> dict[str, Any]:
        """读取并校验 MCP 持久化配置，不触碰连接生命周期。"""

        # 1. 读取 JSON；缺失文件和缺失 servers 都表示空配置
        data = load_json(
            self._config_path,
            default={},
            domain="mcp.registry",
        )

        # 2. 校验顶层 schema
        if not isinstance(data, dict):
            raise ValueError(f"[mcp.registry] 配置根节点必须是对象: {self._config_path}")
        if set(data) - {"servers"}:
            raise ValueError(f"[mcp.registry] 配置包含未知字段: {self._config_path}")

        raw_servers = data.get("servers", {})
        if not isinstance(raw_servers, dict):
            raise ValueError(f"[mcp.registry] servers 必须是对象: {self._config_path}")

        # 3. 校验每个 server 的启动参数
        servers: dict[str, dict[str, Any]] = {}
        for name, raw_config in raw_servers.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"[mcp.registry] server 名称无效: {self._config_path}")
            if not isinstance(raw_config, dict):
                raise ValueError(
                    f"[mcp.registry] server 配置必须是对象: {name} path={self._config_path}"
                )
            if set(raw_config) - {"command", "env", "cwd"}:
                raise ValueError(
                    f"[mcp.registry] server 配置包含未知字段: {name} path={self._config_path}"
                )
            command = raw_config.get("command")
            if not isinstance(command, list) or not command or not all(
                isinstance(item, str) and item for item in command
            ):
                raise ValueError(
                    f"[mcp.registry] server command 无效: {name} path={self._config_path}"
                )
            env = raw_config.get("env")
            if "env" in raw_config and (
                not isinstance(env, dict)
                or not all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in env.items()
                )
            ):
                raise ValueError(
                    f"[mcp.registry] server env 无效: {name} path={self._config_path}"
                )
            cwd = raw_config.get("cwd")
            if cwd is not None and not isinstance(cwd, str):
                raise ValueError(
                    f"[mcp.registry] server cwd 无效: {name} path={self._config_path}"
                )
            servers[name] = raw_config
        return servers

    def _save(self) -> None:
        servers = {
            name: {
                "command": client.command,
                "env": client.env,
                "cwd": client.cwd,
            }
            for name, client in self._clients.items()
        }
        atomic_save_json(
            self._config_path,
            {"servers": servers},
            ensure_ascii=False,
            domain="mcp.registry",
        )

    async def _disconnect_server(self, name: str) -> None:
        for tool_name in self._server_tools.pop(name, []):
            self._tool_registry.unregister(tool_name)
        client = self._clients.pop(name, None)
        if client is not None:
            await client.disconnect()
