"""mcp_add / mcp_remove / mcp_list：agent 用于动态管理 MCP server 的三个工具。"""

from typing import Any

from agent.mcp.registry import McpServerRegistry
from agent.tools.base import Tool


class McpAddTool(Tool):
    """连接并注册一个 MCP server。"""

    def __init__(self, registry: McpServerRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "mcp_add"

    @property
    def description(self) -> str:
        return (
            "连接并注册一个本地 MCP server（stdio 子进程模式）。"
            "command 是启动命令列表，env 是可选的额外环境变量。"
            "连接成功后，该 server 的所有工具立即可用。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "给这个 MCP server 起一个唯一短名称，如 'calendar'",
                },
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "启动命令列表，如 ['python', '/home/user/.akasic/mcp/calendar-mcp/run_server.py']",
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "可选的额外环境变量，如 {'GOOGLE_CLIENT_ID': 'xxx'}",
                },
            },
            "required": ["name", "command"],
        }

    async def execute(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        **_: Any,
    ) -> str:
        return await self._registry.add(name, command, env)


class McpRemoveTool(Tool):
    """注销并断开一个已注册的 MCP server。"""

    def __init__(self, registry: McpServerRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "mcp_remove"

    @property
    def description(self) -> str:
        return "注销并断开一个已注册的 MCP server，同时移除其所有工具。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要注销的 MCP server 名称",
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str, **_: Any) -> str:
        return await self._registry.remove(name)


class McpListTool(Tool):
    """列出当前所有已注册的 MCP server 及其工具。"""

    def __init__(self, registry: McpServerRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "mcp_list"

    @property
    def description(self) -> str:
        return "列出当前所有已注册的 MCP server 及其工具名称。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **_: Any) -> str:
        return self._registry.list_servers()
