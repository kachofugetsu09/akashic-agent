"""McpToolWrapper: 把 MCP server 的远端工具包装成本地 Tool。"""

from typing import Any

from agent.mcp.client import McpClient, McpToolInfo
from agent.tools.base import Tool


class McpToolWrapper(Tool):
    """将单个 MCP 远端工具暴露为标准本地 Tool。

    工具名格式：mcp_{server_name}__{tool_name}
    避免与内置工具冲突，也方便按 server 识别。
    """

    def __init__(
        self,
        client: McpClient,
        info: McpToolInfo,
        *,
        server_name: str | None = None,
    ) -> None:
        self._client = client
        self._info = info
        self._server_name = server_name or client.name

    @property
    def name(self) -> str:
        return f"mcp_{self._server_name}__{self._info.name}"

    @property
    def description(self) -> str:
        return f"[MCP:{self._server_name}] {self._info.description}"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._info.input_schema

    async def execute(self, **kwargs: Any) -> str:
        return await self._client.call(self._info.name, kwargs)

    async def execute_with_timeout(
        self,
        arguments: dict[str, Any],
        execution_timeout: float | None = None,
    ) -> str:
        return await self._client.call(
            self._info.name,
            arguments,
            timeout=execution_timeout,
        )
