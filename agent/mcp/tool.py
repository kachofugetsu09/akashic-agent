"""McpToolWrapper: 把 MCP server 的远端工具包装成本地 Tool。"""

from typing import Any

from agent.mcp.client import McpClient, McpToolInfo
from agent.tools.base import Tool


class McpToolWrapper(Tool):
    """将单个 MCP 远端工具暴露为标准本地 Tool。

    工具名格式：mcp_{server_name}__{tool_name}
    避免与内置工具冲突，也方便按 server 识别。
    """

    def __init__(self, client: McpClient, info: McpToolInfo) -> None:
        self._client = client
        self._info = info

    @property
    def name(self) -> str:
        return f"mcp_{self._client.name}__{self._info.name}"

    @property
    def description(self) -> str:
        return f"[MCP:{self._client.name}] {self._info.description}"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._info.input_schema

    async def execute(self, **kwargs: Any) -> str:
        return await self._client.call(self._info.name, kwargs)
