"""让已提交的旧 MCP 归档在迁移到插件自有 client 前仍能启动。"""

from plugins.mcp.client import McpClient, McpToolExecutionError

__all__ = ["McpClient", "McpToolExecutionError"]

