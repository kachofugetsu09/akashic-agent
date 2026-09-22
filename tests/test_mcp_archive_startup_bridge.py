"""旧归档启动桥测试。"""

from agent.mcp.client import McpClient as ArchivedMcpClient
from agent.mcp.client import McpToolExecutionError as ArchivedMcpToolExecutionError
from plugins.mcp.client import McpClient, McpToolExecutionError


def test_old_mcp_archive_imports_plugin_owned_client() -> None:
    assert ArchivedMcpClient is McpClient
    assert ArchivedMcpToolExecutionError is McpToolExecutionError
