"""旧归档启动桥测试。"""

from agent.mcp.client import McpClient as ArchivedMcpClient
from agent.mcp.client import McpToolExecutionError as ArchivedMcpToolExecutionError


def test_old_mcp_archive_can_import_frozen_client_bridge() -> None:
    assert ArchivedMcpClient.__name__ == "McpClient"
    assert issubclass(ArchivedMcpToolExecutionError, RuntimeError)
