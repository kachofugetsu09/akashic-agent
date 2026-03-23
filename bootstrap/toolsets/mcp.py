from __future__ import annotations

from pathlib import Path

from agent.mcp.manage_tools import McpAddTool, McpListTool, McpRemoveTool
from agent.mcp.registry import McpServerRegistry
from agent.tools.registry import ToolRegistry


def register_mcp_tools(
    tools: ToolRegistry,
    workspace: Path,
) -> McpServerRegistry:
    mcp_registry = McpServerRegistry(
        config_path=workspace / "mcp_servers.json",
        tool_registry=tools,
    )
    tools.register(McpAddTool(mcp_registry), tags=["mcp", "system"], risk="external-side-effect", search_keywords=["添加MCP", "连接MCP", "注册MCP服务器", "mcp add"])
    tools.register(McpRemoveTool(mcp_registry), tags=["mcp", "system"], risk="write", search_keywords=["删除MCP", "移除MCP服务器", "mcp remove"])
    tools.register(McpListTool(mcp_registry), tags=["mcp", "system"], risk="read-only", search_keywords=["MCP列表", "查看MCP服务器", "mcp list"])
    return mcp_registry
