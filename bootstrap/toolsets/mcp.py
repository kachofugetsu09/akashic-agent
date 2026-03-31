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
    tools.register(McpAddTool(mcp_registry), risk="external-side-effect")
    tools.register(McpRemoveTool(mcp_registry), risk="write")
    tools.register(McpListTool(mcp_registry), risk="read-only")
    return mcp_registry
