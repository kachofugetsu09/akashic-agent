from __future__ import annotations

from pathlib import Path

from agent.mcp.manage_tools import McpAddTool, McpListTool, McpRemoveTool
from agent.mcp.registry import McpServerRegistry
from agent.tools.registry import ToolRegistry
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    build_registration_result,
)


class McpToolsetProvider(ToolsetProvider):
    def register(self, registry: ToolRegistry, deps: ToolsetDeps):
        before = set(registry._tools.keys())
        mcp_registry = McpServerRegistry(
            config_path=deps.workspace / "mcp_servers.json",
            tool_registry=registry,
        )
        registry.register(McpAddTool(mcp_registry), risk="external-side-effect")
        registry.register(McpRemoveTool(mcp_registry), risk="write")
        registry.register(McpListTool(mcp_registry), risk="read-only")
        return build_registration_result(
            registry=registry,
            source_name="mcp",
            before=before,
            extras={"mcp_registry": mcp_registry},
        )


def register_mcp_tools(
    tools: ToolRegistry,
    workspace: Path,
) -> McpServerRegistry:
    result = McpToolsetProvider().register(
        tools,
        ToolsetDeps(
            config=None,  # type: ignore[arg-type]
            workspace=workspace,
        ),
    )
    return result.extras["mcp_registry"]
