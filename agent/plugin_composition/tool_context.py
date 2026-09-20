"""Compatibility entry for the source-neutral tool execution context."""

from agent.tool_context import (
    ToolExecutionContext,
    get_current_tool_context,
    tool_execution_context_scope,
)

__all__ = [
    "ToolExecutionContext",
    "get_current_tool_context",
    "tool_execution_context_scope",
]
