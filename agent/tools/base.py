"""兼容入口；工具定义 ABI 由 `agent.plugin_contracts.tool_base` 拥有。

本模块保留原导入路径，避免一次性改动既有启动顺序（host_bridge、
plugin_composition、tool_runtime 等 Core 侧消费者仍按原路径 import）。
新代码和插件应导入 `agent.plugin_contracts.tool_base`。
"""

from agent.plugin_contracts.tool_base import (
    Tool,
    ToolExecutionContext,
    ToolResult,
    get_current_tool_context,
    normalize_tool_parameters,
    normalize_tool_result,
    tool_execution_context_scope,
)

__all__ = [
    "Tool",
    "ToolExecutionContext",
    "ToolResult",
    "get_current_tool_context",
    "normalize_tool_parameters",
    "normalize_tool_result",
    "tool_execution_context_scope",
]
