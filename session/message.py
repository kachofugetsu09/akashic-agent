"""兼容入口；公开消息词汇表由 `agent.plugin_contracts` 拥有。

本模块保留原导入路径，避免一次性改动既有调用点。新代码和插件应导入
`agent.plugin_contracts`（或经 `agent.plugin_composition` 的公开导出）。
"""

from agent.plugin_contracts.message import (
    MAX_METADATA_BYTES,
    Body,
    CallRef,
    ContentPart,
    ContentReferences,
    Control,
    Input,
    Message,
    Output,
    Part,
    ToolCall,
    ToolResult,
    freeze_json,
    freeze_metadata,
)

__all__ = [
    "MAX_METADATA_BYTES",
    "Body",
    "CallRef",
    "ContentPart",
    "ContentReferences",
    "Control",
    "Input",
    "Message",
    "Output",
    "Part",
    "ToolCall",
    "ToolResult",
    "freeze_json",
    "freeze_metadata",
]
