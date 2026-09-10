"""插件公开结构合同。

插件只能依赖本模块和 `agent.plugin_composition`；本模块只定义不可变值词汇表
和不依赖实现的 Protocol，不导入服务实现、存储层或 bootstrap。
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
