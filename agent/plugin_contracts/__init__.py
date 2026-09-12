"""插件公开结构合同。

本模块只承载 Core 自己拥有的值合同，不是所有业务共享代码的收容层。
业务 schema、模型内容解释、存储实现和默认流程仍由相应 owner 提供。
公开模块清单由边界门检查；业务协作可在 Core 外声明合同。
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
