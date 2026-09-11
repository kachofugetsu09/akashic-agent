"""兼容入口；会话模型选择词汇由 `agent.plugin_contracts.session_selection` 拥有。

本模块保留原导入路径，避免一次性改动 Core 侧既有调用点。新代码和插件应导入
`agent.plugin_contracts.session_selection`。
"""

from agent.plugin_contracts.session_selection import (
    LEGACY_MODEL_OVERRIDE_KEY,
    SESSION_MODEL_SELECTION_KEY,
    SessionModelSelection,
    read_session_model_selection,
    write_session_model_selection,
)

__all__ = [
    "LEGACY_MODEL_OVERRIDE_KEY",
    "SESSION_MODEL_SELECTION_KEY",
    "SessionModelSelection",
    "read_session_model_selection",
    "write_session_model_selection",
]
