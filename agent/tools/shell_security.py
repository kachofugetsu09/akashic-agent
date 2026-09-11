"""兼容入口；shell 安全校验由 `agent.plugin_contracts.shell_security` 拥有。

本模块保留原导入路径，避免一次性改动 Core 侧既有调用点。新代码和插件应导入
`agent.plugin_contracts.shell_security`。
"""

from agent.plugin_contracts.shell_security import (
    validate_command,
    validate_network_command,
)

__all__ = [
    "validate_command",
    "validate_network_command",
]
