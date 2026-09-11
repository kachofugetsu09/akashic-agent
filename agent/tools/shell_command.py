"""兼容入口；shell 执行策略由 `agent.plugin_contracts.shell_command` 拥有。

本模块保留原导入路径，避免一次性改动 Core 侧既有调用点。新代码和插件应导入
`agent.plugin_contracts.shell_command`。

再导出的名字按「全库实际被 import 的集合」给出（`resolve_shell`、`ResolvedShell`、
`ShellKind`、`detect_shell_kind`），而不是按模块的公开 API 猜。
"""

from agent.plugin_contracts.shell_command import (
    ResolvedShell,
    ShellKind,
    detect_shell_kind,
    resolve_shell,
)

__all__ = [
    "ResolvedShell",
    "ShellKind",
    "detect_shell_kind",
    "resolve_shell",
]
