"""兼容入口；时间工具由 `agent.plugin_contracts.timekit` 拥有。

本模块保留原导入路径，避免一次性改动既有调用点。新代码和插件应导入
`agent.plugin_contracts.timekit`。
"""

from agent.plugin_contracts.timekit import (
    format_iso,
    local_now,
    parse_iso,
    safe_zone,
    utcnow,
)

__all__ = [
    "format_iso",
    "local_now",
    "parse_iso",
    "safe_zone",
    "utcnow",
]
