"""兼容入口；JSON 原子持久化工具由 `agent.plugin_contracts.json_store` 拥有。

本模块保留原导入路径，避免一次性改动 Core 侧既有调用点。新代码和插件应导入
`agent.plugin_contracts.json_store`。

再导出的名字按「全库实际被 import 的集合」给出（`atomic_write_text`、
`atomic_save_json`、`load_json`、`save_json`）。
"""

from agent.plugin_contracts.json_store import (
    atomic_save_json,
    atomic_write_text,
    load_json,
    save_json,
)

__all__ = [
    "atomic_save_json",
    "atomic_write_text",
    "load_json",
    "save_json",
]
