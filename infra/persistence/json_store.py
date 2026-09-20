"""兼容入口；JSON 原子持久化工具由公开结构合同拥有。"""

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
