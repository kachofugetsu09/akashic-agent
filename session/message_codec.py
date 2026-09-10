"""兼容入口；消息编解码由 `agent.plugin_contracts.message_codec` 拥有。

本模块保留原导入路径，避免一次性改动既有调用点。新代码和插件应导入
`agent.plugin_contracts`。
"""

from agent.plugin_contracts.message_codec import (
    _unique_fields,
    body_to_dict,
    decode_body,
    encode_body,
    json_value,
)

__all__ = [
    "body_to_dict",
    "decode_body",
    "encode_body",
    "json_value",
]

# `_unique_fields` 不在 __all__ 中：它是私有 helper，但不可变的 yoyo 迁移
# `20260907_03_message_metadata.py` 直接 import 它，因此再导出必须保留该名字
# （`from x import _name` 不受 __all__ 限制）。删除它会让从零安装的 workspace
# 无法 apply 迁移。由 tests/test_plugin_contracts.py 守护。
