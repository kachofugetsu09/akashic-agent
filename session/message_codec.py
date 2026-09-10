"""兼容入口；消息编解码由 `agent.plugin_contracts.message_codec` 拥有。

本模块保留原导入路径，避免一次性改动既有调用点。新代码和插件应导入
`agent.plugin_contracts`。
"""

from agent.plugin_contracts.message_codec import (
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
