"""旧 Core 路径兼容；Message 编解码实现由公开值模块拥有。"""
from agent.plugin_contracts.message import (
    body_to_dict as body_to_dict, json_value as json_value,
    encode_body as encode_body, decode_body as decode_body,
    _unique_fields as _unique_fields,
)
