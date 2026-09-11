"""兼容入口；媒体编码工具由 `agent.plugin_contracts.media` 拥有。

本模块保留原导入路径，避免一次性改动既有调用点。新代码和插件应导入
`agent.plugin_contracts.media`。
"""

from agent.plugin_contracts.media import (
    MAX_IMAGE_DATA_URI_TOTAL_BYTES,
    MAX_IMAGE_FILE_BYTES,
    detect_supported_image_mime,
    encode_image_bytes,
    encode_image_data_uri,
    validate_image_attachment_budget,
)

__all__ = [
    "MAX_IMAGE_DATA_URI_TOTAL_BYTES",
    "MAX_IMAGE_FILE_BYTES",
    "detect_supported_image_mime",
    "encode_image_bytes",
    "encode_image_data_uri",
    "validate_image_attachment_budget",
]
