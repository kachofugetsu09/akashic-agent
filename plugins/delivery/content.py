"""兼容入口；原生发送正文读取由结构合同层拥有。

`read_content`/`File`/`AttachmentReadError` 的拥有者已移到
`agent.plugin_contracts.model_content`（它们只读取消息与只读附件端口）；实现侧
按原路径再导出。
"""

from agent.plugin_contracts.model_content import (  # noqa: F401
    AttachmentReadError,
    File,
    read_content,
)

__all__ = ["AttachmentReadError", "File", "read_content"]
