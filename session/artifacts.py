"""兼容入口；附件值词汇表由 `agent.plugin_contracts.artifacts` 拥有。

本模块保留原导入路径，避免一次性改动既有调用点。新代码和插件应导入
`agent.plugin_contracts.artifacts`。
"""

from agent.plugin_contracts.artifacts import (
    AttachmentKind,
    AttachmentReadLease,
    AttachmentReadPort,
    AttachmentRef,
    check_artifact_id,
)

__all__ = [
    "AttachmentKind",
    "AttachmentReadLease",
    "AttachmentReadPort",
    "AttachmentRef",
    "check_artifact_id",
]
