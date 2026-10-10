from __future__ import annotations

from typing import Protocol

from agent.plugin_composition.model import ServiceKey
from session.artifacts import (
    AttachmentKind as AttachmentKind, AttachmentReadLease, AttachmentRef as AttachmentRef,
    check_artifact_id as check_artifact_id,
)


class ArtifactRead(Protocol):
    """只授予有界读取，不暴露附件路径或 repository。"""

    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease: ...


class ArtifactImport(Protocol):
    """导入来源并取得不可变引用，不授予消息写入或删除权。"""

    async def import_source(self, source: str, kind: AttachmentKind) -> AttachmentRef: ...


ARTIFACT_READ = ServiceKey[ArtifactRead]("core.artifact_read")
ARTIFACT_IMPORT = ServiceKey[ArtifactImport]("core.artifact_import")
