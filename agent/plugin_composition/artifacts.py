from __future__ import annotations

from collections.abc import Awaitable, Callable

from agent.plugin_composition.model import ServiceKey
from session.artifacts import AttachmentKind, AttachmentReadLease, AttachmentRef


class _ReadLease:
    """只交付有界字节读取与关闭权，不暴露旧模型路径或附件 repository。"""

    def __init__(self, lease: AttachmentReadLease):
        self._ref = lease.ref
        self._read = lease.read_bytes
        self._close = lease.aclose

    @property
    def ref(self) -> AttachmentRef:
        return self._ref

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        return await self._read(max_bytes=max_bytes)

    async def aclose(self) -> None:
        await self._close()


class ArtifactRead:
    """所有来源共用的窄读取能力；引用由消息或资源 owner 提供。"""

    def __init__(self, acquire: Callable[[AttachmentRef], Awaitable[AttachmentReadLease]] | None):
        self._acquire = acquire

    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease:
        if self._acquire is None:
            raise RuntimeError("candidate 验证期禁止打开正式 artifact")
        return _ReadLease(await self._acquire(ref))


ARTIFACT_READ = ServiceKey[ArtifactRead]("core.artifact_read")


class ArtifactImport:
    """只授予来源导入与不可变引用；不附带消息、读取、删除或任意数据库权限。"""

    def __init__(self, import_source: Callable[[str, AttachmentKind], Awaitable[AttachmentRef]] | None):
        self._import_source = import_source

    async def import_source(self, source: str, kind: AttachmentKind) -> AttachmentRef:
        if self._import_source is None:
            raise RuntimeError("candidate 验证期禁止导入正式 artifact")
        return await self._import_source(source, kind)


ARTIFACT_IMPORT = ServiceKey[ArtifactImport]("core.artifact_import")
