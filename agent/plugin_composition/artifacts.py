from __future__ import annotations

from collections.abc import Awaitable, Callable

from agent.plugin_composition.model import ServiceKey
from session.artifacts import AttachmentReadLease, AttachmentRef


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
