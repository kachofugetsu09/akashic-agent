"""默认回复、完成阶段与状态的公共合同。"""
from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition.messages import MessageReader
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.tasks import ExternalRootPermit, Task
from agent.plugin_contracts import Message
from plugins.sources.contract import (
    SourceGuard,
)


class Completion(Protocol):
    def activity(
        self, reader: MessageReader, source: str
    ) -> AbstractContextManager[None]:
        """返回 Core 活动句柄；可跨任务结算，退出不再访问本 provider。"""
        ...

    def __call__(
        self,
        reader: MessageReader,
        source: str,
        *,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> AbstractAsyncContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class ReplyPreview:
    message_id: str
    text: str = ""
    thinking: str = ""
    call_record_id: str | None = None
    retry_status: str = ""


@dataclass(frozen=True, slots=True)
class ReplyActivity:
    session_id: str
    source: str
    handle: str
    active: bool
    preview: ReplyPreview | None = None


class ReplyStatus(Protocol):
    def snapshot(self, session_id: str) -> tuple[ReplyActivity, ...]: ...
    def follow(
        self, session_id: str
    ) -> AsyncGenerator[tuple[dict[str, object], ...], None]: ...


REPLY_COMPLETION = ServiceKey[Completion]("reply.completion.v1")
REPLY_STATUS = ServiceKey[ReplyStatus]("reply.status.v2")


class ReplyProgramV3(Protocol):
    async def __call__(
        self, task: Task, reader: MessageReader, source: str,
        reminders: Sequence[Mapping[str, object]], *, check_admission: SourceGuard,
    ) -> Message: ...


REPLY_PROGRAM_V3 = ServiceKey[ReplyProgramV3]("reply.program.v3")
