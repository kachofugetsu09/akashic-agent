"""默认回复、完成阶段与状态的公共合同。"""
from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from plugins.ledger.contract import MessageReader
from plugins.models.contract import ModelAttemptStatus
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.tasks import ExternalRootPermit, Task
from plugins.ledger.contract import Message
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
    # 重试中的尝试进度；与 retry_status 分开，持续显示时不撤下本次尝试的草稿。
    attempt: ModelAttemptStatus | None = None
    # 本次物理尝试的首字耗时（毫秒），由 Models 在首段增量时给出；尚未出字时为空。
    first_token_ms: float | None = None
    # 本次物理尝试已到达的非空正文/思考增量数；生成中按它估算解码速度，
    # 结算后由 provider 上报的 usage 精确值取代，不参与持久化。
    output_deltas: int = 0
    # 从首段增量到本次增量到达的解码时长（毫秒），由 status 用单调时钟测量；
    # 尚未出字时为空。与 output_deltas 一起给出生成中的估算速度。
    decode_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ReplyActivity:
    session_id: str
    source: str
    handle: str
    active: bool
    preview: ReplyPreview | None = None


class ReplyStatus(Protocol):
    def snapshot(self, session_id: str) -> tuple[ReplyActivity, ...]: ...
    def follow_active_sessions(self) -> AsyncGenerator[frozenset[str], None]: ...
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
