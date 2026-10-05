"""来源调用回复程序的公共入口；依赖由程序 provider 捕获。"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.messages import MessageReader
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.models import StreamCallback
from agent.plugin_composition.tasks import ExternalRootPermit, Task
from agent.plugin_contracts import Message
from agent.plugin_contracts.models import ContentRenderer
from agent.plugin_contracts.context import MaterialKind
from agent.plugin_contracts.tools import ToolPresentation, ToolView
from agent.plugin_contracts.sources import SourceGuard


class ReplyExecuteV4(Protocol):
    """固定执行前提；输出预算 None 跟随模型上限，未知时用 32768，含推理。"""

    async def __call__(
        self,
        ctx: Context,
        task: Task,
        reader: MessageReader,
        source: str,
        *,
        check_admission: SourceGuard,
        authorize: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]
        ],
        max_output_tokens: int | None,
        max_steps: int,
        render_content: ContentRenderer | None = None,
        tool_view: ToolView | None = None,
        tool_names: Sequence[str] | None = None,
        exclude_material_kinds: frozenset[MaterialKind] = frozenset(),
        prompt_hints: Sequence[str] = (),
        fixed_bindings: Mapping[str, str] | None = None,
        preview: Callable[[str], AbstractContextManager[StreamCallback]] | None = None,
        reminders: Sequence[Mapping[str, object]] = (),
        terminal_tools: frozenset[str] = frozenset(),
        presentation: ToolPresentation | None = None,
    ) -> Message: ...


REPLY_EXECUTE_V4 = ServiceKey[ReplyExecuteV4]("reply.execute.v4")


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
