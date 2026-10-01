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


class ReplyExecute(Protocol):
    """在来源已接纳的 Task 与 Input 上运行；调用者只交入授权与本次选择。"""

    async def __call__(
        self,
        ctx: Context,
        task: Task,
        reader: MessageReader,
        source: str,
        *,
        authorize: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]
        ],
        max_output_tokens: int,
        max_steps: int,
        render_content: ContentRenderer | None = None,
        tool_view: ToolView | None = None,
        tool_names: Sequence[str] | None = None,
        exclude_materials: frozenset[str] = frozenset(),
        prompt_hints: Sequence[str] = (),
        fixed_bindings: Mapping[str, str] | None = None,
        preview: Callable[[str], AbstractContextManager[StreamCallback]] | None = None,
        reminders: Sequence[Mapping[str, object]] = (),
        terminal_tools: frozenset[str] = frozenset(),
        presentation: ToolPresentation | None = None,
    ) -> Message: ...


class ReplyExecuteV2(Protocol):
    """在来源已接纳的 Task 与 Input 上运行；调用者只交入授权与本次选择。"""

    async def __call__(
        self,
        ctx: Context,
        task: Task,
        reader: MessageReader,
        source: str,
        *,
        authorize: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]
        ],
        max_output_tokens: int,
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


class ReplyExecuteV3(Protocol):
    """来源交入不可变控制前提，和输出来源的前提共同核对。

    没有独立 check_admission 时，调用者把 Task.boundary_hint 固定为本次获授的
    同来源 Input/Control seq；多阶段来源按实际阶段输入固定，不用稍后 head 代替。
    跨来源回传必须交入控制来源签发的 check_admission，不能靠输出来源推断权限。
    """

    async def __call__(
        self,
        ctx: Context,
        task: Task,
        reader: MessageReader,
        source: str,
        *,
        authorize: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]
        ],
        max_output_tokens: int,
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
        check_admission: SourceGuard | None = None,
    ) -> Message: ...


REPLY_EXECUTE = ServiceKey[ReplyExecute]("reply.execute.v1")
REPLY_EXECUTE_V2 = ServiceKey[ReplyExecuteV2]("reply.execute.v2")
REPLY_EXECUTE_V3 = ServiceKey[ReplyExecuteV3]("reply.execute.v3")


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
REPLY_PROGRAM = ServiceKey[
    Callable[
        [Task, MessageReader, str, Sequence[Mapping[str, object]]], Awaitable[Message]
    ]
]("reply.program.v2")
REPLY_STATUS = ServiceKey[ReplyStatus]("reply.status.v2")


class ReplyProgramV3(Protocol):
    async def __call__(
        self, task: Task, reader: MessageReader, source: str,
        reminders: Sequence[Mapping[str, object]], *, check_admission: SourceGuard,
    ) -> Message: ...


REPLY_PROGRAM_V3 = ServiceKey[ReplyProgramV3]("reply.program.v3")
