"""一次回复执行程序的公共合同；依赖由提供方捕获。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.messages import MessageReader
from agent.plugin_composition.model import ServiceKey
from plugins.models.contract import StreamCallback
from agent.plugin_composition.tasks import Task
from agent.plugin_contracts import Message
from plugins.models.contract import ContentRenderer
from plugins.context.contract import MaterialKind
from plugins.tools.contract import ToolView
from plugins.tools.contract import ToolPresentation
from plugins.sources.contract import (
    SourceGuard,
)


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
