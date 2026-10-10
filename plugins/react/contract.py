"""单次 Message 到 Message 的模型循环合同。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager
from typing import Protocol

from agent.plugin_composition import RuntimeScope, ServiceKey
from agent.plugin_composition.messages import MessageReader, MessageWriter, OwnerStore
from agent.plugin_composition.models import BoundChatModel, StreamCallback
from agent.plugin_contracts import Message
from plugins.content.contract import ContentView
from plugins.context.contract import ContextBuilder, SummaryReducer
from plugins.models.contract import MessageProjection
from agent.plugin_contracts.tools import ToolMenu, StartCheck

Materials = Mapping[str, object]
Preview = Callable[[str], AbstractContextManager[StreamCallback]]


class OrderedReact(Protocol):
    async def __call__(
        self,
        reader: MessageReader,
        writer: MessageWriter,
        *,
        model: BoundChatModel,
        context: ContextBuilder,
        projection: MessageProjection,
        materials: Callable[[tuple[Message, ...]], Awaitable[Materials]],
        content: ContentView,
        tools: ToolMenu,
        max_output_tokens: int,
        max_steps: int,
        reduce: SummaryReducer | None = None,
        preview: Preview | None = None,
        terminal_tools: frozenset[str] = frozenset(),
        capture_scope: Callable[[], RuntimeScope] | None = None,
        state: OwnerStore,
        check_start: StartCheck,
        max_parallel_calls: int = 1,
    ) -> Message: ...


# v2 也在新 Output 的同一事务核对控制前提，不能把首次启动许可当作永久提交权。
REACT_ORDERED_V2 = ServiceKey[OrderedReact]("react.ordered-start.v2")
