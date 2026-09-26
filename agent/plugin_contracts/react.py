"""单次 Message 到 Message 的模型循环合同。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager
from typing import Protocol

from agent.plugin_composition import RuntimeScope, ServiceKey
from agent.plugin_composition.messages import MessageReader, MessageWriter, OwnerStore
from agent.plugin_composition.models import BoundChatModel, StreamCallback
from agent.plugin_contracts import Message
from agent.plugin_contracts.content import ContentView
from agent.plugin_contracts.context import ContextBuilder, SummaryReducer
from agent.plugin_contracts.models import MessageProjection
from agent.plugin_contracts.tools import ToolMenu

Materials = Mapping[str, object]
Preview = Callable[[str], AbstractContextManager[StreamCallback]]


class React(Protocol):
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
        state: OwnerStore | None = None,
    ) -> Message: ...


REACT = ServiceKey[React]("react.v2")
