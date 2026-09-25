"""上下文构造与材料贡献合同；持久摘要仍由摘要插件拥有。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.models import BoundChatModel, ModelRequest
from agent.plugin_contracts import ContentPart, ContentReferences, Message
from agent.plugin_contracts.models import ContextModel

MaterialData = Mapping[str, object]
SummaryData = Mapping[str, object]
Prepare = Callable[[tuple[Message, ...], str], Awaitable[MaterialData]]


class SummaryReducer(Protocol):
    """摘要 owner 先持久发布再返回；None 表示保留已有摘要，不做缩减。"""

    async def __call__(
        self,
        snapshot: tuple[Message, ...],
        materials: MaterialData,
        request: ModelRequest,
        model: BoundChatModel,
        projection: ContextModel,
        *,
        source: str,
        force: bool,
    ) -> SummaryData | None: ...


class ContextBuilder(Protocol):
    def check_summary(self, part: ContentPart) -> ContentReferences: ...
    def summary_range(
        self, snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]
    ) -> range: ...
    def settled_prefixes(self, messages: tuple[Message, ...]) -> tuple[int, ...]: ...
    def reminder_content(self, materials: MaterialData) -> str | None: ...
    def build(
        self,
        snapshot: Sequence[Message],
        *,
        materials: MaterialData,
        model: ContextModel,
        tools: Sequence[Mapping[str, Any]] = (),
        max_output_tokens: int,
        window_start: str | None = None,
    ) -> ModelRequest: ...
    def build_attempt(
        self,
        snapshot: Sequence[Message],
        *,
        materials: MaterialData,
        model: ContextModel,
        tools: Sequence[Mapping[str, Any]] = (),
        max_output_tokens: int,
        window_start: str | None = None,
    ) -> tuple[ModelRequest, str | None]: ...


class MaterialView(Protocol):
    async def prepare(
        self,
        snapshot: tuple[Message, ...],
        source: str,
        *,
        caller: Context | None = None,
        reminders: tuple[Mapping[str, object], ...] = (),
    ) -> MaterialData: ...
    async def reduce(
        self,
        snapshot: tuple[Message, ...],
        materials: MaterialData,
        request: ModelRequest,
        model: BoundChatModel,
        projection: ContextModel,
        *,
        source: str,
        force: bool,
    ) -> SummaryData | None: ...


class ContextMaterials(Protocol):
    """注册 Effect 归贡献者；bind 固定并保护本次实际使用的材料。"""

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        prepare: Prepare,
        priority: int = 0,
        prompt: bool = False,
        reduce: SummaryReducer | None = None,
    ) -> Effect: ...
    def bind(
        self, *, exclude: frozenset[str] = frozenset()
    ) -> AbstractAsyncContextManager[MaterialView]: ...


CONTEXT = ServiceKey[ContextBuilder]("context.v2")
MATERIALS = ServiceKey[ContextMaterials]("context.materials.v3")
