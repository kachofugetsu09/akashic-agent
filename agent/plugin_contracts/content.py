"""内容注册、校验与一次解码视图的公共合同。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugin_contracts import ContentPart, ContentReferences, Message

ContentCheck = Callable[[ContentPart], ContentReferences]


class ContentView(Protocol):
    def check_metadata(self, metadata: Mapping[str, object]) -> None: ...
    @property
    def prompts(self) -> tuple[str, ...]: ...
    @property
    def checks(self) -> Mapping[str, ContentCheck]: ...
    async def decode(
        self, text: str, references: Sequence[Mapping[str, object]] = ()
    ) -> tuple[tuple[ContentPart, ...], Mapping[str, object]]: ...


class Content(Protocol):
    """注册普通结构声明；活视图只在 bind 的作用域内有效。"""

    def check_text(self, part: ContentPart) -> ContentReferences: ...
    def check_artifact(self, part: ContentPart) -> ContentReferences: ...
    def is_user_input(self, message: Message) -> bool: ...
    def legacy_post_commit_effect(self, message: Message) -> str | None: ...
    async def register(
        self,
        ctx: Context,
        definition: Mapping[str, object],
        *,
        prepare: Callable[[], Mapping[str, object]] | None = None,
    ) -> Effect: ...
    def describe(self) -> Mapping[str, object]: ...
    def save_binding(self, bindings: Bindings) -> str: ...
    def bind(self) -> AbstractAsyncContextManager[ContentView]: ...


CONTENT = ServiceKey[Content]("content.v2")
