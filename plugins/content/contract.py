"""内容注册、校验与一次解码视图的公共合同。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol, TypedDict

from agent.plugin_composition import Context, Effect, ServiceKey
from plugins.ledger.contract import Bindings
from plugins.ledger.contract import ContentPart, ContentReferences, Message

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


# Fleet Citation/Meme 的文本协议使用这些值，Content 仍独占 wire 解码与引用校验。
@dataclass(frozen=True, slots=True)
class Reference:
    """调用方实际取得的引用证据；模型声明不能自己产生这些权限。"""

    ref: str
    resolved_ref: str | None = None
    retrieval_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref:
            raise ValueError("引用身份不能为空")
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in (self.resolved_ref, self.retrieval_ref)
        ):
            raise ValueError("引用的解析目标与查询凭据必须是非空字符串或 None")


@dataclass(frozen=True, slots=True)
class TextSource:
    """原文与其字面区间，供协议同时判定显式声明和召回兜底。"""

    text: str
    literals: tuple[tuple[int, int], ...]

    def allows(self, start: int, end: int) -> bool:
        return not any(left < end and start < right for left, right in self.literals)

    def matches(self, pattern: re.Pattern[str]) -> Iterator[re.Match[str]]:
        return (
            match
            for match in pattern.finditer(self.text)
            if self.allows(match.start(), match.end())
        )


class SpanData(TypedDict):
    start: int
    end: int
    parts: tuple[ContentPart, ...]
