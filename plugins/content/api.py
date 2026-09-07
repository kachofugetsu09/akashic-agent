from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from session.message import ContentPart


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
class Span:
    """替换原文的一个精确区间；零长度区间用于附加引用等非文本事实。"""

    start: int
    end: int
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end < self.start
        ):
            raise ValueError("内容区间必须是有序的非负整数")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("文本协议只能产生内容块，不能提出工具调用")
        if self.start == self.end and any(part.kind == "text" for part in parts):
            raise ValueError("零长度区间只能附加非文本事实")
        object.__setattr__(self, "parts", parts)


ContentCheck = Callable[[ContentPart], tuple[str, ...]]


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


TextDecoder = Callable[[TextSource, tuple[Reference, ...]], Awaitable[Sequence[Span]]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentSchema:
    """内容 owner 的纯 schema 声明，结构化生产者不需要文本协议。"""

    name: str
    content: Mapping[str, ContentCheck]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("内容声明名不能为空")
        if any(
            not isinstance(key, str) or not key or not callable(check)
            for key, check in self.content.items()
        ):
            raise TypeError("内容声明必须映射 kind 到纯 schema 校验函数")
        if "text" in self.content or "tool_call" in self.content:
            raise ValueError("内容声明不能重定义基础文本或工具调用")
        object.__setattr__(self, "content", MappingProxyType(dict(self.content)))


@dataclass(frozen=True, slots=True, kw_only=True)
class TextProtocol(ContentSchema):
    """文本协议一起提供提示、解析与产生的内容 schema。"""

    prompt: str
    decode: TextDecoder

    def __post_init__(self) -> None:
        ContentSchema.__post_init__(self)
        if not isinstance(self.prompt, str) or not callable(self.decode):
            raise TypeError("文本协议必须提供提示文本和 decoder")
