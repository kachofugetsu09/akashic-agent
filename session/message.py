from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal, cast


def freeze_json(value: object) -> object:
    """在消息边界复制 JSON 值，阻止调用者随后改变已接纳内容。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("消息 JSON 不接受非有限浮点数")
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError("消息 JSON 对象的 key 必须是字符串")
        return MappingProxyType(
            {cast(str, key): freeze_json(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item) for item in cast(list[object] | tuple[object, ...], value)
        )
    raise TypeError(f"消息内容必须是 JSON 值，实际为 {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ContentPart:
    """内容类型及其不可变载荷；具体 schema 由声明该类型的能力校验。"""

    kind: str
    value: object

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind or self.kind == "tool_call":
            raise ValueError("内容类型不能为空")
        object.__setattr__(self, "value", freeze_json(self.value))


@dataclass(frozen=True, slots=True)
class ToolCall:
    binding_id: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or not self.binding_id:
            raise ValueError("工具调用必须固定 binding_id")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("工具参数必须是 JSON 对象")
        object.__setattr__(self, "arguments", freeze_json(self.arguments))


@dataclass(frozen=True, slots=True)
class CallRef:
    message_id: str
    part_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_id, str)
            or not self.message_id
            or type(self.part_index) is not int
            or self.part_index < 0
        ):
            raise ValueError("调用引用需要 message_id 和非负 part_index")


type Part = ContentPart | ToolCall


@dataclass(frozen=True, slots=True)
class Input:
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("Input 只能包含内容块")
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True, slots=True)
class Output:
    parts: tuple[Part, ...]
    finish: Literal["continue", "complete", "quiet"]

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if any(not isinstance(part, (ContentPart, ToolCall)) for part in parts):
            raise TypeError("Output 只能包含内容块或工具调用")
        object.__setattr__(self, "parts", parts)
        if self.finish not in {"continue", "complete", "quiet"}:
            raise ValueError("Output finish 无效")
        if self.finish != "continue" and any(
            isinstance(part, ToolCall) for part in self.parts
        ):
            raise ValueError("含工具调用的 Output 必须是 continue")


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_ref: CallRef
    outcome: Literal["success", "denied", "error", "unknown"]
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.call_ref, CallRef):
            raise TypeError("工具结果需要有效的调用引用")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("ToolResult 只能包含内容块")
        object.__setattr__(self, "parts", parts)
        if self.outcome not in {"success", "denied", "error", "unknown"}:
            raise ValueError("ToolResult outcome 无效")


@dataclass(frozen=True, slots=True)
class Control:
    action: Literal["pause", "resume", "abandon", "failure"]
    through_seq: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"pause", "resume", "abandon", "failure"}:
            raise ValueError("Control action 无效")
        if type(self.through_seq) is not int or self.through_seq < 0:
            raise ValueError("Control through_seq 必须非负")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("Control reason 必须是文本")


type Body = Input | Output | ToolResult | Control


@dataclass(frozen=True, slots=True)
class Message:
    """一条已接纳事实；作者、来源与消息用途分别表达独立信息。"""

    message_id: str
    session_id: str
    seq: int
    recorded_at: datetime
    author: str
    source: str
    body: Body

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.message_id, self.session_id, self.author, self.source)
        ):
            raise ValueError("消息身份、Session、作者和来源不能为空")
        if type(self.seq) is not int or self.seq < 0:
            raise ValueError("消息 seq 必须非负")
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.utcoffset() is None
        ):
            raise ValueError("消息接纳时间必须包含时区")
        if not isinstance(self.body, (Input, Output, ToolResult, Control)):
            raise TypeError("消息 body 类型无效")
        if isinstance(self.body, Control) and self.body.through_seq >= self.seq:
            raise ValueError("Control 只能指向之前的已接纳前缀")
