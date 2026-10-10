from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from core.common.frozen_json import freeze_json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, AsyncContextManager, Literal, Protocol, TypeAlias, cast

from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.context import Context
    from agent.plugin_composition.bindings import Bindings


UsageCoverage: TypeAlias = Literal["exact", "partial", "unavailable"]


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    request_count: int = 1
    covered_request_count: int = 0
    coverage: UsageCoverage = "unavailable"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_json_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ModelContinuation:
    """Opaque driver state bound to one exact BoundModelDescriptor.binding_id."""

    binding_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_json_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str | None
    tool_calls: Sequence[ToolCall] = ()
    thinking: str | None = None
    finish_reason: str | None = None
    continuation: ModelContinuation | None = None
    usage: ModelUsage | None = None
    call_record_id: str | None = None
    # 原生响应部件由调用账本保存，随同一 binding 的消息重放。
    provider_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """冻结一次响应，多个请求等待者不能相互改变已结算事实。"""
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.provider_metadata is not None:
            if not isinstance(self.provider_metadata, Mapping):
                raise TypeError("响应协议 metadata 必须是 JSON 对象")
            object.__setattr__(self, "provider_metadata", _freeze_json_mapping(self.provider_metadata))


StreamCallback: TypeAlias = Callable[[dict[str, str]], Awaitable[None]]


def read_content_refs(value: object) -> tuple[tuple[str, int], ...]:
    """在请求与持久记录边界校验同一种消息内容位置编码。"""
    if not isinstance(value, (tuple, list)):
        raise ValueError("内容位置必须是数组")
    refs: list[tuple[str, int]] = []
    for ref in value:
        if (not isinstance(ref, (tuple, list)) or len(ref) != 2
                or not isinstance(ref[0], str) or not ref[0]
                or type(ref[1]) is not int or ref[1] < 0):
            raise ValueError("内容位置需要消息 ID 和非负整数索引")
        refs.append((ref[0], ref[1]))
    if len(set(refs)) != len(refs):
        raise ValueError("请求内容位置不能重复")
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: Sequence[Mapping[str, Any]]
    tools: Sequence[Mapping[str, Any]] = ()
    max_output_tokens: int = 0
    system_prompt: str = ""
    tool_choice: str | Mapping[str, Any] = "auto"
    prompt_cache_key: str | None = None
    on_delta: StreamCallback | None = None
    continuation: ModelContinuation | None = None
    disable_reasoning: bool = False
    request_key: str | None = None
    # 本次完整展示、尚无成功展示回执的消息内容位置；不发送给 provider。
    content_refs: tuple[tuple[str, int], ...] = ()
    content_transformed: bool = False

    def __post_init__(self) -> None:
        """在唯一调用边界冻结请求，adapter 和并行调用不能改写彼此输入。"""
        object.__setattr__(
            self, "messages", _freeze_json_rows(self.messages)
        )
        object.__setattr__(
            self, "tools", _freeze_json_rows(self.tools)
        )
        object.__setattr__(self, "content_refs", read_content_refs(self.content_refs))
        if type(self.content_transformed) is not bool:
            raise ValueError("内容投影标记必须是 bool")
        if isinstance(self.tool_choice, Mapping):
            object.__setattr__(
                self, "tool_choice", _freeze_json_mapping(self.tool_choice)
            )


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    usage: ModelUsage | None = None


def _freeze_json_rows(value: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """接纳普通 Sequence；内部深冻结数组不再逐行遍历。"""
    rows = value if isinstance(value, (list, tuple)) else tuple(value)
    return cast(tuple[Mapping[str, Any], ...], freeze_json(rows))


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """模型请求与消息共用同一 JSON 冻结边界。"""
    return cast(Mapping[str, Any], freeze_json(value))


__all__ = [
    "EmbeddingResult",
    "LLMResponse",
    "ModelContinuation",
    "ModelRequest",
    "ModelUsage",
    "StreamCallback",
    "ToolCall",
    "UsageCoverage",
]
