"""Compaction 与其它插件之间只交换已校验的结构值。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.models import ModelRequest
from agent.plugin_contracts import CallRef, Control, Message, Output, ToolCall, ToolResult


MaterialData = Mapping[str, object]
SummaryData = Mapping[str, object]


class ContextModel(Protocol):
    @property
    def context_window(self) -> int | None: ...

    @property
    def max_tool_schemas(self) -> int | None: ...

    def render(
        self, messages: tuple[Message, ...], *, after_seq: int,
        summary_reference: str | None = None, fresh: bool = False,
    ) -> ModelRequest: ...

    def estimate(self, request: ModelRequest) -> int: ...


class ContextBuilder(Protocol):
    def build_attempt(
        self, snapshot: Sequence[Message], *, materials: MaterialData,
        model: ContextModel, tools: Sequence[Mapping[str, object]] = (),
        max_output_tokens: int, window_start: str | None = None,
    ) -> tuple[ModelRequest, str | None]: ...


class MaterialRegistry(Protocol):
    async def register(
        self, ctx: Context, *, name: str,
        prepare: Callable[[tuple[Message, ...], str], Awaitable[MaterialData]],
        priority: int = 0, prompt: bool = False,
        reduce: Callable[..., Awaitable[SummaryData | None]] | None = None,
    ) -> object: ...


class Turn(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def after_seq(self) -> int: ...

    @property
    def through_seq(self) -> int: ...

    @property
    def ending_message_id(self) -> str | None: ...

    @property
    def status(self) -> str: ...

    @property
    def message_ids(self) -> tuple[str, ...]: ...

    @property
    def observations(self) -> tuple[tuple[CallRef, str], ...]: ...


class TurnProjection(Protocol):
    def project(self, messages: Sequence[Message], source: str) -> tuple[Turn, ...]: ...


class StoredSummary(Protocol):
    @property
    def reference(self) -> str: ...

    @property
    def session_id(self) -> str: ...

    @property
    def generation(self) -> int: ...

    @property
    def parent(self) -> str | None: ...

    @property
    def source_message_ids(self) -> tuple[str, ...]: ...

    @property
    def content(self) -> str: ...


class SummaryLookup(Protocol):
    def head(self, session_id: str) -> StoredSummary | None: ...

    def resolve(self, metadata: Mapping[str, object], *, session_id: str) -> StoredSummary: ...


class CompactionReader(Protocol):
    def source_text(self, messages: Sequence[Message]) -> str: ...

    def window_starts(self, messages: tuple[Message, ...], projection: TurnProjection) -> tuple[int, ...]: ...

    def summary_groups(
        self, groups: tuple[tuple[Message, ...], ...], snapshot: tuple[Message, ...],
    ) -> tuple[tuple[Message, ...], ...]: ...


CONTEXT = ServiceKey[ContextBuilder]("context.v2")
MATERIALS = ServiceKey[MaterialRegistry]("context.materials.v3")
TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")
COMPACTION_SUMMARIES = ServiceKey[SummaryLookup]("compaction.summaries.v1")
COMPACTION_READER = ServiceKey[CompactionReader]("compaction.reader.v1")


def settled_prefixes(messages: tuple[Message, ...]) -> tuple[int, ...]:
    """返回工具已结算或被明确放弃的前缀长度。"""
    pending: dict[CallRef, Message] = {}
    ends: list[int] = []
    for index, message in enumerate(messages):
        body = message.body
        if isinstance(body, Output):
            pending.update(
                (CallRef(message.message_id, position), message)
                for position, part in enumerate(body.parts)
                if isinstance(part, ToolCall)
            )
        elif isinstance(body, ToolResult):
            _ = pending.pop(body.call_ref, None)
        elif isinstance(body, Control) and body.action == "abandon":
            pending = {
                ref: call for ref, call in pending.items()
                if call.source != message.source or call.seq > body.through_seq
            }
        if not pending:
            ends.append(index + 1)
    return tuple(ends)


def summary_range(snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]) -> range:
    """按不可变消息身份定位摘要的连续覆盖区间。"""
    identities = tuple(message.message_id for message in snapshot)
    if not source_message_ids or source_message_ids[0] not in identities:
        raise ValueError("摘要来源缺少实际消息")
    start = identities.index(source_message_ids[0])
    end = start + len(source_message_ids)
    if identities[start:end] != source_message_ids:
        raise ValueError("摘要来源不等于实际连续消息范围")
    return range(start, end)
