"""Markdown memory 消费的外部能力边界。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, cast

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_contracts import ContentPart, ContentReferences, Message
from agent.turn_effects import PostCommitEffect


class MaterialRegistry(Protocol):
    async def register(
        self, ctx: Context, *, name: str,
        prepare: Callable[[tuple[Message, ...], str], Awaitable[Mapping[str, object]]],
        priority: int = 0, prompt: bool = False, reduce: object | None = None,
    ) -> object: ...


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
    def resolve(self, metadata: Mapping[str, object], *, session_id: str) -> StoredSummary: ...


class CompactionReader(Protocol):
    def source_text(self, messages: Sequence[Message]) -> str: ...

    def window_starts(self, messages: tuple[Message, ...], projection: TurnProjection) -> tuple[int, ...]: ...

    def summary_groups(
        self, groups: tuple[tuple[Message, ...], ...], snapshot: tuple[Message, ...],
    ) -> tuple[tuple[Message, ...], ...]: ...


class Turn(Protocol):
    message_ids: tuple[str, ...]
    observations: tuple[tuple[object, str], ...]


class TurnProjection(Protocol):
    def project(self, messages: Sequence[Message], source: str) -> tuple[Turn, ...]: ...


class ContentFacts(Protocol):
    def is_user_input(self, message: Message) -> bool: ...

    def legacy_post_commit_effect(self, message: Message) -> PostCommitEffect | None: ...


MATERIALS = ServiceKey[MaterialRegistry]("context.materials.v3")
COMPACTION_SUMMARIES = ServiceKey[SummaryLookup]("compaction.summaries.v1")
COMPACTION_READER = ServiceKey[CompactionReader]("compaction.reader.v1")
CONTENT = ServiceKey[ContentFacts]("content.v2")
TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")


def check_summary(part: ContentPart) -> ContentReferences:
    """校验已发布摘要的 binding 引用，不解析摘要正文。"""
    value = part.value
    if not isinstance(value, Mapping):
        raise ValueError("context.summary 必须是对象")
    data = dict(cast(Mapping[str, object], value))
    if set(data) != {"reference"} or not isinstance(data["reference"], str) or not data["reference"]:
        raise ValueError("context.summary 必须包含唯一的摘要 binding 引用")
    return ContentReferences(binding_ids=(data["reference"],))


def summary_range(snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]) -> range:
    """按摘要 owner 发布的消息身份定位连续覆盖区间。"""
    identities = tuple(message.message_id for message in snapshot)
    if not source_message_ids or source_message_ids[0] not in identities:
        raise ValueError("摘要来源缺少实际消息")
    start = identities.index(source_message_ids[0])
    end = start + len(source_message_ids)
    if identities[start:end] != source_message_ids:
        raise ValueError("摘要来源不等于实际连续消息范围")
    return range(start, end)
