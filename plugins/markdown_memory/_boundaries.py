"""Markdown memory 消费的外部能力边界。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_contracts import ContentPart, ContentReferences, Message


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


class ContextBuilder(Protocol):
    def check_summary(self, part: ContentPart) -> ContentReferences: ...

    def summary_range(
        self, snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...],
    ) -> range: ...


class ContentFacts(Protocol):
    def is_user_input(self, message: Message) -> bool: ...

    def legacy_post_commit_effect(self, message: Message) -> str | None: ...


MATERIALS = ServiceKey[MaterialRegistry]("context.materials.v3")
CONTEXT = ServiceKey[ContextBuilder]("context.v2")
COMPACTION_SUMMARIES = ServiceKey[SummaryLookup]("compaction.summaries.v1")
COMPACTION_READER = ServiceKey[CompactionReader]("compaction.reader.v1")
CONTENT = ServiceKey[ContentFacts]("content.v2")
TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")
