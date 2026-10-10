"""摘要只读合同；记录发布和迁移仍由 compaction 拥有。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from agent.plugin_composition import ServiceKey
from agent.plugin_contracts import Message
from agent.plugin_contracts.turns import TurnProjection


class StoredSummary(Protocol):
    @property
    def version(self) -> int: ...

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


class PartitionedSummary(StoredSummary, Protocol):
    """compaction v2 的结构化分区；不依赖 compaction 的 class identity。"""

    @property
    def version(self) -> Literal[2]: ...

    @property
    def summary_message_ids(self) -> tuple[str, ...]: ...

    @property
    def omitted_message_ids(self) -> tuple[str, ...]: ...


class SummaryLookup(Protocol):
    def head(self, session_id: str) -> StoredSummary | None: ...
    def resolve(
        self, metadata: Mapping[str, object], *, session_id: str
    ) -> StoredSummary: ...


class CompactionReader(Protocol):
    def source_text(self, messages: Sequence[Message]) -> str: ...

    def window_starts(
        self, messages: tuple[Message, ...], projection: TurnProjection
    ) -> tuple[int, ...]: ...

    def summary_groups(
        self,
        groups: tuple[tuple[Message, ...], ...],
        snapshot: tuple[Message, ...],
    ) -> tuple[tuple[Message, ...], ...]: ...


COMPACTION_SUMMARIES = ServiceKey[SummaryLookup]("compaction.summaries.v1")
COMPACTION_READER = ServiceKey[CompactionReader]("compaction.reader.v1")
