"""独立 JSON 摘要归档示例；只读首代 v2 记录，不取得账本或模型权限。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from plugins.compaction.contract import COMPACTION_SUMMARIES

api_version = 3
name = "summary-archive"
version = "1.0.0"
inject = ()


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[2]
    reference: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    generation: Literal[1]
    parent: None
    source_message_ids: tuple[str, ...] = Field(min_length=1)
    summary_message_ids: tuple[str, ...] = Field(min_length=1)
    omitted_message_ids: tuple[str, ...]
    content: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_partition(self) -> Self:
        """文件边界只接纳来源完整且分区不重叠的首代归档。"""
        source, summarized, omitted = self.source_message_ids, self.summary_message_ids, self.omitted_message_ids
        if (any(not item for item in source) or len(set(source)) != len(source)
                or set(summarized) & set(omitted) or len(source) != len(summarized) + len(omitted)
                or tuple(item for item in source if item in summarized) != summarized
                or tuple(item for item in source if item in omitted) != omitted):
            raise ValueError("摘要来源分区无效")
        return self


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record_ref: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class Archive:
    def __init__(self, rows: tuple[Summary, ...]):
        self._sessions = {row.session_id: row for row in rows}
        if len(self._sessions) != len(rows) or len({row.reference for row in rows}) != len(rows):
            raise ValueError("首代归档不能重复 Session 或摘要身份")

    def head(self, session_id: str) -> Summary | None:
        return self._sessions.get(session_id)

    def resolve(self, metadata: Mapping[str, object], *, session_id: str) -> Summary:
        reference = Reference.model_validate(dict(metadata))
        row = self.head(session_id)
        if row is None or reference.session_id != session_id or reference.record_ref != row.reference:
            raise ValueError("摘要 binding 没有对应 Session 记录")
        return row


async def apply(ctx: Context) -> None:
    raw = await run_file_io((ctx.data_root / "summaries.json").read_bytes)
    rows = TypeAdapter(tuple[Summary, ...]).validate_json(raw)
    await ctx.provide(COMPACTION_SUMMARIES, Archive(rows))
