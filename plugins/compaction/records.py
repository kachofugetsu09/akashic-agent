"""摘要与当前引用同事务发布；旧 generation 始终保留。"""
from __future__ import annotations

import json
from typing import Annotated, Literal, Self, cast
from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator
from agent.plugin_composition import ServiceKey

from session.log import MessageConflict, MessageReader, OwnerStore, OwnerTransaction
from session.message_codec import json_value
from plugins.context.api import summary_range

Text = Annotated[str, Field(min_length=1)]


class SummaryRecord(BaseModel):
    """一次已发布压缩的完整覆盖、模型出处与生成条件。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[1] = 1
    reference: Text
    session_id: Text
    generation: Annotated[int, Field(ge=1)]
    parent: Text | None
    source_message_ids: tuple[Text, ...] = Field(min_length=1)
    content: Text
    model_call_ids: tuple[Text, ...] = Field(min_length=1)
    trigger: Literal["soft_limit", "context_overflow"]
    context_window: Annotated[int, Field(gt=0)]
    max_output_tokens: Annotated[int, Field(gt=0)]
    keep_recent_tokens: Annotated[int, Field(gt=0)]
    tokens_before: Annotated[int, Field(ge=0)]
    tokens_after: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def check_identity(self) -> Self:
        if (self.generation == 1) != (self.parent is None):
            raise ValueError("摘要首代必须没有 parent，后续代必须声明 parent")
        if self.parent == self.reference:
            raise ValueError("摘要不能以自己为 parent")
        if len(set(self.source_message_ids)) != len(self.source_message_ids):
            raise ValueError("摘要来源消息不能重复")
        if len(set(self.model_call_ids)) != len(self.model_call_ids):
            raise ValueError("摘要模型调用不能重复")
        return self


class _Head(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    reference: Text


class SummaryRef(BaseModel):
    """binding 中固定的摘要身份；不再复制 parent 或 generation。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    record_ref: Text
    session_id: Text


class SummaryLookup:
    """归档提供的窄读取口；不能发布摘要、推进 head 或启动模型。"""

    def __init__(self, read: Callable[[str], SummaryRecord | None]):
        self._read = read

    def resolve(self, metadata: Mapping[str, object], *, session_id: str) -> SummaryRecord:
        """只解析 binding 固定的原始记录，并核对完整父链。"""
        reference = SummaryRef.model_validate(dict(metadata))
        if reference.session_id != session_id:
            raise ValueError("摘要 binding 不属于当前 Session")
        record = self._read(reference.record_ref)
        if record is None or record.session_id != session_id:
            raise ValueError("摘要 binding 没有对应 Session 的原始记录")
        current = record
        while current.parent is not None:
            parent = self._read(current.parent)
            if parent is None or parent.session_id != session_id:
                raise ValueError("摘要父链缺失或跨 Session")
            # generation 严格递减同时排除循环，无需维护第二套访问状态。
            if (parent.generation + 1 != current.generation
                    or len(parent.source_message_ids) >= len(current.source_message_ids)
                    or current.source_message_ids[:len(parent.source_message_ids)] != parent.source_message_ids):
                raise ValueError("摘要父链的 generation 或来源前缀不连续")
            current = parent
        return record


COMPACTION_SUMMARIES = ServiceKey[SummaryLookup]("compaction.summaries.v1")


class SummaryRecords:
    """compaction 的唯一发布入口；调用者只能创建记录或读取已有内容。"""

    def __init__(self, state: OwnerStore):
        self._state = state

    def read(self, reference: str) -> SummaryRecord | None:
        row = self._state.read("summary:" + reference)
        if row is None:
            return None
        record = SummaryRecord.model_validate_json(json.dumps(json_value(row.value)))
        if record.reference != reference or row.version != 0:
            raise ValueError("摘要记录的身份或不可变版本损坏")
        return record

    def head(self, session_id: str) -> SummaryRecord | None:
        row = self._state.read("head:" + session_id)
        if row is None:
            return None
        head = _Head.model_validate_json(json.dumps(json_value(row.value)))
        record = self.read(head.reference)
        if record is None or record.session_id != session_id:
            raise ValueError("摘要 head 没有对应 Session 的已发布记录")
        return record

    def publish(
        self, record: SummaryRecord, reader: MessageReader, *, parent: SummaryRecord | None,
    ) -> SummaryRecord:
        """原子检查 parent 与消息前缀，再只增摘要并推进本 Session 指针。"""
        self._state.check_access(reader)
        if reader.session_id != record.session_id:
            raise ValueError("摘要 reader 与记录必须属于同一 Session")

        def commit(tx: OwnerTransaction) -> SummaryRecord:
            # 1. 同一出处允许重放，不能改内容，也不能把 head 倒退到旧代。
            existing = self.read(record.reference)
            if existing is not None:
                if existing != record:
                    raise ValueError("同一摘要出处的内容发生漂移")
                return existing
            current = self.head(record.session_id)
            if current != parent:
                raise MessageConflict("摘要 parent 已被其他发布推进")
            generation = 1 if parent is None else parent.generation + 1
            reference = None if parent is None else parent.reference
            if record.generation != generation or record.parent != reference:
                raise ValueError("摘要 generation 与 parent 不连续")
            if parent is not None and (len(record.source_message_ids) <= len(parent.source_message_ids)
                    or record.source_message_ids[:len(parent.source_message_ids)] != parent.source_message_ids):
                raise ValueError("后续摘要不能撤回已有来源")

            # 2. 源消息与摘要在同一 authority 的事务中校验；读取模块无正文写权。
            _ = summary_range(reader.snapshot(), record.source_message_ids)
            _ = tx.save("summary:" + record.reference,
                        cast(Mapping[str, object], record.model_dump(mode="json")), expected_version=None)
            key = "head:" + record.session_id
            previous = tx.read(key)
            _ = tx.save(key, {"reference": record.reference},
                        expected_version=None if previous is None else previous.version)
            return record

        return self._state.transact(commit)
