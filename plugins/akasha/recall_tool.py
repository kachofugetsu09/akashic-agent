"""归档召回工具从一致学习快照查询，重试只读取原查询出处。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.models import BoundEmbeddingModel
from plugins.content.api import Reference
from plugins.tools.api import CallSource, InvalidArguments, Result
from plugins.tools.plugin import TOOLS
from session.embedding_store import MessageEmbeddings
from session.log import MessageCatalog
from session.message import ContentPart, ContentReferences, Message, Output, ToolCall, ToolResult
from session.message_codec import json_value

from .application.consumer import run_memory_job
from .application.snapshot import read_memory
from .domain.model import MemoryConfig
from .learning import AKASHA_LEARNING, Learning, LearningConfig
from .recalls import ProgramSource, Recall, RecallRecords, ToolSource, query_memory, render_materials


class RecallArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=32000)
    limit: int = Field(default=10, ge=1, le=40)

    @field_validator("query")
    @classmethod
    def check_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("召回问题不能为空白")
        return value


class PreparedRecall(RecallArguments):
    source: ToolSource | None
    learning_binding: str = Field(min_length=1)
    embedding_binding: str = Field(min_length=1)
    max_chars: int = Field(gt=0)


class RecallReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    retrieval_ref: str = Field(min_length=1)


def check_recall(part: ContentPart) -> ContentReferences:
    _ = RecallReference.model_validate_json(json.dumps(json_value(part.value)))
    return ContentReferences()


def tool_references(
    snapshot: tuple[Message, ...], source: str, learning: Learning,
    bindings: Bindings, records: RecallRecords,
) -> tuple[Reference, ...]:
    """只有实际 Akasha 调用产生且属于该 CallRef 的查询记录能授予本地引用。"""
    turns = learning.projection.project(snapshot, source)
    if not turns or turns[-1].status != "open":
        return ()
    by_id = {message.message_id: message for message in snapshot}
    references: dict[str, Reference] = {}
    for call_ref, identity in turns[-1].observations:
        result = by_id[identity].body
        if not isinstance(result, ToolResult) or result.outcome != "success":
            continue
        for part in result.parts:
            if part.kind != "akasha.recall":
                continue
            request = by_id[call_ref.message_id]
            if not isinstance(request.body, Output):
                raise ValueError("召回出处不是工具请求")
            call = request.body.parts[call_ref.part_index]
            if not isinstance(call, ToolCall):
                raise ValueError("召回出处没有指向 ToolCall")
            metadata = bindings.describe(call.binding_id, TOOLS)
            tool = metadata["tool"]
            if not isinstance(tool, Mapping) or tool["owner"] != learning.owner:
                raise ValueError("其他工具不能伪造 Akasha 召回出处")
            marker = RecallReference.model_validate_json(json.dumps(json_value(part.value)))
            recall = records.read(marker.retrieval_ref)
            if (recall is None or not isinstance(recall.source, ToolSource)
                or recall.source.session_id != request.session_id or recall.source.call_ref != call_ref):
                raise ValueError("召回记录不属于实际工具调用")
            for message_id in recall.presented_message_ids:
                references[message_id] = Reference(message_id, resolved_ref=message_id,
                                                   retrieval_ref=marker.retrieval_ref)
    return tuple(references.values())


class RecallTool:
    """查询记录是幂等与恢复 owner；工具 receipt 不要求再次调用 embedding。"""

    idempotent = True

    def __init__(
        self, *, memory: Path, legacy_index: Path | None, config: MemoryConfig,
        catalog: MessageCatalog, embeddings: MessageEmbeddings, bindings: Bindings,
        select_learning: Callable[[], tuple[str, str]], records: RecallRecords,
        open_embedding: Callable[[str], AbstractAsyncContextManager[BoundEmbeddingModel]], max_chars: int = 12000,
    ):
        if max_chars <= 0:
            raise ValueError("召回文本预算必须为正")
        self._memory = memory
        self._legacy_index = legacy_index
        self._config = config
        self._catalog = catalog
        self._embeddings = embeddings
        self._bindings = bindings
        self._select_learning = select_learning
        self._records = records
        self._open_embedding = open_embedding
        self._max_chars = max_chars

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        """只固定用户查询和实际 CallRef；后来输入不会改变参数出处。"""
        try:
            request = RecallArguments.model_validate(json_value(arguments))
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        origin = None if source is None else ToolSource(
            session_id=source.messages[0].session_id, call_ref=source.call_ref,
        )
        binding, model_id = self._select_learning()
        return PreparedRecall(**request.model_dump(), source=origin,
                              learning_binding=binding,
                              embedding_binding=model_id,
                              max_chars=self._max_chars).model_dump(mode="json")

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """缺少原查询时才嵌入和读图；发布成功的重试返回原材料。"""
        existing = await self.query(key)
        if existing is not None:
            return existing
        request = PreparedRecall.model_validate_json(json.dumps(json_value(arguments)))
        identity = "tool:" + key
        source = request.source or ProgramSource(key=key, query=request.query)
        # 1. 打开工具实际固定的学习规则；读副本也校验图的 embedding 空间。
        async with self._bindings.open(request.learning_binding, AKASHA_LEARNING) as (learning, metadata):
            rule = LearningConfig.model_validate(dict(metadata))
            async with read_memory(
                self._memory, legacy_index=self._legacy_index, catalog=self._catalog,
                embeddings=self._embeddings, bindings=self._bindings, config=self._config,
                embedding_space=(rule.embedding_model, rule.dimension),
            ) as (cycle, state):
                async with self._open_embedding(request.embedding_binding) as model:
                    if model.descriptor.identity != rule.embedding_model:
                        raise ValueError("实际 embedding 模型不匹配已准备的学习 binding")
                    values = (await model.embed([request.query])).vectors
                if len(values) != 1 or len(values[0]) != rule.dimension:
                    raise ValueError("召回 embedding 数量或维度不匹配")
                dense = np.asarray(values[0], dtype=np.float32)
                if not np.isfinite(dense).all():
                    raise ValueError("召回 embedding 必须是有限向量")
                norm = float(np.linalg.norm(dense))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise ValueError("召回 embedding 必须具有有限正模长")
                dense = dense / norm
                stamp = datetime.now(UTC)
                recall = await run_memory_job(lambda: query_memory(
                    cycle, state, learning_binding=request.learning_binding, text=request.query,
                    dense=dense, stamp=stamp, source=source, limit=request.limit,
                ))
            # 2. 材料和呈现出处固定后一次发布；失败不能暴露成功 Result。
            material = render_materials(identity, recall, learning, self._catalog, max_chars=request.max_chars)
            recall = recall.model_copy(update={
                "max_chars": request.max_chars,
                "presented_message_ids": tuple(dict.fromkeys(ref.ref for ref in material.references)),
            })
            _ = self._records.save(identity, recall)
            return self._result(identity, recall, material.context)

    async def query(self, key: str) -> Result | None:
        """工具外部结果恢复只读实际查询记录；不读当前图或重跑模型。"""
        identity = "tool:" + key
        recall = self._records.read(identity)
        if recall is None:
            return None
        async with self._bindings.open(recall.learning_binding, AKASHA_LEARNING) as (learning, _metadata):
            material = render_materials(identity, recall, learning, self._catalog, max_chars=recall.max_chars)
        if tuple(dict.fromkeys(ref.ref for ref in material.references)) != recall.presented_message_ids:
            raise ValueError("原查询呈现的材料发生变化，不能用当前结果冒充恢复")
        return self._result(identity, recall, material.context)

    @staticmethod
    def _result(identity: str, recall: Recall, parts: tuple[ContentPart, ...]) -> Result:
        metadata = json.dumps({
            "retrieval_ref": identity, "graph_version": recall.graph_version,
            "count": len(recall.hits), "presented_message_ids": recall.presented_message_ids,
        }, ensure_ascii=False)
        return Result("success", (
            ContentPart("text", metadata), *parts,
            ContentPart("akasha.recall", {"retrieval_ref": identity}),
        ))
