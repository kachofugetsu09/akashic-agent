"""消息学习与实际召回共用一个运行 owner，不保存第二份对话状态。"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from dataclasses import replace
from uuid import uuid4

from agent.plugin_composition.bindings import Bindings
from plugins.context.api import Materials
from session.embedding_store import MessageEmbeddings
from session.log import MessageCatalog
from session.message import Input, Message

from .application.consumer import MessageConsumer, run_memory_job
from .learning import AKASHA_LEARNING, Learning, LearningConfig
from .application.cycle import MemoryCycle
from .infrastructure.consumption import Consumption
from .projection import input_features
from .recalls import ContextSource, RecallRecords, query_memory, render_materials
from .recall_tool import tool_references


class MessageMemory:
    """串行拥有学习、查询和关闭；线程未排空时不能释放图 writer。"""

    def __init__(
        self, consumer: MessageConsumer, *, catalog: MessageCatalog,
        embeddings: MessageEmbeddings, bindings: Bindings, learning_binding: str,
        records: RecallRecords, embed_batch: Callable[[list[str]], Awaitable[list[list[float]]]],
        limit: int = 40, max_chars: int = 12000,
    ):
        if not 1 <= limit <= 40 or max_chars <= 0:
            raise ValueError("召回数量或文本预算无效")
        self._consumer = consumer
        self._catalog = catalog
        self._embeddings = embeddings
        self._bindings = bindings
        self._learning_binding = learning_binding
        self._records = records
        self._embed_batch = embed_batch
        self._limit = limit
        self._max_chars = max_chars
        self._lock = asyncio.Lock()
        self._closing = False

    def _check_open(self) -> None:
        if self._closing:
            raise RuntimeError("Akasha 消息运行已关闭")
        _ = self._consumer.cycle

    async def close(self) -> None:
        """关闭一旦开始就排空并释放 writer，随后再向调用者传播取消。"""
        self._closing = True
        async def drain() -> None:
            async with self._lock:
                self._consumer.close()
        job = asyncio.create_task(drain())
        cancelled = False
        while not job.done():
            try:
                await asyncio.shield(job)
            except asyncio.CancelledError:
                cancelled = True
        job.result()
        if cancelled:
            raise asyncio.CancelledError

    async def consume(self) -> int:
        """按固定 binding 追赶已接纳日志，不与查询或关闭并发修改图。"""
        async with self._lock:
            self._check_open()
            return await self._consumer.consume(
                catalog=self._catalog, learning_binding=self._learning_binding,
                embeddings=self._embeddings, bindings=self._bindings, embed_batch=self._embed_batch,
            )

    async def prepare(self, snapshot: tuple[Message, ...], source: str) -> Materials:
        """Context 查询与学习、关闭共用串行锁；查询本身不推进学习图。"""
        async with self._lock:
            self._check_open()
            async with self._bindings.open(self._learning_binding, AKASHA_LEARNING) as (learning, metadata):
                rule = LearningConfig.model_validate(dict(metadata))
                self._consumer.check_embedding_space(rule.embedding_model, rule.dimension, self._bindings)
                return await prepare_materials(
                    snapshot, source, cycle=self._consumer.cycle, state=self._consumer.state,
                    catalog=self._catalog, embeddings=self._embeddings, bindings=self._bindings,
                    learning_binding=self._learning_binding, learning=learning, rule=rule,
                    records=self._records, embed_batch=self._embed_batch,
                    limit=self._limit, max_chars=self._max_chars,
                )


async def prepare_materials(
    snapshot: tuple[Message, ...], source: str, *, cycle: MemoryCycle, state: Consumption,
    catalog: MessageCatalog, embeddings: MessageEmbeddings, bindings: Bindings,
    learning_binding: str, learning: Learning, rule: LearningConfig, records: RecallRecords,
    embed_batch: Callable[[list[str]], Awaitable[list[list[float]]]], limit: int, max_chars: int,
) -> Materials:
    """在已核对空间的图上查询真实输入前缀，发布出处后交付材料。"""
    if not snapshot:
        return Materials("")
    # 1. 使用调用者已经固定的真实前缀，后来输入不会进入本次 cue。
    session_id = snapshot[0].session_id
    if catalog.reader(session_id).snapshot(through_seq=snapshot[-1].seq) != snapshot:
        raise ValueError("召回材料需要完整且真实的 Message 前缀")
    projected = learning.projection.project(snapshot, source)
    if not projected or projected[-1].status != "open":
        return Materials("")
    members = set(projected[-1].message_ids)
    inputs = tuple(message for message in snapshot
                   if message.message_id in members and isinstance(message.body, Input))
    if not any(learning.text(message).strip() for message in inputs):
        return Materials("")
    # 2. 空间身份必须在嵌入和写向量之前核对；历史向量只读取。
    vectors = embeddings.bind(learning.text)
    missing = [message for message in inputs if learning.text(message).strip()
               and vectors.read(message, model=rule.embedding_model, dimension=rule.dimension) is None]
    if missing:
        values = await embed_batch([learning.text(message) for message in missing])
        if len(values) != len(missing) or any(len(value) != rule.dimension for value in values):
            raise ValueError("召回 embedding 数量或维度不匹配")
        for message, value in zip(missing, values):
            vectors.save(message, model=rule.embedding_model, embedding=value)
    text, dense = input_features(inputs, text=learning.text, embeddings=vectors,
                                 embedding_model=rule.embedding_model, dimension=rule.dimension)
    # 3. 图读取移出事件循环；取消仍先排空，再释放 binding 与串行锁。
    stamp = datetime.now(UTC)
    origin = ContextSource(session_id=session_id, source=source, through_seq=snapshot[-1].seq)
    recall = await run_memory_job(lambda: query_memory(
        cycle, state, learning_binding=learning_binding,
        text=text, dense=dense, stamp=stamp, source=origin, limit=limit,
    ))
    identity = uuid4().hex
    material = render_materials(identity, recall, learning, catalog, max_chars=max_chars)
    recall = recall.model_copy(update={
        "max_chars": max_chars,
        "presented_message_ids": tuple(dict.fromkeys(ref.ref for ref in material.references)),
    })
    _ = records.save(identity, recall)
    references = {ref.ref: ref for ref in material.references}
    # 同一消息有多次真实查询时，当前工具结果的精确出处供后续 Citation 使用。
    references.update((ref.ref, ref) for ref in tool_references(
        snapshot, source, learning, bindings, records,
    ))
    return replace(material, references=tuple(references.values()))
