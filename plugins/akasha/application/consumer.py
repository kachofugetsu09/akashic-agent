"""学习图与消息消费进度只在同一份快照中前进。"""
from __future__ import annotations

from dataclasses import replace
import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from itertools import groupby
from functools import partial
from pathlib import Path

from agent.plugin_composition.bindings import Bindings
from session.log import MessageCatalog
from session.embedding_store import MessageEmbeddings

from ..domain.features import BurstAwareFeaturePool
from ..domain.model import ContextState, EmbeddingSpaceMismatchError, MemoryConfig, Turn
from ..infrastructure.consumption import Applied, Consumption, LegacyPrefix, turns_digest, load_legacy_prefix, legacy_embedding_model
from ..infrastructure.lease import WriterLease
from ..infrastructure.persistence import load_consumption, load_memory_state, write_memory_database
from .cycle import MemoryCycle


class MessageConsumer:
    """拥有一个学习快照；调用者串行提交已用固定规则验证的材料。"""

    def __init__(self, path: Path, *, turns: list[Turn], state: Consumption, config: MemoryConfig):
        state.check_turns(turns)
        self.path = path
        self.state = state
        self.config = config
        self._error: BaseException | None = None
        self._closed = False
        self._embedding_model: str | None = None
        self._lease = WriterLease(path)
        try:
            # 1. 恢复只能装载已发布图；缺图或缺切换记录不能触发历史重学。
            if path.exists():
                if load_consumption(path) != state:
                    raise ValueError("学习快照与已解析的消费出处不一致")
                graph, events, evidence, context, recalls, burst = load_memory_state(
                    path, turns=turns, config=config, source_index_sha256=None,
                )
                self._cycle = MemoryCycle.restore(
                    config=config, turns=turns, graph=graph, events=events,
                    evidence=evidence, context=context, recalls=recalls, burst_members=burst,
                )
                if turns:
                    self._cycle.feature_pool = BurstAwareFeaturePool(turns, appendable=True)
            else:
                if turns or state.legacy_prefix.count or state.applied:
                    raise ValueError("已有消费进度缺少学习图，不能自动重放")
                self._cycle = MemoryCycle(config)
                self._cycle.context = ContextState((), None, ())
                _ = write_memory_database(
                    path, turns=[], graph=self._cycle.graph, events=[], evidence=[], captures=[],
                    context=self._cycle.context, burst_members={}, config=config, metadata={},
                    consumption=state,
                )
        except BaseException:
            self._lease.close()
            raise

    @classmethod
    async def load(
        cls, path: Path, *, legacy_index: Path | None, catalog: MessageCatalog,
        embeddings: MessageEmbeddings, bindings: Bindings,
        config: MemoryConfig,
    ) -> MessageConsumer:
        """先按原绑定还原材料，再取得唯一 writer 装载图；缺失来源不自动重学。"""
        from ..learning import AKASHA_LEARNING, LearningConfig

        # 1. 第一次启用固定已有日志上界，重启前也必须把空图与起点一起发布。
        if not path.exists():
            if legacy_index is not None and legacy_index.exists():
                raise ValueError("旧索引仍存在但学习图缺失，需要显式恢复")
            state = Consumption(
                legacy_prefix=LegacyPrefix(count=0, index_state_sha256="0" * 64,
                                           turns_digest=turns_digest([])),
                cutover_heads=tuple(sorted(catalog.snapshot_heads().items())),
            )
            return cls(path, turns=[], state=state, config=config)
        state = load_consumption(path)
        if state is None:
            raise ValueError("旧学习图尚未完成 yoyo 消费切换")
        turns = load_legacy_prefix(state, legacy_index)
        space = None
        if state.legacy_prefix.count:
            if legacy_index is None:
                raise RuntimeError("已恢复旧前缀缺少其索引路径")
            space = legacy_embedding_model(legacy_index)

        # 2. 后缀逐项打开原算法闭包；不开模型、不嵌入，也不调用 commit。
        for identity, entries in groupby(state.applied, key=lambda entry: entry.learning_binding):
            async with bindings.open(identity, AKASHA_LEARNING) as (learning, metadata):
                rule = LearningConfig.model_validate(dict(metadata))
                try:
                    _check_embedding_space(rule.embedding_model, rule.dimension, space, turns)
                except EmbeddingSpaceMismatchError as error:
                    raise ValueError("已发布 Akasha 学习图的 embedding 空间不一致") from error
                space = rule.embedding_model
                for entry in entries:
                    turns.append(learning.restore(
                        catalog, embeddings, rule, entry, previous=turns, state=state, bindings=bindings,
                    ))
        consumer = cls(path, turns=turns, state=state, config=config)
        consumer._embedding_model = space
        return consumer

    @property
    def cycle(self) -> MemoryCycle:
        if self._closed:
            raise RuntimeError("学习消费者已关闭")
        if self._error is not None:
            raise RuntimeError("学习发布失败；必须重新读取耐久快照后恢复") from self._error
        return self._cycle

    def close(self) -> None:
        self._closed = True
        self._lease.close()

    def check_embedding_space(self, model: str, dimension: int, bindings: Bindings) -> None:
        """查询与学习都在模型调用前核对同一张图的固定空间。"""
        from ..learning import AKASHA_LEARNING, LearningConfig

        space = self._embedding_model
        if space is None and self.state.applied:
            prior = LearningConfig.model_validate(dict(bindings.describe(
                self.state.applied[-1].learning_binding, AKASHA_LEARNING,
            )))
            space = prior.embedding_model
        if space is None and self.state.legacy_prefix.count:
            raise RuntimeError("旧学习图必须通过 load 恢复其 embedding 身份")
        _check_embedding_space(model, dimension, space, self.cycle.turns)

    async def consume(
        self, *, catalog: MessageCatalog, learning_binding: str,
        embeddings: MessageEmbeddings, bindings: Bindings,
        embed_batch: Callable[[list[str]], Awaitable[list[list[float]]]],
    ) -> int:
        """追赶一个固定日志前缀；在线只补缺向量，学习与进度仍一次发布。"""
        from session.message import Input, Output
        from ..projection import applied_source

        from ..learning import AKASHA_LEARNING, LearningConfig

        # 1. 实际执行原绑定的纯规则，不能另传一套不相符的算法和配置。
        async with bindings.open(learning_binding, AKASHA_LEARNING) as (learning, metadata):
            rule = LearningConfig.model_validate(dict(metadata))
            # 图空间从原出处恢复；切换模型不能把新向量接到旧图中。
            self.check_embedding_space(rule.embedding_model, rule.dimension, bindings)
            heads = catalog.snapshot_heads()
            cutover = dict(self.state.cutover_heads)
            applied = {entry.ending[1] for entry in self.state.applied}
            records = embeddings.bind(learning.text)
            count = 0
            for sample in learning.samples(catalog, rule, heads=heads):
                if sample.ending.seq <= cutover.get(sample.ending.session_id, -1) or sample.ending.message_id in applied:
                    continue
                inputs = [message for message in sample.messages if isinstance(message.body, Input)]
                if not any(learning.text(message).strip() for message in inputs) or not learning.text(sample.ending).strip():
                    continue
                if not isinstance(sample.ending.body, Output) or sample.ending.body.finish != "complete":
                    continue
                # 2. 固定消息向量先于学习发布，缺失之外的空间差异直接失败。
                missing = [message for message in (*inputs, sample.ending)
                           if learning.text(message).strip()
                           and records.read(message, model=rule.embedding_model, dimension=rule.dimension) is None]
                if missing:
                    vectors = await embed_batch([learning.text(message) for message in missing])
                    if len(vectors) != len(missing) or any(len(vector) != rule.dimension for vector in vectors):
                        raise ValueError("embedding 返回数量或维度不匹配固定学习空间")
                    for message, vector in zip(missing, vectors):
                        records.save(message, model=rule.embedding_model, embedding=vector)
                turn = learning.make_turn(sample, rule, embeddings, previous=self.cycle.turns,
                                          state=self.state, bindings=bindings)
                if turn is None:
                    raise RuntimeError("已接纳的学习样本没有产生节点")
                entry = applied_source(sample, learning_binding=learning_binding)
                count += await run_memory_job(partial(self.apply, turn, entry))
                self._embedding_model = rule.embedding_model
            return count

    def apply(self, turn: Turn, entry: Applied) -> bool:
        """重复通知不强化；提交失败后停止本实例，避免猜测外部发布是否成功。"""
        # 1. 同一 ending 的重复消费必须有完全相同的固定出处。
        cycle = self.cycle
        for applied in self.state.applied:
            if applied.ending[1] == entry.ending[1]:
                if applied != entry:
                    raise ValueError("重复学习通知的出处不一致")
                return False
        state = self.state.append(entry)
        now = datetime.fromisoformat(turn.committed_at)
        if now.utcoffset() is None:
            raise ValueError("学习时间必须包含时区")
        gap = None
        if cycle.turns:
            previous = cycle.turns[-1]
            last = datetime.fromisoformat(previous.committed_at)
            gap = (now - last).total_seconds()
            if gap < 0:
                raise ValueError("新学习材料早于已发布图，不能自动重排")
            if self.state.applied:
                last_entry = self.state.applied[-1]
                if (now, entry.session_id, entry.ending) <= (last, last_entry.session_id, last_entry.ending):
                    raise ValueError("新学习材料违反已发布的全局顺序")
        turn = replace(turn, node_id=cycle.state_version, inter_gap_seconds=gap)
        state.check_turns([*cycle.turns, turn])

        # 2. MemoryCycle 唯一学习一次；图与消费出处在一个完整文件中一起发布。
        try:
            _ = cycle.commit(turn, None)
            if cycle.context is None:
                raise RuntimeError("已学习图缺少 context")
            _ = write_memory_database(
                self.path, turns=cycle.turns, graph=cycle.graph, events=cycle.events,
                evidence=cycle.evidence, captures=[], context=cycle.context,
                burst_members=cycle.burst_members, config=self.config, metadata={},
                recalls=cycle.recalls, consumption=state,
            )
        except BaseException as error:
            # 发布可能已经 replace；回退 Python 指针不能证明文件没有提交。
            self._error = error
            raise
        self.state = state
        if cycle.feature_pool is None:
            cycle.feature_pool = BurstAwareFeaturePool(cycle.turns, appendable=True)
        return True


def _check_embedding_space(model: str, dimension: int, previous_model: str | None, turns: list[Turn]) -> None:
    """图不能混合向量空间；身份来自原学习出处，维度来自实际恢复的材料。"""
    if previous_model is not None and previous_model != model:
        raise EmbeddingSpaceMismatchError("学习图不能混用不同 embedding 空间，需显式重建")
    if any(len(vector) != dimension for turn in turns
           for vector in (turn.user_dense, turn.assistant_dense) if vector is not None):
        raise EmbeddingSpaceMismatchError("新学习 binding 的维度不匹配已有图，需显式重建")


async def run_memory_job[T](work: Callable[[], T]) -> T:
    """CPU 与文件发布移出事件循环；取消后先排空线程，再允许关闭图 writer。"""
    job = asyncio.create_task(asyncio.to_thread(work))
    cancelled = False
    while not job.done():
        try:
            _ = await asyncio.shield(job)
        except asyncio.CancelledError:
            cancelled = True
    result = job.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
