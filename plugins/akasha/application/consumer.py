"""学习图与消息消费进度只在同一份快照中前进。"""
from __future__ import annotations

from dataclasses import replace
import asyncio
import hashlib
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from itertools import groupby
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from plugins.ledger.contract import Bindings
from plugins.ledger.contract import EmbeddingRecords, MessageCatalog, MessageEmbeddings, Message, Input

from ..domain.features import BurstAwareFeaturePool
from ..domain.model import (
    ContextState,
    EmbeddingSpaceMismatchError,
    MemoryConfig,
    MemoryRebuildRequiredError,
    Turn,
)
from ..infrastructure.consumption import Applied, Consumption, Skipped
from ..infrastructure.lease import WriterLease
from ..infrastructure.lexical import TermCache, Terms, terms as lexical_terms
from ..infrastructure.persistence import load_consumption, load_memory_state, write_memory_database
from .cycle import MemoryCycle

if TYPE_CHECKING:
    from ..projection import Sample, SessionCursor
    from ..learning import Learning, LearningConfig


type RestoreEmbeddings = Callable[["LearningConfig", list[str]], Awaitable[list[list[float]]]]

# 在线连续学习最多攒这么多条再发布一次；未发布的尾部崩溃后由日志按原序重学。
PUBLISH_EVERY = 16
# 启动还原每块最多解码这么多条已学样本；整图一次解码会让大工具输出同时驻留。
LOAD_CHUNK = 256
# 落盘续读点最多这么久写一次；落后的续读点只多投影一段，已学和已跳过的闭段仍被过滤。
CURSOR_SAVE_SECONDS = 60.0
# 决定"哪些消息组成样本、哪些样本被准入"的代码所在的包；任一源码变了，落盘续读点作废。
_PROJECTION_PACKAGES = ("akasha", "turn_projection", "ledger")


# 投影与准入实现的源码指纹：规则相同但实现变了时，旧续读点可能越过新实现会接纳的闭段。
@cache
def _projection_code() -> str:
    plugins = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for package in _PROJECTION_PACKAGES:
        for path in sorted((plugins / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(plugins).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class _SavedCursors(BaseModel):
    """随已发布快照保存的投影续读点；只在同一学习规则、同一前缀 revision、同一消费进度前缀下复用。

    learning binding 的身份含插件 generation，每次部署都会变；投影只取决于规则与投影实现，
    所以落盘续读点按规则加投影代码指纹匹配，内存续读点仍按 binding。
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[2] = 2
    learning_rule: str
    prefix_revision: int
    applied: tuple[int, str | None]
    skipped: tuple[int, str | None]
    cursors: dict[str, tuple[int, dict[str, int]]]


class MessageConsumer:
    """拥有一个学习快照；调用者串行提交已用固定规则验证的材料。"""

    def __init__(
        self, path: Path, *, turns: list[Turn], state: Consumption, config: MemoryConfig,
        deferred_publish: bool = False,
    ):
        state.check_turns(turns)
        self.path = path
        self.state = state
        # 磁盘上已发布快照的消费进度；连续发布以它为增量起点。
        self._published = state
        self.config = config
        # 完整重建写的是丢弃用候选文件：每次学习都重写整库没有恢复价值，
        # 因此允许把发布推迟到重放结束的一次原子写入。
        self._deferred_publish = deferred_publish
        self._error: BaseException | None = None
        self._closed = False
        self._embedding_model: str | None = None
        # 投影续读点只在一批成功消费后前进；换学习绑定或前缀被改写时从头投影。
        # 新实例从落盘续读点恢复，重启不再整段解码全部历史。
        self._cursors: tuple[str, int | None, dict[str, SessionCursor]] = ("", None, {})
        self._cursor_path = path.with_name(path.name + ".cursors.json")
        self._cursor_file_read = False
        self._cursor_rule = ""
        self._cursors_saved_at: float | None = None
        self._lease = WriterLease(path)
        try:
            # 1. 恢复只能装载已发布图；缺图不能触发历史重学。
            if path.exists():
                if load_consumption(path) != state:
                    raise ValueError("学习快照与已解析的消费出处不一致")
                graph, events, evidence, context, recalls, burst = load_memory_state(
                    path, turns=turns, config=config,
                )
                self._cycle = MemoryCycle.restore(
                    config=config, turns=turns, graph=graph, events=events,
                    evidence=evidence, context=context, recalls=recalls, burst_members=burst,
                )
                if turns:
                    self._cycle.feature_pool = BurstAwareFeaturePool(self._cycle.turns, appendable=True)
            else:
                if turns or state.applied:
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
    async def create(
        cls, path: Path, *, turns: list[Turn], state: Consumption, config: MemoryConfig,
        deferred_publish: bool = False,
    ) -> MessageConsumer:
        """在后台取得 writer；取消期间构造成功的资源仍必须关闭。"""
        consumer: MessageConsumer | None = None

        def build() -> None:
            nonlocal consumer
            consumer = cls(path, turns=turns, state=state, config=config,
                           deferred_publish=deferred_publish)

        try:
            await run_memory_job(build)
        except BaseException:
            # run_memory_job 已排空线程，构造结果不会在关闭后才出现。
            if consumer is not None:
                consumer.close()
            raise
        assert consumer is not None
        return consumer

    @classmethod
    async def load(
        cls, path: Path, *, catalog: MessageCatalog,
        embeddings: MessageEmbeddings, bindings: Bindings,
        config: MemoryConfig, cutover: bool = True,
        restore_embeddings: RestoreEmbeddings | None = None,
        term_cache: Path | None = None,
    ) -> MessageConsumer:
        """先按原绑定还原材料，再取得唯一 writer 装载图；缺失来源不自动重学。"""
        from ..learning import AKASHA_LEARNING, LearningConfig

        # 1. 第一次启用固定已有日志上界，重启前也必须把空图与起点一起发布。
        #    独立图的成员从接纳起就固定路由到它，不设上界而从头学习。
        if not path.exists():
            heads = catalog.snapshot_heads().items() if cutover else ()
            state = Consumption(cutover_heads=tuple(sorted(heads)))
            return await cls.create(path, turns=[], state=state, config=config)
        state = await run_memory_job(partial(load_consumption, path))
        if state is None:
            raise MemoryRebuildRequiredError("学习图缺少当前消费出处，需要显式重建")
        # 2. 按原算法验证出处，缺失向量由明确的模型端口补算；不调用学习 commit。
        #    分词是纯函数，按正文哈希复用上次结果，只有新正文现场切分。
        cache = None if term_cache is None else TermCache(term_cache)
        if cache is not None:
            await run_memory_job(cache.load)
        terms = lexical_terms if cache is None else cache.terms
        turns: list[Turn] = []
        space: str | None = None
        for identity, grouped in groupby(state.applied, key=lambda entry: entry.learning_binding):
            entries = tuple(grouped)
            async with bindings.open(identity, AKASHA_LEARNING) as (learning, metadata):
                rule = LearningConfig.model_validate(dict(metadata))
                await run_memory_job(lambda: _check_embedding_space(
                    rule.embedding_model, rule.dimension, space, turns,
                ))
                # 分块还原：同一时刻只解码一块样本的原消息，Turn 只留文本与向量。
                for offset in range(0, len(entries), LOAD_CHUNK):
                    await _restore_chunk(
                        entries[offset:offset + LOAD_CHUNK], turns, catalog=catalog,
                        embeddings=embeddings, bindings=bindings, learning=learning, rule=rule,
                        state=state, terms=terms, restore_embeddings=restore_embeddings,
                    )
                space = rule.embedding_model
        if cache is not None:
            cache.release()
            await run_memory_job(cache.save)
        consumer = await cls.create(path, turns=turns, state=state, config=config)
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

    def flush(self) -> None:
        """把内存中已提交、尚未发布的学习尾部写成一份完整快照。"""
        if self._closed or self._deferred_publish or self.state == self._published:
            return
        try:
            _ = self.publish_snapshot_for(self.state)
        except BaseException as error:
            self._error = error
            raise

    def check_embedding_space(self, model: str, dimension: int, bindings: Bindings) -> None:
        """查询与学习都在模型调用前核对同一张图的固定空间。"""
        from ..learning import AKASHA_LEARNING, LearningConfig

        space = self._embedding_model
        if space is None and self.state.applied:
            prior = LearningConfig.model_validate(dict(bindings.describe(
                self.state.applied[-1].learning_binding, AKASHA_LEARNING,
            )))
            space = prior.embedding_model
        if space is None and self.cycle.turns:
            raise EmbeddingSpaceMismatchError("已学习图必须通过 load 恢复其 embedding 身份")
        _check_embedding_space(model, dimension, space, self.cycle.turns)

    async def consume(
        self, *, catalog: MessageCatalog, learning_binding: str,
        embeddings: MessageEmbeddings, bindings: Bindings,
        embed_batch: Callable[[list[str]], Awaitable[list[list[float]]]],
        skip_missing_embeddings: bool = False,
        member: Callable[[str], bool] | None = None,
    ) -> int:
        """追赶一个固定日志前缀；在线只补缺向量，学习与进度仍一次发布。

        member 选出路由到本图的 Session；每张图只消费自己的成员。
        """
        from plugins.ledger.contract import Input, Output
        from ..projection import Sample, applied_source

        from ..learning import AKASHA_LEARNING, LearningConfig

        # 1. 实际执行原绑定的纯规则，不能另传一套不相符的算法和配置。
        async with bindings.open(learning_binding, AKASHA_LEARNING) as (learning, metadata):
            rule = LearningConfig.model_validate(dict(metadata))
            # 图空间从原出处恢复；切换模型不能把新向量接到旧图中。
            await run_memory_job(partial(
                self.check_embedding_space, rule.embedding_model, rule.dimension, bindings,
            ))
            heads = {
                session: head for session, head in catalog.snapshot_heads().items()
                if member is None or member(session)
            }
            cutover = dict(self.state.cutover_heads)
            applied = {entry.ending[1] for entry in self.state.applied}
            skipped = {item.ending[1] for item in self.state.skipped}

            def fresh(sample: Sample) -> bool:
                """切换上界之前、已学习或已明确跳过的闭段不再处理。"""
                ending = sample.ending
                return (ending.seq > cutover.get(ending.session_id, -1)
                        and ending.message_id not in applied and ending.message_id not in skipped)

            records = embeddings.bind(learning.text)
            count = 0
            # 投影只读续读点之后的尾部；与图发布共用排空后才取消的后台执行方式。
            # revision 先于投影读取并随续读点保存：投影期间的前缀改写会让下次检查失配。
            revision = await run_memory_job(catalog.prefix_revision)
            rule_key = f"{rule.model_dump_json()}|{await run_memory_job(_projection_code)}"
            cursors = await run_memory_job(partial(self._resume_cursors, learning_binding, rule_key, revision))
            samples, advanced = await run_memory_job(partial(
                learning.samples_since, catalog, rule, heads=heads, cursors=cursors, keep=fresh,
            ))
            for sample in samples:
                inputs = [message for message in sample.messages if isinstance(message.body, Input)]
                if not any(learning.text(message).strip() for message in inputs) or not learning.text(sample.ending).strip():
                    continue
                if not isinstance(sample.ending.body, Output) or sample.ending.body.finish != "complete":
                    continue
                # 2. 固定消息向量先于学习发布，缺失之外的空间差异直接失败。
                missing = await run_memory_job(lambda: [
                    message for message in (*inputs, sample.ending)
                    if learning.text(message).strip()
                    and records.read(message, model=rule.embedding_model, dimension=rule.dimension) is None
                ])
                if missing:
                    if skip_missing_embeddings:
                        # 缺少固定向量的闭段明确跳过；跳过必须持久，避免在线路径稍后乱序补学。
                        count += await run_memory_job(partial(
                            self.skip, sample, reason="missing-embedding",
                        ))
                        continue
                    vectors = await embed_batch([learning.text(message) for message in missing])
                    if len(vectors) != len(missing) or any(len(vector) != rule.dimension for vector in vectors):
                        raise ValueError("embedding 返回数量或维度不匹配固定学习空间")
                    def save_vectors() -> None:
                        for message, vector in zip(missing, vectors):
                            records.save(message, model=rule.embedding_model, embedding=vector)
                    await run_memory_job(save_vectors)

                def learn_sample() -> bool:
                    """材料依赖当前图；发布完成后调用方才处理下一条经历。"""
                    turn = learning.make_turn(sample, rule, embeddings, previous=self.cycle.turns,
                                              state=self.state, bindings=bindings)
                    if turn is None:
                        raise RuntimeError("已接纳的学习样本没有产生节点")
                    entry = applied_source(sample, learning_binding=learning_binding)
                    applied = self.apply(turn, entry, publish=False)
                    self._embedding_model = rule.embedding_model
                    if len(self.state.applied) - len(self._published.applied) >= PUBLISH_EVERY:
                        self.flush()
                    return applied

                count += await run_memory_job(learn_sample)
            # 3. 本批学完统一发布一次；批内攒满 PUBLISH_EVERY 已在途中发布。
            await run_memory_job(self.flush)
            # 4. 整批处理并发布后才前进续读点；失败或取消时下次仍从旧续读点重读。
            self._cursors = (learning_binding, revision, advanced)
            self._cursor_rule = rule_key
            await run_memory_job(self._save_cursors)
            return count

    # 内存续读点按 binding 优先；新实例读一次落盘续读点，按规则匹配。规则、revision 或消费进度对不上都从头投影。
    def _resume_cursors(self, binding: str, rule_key: str, revision: int | None) -> Mapping[str, SessionCursor]:
        from ..projection import SessionCursor

        known, known_revision, cursors = self._cursors
        if revision is None:
            return {}
        if known:
            return cursors if known == binding and known_revision == revision else {}
        if self._cursor_file_read:
            return {}
        self._cursor_file_read = True
        try:
            saved = _SavedCursors.model_validate_json(self._cursor_path.read_bytes())
        except (OSError, ValueError):
            # 缺失、损坏或旧格式都只是没有续读点；ValidationError 是 ValueError 子类。
            return {}
        if (saved.learning_rule != rule_key or saved.prefix_revision != revision
                or not _progress_prefix(saved, self._published)):
            return {}
        return {session: SessionCursor(head, boundaries)
                for session, (head, boundaries) in saved.cursors.items()}

    # 只保存已发布进度对应的续读点；重建候选不落盘，节流写入后原子替换。
    def _save_cursors(self) -> None:
        binding, revision, cursors = self._cursors
        if not binding or not self._cursor_rule or revision is None or self._closed or self._deferred_publish:
            return
        # 续读点越过的闭段必须都已发布；否则崩溃后已发布进度回退，越过的样本永不重学。
        if self.state != self._published:
            return
        now = time.monotonic()
        if self._cursors_saved_at is not None and now - self._cursors_saved_at < CURSOR_SAVE_SECONDS:
            return
        published = self._published
        saved = _SavedCursors(
            learning_rule=self._cursor_rule, prefix_revision=revision,
            applied=(len(published.applied), published.applied[-1].ending[1] if published.applied else None),
            skipped=(len(published.skipped), published.skipped[-1].ending[1] if published.skipped else None),
            cursors={session: (cursor.head, dict(cursor.boundaries)) for session, cursor in cursors.items()},
        )
        temporary = self._cursor_path.with_name(self._cursor_path.name + ".tmp")
        _ = temporary.write_text(saved.model_dump_json(), encoding="utf-8")
        os.replace(temporary, self._cursor_path)
        self._cursors_saved_at = now

    def publish_snapshot(self) -> str:
        """按当前内存状态发布一次完整快照；延迟发布的重建用它收口。"""

        return self.publish_snapshot_for(self.state)

    def publish_snapshot_for(self, state: Consumption) -> str:
        """用给定消费状态发布当前图；调用者负责保证它与图对应。"""

        cycle = self.cycle
        if cycle.context is None:
            raise RuntimeError("已学习图缺少 context")
        digest = write_memory_database(
            self.path, turns=cycle.turns, graph=cycle.graph, events=cycle.events,
            evidence=cycle.evidence, captures=[], context=cycle.context,
            burst_members=cycle.burst_members, config=self.config, metadata={},
            recalls=cycle.recalls, consumption=state,
            previous_consumption=None if self._deferred_publish else self._published,
        )
        self._published = state
        return digest

    def skip(self, sample: Sample, *, reason: str) -> bool:
        """把一个闭段记为明确跳过；重复记事必须完全一致。"""

        entry = Skipped(
            session_id=sample.ending.session_id,
            ending=(sample.ending.seq, sample.ending.message_id),
            reason=reason,
        )
        for existing in self.state.skipped:
            if existing.ending[1] == entry.ending[1]:
                if existing != entry:
                    raise ValueError("重复跳过记事的出处不一致")
                return False
        state = self.state.mark_skipped(entry)
        if self._deferred_publish:
            self.state = state
            return True
        try:
            _ = self.publish_snapshot_for(state)
        except BaseException as error:
            self._error = error
            raise
        self.state = state
        return True


    def apply(self, turn: Turn, entry: Applied, *, publish: bool = True) -> bool:
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
            if publish and not self._deferred_publish:
                _ = self.publish_snapshot_for(state)
        except BaseException as error:
            # 发布可能已经 replace；回退 Python 指针不能证明文件没有提交。
            self._error = error
            raise
        self.state = state
        if cycle.feature_pool is None:
            cycle.feature_pool = BurstAwareFeaturePool(cycle.turns, appendable=True)
        return True


# 按原前缀还原一块已学样本：核对出处、补算缺失向量，再逐条接到 turns 之后。
async def _restore_chunk(
    entries: Sequence[Applied], turns: list[Turn], *, catalog: MessageCatalog,
    embeddings: MessageEmbeddings, bindings: Bindings, learning: Learning, rule: LearningConfig,
    state: Consumption, terms: Callable[[str], Terms], restore_embeddings: RestoreEmbeddings | None,
) -> None:
    # 1. 按原算法验证出处；分块不改变还原顺序，后一条只能读取已完成的 previous。
    samples = await run_memory_job(lambda: tuple(
        learning.load_sample(catalog, rule, entry) for entry in entries
    ))
    records = embeddings.bind(learning.text)
    verified: dict[str, tuple[float, ...]] = {}
    def missing_vectors() -> list[Message]:
        messages = {message.message_id: message for sample in samples
                    for message in sample.messages
                    if isinstance(message.body, Input) or message == sample.ending}
        missing: list[Message] = []
        for message in messages.values():
            if not learning.text(message).strip():
                continue
            vector = records.read(message, model=rule.embedding_model, dimension=rule.dimension)
            if vector is None:
                missing.append(message)
            else:
                verified[message.message_id] = vector
        return missing
    # 2. 缺失向量由明确的模型端口补算；不调用学习 commit。
    missing = await run_memory_job(missing_vectors)
    if missing:
        if restore_embeddings is None:
            raise MemoryRebuildRequiredError("派生向量缺失；当前只读调用没有模型补算权限")
        vectors = await restore_embeddings(rule, [learning.text(message) for message in missing])
        if len(vectors) != len(missing) or any(len(vector) != rule.dimension for vector in vectors):
            raise ValueError("补算向量数量或维度不匹配原学习空间")
        def save_vectors() -> None:
            for message, vector in zip(missing, vectors, strict=True):
                records.save(message, model=rule.embedding_model, embedding=vector)
        await run_memory_job(save_vectors)
    # 3. 用已验证的样本还原材料。
    def restore_entries() -> None:
        for sample in samples:
            turns.append(learning.restore(
                sample, _VerifiedVectors(embeddings, verified), rule,
                previous=turns, state=state, bindings=bindings, terms=terms,
            ))
    await run_memory_job(restore_entries)


# 保存时的已发布进度必须仍是当前进度的前缀：消费只追加，换了图或重建后的不同历史不复用。
def _progress_prefix(saved: _SavedCursors, state: Consumption) -> bool:
    applied_count, applied_last = saved.applied
    skipped_count, skipped_last = saved.skipped
    if applied_count > len(state.applied) or skipped_count > len(state.skipped):
        return False
    if (applied_last is None) != (applied_count == 0) or (skipped_last is None) != (skipped_count == 0):
        return False
    return ((applied_count == 0 or state.applied[applied_count - 1].ending[1] == applied_last)
            and (skipped_count == 0 or state.skipped[skipped_count - 1].ending[1] == skipped_last))


def _check_embedding_space(model: str, dimension: int, previous_model: str | None, turns: list[Turn]) -> None:
    """图不能混合向量空间；身份来自原学习出处，维度来自实际恢复的材料。"""
    if previous_model is not None and previous_model != model:
        raise EmbeddingSpaceMismatchError("学习图不能混用不同 embedding 空间，需显式重建")
    if any(len(vector) != dimension for turn in turns
           for vector in (turn.user_dense, turn.assistant_dense) if vector is not None):
        raise EmbeddingSpaceMismatchError("新学习 binding 的维度不匹配已有图，需显式重建")


# 恢复复用刚核对过正文出处的向量，不再逐条重读；未命中（如刚补算的）仍走原存储与校验。
class _VerifiedVectors:
    def __init__(self, embeddings: MessageEmbeddings, vectors: Mapping[str, tuple[float, ...]]):
        self._embeddings = embeddings
        self._vectors = vectors

    def bind(self, text: Callable[[Message], str]) -> EmbeddingRecords:
        return _VerifiedRecords(self._embeddings.bind(text), self._vectors)


class _VerifiedRecords:
    def __init__(self, records: EmbeddingRecords, vectors: Mapping[str, tuple[float, ...]]):
        self._records = records
        self._vectors = vectors

    def read(self, message: Message, *, model: str, dimension: int) -> tuple[float, ...] | None:
        vector = self._vectors.get(message.message_id)
        if vector is not None and len(vector) == dimension:
            return vector
        return self._records.read(message, model=model, dimension=dimension)

    def save(self, message: Message, *, model: str, embedding: Sequence[float]) -> None:
        self._records.save(message, model=model, embedding=embedding)


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
