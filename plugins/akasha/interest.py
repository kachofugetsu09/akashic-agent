from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import threading
from typing import Literal

import numpy as np

from plugins.ledger.contract import EmbeddingRecords, MessageCatalog, MessageEmbeddings
from plugins.ledger.contract import Input, Output
from plugins.akasha.contract import SEMANTIC_INTEREST as SEMANTIC_INTEREST

from .application.consumer import run_memory_job
from .learning import Learning, LearningConfig
from .projection import CausalKey, Sample, SessionCursor, input_features

type Embed = Callable[[list[str]], Awaitable[list[list[float]]]]

# 兴趣只取 cutoff 前最新的这么多个原型；增量缓存多留一倍，cutoff 过滤后不足且曾截断时整段重算。
PROTOTYPES = 256
_KEEP = 2 * PROTOTYPES


@dataclass(slots=True)
class _Prototypes:
    """同一学习空间与前缀 revision 下的投影续读点、已算原型，以及向量尚未写入、下次重试的样本。"""

    rule: tuple[str, int, tuple[str, ...]]
    revision: int
    cursors: dict[str, SessionCursor]
    items: list[tuple[CausalKey, np.ndarray]]
    pending: list[Sample]
    truncated: bool = False


class SemanticInterest:
    """用已完成对话的固定向量衡量候选兴趣，不重建历史向量或写学习图。"""

    def __init__(self, learning: Learning, catalog: MessageCatalog, embeddings: MessageEmbeddings,
                 select: Callable[[], Awaitable[tuple[LearningConfig, Embed]]]):
        self._learning = learning
        self._catalog = catalog
        self._embeddings = embeddings
        self._select = select
        # 每批新 Content 都要评分；按 Session 续读只投影新增闭段，不再每次整段解码全部历史。
        self._cache: _Prototypes | None = None
        self._cache_lock = threading.Lock()

    async def score(self, texts: Sequence[str], *, cutoff: str) -> tuple[float, ...]:
        """沿用 0.9 输入加 0.1 回复的最大余弦四次方，只采用 cutoff 前的证据。"""
        through = datetime.fromisoformat(cutoff)
        if through.tzinfo is None or through.utcoffset() is None:
            raise ValueError("兴趣截止时间必须包含时区")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("兴趣候选必须是字符串")
        rule, embed = await self._select()
        prototypes = await run_memory_job(lambda: self._prototypes(rule, through))

        # 2. 仅嵌入本轮非空候选；无历史证据返回零，provider 错误保持失败。
        scores = [0.0] * len(texts)
        indexed = [(index, text) for index, text in enumerate(texts) if text.strip()]
        if not indexed or not prototypes:
            return tuple(scores)
        vectors = await embed([text for _, text in indexed])
        if len(vectors) != len(indexed):
            raise ValueError("兴趣候选 embedding 数量不一致")
        for (index, _), vector in zip(indexed, vectors, strict=True):
            candidate = np.asarray(vector, dtype=np.float32)
            if candidate.shape != (rule.dimension,) or not np.all(np.isfinite(candidate)):
                raise ValueError("兴趣候选 embedding 不属于固定向量空间")
            norm = float(np.linalg.norm(candidate))
            if norm > 0:
                similarity = max(float(np.dot(candidate / norm, prototype)) for prototype in prototypes)
                scores[index] = min(0.999, max(0.0, similarity) ** 4)
        return tuple(scores)

    def _prototypes(self, rule: LearningConfig, through: datetime) -> list[np.ndarray]:
        """Read the fixed historical inputs off the host event loop."""
        with self._cache_lock:
            # 1. 固定消息上界；前缀被改写或学习空间变化时丢弃缓存，从头投影。
            heads = self._catalog.snapshot_heads()
            revision = self._catalog.prefix_revision()
            identity = (rule.embedding_model, rule.dimension, tuple(rule.sources))
            cache = self._cache
            if cache is None or revision is None or cache.rule != identity or cache.revision != revision:
                cache = _Prototypes(identity, -1 if revision is None else revision, {}, [], [])
            # 2. 学习准入继续由 Akasha 独占，内部和未完成工作没有样本；只投影续读点之后的闭段。
            samples, cursors = self._learning.samples_since(
                self._catalog, rule, heads=heads, cursors=cache.cursors, keep=_complete,
            )
            records = self._embeddings.bind(self._learning.text)
            pending: list[Sample] = []
            fresh: list[tuple[CausalKey, np.ndarray]] = []
            for sample in (*cache.pending, *samples):
                if sample.ending.session_id not in heads:
                    continue
                # cutoff 之后的样本不读向量，留到 cutoff 推进后再算，与原算法同样不受其向量影响。
                if sample.key[0] > through:
                    pending.append(_compact(sample))
                    continue
                prototype = self._prototype(sample, rule, records)
                if not isinstance(prototype, str):
                    fresh.append((sample.key, prototype))
                elif prototype == "missing":
                    pending.append(_compact(sample))
            # 3. 删除的 Session 不再提供证据；缓存只留最新原型，截断后 cutoff 不足时整段重算。
            items = sorted((item for item in (*cache.items, *fresh) if item[0][1] in heads), key=lambda item: item[0])
            truncated = cache.truncated or len(items) > _KEEP
            cache.items, cache.pending, cache.cursors, cache.truncated = items[-_KEEP:], pending, cursors, truncated
            if revision is not None:
                self._cache = cache
            selected = [vector for key, vector in cache.items if key[0] <= through]
            if len(selected) < PROTOTYPES and truncated:
                self._cache = None
                return self._full(rule, through, heads)
            return selected[-PROTOTYPES:]

    # 缓存被截断而 cutoff 早于保留窗口时，按原算法整段重算一次，不写缓存。
    def _full(self, rule: LearningConfig, through: datetime, heads: Mapping[str, int]) -> list[np.ndarray]:
        records = self._embeddings.bind(self._learning.text)
        samples = self._learning.samples(self._catalog, rule, heads=heads)
        prototypes: list[np.ndarray] = []
        for sample in samples:
            if sample.key[0] > through or not _complete(sample):
                continue
            prototype = self._prototype(sample, rule, records)
            if not isinstance(prototype, str):
                prototypes.append(prototype)
        return prototypes[-PROTOTYPES:]

    # 0.9 输入加 0.1 回复的归一化原型；"skip" 表示样本永远不提供证据，"missing" 表示固定向量尚未写入。
    def _prototype(
        self, sample: Sample, rule: LearningConfig, records: EmbeddingRecords,
    ) -> np.ndarray | Literal["skip", "missing"]:
        ending = sample.ending
        users = [message for message in sample.messages
                 if isinstance(message.body, Input) and self._learning.text(message).strip()]
        if not users or not self._learning.text(ending).strip():
            return "skip"
        vectors = [records.read(message, model=rule.embedding_model, dimension=rule.dimension)
                   for message in (*users, ending)]
        if any(vector is None for vector in vectors):
            return "missing"
        _, user = input_features(users, text=self._learning.text, embeddings=records,
                                 embedding_model=rule.embedding_model, dimension=rule.dimension)
        if user is None:
            return "skip"
        combined = 0.9 * user + 0.1 * np.asarray(vectors[-1], dtype=np.float32)
        norm = float(np.linalg.norm(combined))
        return combined / norm if norm > 0 else "skip"


# 只有完整回复的闭段可能成为原型；其余样本不让正文常驻。
def _complete(sample: Sample) -> bool:
    ending = sample.ending
    return isinstance(ending.body, Output) and ending.body.finish == "complete"


# 等待向量或晚于 cutoff 的样本只留计算原型要读的输入与回复，不持有工具输出等成员正文。
def _compact(sample: Sample) -> Sample:
    kept = tuple(message for message in sample.messages
                 if isinstance(message.body, Input) or message == sample.ending)
    return Sample(ending=sample.ending, messages=kept, observations=())
