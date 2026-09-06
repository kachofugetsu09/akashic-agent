from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime

import numpy as np

from agent.plugin_composition import ServiceKey
from session.embedding_store import MessageEmbeddings
from session.log import MessageCatalog
from session.message import Input, Output

from .learning import Learning, LearningConfig
from .projection import input_features

type Embed = Callable[[list[str]], Awaitable[list[list[float]]]]


class SemanticInterest:
    """用已完成对话的固定向量衡量候选兴趣，不重建历史向量或写学习图。"""

    def __init__(self, learning: Learning, catalog: MessageCatalog, embeddings: MessageEmbeddings,
                 select: Callable[[], tuple[LearningConfig, Embed]]):
        self._learning = learning
        self._catalog = catalog
        self._embeddings = embeddings
        self._select = select

    async def score(self, texts: Sequence[str], *, cutoff: str) -> tuple[float, ...]:
        """沿用 0.9 输入加 0.1 回复的最大余弦四次方，只采用 cutoff 前的证据。"""
        through = datetime.fromisoformat(cutoff)
        if through.tzinfo is None or through.utcoffset() is None:
            raise ValueError("兴趣截止时间必须包含时区")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("兴趣候选必须是字符串")
        rule, embed = self._select()
        records = self._embeddings.bind(self._learning.text)
        prototypes: list[np.ndarray] = []
        # 1. 固定消息上界；学习准入继续由 Akasha 独占，内部和未完成工作没有样本。
        samples = self._learning.samples(self._catalog, rule, heads=self._catalog.snapshot_heads())
        for sample in samples:
            ending = sample.ending
            if ending.recorded_at > through or not isinstance(ending.body, Output) or ending.body.finish != "complete":
                continue
            users = [message for message in sample.messages
                     if isinstance(message.body, Input) and self._learning.text(message).strip()]
            if not users or not self._learning.text(ending).strip():
                continue
            vectors = [records.read(message, model=rule.embedding_model, dimension=rule.dimension)
                       for message in (*users, ending)]
            if any(vector is None for vector in vectors):
                continue
            _, user = input_features(users, text=self._learning.text, embeddings=records,
                                     embedding_model=rule.embedding_model, dimension=rule.dimension)
            if user is None:
                continue
            combined = 0.9 * user + 0.1 * np.asarray(vectors[-1], dtype=np.float32)
            norm = float(np.linalg.norm(combined))
            if norm > 0:
                prototypes.append(combined / norm)
        prototypes = prototypes[-256:]

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


SEMANTIC_INTEREST = ServiceKey[SemanticInterest]("akasha.semantic-interest.v1")
