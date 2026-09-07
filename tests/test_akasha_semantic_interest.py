from datetime import datetime, timedelta, timezone

import pytest

from plugins.akasha.interest import SemanticInterest
from plugins.akasha.learning import Learning, LearningConfig
from plugins.turn_projection.plugin import TurnProjection
from session.embedding_store import MessageEmbeddings
from session.log import MessageLog, SessionAttributes
from session.message import ContentPart, ContentReferences, Input, Output


@pytest.mark.asyncio
async def test_interest_uses_completed_eligible_cached_inputs_and_cutoff(tmp_path):
    log = MessageLog(tmp_path / "messages.db")
    learning = Learning(TurnProjection(), owner="akasha")
    embeddings = MessageEmbeddings(log)
    records = embeddings.bind(learning.text)
    rule = LearningConfig(embedding_model="fixed", dimension=2, sources=("conversation",))
    calls = []

    async def embed(texts):
        calls.append(texts)
        return [[0.0, 1.0] for _ in texts]

    interest = SemanticInterest(learning, log.catalog(), embeddings, lambda: (rule, embed))

    def append(identity, body, *, session="chat", source="conversation", vector=(1.0, 0.0)):
        writer = log.writer(session, author="fixture", source=source, body_types=(type(body),),
                            content={"text": lambda part: ContentReferences()})
        message = writer.append(identity, body)
        if vector is not None:
            records.save(message, model="fixed", embedding=vector)
        return message

    def text(value):
        return (ContentPart("text", value),)

    try:
        old = datetime.now(timezone.utc) - timedelta(days=1)
        append("input1", Input(text("first")))
        append("input2", Input(text("second")), vector=(0.0, 1.0))
        ending = append("answer", Output(text("answer"), "complete"))
        cutoff = ending.recorded_at.isoformat()
        score = await interest.score(["candidate"], cutoff=cutoff)
        assert 0.18 < score[0] < 0.19  # 两输入归一化均值与回复加权，不丢首个输入。

        append("open", Input(text("unanswered")), vector=(0.0, 1.0))
        log.ensure_session("hidden", SessionAttributes(visibility="internal", learning="excluded"))
        append("hidden-input", Input(text("private")), session="hidden", vector=(0.0, 1.0))
        append("hidden-answer", Output(text("private"), "complete"), session="hidden", vector=(0.0, 1.0))
        append("later-answer", Output(text("future"), "complete"), vector=(0.0, 1.0))
        assert await interest.score(["candidate"], cutoff=cutoff) == score
        assert await interest.score(["candidate"], cutoff=old.isoformat()) == (0.0,)
        assert calls == [["candidate"], ["candidate"]]

        # 缺历史缓存不调用模型补齐；已有空间损坏仍由向量记录边界明确失败。
        append("uncached-input", Input(text("uncached")), session="missing", vector=None)
        append("uncached-answer", Output(text("uncached"), "complete"), session="missing", vector=None)
        assert await interest.score(["candidate"], cutoff=cutoff) == score
        async def unavailable(texts):
            raise ConnectionError("provider unavailable")
        failing = SemanticInterest(learning, log.catalog(), embeddings, lambda: (rule, unavailable))
        with pytest.raises(ConnectionError, match="provider unavailable"):
            await failing.score(["candidate"], cutoff=cutoff)
    finally:
        log.close()
