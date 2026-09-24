import asyncio
import sqlite3
from contextlib import closing
from plugins.content.api import legacy_post_commit_effect
from datetime import datetime, timedelta, timezone

import numpy as np
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
    learning = Learning(TurnProjection(), owner="akasha", post_commit_effect=legacy_post_commit_effect)
    embeddings = MessageEmbeddings(log)
    records = embeddings.bind(learning.text)
    rule = LearningConfig(embedding_model="fixed", dimension=2, sources=("conversation",))
    calls = []

    async def embed(texts):
        calls.append(texts)
        return [[0.0, 1.0] for _ in texts]

    async def select():
        return rule, embed

    interest = SemanticInterest(learning, log.catalog(), embeddings, select)

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
        user = np.asarray((1.0, 1.0), dtype=np.float32)
        user /= np.linalg.norm(user)
        combined = 0.9 * user + 0.1 * np.asarray((1.0, 0.0), dtype=np.float32)
        expected = float((combined[1] / np.linalg.norm(combined)) ** 4)
        assert score == pytest.approx((expected,))  # 两输入均值、回复加权与原同步公式一致。

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
        async def select_unavailable():
            return rule, unavailable
        failing = SemanticInterest(learning, log.catalog(), embeddings, select_unavailable)
        with pytest.raises(ConnectionError, match="provider unavailable"):
            await failing.score(["candidate"], cutoff=cutoff)
    finally:
        log.close()


@pytest.mark.asyncio
async def test_interest_keeps_global_sample_order_and_schedules_peer_between_sessions(tmp_path):
    log = MessageLog(tmp_path / "messages.db")
    learning = Learning(TurnProjection(), owner="akasha", post_commit_effect=legacy_post_commit_effect)
    embeddings = MessageEmbeddings(log)
    records = embeddings.bind(learning.text)
    rule = LearningConfig(embedding_model="fixed", dimension=2, sources=("conversation",))

    async def embed(texts):
        assert texts == ["oldest", "recent"]
        return [[-1.0, 0.0], [0.0, 1.0]]

    async def select():
        return rule, embed

    interest = SemanticInterest(learning, log.catalog(), embeddings, select)

    def append_turn(index, session, vector):
        writer = log.writer(session, author="fixture", source="conversation", body_types=(Input, Output),
                            content={"text": lambda part: ContentReferences()})
        for suffix, body in (("input", Input((ContentPart("text", "ask"),))),
                             ("answer", Output((ContentPart("text", "reply"),), "complete"))):
            message = writer.append(f"{index}-{suffix}", body)
            records.save(message, model="fixed", embedding=vector)

    try:
        # Session 目录按 a、z 读取，但全局时间最早的 z 样本必须被 256 上限裁掉。
        for index in range(257):
            session = "z" if index == 0 else "a"
            vector = (-1.0, 0.0) if index == 0 else (0.0, 1.0)
            append_turn(index, session, vector)

        heads = log.catalog().snapshot_heads()
        original = learning.samples(log.catalog(), rule, heads=heads)
        assert len(original) == 257
        assert original[0].ending.session_id == "z"
        assert original[-1].ending.session_id == "a"

        async def peer_append():
            # 第一轮 session 后加入 z；本次固定 heads 不能吸收这个新样本。
            append_turn(257, "z", (-1.0, 0.0))

        peer = asyncio.create_task(peer_append())
        scores = await interest.score(["oldest", "recent"], cutoff=datetime.now(timezone.utc).isoformat())
        assert peer.done()
        assert scores == (0.0, 0.999)
    finally:
        log.close()


@pytest.mark.asyncio
async def test_interest_cancel_during_skipped_vector_samples_never_persists_partial_work(tmp_path, monkeypatch):
    path = tmp_path / "messages.db"
    log = MessageLog(path)
    learning = Learning(TurnProjection(), owner="akasha", post_commit_effect=legacy_post_commit_effect)
    embeddings = MessageEmbeddings(log)
    rule = LearningConfig(embedding_model="fixed", dimension=2, sources=("conversation",))
    embed_calls = 0

    async def embed(texts):
        nonlocal embed_calls
        embed_calls += 1
        return [[1.0, 0.0] for _ in texts]

    async def select():
        return rule, embed

    interest = SemanticInterest(learning, log.catalog(), embeddings, select)
    writer = log.writer("chat", author="fixture", source="conversation", body_types=(Input, Output),
                        content={"text": lambda part: ContentReferences()})
    for index in range(40):
        writer.append(f"{index}-input", Input((ContentPart("text", "ask"),)))
        writer.append(f"{index}-answer", Output((ContentPart("text", "reply"),), "complete"))

    def counts():
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            return (db.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                    db.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0])

    before = counts()
    original_text = learning.text
    scheduled = False
    score_task = None

    def cancel_on_first_sample(message):
        nonlocal scheduled
        if not scheduled:
            scheduled = True
            assert score_task is not None
            asyncio.get_running_loop().call_soon(score_task.cancel)
        return original_text(message)

    monkeypatch.setattr(learning, "text", cancel_on_first_sample)
    try:
        score_task = asyncio.create_task(interest.score(["candidate"], cutoff=datetime.now(timezone.utc).isoformat()))
        with pytest.raises(asyncio.CancelledError):
            await score_task
        assert scheduled
        assert embed_calls == 0
        assert counts() == before == (80, 0)
    finally:
        log.close()
