"""真实消息、学习图及召回在派生库丢失后的恢复验收。"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import sqlite3
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition import CompositionRoot, Context
from plugins.akasha.application.consumer import MessageConsumer
from plugins.akasha.application.snapshot import read_memory
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.ledger import read_overview, list_skipped
from plugins.akasha.infrastructure.persistence import MemoryRebuildRequiredError
from plugins.akasha.learning import AKASHA_LEARNING, Learning, LearningConfig
from plugins.akasha.contract import ProgramSource
from plugins.akasha.recalls import query_memory
from plugins.content.api import legacy_post_commit_effect
from plugins.content.plugin import check_text
from plugins.ledger.bindings import Bindings
from plugins.ledger.contract import ContentPart, Input, Output, SessionAttributes
from plugins.ledger.embedding_store import MessageEmbeddings
from plugins.ledger.log import MessageLog
from plugins.turn_projection.plugin import TurnProjection


def vector(text: str) -> list[float]:
    """固定本地模型：相同文本总是进入同一有限三维空间。"""
    return [float(value + 1) for value in hashlib.sha256(text.encode()).digest()[:3]]


async def run(workspace: Path) -> dict[str, object]:
    """先实际学习，再删除派生副本，比较同一查询和原存储的完整内容。"""
    path = workspace / 'sessions.db'
    derived = workspace / 'sessions-derived.db'
    graph = workspace / 'memory.db'
    log = MessageLog(path)
    store = MessageEmbeddings(log, derived)
    learning = Learning(TurnProjection(), owner='akasha', post_commit_effect=legacy_post_commit_effect)
    rule = LearningConfig(embedding_model='scenario-fixed', dimension=3, sources=('conversation',))
    root = CompositionRoot('derived-recovery')
    async def provide(ctx: Context) -> None:
        await ctx.provide(AKASHA_LEARNING, learning)
    await root.mount(provide, name='learning')
    bindings = Bindings(log, root.context)
    log.save_binding('learning', {'version': 1, 'service': AKASHA_LEARNING.name,
                                'root_ref': 'fixture-archive', 'metadata': rule.model_dump()})
    encoded: list[str] = []
    async def embed(texts: list[str]) -> list[list[float]]:
        encoded.extend(texts)
        return [vector(text) for text in texts]
    async def restore(original: LearningConfig, texts: list[str]) -> list[list[float]]:
        assert original == rule
        return await embed(texts)
    try:
        consumer = await MessageConsumer.load(graph, catalog=log.catalog(), embeddings=store,
                                             bindings=bindings, config=MemoryConfig(), cutover=False)
        try:
            for index, text in enumerate(('tea by the lake', 'lunch in the garden', 'tea at home')):
                session = f'conversation:{index}'
                log.ensure_session(session, SessionAttributes())
                log.writer(session, author='user', source='conversation', body_types=(Input,),
                           content={'text': check_text}).append(
                               f'input:{index}', Input((ContentPart('text', text),)))
                log.writer(session, author='assistant', source='conversation', body_types=(Output,),
                           content={'text': check_text}).append(
                               f'output:{index}', Output((ContentPart('text', 'remember ' + text),), 'complete'))
                assert await consumer.consume(catalog=log.catalog(), learning_binding='learning',
                                              embeddings=store, bindings=bindings, embed_batch=embed) == 1
        finally:
            consumer.close()
        assert len(encoded) == 6
        stamp = datetime.now(UTC)
        async def query():
            async with read_memory(graph, catalog=log.catalog(), embeddings=store, bindings=bindings,
                                   config=MemoryConfig(), embedding_space=('scenario-fixed', 3),
                                   restore_embeddings=restore) as (cycle, state):
                dense = np.asarray(vector('tea by the lake'), dtype=np.float32)
                dense /= np.linalg.norm(dense)
                return query_memory(cycle, state, learning_binding='learning', text='tea by the lake',
                                    dense=dense, stamp=stamp, source=ProgramSource(key='probe', query='tea by the lake'),
                                    limit=3)
        before = await query()
        assert before.hits
        assert len(encoded) == 6
        store.close()
        log.close()
        original = path.read_bytes()
        published = graph.read_bytes()
        assert read_overview(graph)["learned"] == 3
        assert list_skipped(graph) == []
        # 仅损坏隔离副本，证明 UI 读边界不会把坏进度显示成空数据。
        broken = workspace / "broken-memory.db"
        broken.write_bytes(published)
        with sqlite3.connect(broken) as connection:
            connection.execute("UPDATE metadata SET value='{}' WHERE key='consumer_state_json'")
        for read in (read_overview, list_skipped):
            try:
                read(broken)
            except MemoryRebuildRequiredError:
                pass
            else:
                raise AssertionError("损坏的学习进度被隐藏")
        # 丢失的是可重建副本；不改源消息或学习事实来配合测试。
        derived.unlink()
        log = MessageLog(path)
        store = MessageEmbeddings(log, derived)
        bindings = Bindings(log, root.context)
        after = await query()
        assert after == before
        assert len(encoded) == 12
        again = await query()
        assert again == before and len(encoded) == 12
        consumer = await MessageConsumer.load(graph, catalog=log.catalog(), embeddings=store,
            bindings=bindings, config=MemoryConfig(), restore_embeddings=restore)
        consumer.close()
        assert len(encoded) == 12
        store.close()
        log.close()
        assert path.read_bytes() == original
        assert graph.read_bytes() == published
        return {'learned_turns': 3, 'rebuilt_vectors': 6, 'same_nonempty_recall': True,
                'warm_query_no_reembedding': True, 'online_restore': True,
                'invalid_progress_fails': True, 'messages_bytes_unchanged': True, 'graph_bytes_unchanged': True}
    finally:
        await root.dispose()
        store.close()
        log.close()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='akashic-derived-recovery-') as folder:
        print(json.dumps(asyncio.run(run(Path(folder)))))
