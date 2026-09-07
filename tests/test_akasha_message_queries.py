import asyncio
from contextlib import asynccontextmanager, closing
import threading

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.akasha.application.consumer import MessageConsumer
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.infrastructure.persistence import logical_state_sha256
from plugins.akasha.infrastructure.lease import WriterLease
from plugins.akasha.learning import AKASHA_LEARNING, LearningConfig
from plugins.akasha.recalls import RecallRecords
from plugins.akasha.runtime import MessageMemory
from session.embedding_store import MessageEmbeddings
from session.log import MessageLog, OwnerTransaction
from session.message import ContentPart, ContentReferences, Control, Input, Output
from tests.test_akasha_learning_binding import manager, sources


@asynccontextmanager
async def memory_runtime(tmp_path, *, max_chars=12000):
    root = tmp_path / "plugins"
    sources(root)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [root], log)
    runtime = None
    consumer = None
    calls = []
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        embeddings = MessageEmbeddings(log)
        consumer = await MessageConsumer.load(tmp_path / "memory.db", legacy_index=None,
            catalog=log.catalog(), embeddings=embeddings, bindings=bindings, config=MemoryConfig())
        async with lease_runtime_snapshot(host.snapshot_store):
            rule = LearningConfig(embedding_model="fixed", dimension=2, sources=("chat",))
            binding = bindings.bind(AKASHA_LEARNING, rule.model_dump())
        async def embed(texts):
            calls.append(texts)
            return [[0.6, 0.8] for _ in texts]
        records = RecallRecords(log.owner("plugin:akasha"))
        runtime = MessageMemory(consumer, catalog=log.catalog(), embeddings=embeddings, bindings=bindings,
            learning_binding=binding, records=records, embed_batch=embed, max_chars=max_chars)
        def write(identity, text, kind=Input, source="chat"):
            body = Input((ContentPart("text", text),)) if kind is Input else Output((ContentPart("text", text),), "complete")
            return log.writer("s", author="test", source=source, body_types=(kind,),
                              content={"text": lambda part: ContentReferences()}).append(identity, body)
        yield runtime, consumer, log, records, calls, write
    finally:
        if runtime is not None:
            await runtime.close()
        elif consumer is not None:
            consumer.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_context_query_uses_fixed_multimessage_input_and_published_references(tmp_path):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        write("old-u1", "first detail")
        write("old-u2", "important correction")
        write("old-a", "learned answer", Output)
        assert await runtime.consume() == 1
        graph = logical_state_sha256(tmp_path / "memory.db")
        first = write("q1", "remember the detail")
        log.writer("s", author="user", source="chat", body_types=(Control,), content={}).append(
            "pause", Control("pause", first.seq))
        write("q2", "and the correction")
        write("other", "independent source", source="timer")
        snapshot = log.reader("s").snapshot()
        write("later", "later input must stay out")
        material = await runtime.prepare(snapshot, "chat")
        assert material.system_prompt == ""
        assert calls[-1] == ["remember the detail", "and the correction"]
        assert [ref.ref for ref in material.references] == ["old-u1", "old-u2", "old-a"]
        assert "learned answer" in material.context[0].value
        identity = material.references[0].retrieval_ref
        record = records.read(identity)
        assert record.source.through_seq == snapshot[-1].seq
        assert record.graph_version == 1
        assert record.hits[0].message_ids == ("old-u1", "old-u2", "old-a")
        assert record.presented_message_ids == tuple(ref.ref for ref in material.references)
        assert logical_state_sha256(tmp_path / "memory.db") == graph
        assert len(consumer.state.applied) == 1
        write("new-a", "new answer", Output)
        assert await runtime.consume() == 1
        assert records.read(identity) == record
        assert consumer.cycle.state_version == 2
        # 消费复用查询阶段固定的两个向量，只补后来输入与答案。
        assert calls[-1] == ["later input must stay out", "new answer"]
    with closing(MessageLog(tmp_path / "sessions.db")) as restored:
        assert RecallRecords(restored.owner("plugin:akasha")).read(identity) == record


@pytest.mark.asyncio
async def test_failed_query_record_cannot_return_materials_and_empty_hit_is_recorded(tmp_path, monkeypatch):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        write("query", "no old memories")
        snapshot = log.reader("s").snapshot()
        save = OwnerTransaction.save
        def fail(self, *args, **kwargs):
            save(self, *args, **kwargs)
            raise OSError("query publication failed")
        with monkeypatch.context() as patch:
            patch.setattr(OwnerTransaction, "save", fail)
            with pytest.raises(OSError, match="query publication failed"):
                await runtime.prepare(snapshot, "chat")
        assert consumer.state.applied == ()
        observed = []
        original = records.save
        def capture(identity, record):
            result = original(identity, record)
            observed.append(record)
            return result
        monkeypatch.setattr(records, "save", capture)
        material = await runtime.prepare(snapshot, "chat")
        assert material.context == material.references == ()
        assert len(observed) == 1 and observed[0].hits == ()
        assert calls == [["no old memories"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_cancelled_query_drains_graph_reader_before_close(tmp_path, monkeypatch, cancel_close):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        write("query", "read in background")
        entered = asyncio.Event()
        released = threading.Event()
        loop = asyncio.get_running_loop()
        retrieve = consumer.cycle.retrieve
        def blocking(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert released.wait(10), "test did not release query"
            return retrieve(*args, **kwargs)
        monkeypatch.setattr(consumer.cycle, "retrieve", blocking)
        query = asyncio.create_task(runtime.prepare(log.reader("s").snapshot(), "chat"))
        await asyncio.wait_for(entered.wait(), 10)
        query.cancel()
        close_started = asyncio.Event()
        async def close_runtime():
            close_started.set()
            await runtime.close()
        closing = asyncio.create_task(close_runtime())
        try:
            # close 已开始接纳等待，但 writer 仍需等线程完成。
            await close_started.wait()
            assert not closing.done()
            assert consumer.cycle.state_version == 0
            if cancel_close:
                closing.cancel()
        finally:
            released.set()
        with pytest.raises(asyncio.CancelledError):
            await query
        if cancel_close:
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            await closing
        lease = WriterLease(tmp_path / "memory.db")
        lease.close()
        with pytest.raises(RuntimeError, match="已关闭"):
            await runtime.consume()


@pytest.mark.asyncio
async def test_context_rejects_truncated_prefix_and_does_not_use_abandoned_input(tmp_path):
    async with memory_runtime(tmp_path) as (runtime, consumer, log, records, calls, write):
        first = write("abandoned", "discarded query")
        log.writer("s", author="user", source="chat", body_types=(Control,), content={}).append(
            "abandon", Control("abandon", first.seq))
        assert (await runtime.prepare(log.reader("s").snapshot(), "chat")).context == ()
        assert calls == []
        write("current", "actual query")
        snapshot = log.reader("s").snapshot()
        with pytest.raises(ValueError, match="完整且真实"):
            await runtime.prepare(snapshot[1:], "chat")
        await runtime.prepare(snapshot, "chat")
        assert calls == [["actual query"]]


@pytest.mark.asyncio
async def test_budget_records_exact_presented_members_without_losing_learning_members(tmp_path, monkeypatch):
    async with memory_runtime(tmp_path, max_chars=100) as (runtime, consumer, log, records, calls, write):
        for number in (1, 2):
            write(f"u{number}", "detail")
            write(f"a{number}", "answer", Output)
            assert await runtime.consume() == 1
        write("q", "recall")
        material = await runtime.prepare(log.reader("s").snapshot(), "chat")
        assert len(material.context[0].value) <= 100
        assert len(material.references) == 1
        record = records.read(material.references[0].retrieval_ref)
        assert [hit.message_ids for hit in record.hits] == [("u2", "a2"), ("u1", "a1")]
        assert record.presented_message_ids == tuple(ref.ref for ref in material.references)
        assert record.presented_message_ids == ("u2",)
