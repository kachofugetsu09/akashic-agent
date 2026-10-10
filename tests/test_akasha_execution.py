"""Akasha 保持逐条因果发布；后台工作不能冻结宿主或提前释放 owner。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import threading

import pytest

from agent.plugin_composition import CompositionRoot, Context
from plugins.ledger.bindings import Bindings
from plugins.akasha.application.consumer import MessageConsumer
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.infrastructure.consumption import Consumption
from plugins.akasha.infrastructure.lease import WriterLease
from plugins.akasha.infrastructure.persistence import load_consumption, logical_state_sha256
from plugins.akasha.learning import AKASHA_LEARNING, Learning, LearningConfig
from plugins.akasha.recalls import RecallRecords
from plugins.akasha.runtime import MessageMemory
from plugins.content.api import legacy_post_commit_effect
from plugins.content.plugin import check_text
from plugins.turn_projection.plugin import TurnProjection
from plugins.ledger.embedding_store import MessageEmbeddings
from plugins.ledger.log import MessageLog, SessionAttributes
from plugins.ledger.contract import ContentPart, Input, Output


class WorkerGate:
    """在真实同步边界暂停 worker；测试线程可确定地推进事件循环。"""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.entered = asyncio.Event()
        self.release = threading.Event()

    def stop(self) -> None:
        assert threading.get_ident() != self.loop_thread, "Akasha work blocked the event loop"
        self.loop.call_soon_threadsafe(self.entered.set)
        assert self.release.wait(10), "test did not release worker"

    async def wait(self, job: asyncio.Task[object]) -> None:
        """等待屏障或直接报告工作失败，避免用超时掩盖原异常。"""
        arrival = asyncio.create_task(self.entered.wait())
        try:
            done, _ = await asyncio.wait((arrival, job), timeout=5,
                                         return_when=asyncio.FIRST_COMPLETED)
            if job in done:
                job.result()
                raise AssertionError("work finished without reaching the gate")
            assert arrival in done, "worker did not reach the gate"
        finally:
            arrival.cancel()
            await asyncio.gather(arrival, return_exceptions=True)


async def loop_turn() -> None:
    """让已排队 Task 获得执行机会，不依赖计时 sleep。"""
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await ready.wait()


@asynccontextmanager
async def memory_fixture(path: Path) -> AsyncIterator[tuple[MessageMemory, MessageConsumer, Learning, Bindings, MessageLog, MessageEmbeddings, CompositionRoot, asyncio.Event]]:
    """使用真实消息、固定向量、provider scope、学习器和磁盘 writer。"""
    log = MessageLog(path / "sessions.db")
    log.ensure_session("conversation", SessionAttributes())
    embeddings = MessageEmbeddings(log, log._path.with_name("sessions-derived.db"))
    root = CompositionRoot("akasha-execution")
    learning = Learning(TurnProjection(), owner="akasha", post_commit_effect=legacy_post_commit_effect)
    bindings = Bindings(log, root.context)
    rule = LearningConfig(embedding_model="fixed", dimension=3, sources=("conversation",))
    identity = "fixed-learning"
    log.save_binding(identity, {"version": 1, "service": AKASHA_LEARNING.name,
        "root_ref": "fixture-archive", "metadata": rule.model_dump()})
    closed = asyncio.Event()

    async def provide(ctx: Context) -> None:
        await ctx.provide(AKASHA_LEARNING, learning)
        await ctx.effect(lambda: closed.set)

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    await root.mount(provide, name="learning")
    assert root.receipt().ready, repr(root.receipt())
    consumer = await MessageConsumer.load(path / "memory.db", catalog=log.catalog(),
        embeddings=embeddings, bindings=bindings, config=MemoryConfig(), cutover=False)
    memory = MessageMemory(consumer, catalog=log.catalog(), embeddings=embeddings,
        bindings=bindings, learning_binding=identity, records=RecallRecords(log.owner("akasha")),
        embed_batch=embed)
    try:
        yield memory, consumer, learning, bindings, log, embeddings, root, closed
    finally:
        await memory.close()
        await root.dispose()
        embeddings.close()
        log.close()


def append_pair(log: MessageLog, embeddings: MessageEmbeddings, learning: Learning, number: int) -> None:
    """真实已提交问答和固定向量是学习的唯一输入。"""
    for author, kind, body in (
        ("user", Input, Input((ContentPart("text", f"remember episode {number}"),))),
        ("assistant", Output, Output((ContentPart("text", f"episode {number} happened"),), "complete")),
    ):
        message = log.writer("conversation", author=author, source="conversation",
            body_types=(kind,), content={"text": check_text}).append(f"{author}-{number}", body)
        embeddings.bind(learning.text).save(message, model="fixed", embedding=[1.0, 0.0, 0.0])


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before", "after"])
@pytest.mark.parametrize("fail", [False, True])
async def test_learning_waits_for_prior_durable_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, fail: bool) -> None:
    """MEM-009 / O：同图 t+1 等待 t 发布，宿主仍运行；失败不能越过前驱。"""
    async with memory_fixture(tmp_path) as (memory, consumer, learning, _, log, embeddings, _, _):
        for number in range(2):
            append_pair(log, embeddings, learning, number)
        gate = WorkerGate()
        original_publish = consumer.publish_snapshot_for
        original_make = learning.make_turn
        observed: list[tuple[int, int]] = []

        def make_turn(*args, **kwargs):
            assert threading.get_ident() != gate.loop_thread
            disk = load_consumption(consumer.path)
            assert disk is not None
            observed.append((len(kwargs["previous"]), len(disk.applied)))
            return original_make(*args, **kwargs)

        def publish(state: Consumption) -> str:
            if len(state.applied) != 1:
                return original_publish(state)
            if boundary == "after":
                result = original_publish(state)
            gate.stop()
            if fail:
                raise OSError("controlled publication failure")
            return result if boundary == "after" else original_publish(state)

        monkeypatch.setattr(learning, "make_turn", make_turn)
        monkeypatch.setattr(consumer, "publish_snapshot_for", publish)
        first = asyncio.create_task(memory.consume())
        tasks: list[asyncio.Task[object]] = [first]
        try:
            await gate.wait(first)
            follower = asyncio.create_task(memory.consume())
            query = asyncio.create_task(memory.prepare(log.reader("conversation").snapshot(), "conversation"))
            tasks.extend((follower, query))
            await loop_turn()
            assert observed == [(0, 0)]
            assert not any(task.done() for task in tasks)
            disk = load_consumption(consumer.path)
            assert disk is not None
            assert len(disk.applied) == (1 if boundary == "after" else 0)
            assert len(consumer.state.applied) == 0
            gate.release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            if fail:
                assert isinstance(results[0], OSError)
                assert all(isinstance(result, RuntimeError) for result in results[1:])
                assert observed == [(0, 0)]
            else:
                assert results[:2] == [2, 0]
                assert isinstance(results[2], dict)
                assert observed == [(0, 0), (1, 1)]
                disk = load_consumption(consumer.path)
                assert disk is not None
                assert [entry.ending[1] for entry in disk.applied] == ["assistant-0", "assistant-1"]
        finally:
            gate.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_restore_keeps_order_and_binding_until_worker_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool) -> None:
    """PLG-003 / MEM-009：恢复按前缀运行；取消和卸载必须等待真实工作结束。"""
    async with memory_fixture(tmp_path) as (memory, consumer, learning, bindings, log, embeddings, root, closed):
        for number in range(2):
            append_pair(log, embeddings, learning, number)
        assert await memory.consume() == 2
        before = logical_state_sha256(consumer.path)
        await memory.close()
        gate = WorkerGate()
        original = learning.restore
        seen: list[int] = []

        def restore(*args, **kwargs):
            assert threading.get_ident() != gate.loop_thread, "restore must not run on the host loop"
            position = len(kwargs["previous"])
            assert position == len(seen)
            turn = original(*args, **kwargs)
            seen.append(position)
            if position == 0:
                gate.stop()
            return turn

        monkeypatch.setattr(learning, "restore", restore)
        load = asyncio.create_task(MessageConsumer.load(consumer.path, catalog=log.catalog(),
            embeddings=embeddings, bindings=bindings, config=MemoryConfig()))
        disposal = None
        try:
            await gate.wait(load)
            await loop_turn()
            assert seen == [0] and not load.done()
            if cancel:
                load.cancel()
                await loop_turn()
                load.cancel()
                disposal = asyncio.create_task(root.dispose())
                await loop_turn()
                assert not load.done() and not disposal.done() and not closed.is_set()
            gate.release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await load
                assert disposal is not None
                await disposal
                assert closed.is_set()
            else:
                restored = await load
                assert [turn.turn_id for turn in restored.cycle.turns] == [turn.turn_id for turn in consumer._cycle.turns]
                restored.close()
            assert seen == [0, 1]
            assert logical_state_sha256(consumer.path) == before
        finally:
            gate.release.set()
            await asyncio.gather(load, return_exceptions=True)
            if disposal is not None:
                await disposal


@pytest.mark.asyncio
async def test_cancelled_creation_closes_the_completed_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLG-006：取消不能丢掉后台刚取得的 writer，旧任务排空后才能重新打开。"""
    gate = WorkerGate()
    original = MessageConsumer.__init__

    def build(self, *args, **kwargs):
        assert threading.get_ident() != gate.loop_thread
        original(self, *args, **kwargs)
        gate.stop()

    monkeypatch.setattr(MessageConsumer, "__init__", build)
    path = tmp_path / "memory.db"
    job = asyncio.create_task(MessageConsumer.create(path, turns=[], state=Consumption(cutover_heads=()), config=MemoryConfig()))
    try:
        await gate.wait(job)
        job.cancel()
        await loop_turn()
        job.cancel()
        await loop_turn()
        assert not job.done()
        with pytest.raises(RuntimeError, match="already has a writer"):
            WriterLease(path)
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await job
        lease = WriterLease(path)
        lease.close()
        assert load_consumption(path) == Consumption(cutover_heads=())
    finally:
        gate.release.set()
        await asyncio.gather(job, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_learning_drains_before_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """取消只阻止后续经历；当前发布与 writer 释放必须按顺序完成。"""
    async with memory_fixture(tmp_path) as (memory, consumer, learning, _, log, embeddings, _, _):
        for number in range(2):
            append_pair(log, embeddings, learning, number)
        gate = WorkerGate()
        original = consumer.publish_snapshot_for

        def publish(state: Consumption) -> str:
            result = original(state)
            gate.stop()
            return result

        monkeypatch.setattr(consumer, "publish_snapshot_for", publish)
        job = asyncio.create_task(memory.consume())
        close = None
        try:
            await gate.wait(job)
            job.cancel()
            await loop_turn()
            job.cancel()
            close = asyncio.create_task(memory.close())
            await loop_turn()
            assert not job.done() and not close.done()
            with pytest.raises(RuntimeError, match="already has a writer"):
                WriterLease(consumer.path)
            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await job
            await close
            disk = load_consumption(consumer.path)
            assert disk is not None
            assert [entry.ending[1] for entry in disk.applied] == ["assistant-0"]
            assert consumer.state == disk
            lease = WriterLease(consumer.path)
            lease.close()
        finally:
            gate.release.set()
            await asyncio.gather(job, return_exceptions=True)
            if close is not None:
                await close


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_rebuild_finishes_publication_receipt_before_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool) -> None:
    """重建仍复用逐条学习；替换后取消也要排空耐久回执。"""
    from plugins.akasha.application import rebuild

    async with memory_fixture(tmp_path) as (memory, consumer, learning, bindings, log, embeddings, _, _):
        for number in range(2):
            append_pair(log, embeddings, learning, number)
        assert await memory.consume() == 2
        before = logical_state_sha256(consumer.path)
        identity = consumer.state.applied[0].learning_binding
        await memory.close()
        gate = WorkerGate()
        original = rebuild._write_manifest

        def write_manifest(*args):
            gate.stop()
            return original(*args)

        async def unexpected_embedding(texts: list[str]) -> list[list[float]]:
            raise AssertionError("fixed embeddings must be reused")

        monkeypatch.setattr(rebuild, "_write_manifest", write_manifest)
        backups = tmp_path / "backups"
        job = asyncio.create_task(rebuild.rebuild_from_catalog(
            catalog=log.catalog(), embeddings=embeddings, bindings=bindings,
            config=MemoryConfig(), learning_binding=identity, embed_batch=unexpected_embedding,
            memory_path=consumer.path, backup_root=backups,
        ))
        try:
            await gate.wait(job)
            assert logical_state_sha256(consumer.path) == before
            if cancel:
                job.cancel()
                await loop_turn()
                assert not job.done()
            gate.release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await job
            else:
                report = await job
                assert report.turns == 2 and report.embedded_messages == 0
            assert len(list(backups.glob("*/manifest.json"))) == 1
            assert len(list(backups.glob("*/memory-before.db"))) == 1
            assert logical_state_sha256(consumer.path) == before
        finally:
            gate.release.set()
            await asyncio.gather(job, return_exceptions=True)
