"""用临时 Root、真实 SQLite 和 Wake 验证 Drift 的异步存储边界。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
    MessageWriters, OwnerState, SessionAdmission,
)
from agent.plugin_composition.tasks import TASKS, PluginTasks
from agent.plugin_contracts.proactive import DRIFT_WAKE, DRIFT_DELIVERY
from plugins.drift.plugin import apply as apply_drift
from plugins.drift.store import DriftStore
from plugins.wake.api import DRIFT_WAKE as ASYNC_WAKE, DRIFT_DELIVERY as ASYNC_DELIVERY, DeliveryTarget
from plugins.wake.request import Request, TOOLS
from plugins.wake.source import Source
from plugins.wake.state import WakeState
from session.log import MessageLog


async def loop_turn() -> None:
    """让排队的取消与关闭推进一次，不以 sleep 猜测顺序。"""
    done = asyncio.Event()
    asyncio.get_running_loop().call_soon(done.set)
    await done.wait()


async def storage(directory: Path, baseline: bool) -> dict[str, object]:
    """核对真实锁等待、提交、回滚、重放及 owner 的取消排空。"""
    root = CompositionRoot("drift-io")
    provider = await root.mount(apply_drift, name="drift", runtime=PluginRuntime(
        "drift", "scenario", directory, directory, directory, {}))
    contexts = []

    async def consume(ctx):
        contexts.append(ctx)

    consumer = await root.mount(consume, name="consumer", inject=(
        DRIFT_WAKE, DRIFT_DELIVERY, ASYNC_WAKE, ASYNC_DELIVERY))
    ctx = contexts[0]
    store = DriftStore(directory / "drift.sqlite3")
    now = datetime.now(UTC)
    ref = store.propose("lock", "one", {"text": "unchanged"}, now)["ref"]
    gate, release, connected = threading.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    metrics: dict[str, Any] = {}

    # 1. 独立连接持有真实写锁。只有 loop 回调可以正常释放；线程超时只是防挂死。
    def lock_database():
        with closing(sqlite3.connect(store.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connected.set()
            metrics["released_by_loop"] = release.wait(0.4)
            connection.rollback()

    holder = threading.Thread(target=lock_database)
    holder.start()
    assert await asyncio.to_thread(connected.wait, 2)
    key = DRIFT_WAKE if baseline else ASYNC_WAKE
    async with ctx.runtime_scope():
        service = ctx.require(key)
        started = time.perf_counter()

        def heartbeat():
            metrics["loop_lag_ms"] = (time.perf_counter() - started) * 1000
            release.set()

        loop.call_soon(heartbeat)
        accepted = {"session_id": "one", "turn_id": "input"}
        result = (service.select(ref, accepted, now) if baseline
                  else await service.select(ref, accepted, now))
        assert result["selected"]
    await asyncio.to_thread(holder.join, 2)
    await loop_turn()
    assert metrics["released_by_loop"] is (not baseline), metrics
    if baseline:
        await root.dispose()
        return metrics

    # 2. 所有 v2 查询和事务都在 worker 打开/使用/关闭自己的真实连接。
    original_connect = sqlite3.connect
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    pause_commit = False
    fail_commit = False
    connections = []

    class Connection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert threading.get_ident() != loop_thread
            self.thread = threading.get_ident()
            self.closed = False
            connections.append(self)

        def commit(self):
            if pause_commit:
                loop.call_soon_threadsafe(entered.set)
                assert gate.wait(5), "worker 没有被释放"
            if fail_commit:
                raise OSError("controlled commit failure")
            super().commit()

        def close(self):
            assert threading.get_ident() == self.thread
            super().close()
            self.closed = True

    def connect(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=Connection)

    async with ctx.runtime_scope():
        wake, delivery = ctx.require(ASYNC_WAKE), ctx.require(ASYNC_DELIVERY)
        token = result["selection_token"]
        with patch("sqlite3.connect", connect):
            assert (await wake.selection(accepted))["selection_token"] == token
            assert not (await wake.snapshot(now))["proposals"]
            assert (await delivery.lookup(accepted))["status"] == "selected"
            assert (await wake.transition(token, "ready_for_delivery"))["changed"]
            assert (await delivery.settle(token, "notice"))["settled"]
            assert (await delivery.settle(token, "notice"))["duplicate"]
            assert (await delivery.lookup(accepted))["settlement_ref"] == "notice"
            try:
                await delivery.settle(token, "other-notice")
            except RuntimeError as error:
                assert "identity conflict" in str(error)
            else:
                raise AssertionError("冲突的结算身份没有失败")
        assert all(connection.closed for connection in connections)

        # 3. 两个真实事务抢相同版本，只能有一个提交领取。
        ref = store.propose("race", "one", {}, now)["ref"]
        claims = await asyncio.gather(*(wake.select(ref, {"session_id": str(i), "turn_id": "race"}, now)
                                       for i in range(2)))
        assert sum(claim["selected"] for claim in claims) == 1
        ref = store.propose("rollback", "one", {}, now)["ref"]
        fail_commit = True
        with patch("sqlite3.connect", connect):
            try:
                await wake.select(ref, {"session_id": "rollback", "turn_id": "input"}, now)
            except OSError as error:
                assert str(error) == "controlled commit failure"
            else:
                raise AssertionError("提交失败被隐藏")
        fail_commit = False
        assert store.selection({"session_id": "rollback", "turn_id": "input"}) is None

    # 4. 取消发生在 COMMIT 前；consumer 关闭必须等待物理事务及 scope 退出。
    ref = store.propose("cancel", "one", {}, now)["ref"]
    accepted = {"session_id": "cancel", "turn_id": "input"}
    pause_commit = True

    async def claim():
        async with ctx.runtime_scope():
            return await ctx.require(ASYNC_WAKE).select(ref, accepted, now)

    with patch("sqlite3.connect", connect):
        job = asyncio.create_task(claim())
        disposal = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            job.cancel()
            await loop_turn()
            job.cancel()
            disposal = asyncio.create_task(consumer.dispose())
            await loop_turn()
            assert not job.done() and not disposal.done()
            assert any(not connection.closed for connection in connections)
            gate.set()
            try:
                await job
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("丢失调用者取消")
            await disposal
        finally:
            gate.set()
            await asyncio.gather(job, return_exceptions=True)
            if disposal is not None:
                await disposal
    assert all(connection.closed for connection in connections)
    saved = DriftStore(store.path).selection(accepted)
    assert saved is not None, "已提交的领取不能随取消丢失"
    assert store.select(cast(Mapping[str, object], ref), accepted, now)["selected"] is False
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT payload_json FROM proposals WHERE proposal_id='lock'").fetchone()[0] == '{"text":"unchanged"}'
    await provider.dispose()
    await root.dispose()
    return {**metrics, "worker_connections": len(connections), "cases": [
        "lock", "reads", "settlement_replay", "settlement_conflict", "claim_cas", "rollback", "cancel_drain", "reopen"]}


async def wake_flow(directory: Path) -> dict[str, object]:
    """运行真实 Wake decline 路径；它不需要模型、工具执行或外部发送。"""
    root = CompositionRoot("drift-wake-flow")
    log, tasks = MessageLog(directory / "sessions.db"), PluginTasks()
    state = WakeState(directory / "wake.db")
    state.initialize()
    # decline 不打开这些资源；持久引用仍必须有真实 descriptor 行。
    log.save_binding("unused", {"scenario": "decline-only-unused-binding"})
    await root.mount(apply_drift, name="drift", runtime=PluginRuntime(
        "drift", "scenario", directory, directory, directory, {}))

    async def providers(ctx):
        await ctx.provide(MESSAGE_CATALOG, log.catalog())
        await ctx.provide(MESSAGE_WRITERS, MessageWriters(log))
        await ctx.provide(OWNER_STATE, OwnerState(log))
        await ctx.provide(SESSION_ADMISSION, SessionAdmission(log))
        await ctx.provide(TASKS, tasks)

    contexts = []
    async def consumer(ctx):
        contexts.append(ctx)

    await root.mount(providers, name="core")
    await root.mount(consumer, name="wake", inject=(MESSAGE_CATALOG, MESSAGE_WRITERS,
        OWNER_STATE, SESSION_ADMISSION, TASKS, ASYNC_WAKE, ASYNC_DELIVERY),
        runtime=PluginRuntime("wake", "scenario", directory, directory, directory, {}))
    store, now = DriftStore(directory / "drift.sqlite3"), datetime.now(UTC)
    store.propose("decline", "one", {"wake_action": "decline"}, now)
    ctx = contexts[0]
    try:
        async with ctx.runtime_scope():
            proposals = (await ctx.require(ASYNC_WAKE).snapshot(now))["proposals"]
            request = Request(flow_id="a" * 32, owner="drift", now=now, timezone="UTC",
                target=DeliveryTarget(channel="unused", recipient="unused", session_id="target"),
                sink={"name": "unused", "address": "unused", "binding_id": "unused"},
                program_binding="unused", tools={name: "unused" for name in TOOLS["drift"]},
                snapshot_seq=0, proposals=tuple(dict(p) for p in proposals), rules="", history="")
            source = Source(ctx, state)
            await source.accept(request)
            original = log.reader(request.session_id).snapshot()[0]
            task = await source.start(request.flow_id)
            assert task is not None
            await task.join()
            assert not source.pending()
            rows = log.reader(request.session_id).snapshot()
            assert rows[0] == original and len(rows) == 2
            assert state.get_run(request.flow_id)["decision"] == "skip"
            assert await source.start(request.flow_id) is None
        assert store.snapshot(now)["proposals"] == ()
        with closing(sqlite3.connect(store.path)) as connection:
            assert connection.execute("SELECT status FROM proposals").fetchone()[0] == "await_change"
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        return {"decision": "skip", "status": "await_change", "messages": len(rows)}
    finally:
        await tasks.close()
        await root.dispose()
        log.close()


async def main(baseline: bool) -> None:
    with TemporaryDirectory(prefix="drift-io-") as name:
        path = Path(name)
        result = {"storage": await storage(path / "store", baseline)}
        if not baseline:
            (path / "wake").mkdir()
            result["wake"] = await wake_flow(path / "wake")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="实测保留的 v1 同步能力")
    asyncio.run(main(parser.parse_args().baseline))
