"""在临时 SQLite 日志上验证类型唤醒、并发接纳、回滚和重开追赶。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.ledger.log import MessageLog
from plugins.ledger.contract import Control, Input


async def turn() -> None:
    """让已入队的通知和消费者各执行一轮，不依赖计时 sleep。"""
    for _ in range(3):
        ready = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(ready.set_result, None)
        await ready


async def run(path: Path) -> None:
    """通过真实 writer、owner 事务和订阅读取验证可观察消息合同。"""
    log = MessageLog(path)
    writer = log.writer("s", author="user", source="conversation", body_types=(Input, Control), content={})
    owner = log.owner("scenario")
    streams = []
    try:
        # 1. 初始追赶包含全部历史；普通消息不触发 Control 消费者。
        writer.append("input", Input(()))
        stream = log.catalog().follow(wake_on=Control)
        streams.append(stream)
        assert dict(await anext(stream)) == {"s": 0}
        pending = asyncio.create_task(anext(stream))
        await turn()
        writer.append("ordinary", Input(()))
        await turn()
        assert not pending.done()
        writer.append("control", Control("abandon", 1))
        assert dict(await asyncio.wait_for(pending, 2)) == {"s": 2}
        pending = asyncio.create_task(anext(stream))
        await turn()

        # 2. 事务回滚既不追加消息，也不制造唤醒；后续正常提交仍可交付。
        def rollback(transaction):
            transaction.append(writer, "rolled-back", Control("pause", 2))
            raise ValueError("scenario rollback")

        try:
            owner.transact(rollback)
        except ValueError as error:
            assert str(error) == "scenario rollback"
        else:
            raise AssertionError("transaction should fail")
        writer.append("after-rollback", Input(()))
        await turn()
        assert not pending.done()
        assert all(message.message_id != "rolled-back" for message in log.reader("s").snapshot())

        # 3. 在 insert 与 commit 之间接纳新订阅，必须先见旧快照再收到新提交。
        inserted = threading.Event()
        release = threading.Event()

        def held(transaction):
            transaction.append(writer, "during-subscribe", Control("pause", 3))
            inserted.set()
            if not release.wait(5):
                raise TimeoutError("subscription did not reach the commit barrier")

        commit = asyncio.create_task(owner.transact_async(held))
        assert await asyncio.to_thread(inserted.wait, 5)
        late = log.catalog().follow(wake_on=Control)
        streams.append(late)
        assert dict(await anext(late)) == {"s": 3}
        late_next = asyncio.create_task(anext(late))
        await turn()
        release.set()
        await commit
        assert dict(await asyncio.wait_for(late_next, 2)) == {"s": 4}
        assert dict(await asyncio.wait_for(pending, 2)) == {"s": 4}

        # 4. 不同类型的异步提交均保留通知；无筛选订阅仍看到所有提交。
        inputs = log.catalog().follow(wake_on=Input)
        all_messages = log.catalog().follow()
        streams.extend((inputs, all_messages))
        await anext(inputs)
        await anext(all_messages)
        pending = asyncio.create_task(anext(stream))
        input_next = asyncio.create_task(anext(inputs))
        all_next = asyncio.create_task(anext(all_messages))
        await turn()
        await asyncio.gather(
            writer.append_async("async-input", Input(())),
            writer.append_async("async-control", Control("resume", 4)),
        )
        for task in (pending, input_next, all_next):
            assert (await asyncio.wait_for(task, 2))["s"] >= 5

        # 5. 另一连接的写入由显式轮询追赶；关闭唤醒全部类型并正常结束。
        polled = log.catalog().follow(wake_on=Control, poll_interval=0.01)
        streams.append(polled)
        assert (await anext(polled))["s"] == 6
        external = MessageLog(path)
        try:
            external.writer("s", author="user", source="conversation", body_types=(Input,), content={}).append("external", Input(()))
        finally:
            external.close()
        assert (await asyncio.wait_for(anext(polled), 2))["s"] == 7
        closed = log.catalog().follow(wake_on=Control)
        streams.append(closed)
        await anext(closed)
        closed_next = asyncio.create_task(anext(closed))
        await turn()
        before = log.reader("s").snapshot()
        log.close()
        try:
            await asyncio.wait_for(closed_next, 2)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("closed log should end the stream")
    finally:
        for stream in streams:
            await stream.aclose()
        log.close()

    reopened = MessageLog(path)
    try:
        assert reopened.reader("s").snapshot() == before
        stream = reopened.catalog().follow(wake_on=Control)
        assert dict(await anext(stream)) == {"s": 7}
        await stream.aclose()
    finally:
        reopened.close()
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    print(json.dumps({"messages": len(before), "checks": ["type filter", "rollback", "subscribe during commit", "concurrent commits", "external poll", "close", "reopen equality"]}))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-message-wake-") as directory:
        asyncio.run(run(Path(directory) / "sessions.db"))
