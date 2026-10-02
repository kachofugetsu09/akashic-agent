"""真实 Source/ReAct/Models/MessageLog 在 Output 写锁下的隔离与取消证据。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.plugin_composition.models import LLMResponse
from session.log import MessageLog, MessageWriter, WriterExpired
from session.message import Input, Output
from tests.support.message_react import runtime


async def check(directory: Path, *, owner: bool, cancel: bool, stage: str) -> dict:
    """只拦住独立 SQLite 写锁；模型驱动本地返回，所有提交与退出走真实组件。"""
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    lock_requested, locked, release = (threading.Event() for _ in range(3))
    notified = asyncio.Event()
    stamps = []
    original_prepare, original_write = MessageWriter._prepare, MessageLog._write
    original_insert = MessageWriter._insert

    def blocker():
        if not lock_requested.wait(5):
            return
        with sqlite3.connect(directory / "sessions.db") as connection:
            connection.execute("BEGIN IMMEDIATE")
            locked.set()
            release.wait(1.0)  # 基线阻塞 loop 时也能独立退出。
            connection.rollback()

    thread = threading.Thread(target=blocker)
    thread.start()

    def prepare(writer, body, metadata):
        assert threading.get_ident() == loop_thread, "内容校验离开原 loop"
        prepared = original_prepare(writer, body, metadata)
        if isinstance(body, Output) and stage == "before":
            lock_requested.set()
            assert locked.wait(3)
            stamps.append(time.perf_counter())
        return prepared

    def write(log, callback):
        if locked.is_set():
            loop.call_soon_threadsafe(notified.set)
        return original_write(log, callback)

    def insert(writer, identity, prepared, head):
        message = original_insert(writer, identity, prepared, head)
        if isinstance(prepared.body, Output) and stage == "after":
            stamps.append(time.perf_counter())
            loop.call_soon_threadsafe(notified.set)
            release.wait(1.0)
        return message

    async def complete(_request):
        return LLMResponse("durable output")

    async def invoke(*_args):
        raise AssertionError("本场景没有工具调用")

    try:
        async with runtime(directory, complete, invoke, state_owner="react" if owner else None) as (source, log, models, run):
            await source.accept("original", Input(()))
            original = log.reader("s").snapshot()
            digest = hashlib.sha256(repr(original).encode()).hexdigest()
            with patch.object(MessageWriter, "_prepare", prepare), patch.object(MessageLog, "_write", write), patch.object(MessageWriter, "_insert", insert):
                task = await source.start(run)
                await asyncio.wait_for(notified.wait(), 4)
                lag = time.perf_counter() - stamps[0]
                baseline = os.environ.get("BASELINE") == "1"
                if not baseline:
                    assert lag < 0.2, lag
                    assert not task.done
                    # 无关来源的只读请求与停止控制能在锁未释放时推进。
                    assert await log.reader("peer").snapshot_async(through_seq=-1) == ()
                    if cancel:
                        task.cancel()
                        task.cancel()
                        await asyncio.get_running_loop().run_in_executor(None, lambda: None)
                        assert not task.done
                release.set()
                try:
                    await task.join()
                except BaseException as error:
                    assert cancel
                    if isinstance(error, BaseExceptionGroup):
                        assert error.subgroup(lambda e: not isinstance(e, (BaseExceptionGroup, asyncio.CancelledError, WriterExpired))) is None
                    else:
                        assert isinstance(error, (asyncio.CancelledError, WriterExpired))
            rows = log.reader("s").snapshot()
            assert rows[:len(original)] == original
            assert hashlib.sha256(repr(rows[:len(original)]).encode()).hexdigest() == digest
            outputs = [m for m in rows if isinstance(m.body, Output)]
            # 在写锁等待中撤权，事务尚未检查 grant，因此不能追加 Output。
            assert len(outputs) == (0 if stage == "before" and cancel and not baseline else 1)
            with sqlite3.connect(directory / "sessions.db") as connection:
                assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
            return {"owner": owner, "cancel": cancel, "stage": stage, "loop_delay_seconds": lag,
                    "outputs": len(outputs), "original_messages_equal": True}
    finally:
        release.set()
        lock_requested.set()
        thread.join(3)
        assert not thread.is_alive()


async def main(directory):
    results = []
    for stage in (("after",) if os.environ.get("BASELINE") == "1" else ("before", "after")):
        for owner, cancel in ((False, False), (True, False), (False, True), (True, True)):
            target = directory / f"{stage}-{owner}-{cancel}"
            target.mkdir()
            results.append(await check(target, owner=owner, cancel=cancel, stage=stage))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-react-output-") as directory:
        results = asyncio.run(main(Path(directory)))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))
