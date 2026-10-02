"""真实工具调用各持久阶段的慢提交、停止和结果完整性场景。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.plugin_composition.models import LLMResponse, ToolCall as ModelToolCall
from plugins.tools.api import Result
from session.log import OwnerTransaction
from session.message import ContentPart, Control, Input, ToolResult
from tests.support.message_react import runtime


async def check(directory: Path, phase: str, cancel: bool) -> dict:
    """阻塞实际 SQLite 事务的 commit 前阶段，不替换工具执行或回执。"""
    loop = asyncio.get_running_loop()
    reached, release = asyncio.Event(), threading.Event()
    stamps, requests, effects = [], [], []
    original_save = OwnerTransaction.save

    def save(transaction, key, value, **kwargs):
        row = original_save(transaction, key, value, **kwargs)
        if transaction._store._owner == "tools" and value.get("phase") == phase and not stamps:
            stamps.append(time.perf_counter())
            loop.call_soon_threadsafe(reached.set)
            release.wait(1.0)
        return row

    async def complete(request):
        requests.append(request)
        if len(requests) == 1:
            return LLMResponse(None, [ModelToolCall("local", "example", {})])
        return LLMResponse("finished")

    async def invoke(key, _arguments):
        with (directory / "effect.txt").open("a") as stream:
            stream.write(key + "\n")
            stream.flush()
        effects.append(key)
        return Result("success", (ContentPart("text", "effect saved"),))

    try:
        async with runtime(directory, complete, invoke, state_owner="react") as (source, log, _, run):
            await source.accept("original", Input(()))
            original = log.reader("s").snapshot()
            with patch.object(OwnerTransaction, "save", save):
                task = await source.start(run)
                await asyncio.wait_for(reached.wait(), 4)
                lag = time.perf_counter() - stamps[0]
                baseline = os.environ.get("BASELINE") == "1"
                if not baseline:
                    assert lag < 0.2, lag
                    assert not task.done
                    assert await log.reader("peer").snapshot_async(through_seq=-1) == ()
                    if cancel:
                        task.cancel()
                        task.cancel()
                        await loop.run_in_executor(None, lambda: None)
                        assert not task.done
                release.set()
                try:
                    await task.join()
                except asyncio.CancelledError:
                    assert cancel
            rows = log.reader("s").snapshot()
            assert rows[:len(original)] == original
            records = log.owner("tools").list()
            assert len(records) == 1
            record = records[0][1].value
            results = [m for m in rows if isinstance(m.body, ToolResult)]
            assert len(results) == int(record["phase"] == "done")
            if results:
                assert record["result"]["message_id"] == results[0].message_id
                assert record["result"]["seq"] == results[0].seq
            if cancel and not baseline:
                assert len(effects) == int(phase in {"started", "done"})
                assert record["phase"] == ("prepared" if phase in {"requested", "prepared"} else "done"), (phase, record["phase"])
            else:
                assert len(effects) == 1 and record["phase"] == "done"
            effect_file = directory / "effect.txt"
            assert (effect_file.read_text().splitlines() if effect_file.exists() else []) == effects
            with sqlite3.connect(directory / "sessions.db") as connection:
                assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
            return {"phase": phase, "cancel": cancel, "loop_delay_seconds": lag,
                    "effects": len(effects), "saved_phase": record["phase"], "results": len(results)}
    finally:
        release.set()


async def abandon_race(directory: Path) -> dict:
    """真实放弃先提交，迟到的成功结果只能采用其回执，不能覆盖正文。"""
    loop = asyncio.get_running_loop()
    invoked, finish_tool, committing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release = threading.Event()
    original_save = OwnerTransaction.save
    receipts = []

    def save(transaction, key, value, **kwargs):
        row = original_save(transaction, key, value, **kwargs)
        if transaction._store._owner == "tools" and value.get("phase") == "done" and not receipts:
            receipts.append(value)
            loop.call_soon_threadsafe(committing.set)
            release.wait(1.0)
        return row

    async def complete(_request):
        return LLMResponse(None, [ModelToolCall("local", "example", {})])

    async def invoke(key, _arguments):
        (directory / "effect.txt").write_text(key)
        invoked.set()
        await finish_tool.wait()
        return Result("success", (ContentPart("text", "late success"),))

    try:
        async with runtime(directory, complete, invoke, state_owner="react") as (source, log, _, run):
            await source.accept("original", Input(()))
            task = await source.start(run)
            await asyncio.wait_for(invoked.wait(), 3)
            original = log.reader("s").snapshot()
            head = original[-1].seq
            with patch.object(OwnerTransaction, "save", save):
                await source.control("abandon", Control("abandon", head), expected_head=head, handle=task.handle)
                await asyncio.wait_for(committing.wait(), 3)
                assert not task.done
                finish_tool.set()
                # 真实结果现在可竞争同一回执；SQL 第一笔仍未提交。
                assert await log.reader("peer").snapshot_async(through_seq=-1) == ()
                release.set()
                try:
                    await asyncio.wait_for(task.join(), 3)
                except asyncio.CancelledError:
                    pass
            rows = log.reader("s").snapshot()
            assert rows[:len(original)] == original
            results = [m for m in rows if isinstance(m.body, ToolResult)]
            assert len(results) == 1 and results[0].body.outcome == "interrupted"
            record = log.owner("tools").list()[0][1].value
            assert record["phase"] == "done" and record["result"]["message_id"] == results[0].message_id
            assert (directory / "effect.txt").read_text()
            return {"case": "abandon_before_late_success", "results": 1, "outcome": "interrupted"}
    finally:
        release.set()
        finish_tool.set()


async def main(directory):
    results = []
    for phase in ("requested", "prepared", "started", "done"):
        for cancel in (False, True):
            target = directory / f"{phase}-{cancel}"
            target.mkdir()
            results.append(await check(target, phase, cancel))
    if os.environ.get("BASELINE") != "1":
        target = directory / "abandon"
        target.mkdir()
        results.append(await abandon_race(target))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-tool-result-") as directory:
        results = asyncio.run(main(Path(directory)))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))
