"""真实 Delivery 当前接口的消息、目的地和 cursor 提交隔离。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import threading
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.plugin_composition import CompositionRoot, PluginRuntime
from plugins.ledger.contract import BINDINGS
from plugins.ledger.bindings import Bindings
from plugins.ledger.contract import MESSAGE_CATALOG, OWNER_STATE, OwnerState
from agent.plugin_composition.tasks import TASKS, PluginTasks
from plugins.delivery.contract import (
    DELIVERY_GUARDED_START as DELIVERY,
)
from plugins.delivery import plugin
from plugins.ledger.log import MessageLog, OwnerTransaction
from plugins.ledger.contract import ContentPart, ContentReferences, Output


async def check(directory: Path, action: str, cancel: bool) -> dict:
    """在真实事务中阻塞写入，核对取消后整笔事实以及原 scope 的内容校验。"""
    log = MessageLog(directory / "sessions.db")
    root, tasks = CompositionRoot("delivery-prepare"), PluginTasks()
    loop, loop_thread = asyncio.get_running_loop(), threading.get_ident()
    reached, release = asyncio.Event(), threading.Event()
    stamps, contexts = [], []
    original_save = OwnerTransaction.save
    sink = {"name": "local", "binding_id": "local", "address": "one"}
    extra = {"name": "extra", "binding_id": "local", "address": "two"}
    for binding in ("local",):
        log.save_binding(binding, {"scenario": True})

    async def storage(ctx):
        for key, value in ((MESSAGE_CATALOG, log.catalog()), (OWNER_STATE, OwnerState(log)),
                           (TASKS, tasks), (BINDINGS, Bindings(log, root.context))):
            await ctx.provide(key, value)

    async def consumer(ctx):
        contexts.append(ctx)

    def save(tx, key, value, **kwargs):
        row = original_save(tx, key, value, **kwargs)
        selected = key.startswith("selection:") if action != "add" else key.startswith("delivery:")
        if selected and not stamps:
            stamps.append(time.perf_counter())
            loop.call_soon_threadsafe(reached.set)
            release.wait(1.0)
        return row

    def check_content(_part):
        assert threading.get_ident() == loop_thread
        return ContentReferences()

    writer = log.writer("s", author="scenario", source="scenario", body_types=(Output,), content={"text": check_content})
    original = writer.append("original", Output((ContentPart("text", "original body"),), "complete"))
    try:
        for name, apply, inject in (("storage", storage, ()), ("delivery", plugin.apply, plugin.inject),
                                    ("consumer", consumer, (DELIVERY,))):
            await root.mount(apply, name=name, inject=inject,
                runtime=PluginRuntime(name, name, directory, directory, directory, {}))
        ctx = contexts[0]
        async with ctx.runtime_scope():
            delivery = ctx.require(DELIVERY).open(ctx)
            if action == "add":
                await delivery.prepare_async(log.reader("s"), original, (sink,))

        async def operation():
            async with ctx.runtime_scope():
                delivery = ctx.require(DELIVERY).open(ctx)
                if action == "publish":
                    return await delivery.publish_async(writer, "new", Output((ContentPart("text", "new body"),), "complete"), (sink,))
                if action == "prepare":
                    return await delivery.prepare_async(log.reader("s"), original, (sink,))
                if action == "consume":
                    return await delivery.consume_async(log.reader("s"), original, (sink,))
                return await delivery.add_async("original", extra)

        with patch.object(OwnerTransaction, "save", save):
            job = asyncio.create_task(operation())
            await asyncio.wait_for(reached.wait(), 3)
            lag = time.perf_counter() - stamps[0]
            assert lag < 0.2, lag
            assert not job.done()
            assert await log.reader("peer").snapshot_async(through_seq=-1) == ()
            if cancel:
                job.cancel()
                job.cancel()
                await loop.run_in_executor(None, lambda: None)
                assert not job.done()
            release.set()
            try:
                await job
            except asyncio.CancelledError:
                assert cancel
        assert log.reader("s").get("original") == original
        identity = "new" if action == "publish" else "original"
        async with ctx.runtime_scope():
            delivery = ctx.require(DELIVERY).open(ctx)
            selected = delivery.selection(identity)
            assert selected is not None and selected.sinks == ("local",)
            assert delivery.destination(identity, "local").address == "one"
            if action == "add":
                assert delivery.destination(identity, "extra").address == "two"
            if action == "consume":
                assert delivery.cursor("s") == original.seq
        assert log.reader("s").get(identity) is not None
        with sqlite3.connect(directory / "sessions.db") as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        return {"action": action, "cancel": cancel, "loop_delay_seconds": lag, "atomic_facts": True}
    finally:
        release.set()
        await tasks.close()
        await root.dispose()
        writer.expire()
        log.close()


async def main(directory):
    results = []
    for action in ("publish", "prepare", "consume", "add"):
        for cancel in (False, True):
            path = directory / f"{action}-{cancel}"
            path.mkdir()
            results.append(await check(path, action, cancel))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-delivery-prepare-") as directory:
        results = asyncio.run(main(Path(directory)))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))
