"""两个 generation 的真实调度文件提交与取消排空。"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from typing import cast
from unittest.mock import patch

from agent.plugin_composition.tasks import TaskAdmission
from plugins.scheduler.schedule import ScheduledJob
from plugins.scheduler import store as store_module
from plugins.scheduler.store import JobStore
from plugins.scheduler.tools import ScheduleTool


async def check(root: Path) -> None:
    """取消调用方不丢已开始的提交；后代写入必须重读前代事实。"""
    first, second = JobStore(root / "schedules.json"), JobStore(root / "schedules.json")
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = store_module.atomic_save_json

    def blocked(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise RuntimeError("Scheduler 写文件阻塞了事件循环")
        return original(*args, **kwargs)

    async def add(store, key):
        job = ScheduledJob(id=key, trigger="at", tier="instant", channel="web", chat_id="one",
                           message=key, timezone="UTC", fire_at=datetime.now(UTC) + timedelta(hours=1))
        tool = ScheduleTool(store, cast(TaskAdmission, None), "schedule")
        return await tool.invoke(key, {"job": store.encode_job(job), "response": key})

    with patch.object(store_module, "atomic_save_json", blocked):
        writer = asyncio.create_task(add(first, "first"))
        next_writer = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            writer.cancel()
            next_writer = asyncio.create_task(add(second, "second"))
            checkpoint = loop.create_future()
            loop.call_soon(checkpoint.set_result, None)
            await checkpoint
            assert not writer.done() and not next_writer.done()
            release.set()
            try:
                await writer
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("写入取消丢失")
            await next_writer
            state = second.read()
            assert set(state.jobs) == set(state.operations) == {"first", "second"}
            assert state.operations["first"].response == "first"
        finally:
            release.set()
            await asyncio.gather(writer, *(() if next_writer is None else (next_writer,)), return_exceptions=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-scheduler-io-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: cancellation drains commit; next generation preserves both jobs and receipts")
