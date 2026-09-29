"""真实 Root、客户端 adapter 与 RPC 的调度慢读和取消排空。"""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch

from agent.plugin_composition import CompositionRoot
from plugins.akashic_clients.runtime_inspection import ScopedRpcRuntimeInspection
from plugins.akashic_clients.services import RuntimeInspectionError
from plugins.runtime_inspection import plugin as inspection
from plugins.scheduler.inspection import SCHEDULER_INSPECTION, SchedulerInspectionProvider
from plugins.scheduler.schedule import ScheduledJob
from plugins.scheduler.store import JobStore


async def check(root_path: Path) -> None:
    """慢 load 不冻结 RPC loop，取消后仍保留正在读取的 provider。"""
    root = CompositionRoot("scheduler-inspection-io")
    store = JobStore(root_path / "schedules.json")
    job = ScheduledJob(id="job", tier="instant", trigger="at", channel="web", chat_id="one",
                       message="hello", timezone="UTC", fire_at=datetime.now(UTC) + timedelta(hours=1))
    await store.add("add", job, "created")

    @asynccontextmanager
    async def scope():
        async with root.context.runtime_scope():
            yield root.context

    client = ScopedRpcRuntimeInspection(scope)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = store.load

    def blocked():
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise RuntimeError("调度诊断 RPC 阻塞了事件循环")
        return original()

    async def provide(ctx):
        await ctx.provide(SCHEDULER_INSPECTION, SchedulerInspectionProvider(store))

    try:
        await root.mount(inspection.apply, name="inspection")
        try:
            await client.list_jobs()
        except RuntimeInspectionError as error:
            assert error.code == "scheduler_unavailable"
        else:
            raise AssertionError("缺少 provider 不能返回成功空列表")
        owner = await root.mount(provide, name="schedules")
        items = (await client.list_jobs())["items"]
        assert isinstance(items, list) and len(items) == 1
        assert (await client.get_job("job"))["id"] == "job"
        with patch.object(store, "load", blocked):
            request = asyncio.create_task(client.list_jobs())
            closing = None
            try:
                await asyncio.wait_for(entered.wait(), 2)
                request.cancel()
                closing = asyncio.create_task(owner.dispose())
                checkpoint = loop.create_future()
                loop.call_soon(checkpoint.set_result, None)
                await checkpoint
                assert not request.done() and not closing.done()
                release.set()
                try:
                    await request
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("RPC 取消丢失")
                await closing
            finally:
                release.set()
                await asyncio.gather(request, *(() if closing is None else (closing,)), return_exceptions=True)
    finally:
        release.set()
        await root.dispose()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-scheduler-inspection-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: v3 jobs RPC, missing provider, slow load yields and drains before unload")
