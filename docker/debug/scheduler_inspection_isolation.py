"""真实 Root、客户端 adapter 与 RPC 的调度慢读和取消排空。"""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch

from agent.plugin_composition import CompositionRoot
from plugins.gateway.contract import RpcMethod
from plugins.runtime_inspection import plugin as inspection
from plugins.scheduler.contract import SCHEDULER_INSPECTION
from plugins.scheduler.inspection import SchedulerInspectionProvider
from plugins.scheduler.schedule import ScheduledJob
from plugins.scheduler.store import JobStore


# 直接调用 runtime_inspection 发布的定时任务 RPC；不可用时返回 {"unavailable": {code, message}}。
class _Jobs:
    def __init__(self, open_scope) -> None:
        self._open_scope = open_scope

    async def _call(self, name: str, payload: dict[str, object]) -> dict[str, object]:
        async with self._open_scope() as scope:
            with scope.borrow(RpcMethod.key(name)) as method:
                assert isinstance(method, RpcMethod), name
                return dict(await method.invoke(method.params.model_validate(payload), None))

    async def list_jobs(self) -> dict[str, object]:
        return await self._call("inspection/jobs.list", {})

    async def get_job(self, job_id: str) -> dict[str, object]:
        return await self._call("inspection/jobs.get", {"job_id": job_id})


async def check(root_path: Path) -> None:
    """慢 load 不冻结 RPC loop，取消后仍保留正在读取的 provider。"""
    root = CompositionRoot("scheduler-inspection-io")
    store = JobStore(root_path / "schedules.json")
    job = ScheduledJob(id="job", action="notify", trigger="at", channel="web", chat_id="one", session_id="web:one",
                       text="hello", timezone="UTC", fire_at=datetime.now(UTC) + timedelta(hours=1))
    await store.add("add", job, "created")

    @asynccontextmanager
    async def scope():
        async with root.context.runtime_scope():
            yield root.context

    client = _Jobs(scope)
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
        missing = await client.list_jobs()
        assert missing.get("unavailable", {}).get("code") == "scheduler_unavailable", missing
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
