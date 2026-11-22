"""测试共用的同步边界屏障：在真实同步边界暂停 worker，测试线程可确定地推进事件循环。"""
import asyncio
import threading


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
