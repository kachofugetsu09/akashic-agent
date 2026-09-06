import asyncio

import pytest

from agent.plugin_composition.tasks import Tasks


@pytest.mark.asyncio
async def test_activity_blocks_idle_work_but_allows_short_work_and_preserves_fifo():
    tasks = Tasks()
    order = []
    one, two, cancelled, short = (asyncio.Event() for _ in range(4))

    async def work(label, started, *, idle):
        started.set()
        async with tasks.exclusive("key", idle=idle):
            order.append(label)

    try:
        with tasks.activity("key"):
            first = asyncio.create_task(work("one", one, idle=True))
            await one.wait()
            removed = asyncio.create_task(work("cancelled", cancelled, idle=True))
            await cancelled.wait()
            last = asyncio.create_task(work("two", two, idle=True))
            await two.wait()
            removed.cancel()
            with pytest.raises(asyncio.CancelledError):
                await removed
            urgent = asyncio.create_task(work("short", short, idle=False))
            await urgent
            assert order == ["short"]
            assert not first.done() and not last.done()
        await asyncio.gather(first, last)
        assert order == ["short", "one", "two"]
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_new_activity_during_exclusive_blocks_waiting_idle_work():
    tasks = Tasks()
    queued, active, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    entered = []

    async def background():
        queued.set()
        async with tasks.exclusive("key", idle=True):
            entered.append("background")

    async def foreground():
        with tasks.activity("key"):
            active.set()
            await release.wait()

    try:
        async with tasks.exclusive("key"):
            blocked = asyncio.create_task(background())
            await queued.wait()
            human = asyncio.create_task(foreground())
            await active.wait()
        # 短排他权已归还，但新活动仍阻止背景工作开始。
        async with tasks.exclusive("key"):
            assert entered == []
        release.set()
        await asyncio.gather(human, blocked)
        assert entered == ["background"]
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_idle_waiter_observes_new_activity_on_the_same_key():
    tasks = Tasks()
    waiting, first_active, first_done, second_active, second_done = (asyncio.Event() for _ in range(5))
    order = []

    async def wait():
        waiting.set()
        await tasks.wait_idle("key")
        order.append("idle")

    async def first():
        with tasks.activity("key"):
            first_active.set()
            await first_done.wait()
        # 在 waiter 重新取得锁前立即进入第二段活动，不能丢失原 key 的状态。
        with tasks.activity("key"):
            second_active.set()
            await second_done.wait()

    active = asyncio.create_task(first())
    await first_active.wait()
    waiter = asyncio.create_task(wait())
    await waiting.wait()
    first_done.set()
    await second_active.wait()
    assert order == []
    second_done.set()
    await asyncio.gather(active, waiter)
    assert order == ["idle"]
    await tasks.close()


@pytest.mark.asyncio
async def test_close_wakes_queued_work_without_revoking_existing_resource_cleanup():
    tasks = Tasks()
    started = asyncio.Event()

    async def queued():
        started.set()
        async with tasks.exclusive("key", idle=True):
            pytest.fail("closed queue admitted work")

    with tasks.activity("key"):
        pending = asyncio.create_task(queued())
        await started.wait()
        closing = asyncio.create_task(tasks.close())
        with pytest.raises(RuntimeError, match="关闭"):
            await pending
        assert not closing.done()
    await closing
    with pytest.raises(RuntimeError, match="关闭"):
        await tasks.wait_idle("key")
