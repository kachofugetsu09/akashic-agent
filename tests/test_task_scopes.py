import asyncio

import pytest

from agent.plugin_composition.tasks import StaleTask, TaskBusy, Tasks


@pytest.mark.asyncio
async def test_cancel_revokes_writer_before_started_work_drains_and_fences_old_handle():
    tasks = Tasks()
    started = asyncio.Event()
    drain = asyncio.Event()
    cancelled = asyncio.Event()
    resources = []

    async def operation(scope):
        scope.on_close(lambda: resources.append("revoked"))
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            await drain.wait()
        return "settled"

    old = await tasks.admit("key", lambda slot: slot.start(operation))
    await started.wait()
    await tasks.admit("key", lambda slot: slot.require(old.handle).cancel())
    assert resources == ["revoked"]
    await cancelled.wait()
    old.cancel()  # 重复取消不能再次打断已经开始的结算。
    with pytest.raises(TaskBusy):
        await tasks.admit("key", lambda slot: slot.start(operation))
    drain.set()
    assert await old.join() == "settled"
    new = await tasks.admit("key", lambda slot: slot.start(operation))
    assert new.handle != old.handle
    with pytest.raises(StaleTask):
        await tasks.admit("key", lambda slot: slot.require(old.handle).cancel())
    assert new.active
    await tasks.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_failed_admission_cannot_leave_a_new_task_running(eager):
    tasks = Tasks()
    started = []
    accepted = []

    async def operation(scope):
        started.append(scope.handle)

    def fail(slot):
        accepted.append(slot.start(operation))
        return operation(accepted[-1])

    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        with pytest.raises(TypeError, match="同步"):
            await tasks.admit("key", fail)
    finally:
        loop.set_task_factory(old_factory)
    with pytest.raises(asyncio.CancelledError):
        await accepted[0].join()
    assert started == []
    replacement = await tasks.admit("key", lambda slot: slot.start(operation))
    await replacement.join()
    assert started == [replacement.handle]
    await tasks.close()


@pytest.mark.asyncio
async def test_eager_factory_cannot_reenter_before_slot_registration():
    tasks = Tasks()
    admitted = []

    async def operation(scope):
        current = await tasks.admit("key", lambda slot: slot.require(scope.handle))
        assert current is scope
        with pytest.raises(TaskBusy):
            await tasks.admit("key", lambda slot: slot.start(operation))
        admitted.append(scope.handle)

    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        task = await tasks.admit("key", lambda slot: slot.start(operation))
        await task.join()
    finally:
        loop.set_task_factory(old_factory)
        await tasks.close()
    assert admitted == [task.handle]


@pytest.mark.asyncio
async def test_stale_admission_and_async_callbacks_cannot_start_late_work():
    tasks = Tasks()
    slot = await tasks.admit("key", lambda slot: slot)
    with pytest.raises(RuntimeError, match="准入已结束"):
        slot.start(lambda _: asyncio.sleep(0))
    started = []

    async def admission(slot):
        started.append(True)

    with pytest.raises(TypeError, match="同步"):
        await tasks.admit("key", admission)
    assert started == []
    await tasks.close()


@pytest.mark.asyncio
async def test_effect_start_and_control_share_the_same_admission_order():
    tasks = Tasks()
    ready = asyncio.Event()
    proceed = asyncio.Event()
    effects = []

    async def operation(scope):
        ready.set()
        try:
            await proceed.wait()
        except asyncio.CancelledError:
            pass
        await tasks.admit(
            "source",
            lambda slot: (slot.require(scope.handle), effects.append("started")),
        )

    task = await tasks.admit("source", lambda slot: slot.start(operation))
    await ready.wait()
    await tasks.admit("source", lambda slot: slot.require(task.handle).cancel())
    proceed.set()
    with pytest.raises(StaleTask):
        await task.join()
    assert effects == []
    await tasks.close()
