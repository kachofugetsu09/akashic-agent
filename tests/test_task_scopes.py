import asyncio
import inspect

import pytest

from agent.plugin_composition import (
    CompositionError,
    CompositionRoot,
    Context,
    FiberState,
    PluginRuntime,
    RUNTIME_STARTED,
    RUNTIME_STARTING,
    RUNTIME_STOPPING,
)
from agent.plugin_composition.tasks import (
    TASKS, PluginTasks, StaleTask, TaskBusy, TaskServiceClosed, Tasks,
)


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
    assert started == []
    # 拒绝已完成清理，立即使用同 key 不需要额外 sleep 或 join。
    replacement = await tasks.admit("key", lambda slot: slot.start(operation))
    with pytest.raises(asyncio.CancelledError):
        await accepted[0].join()
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


@pytest.mark.asyncio
async def test_plugin_task_scope_survives_parent_exit_until_child_finishes():
    """Task captures local scope and keeps owner resources until its own exit."""

    root = CompositionRoot("task-local-scope-root")
    contexts: list[Context] = []
    cleanup_started = asyncio.Event()
    cleanup_done = asyncio.Event()
    cleanup_count = 0

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

        async def cleanup() -> None:
            nonlocal cleanup_count
            cleanup_started.set()
            cleanup_count += 1
            cleanup_done.set()

        await ctx.effect(lambda: cleanup, label="owner-resource")

    fiber = await root.mount(apply, name="task-owner")
    ctx = contexts[0]
    unrelated_events: list[str] = []
    unrelated_contexts: list[Context] = []

    async def apply_unrelated(ctx: Context) -> None:
        unrelated_contexts.append(ctx)
        await ctx.on(RUNTIME_STARTING, lambda _event: unrelated_events.append("starting"))
        await ctx.on(RUNTIME_STARTED, lambda _event: unrelated_events.append("started"))
        await ctx.on(RUNTIME_STOPPING, lambda _event: unrelated_events.append("stopping"))

    unrelated = await root.mount(apply_unrelated, name="unrelated-owner")
    unrelated_ctx = unrelated_contexts[0]
    unrelated_events_before = tuple(unrelated_events)
    tasks = Tasks()
    entered = asyncio.Event()
    parent_scope_exited = asyncio.Event()
    owner_unloading = asyncio.Event()
    derived_scope_entered = asyncio.Event()
    release = asyncio.Event()
    marker = asyncio.Event()
    order: list[str] = []

    async def operation(task):
        task.on_close(lambda: order.append("task-cleanup"))
        entered.set()
        await parent_scope_exited.wait()
        await owner_unloading.wait()
        derived_scope = ctx.capture_runtime_scope()
        async with derived_scope:
            assert ctx.fiber._fiber._call_owned_by_current_task() is not None
            order.append("derived-scope-entered")
            derived_scope_entered.set()
            await release.wait()
        order.append("operation-done")

    async with ctx.runtime_scope():
        child = await tasks.admit("key", lambda slot: slot.start(operation))
        await entered.wait()

    parent_scope_exited.set()
    dispose_task = asyncio.create_task(fiber.dispose())
    asyncio.get_running_loop().call_soon(marker.set)
    await marker.wait()
    assert fiber.state is FiberState.UNLOADING
    owner_unloading.set()
    await derived_scope_entered.wait()
    assert not cleanup_started.is_set()
    assert cleanup_count == 0
    assert order == ["derived-scope-entered"]

    unrelated_state = unrelated.state
    async with unrelated_ctx.runtime_scope():
        assert unrelated_ctx is unrelated_contexts[0]
        assert unrelated.state is unrelated_state is FiberState.ACTIVE
    assert unrelated.state is unrelated_state is FiberState.ACTIVE
    assert tuple(unrelated_events) == unrelated_events_before

    release.set()
    await child.join()
    await dispose_task
    assert order == ["derived-scope-entered", "operation-done", "task-cleanup"]
    assert cleanup_started.is_set()
    assert cleanup_done.is_set()
    assert cleanup_count == 1
    assert fiber.state is FiberState.DISPOSED
    await tasks.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_raw_child_does_not_inherit_parent_plugin_scope():
    """A raw child Task inherits values, but cannot capture the parent permit."""

    root = CompositionRoot("task-raw-child-root")
    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

    fiber = await root.mount(apply, name="task-owner")
    ctx = contexts[0]
    tasks = Tasks()
    errors: list[str | None] = []

    async def operation(_task):
        with pytest.raises(CompositionError) as excinfo:
            ctx.capture_runtime_scope()
        errors.append(excinfo.value.code)

    async def raw_child() -> None:
        task = await tasks.admit("key", lambda slot: slot.start(operation))
        await task.join()

    async with ctx.runtime_scope():
        await asyncio.create_task(raw_child())

    assert errors == ["OWNER_CALL_CONTEXT"]
    await tasks.close()
    await fiber.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_tasks_uses_root_service_for_undeclared_fiber(tmp_path):
    """A scoped Fiber may use the host Task service without declaring it."""

    root = CompositionRoot("task-root-fallback")
    plugin_tasks = PluginTasks()
    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

    await root.context.provide(TASKS, plugin_tasks)
    fiber = await root.mount(
        apply,
        name="task-owner",
        runtime=PluginRuntime(
            "task-owner",
            "task-root-fallback",
            tmp_path,
            tmp_path,
            tmp_path,
            {},
        ),
    )
    ctx = contexts[0]
    assert TASKS not in ctx._declared_dependencies()
    assert ctx.require(TASKS) is plugin_tasks
    settled: list[str] = []

    async def operation(_task):
        settled.append("done")
        return "settled"

    async with ctx.runtime_scope():
        admission = ctx.require(TASKS).open(ctx)
        task = await admission.admit("real-task", lambda slot: slot.start(operation))
    assert await task.join() == "settled"
    assert settled == ["done"]
    await plugin_tasks.close()
    await fiber.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_cancel_before_task_operation_releases_captured_scope():
    """Public cancel before first user instruction still releases local scope."""

    root = CompositionRoot("task-cancel-scope-root")
    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

    fiber = await root.mount(apply, name="task-owner")
    ctx = contexts[0]
    tasks = Tasks()
    started: list[str] = []

    async def operation(_task):
        started.append("started")

    async with ctx.runtime_scope():
        task = await tasks.admit("key", lambda slot: slot.start(operation))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task.join()

    assert started == []
    assert not ctx.fiber._fiber._in_flight_calls
    await tasks.close()
    await fiber.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_failed_admission_releases_captured_scope_before_operation():
    """Admission callback failure drains captured scope without running user work."""

    root = CompositionRoot("task-admission-failure-root")
    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

    fiber = await root.mount(apply, name="task-owner")
    ctx = contexts[0]
    tasks = Tasks()
    accepted = []
    started: list[str] = []

    async def operation(_task):
        started.append("started")

    def fail(slot):
        accepted.append(slot.start(operation))
        return operation(accepted[-1])

    async with ctx.runtime_scope():
        with pytest.raises(TypeError, match="同步"):
            await tasks.admit("key", fail)

    assert started == []
    assert not ctx.fiber._fiber._in_flight_calls
    with pytest.raises(asyncio.CancelledError):
        await accepted[0].join()
    await tasks.close()
    await fiber.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_task_factory_failure_closes_run_and_releases_local_scope():
    """Synchronous create_task failure closes _run and returns captured permit."""

    root = CompositionRoot("task-factory-failure-root")
    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

    fiber = await root.mount(apply, name="task-owner")
    ctx = contexts[0]
    tasks = Tasks()
    created_coroutines: list[object] = []
    ran: list[str] = []

    def failing_factory(_loop, coroutine, **_kwargs):
        created_coroutines.append(coroutine)
        raise RuntimeError("task factory failed")

    async def operation(_task):
        ran.append("ran")

    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    try:
        async with ctx.runtime_scope():
            loop.set_task_factory(failing_factory)
            try:
                with pytest.raises(RuntimeError, match="task factory failed"):
                    await tasks.admit("key", lambda slot: slot.start(operation))
            finally:
                loop.set_task_factory(old_factory)
    finally:
        loop.set_task_factory(old_factory)

    assert len(created_coroutines) == 1
    assert inspect.getcoroutinestate(created_coroutines[0]) == inspect.CORO_CLOSED
    assert ran == []
    assert not ctx.fiber._fiber._in_flight_calls

    async def replacement(_task):
        ran.append("replacement")

    task = await tasks.admit("key", lambda slot: slot.start(replacement))
    await task.join()
    assert ran == ["replacement"]
    await tasks.close()
    await fiber.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_task_service_distinguishes_formal_and_closed_admission():
    """关闭服务使用 typed signal，而非 formal=False 的普通拒绝。"""

    tasks = Tasks()
    await tasks.close()
    with pytest.raises(TaskServiceClosed) as closed:
        await tasks.admit("closed", lambda slot: slot)

    root = CompositionRoot("task-service-closed-signal-root")
    plugin_tasks = PluginTasks(formal=False)
    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

    fiber = None
    try:
        await root.context.provide(TASKS, plugin_tasks)
        fiber = await root.mount(apply, name="task-service-owner")
        ctx = contexts[0]
        async with ctx.runtime_scope():
            with pytest.raises(RuntimeError) as non_formal:
                plugin_tasks.open(ctx)
        assert type(non_formal.value).__name__ == "RuntimeError"
        assert str(non_formal.value) == "当前不能接纳正式 Task"
    finally:
        if fiber is not None:
            await fiber.dispose()
        await root.dispose()


@pytest.mark.asyncio
async def test_plugin_task_close_rejects_new_owner_until_old_task_drains(tmp_path):
    """关闭先拒绝新准入，旧 Task 物理排空后才能重新打开服务。"""

    root = CompositionRoot("task-service-drain-root")
    plugin_tasks = PluginTasks()
    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

    fiber = None
    task = None
    closing = None
    task_joined = False
    closing_joined = False
    release = asyncio.Event()
    main_error = None
    cleanup_errors = []
    try:
        await root.context.provide(TASKS, plugin_tasks)
        fiber = await root.mount(
            apply,
            name="task-service-owner",
            runtime=PluginRuntime(
                "task-service-owner", "task-service-drain-root", tmp_path,
                tmp_path, tmp_path, {},
            ),
        )
        ctx = contexts[0]
        started = asyncio.Event()
        closing_started = asyncio.Event()

        async def operation(_task):
            started.set()
            try:
                await asyncio.Future()
            finally:
                closing_started.set()
                await release.wait()

        async with ctx.runtime_scope():
            admission = plugin_tasks.open(ctx)
            task = await admission.admit("source", lambda slot: slot.start(operation))
        await started.wait()

        closing = asyncio.create_task(plugin_tasks.close())
        await closing_started.wait()
        async with ctx.runtime_scope():
            with pytest.raises(TaskServiceClosed, match="当前不能接纳正式 Task"):
                plugin_tasks.open(ctx)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task.join()
        task_joined = True
        await closing
        closing_joined = True

        plugin_tasks.start()
        async with ctx.runtime_scope():
            reopened = plugin_tasks.open(ctx)
        assert reopened is not admission
        await plugin_tasks.close()
    except BaseException as error:
        main_error = error
    finally:
        # Release the operation before observing any already-created waiter.
        release.set()
        if task is not None and not task_joined:
            try:
                await task.join()
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                cleanup_errors.append(error)
        if closing is not None and not closing_joined:
            try:
                await closing
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            await plugin_tasks.close()
        except BaseException as error:
            cleanup_errors.append(error)
        if fiber is not None:
            try:
                await fiber.dispose()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            await root.dispose()
        except BaseException as error:
            cleanup_errors.append(error)
    if main_error is not None:
        if cleanup_errors:
            raise BaseExceptionGroup(
                "task scope body and cleanup failed",
                [main_error, *cleanup_errors],
            )
        raise main_error
    if cleanup_errors:
        raise BaseExceptionGroup("task scope cleanup failed", cleanup_errors)
