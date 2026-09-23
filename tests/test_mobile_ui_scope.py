"""Mobile query lifecycle tests use the public provider over a real Root."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from threading import Event

import pytest

from agent.plugin_composition import (
    CompositionRoot,
    MobileUiDefinition,
    MobileUiPluginUnavailable,
    MobileUiQueryTimeout,
    MobileUiRpcExecutionError,
    PluginRuntime,
    UI_SLOTS,
)
from agent.plugins.mobile_ui import PluginMobileUiProvider
from plugins.ui import plugin as ui_plugin


def _write_assets(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "mobile.js").write_text("export const mobile = true;")
    (path / "mobile.css").write_text(".mobile {}")


async def _mount_query_owner(root: CompositionRoot, code: Path, handler, *, name="mobile"):
    state: dict[str, object] = {}

    async def apply(ctx):
        state["context"] = ctx
        state["effect"] = await ctx.require(UI_SLOTS).register_mobile(
            ctx,
            MobileUiDefinition(module="mobile.js", stylesheet="mobile.css"),
            query=handler,
        )

        async def setup_cleanup():
            async def cleanup():
                state["cleanup_count"] = int(state.get("cleanup_count", 0)) + 1

            return cleanup

        state["cleanup_count"] = 0
        state["cleanup_effect"] = await ctx.effect(
            setup_cleanup, label="test:mobile-ui-cleanup",
        )

    fiber = await root.mount(
        apply,
        name=name,
        inject=(UI_SLOTS,),
        runtime=PluginRuntime(
            plugin_id=name,
            generation_id=name + "-generation",
            plugin_dir=code,
            data_dir=code / "data",
            workspace=code,
            config={},
        ),
    )
    return fiber, state


async def _mobile_fixture(tmp_path: Path, handler):
    root = CompositionRoot("mobile-scope")
    await root.mount(ui_plugin.apply, name="ui")
    code = tmp_path / "mobile"
    _write_assets(code)
    owner, state = await _mount_query_owner(root, code, handler)
    provider = PluginMobileUiProvider(root)
    catalog = await provider.catalog()
    return root, provider, owner, state, catalog["items"][0]["revision"]


@pytest.mark.asyncio
async def test_old_query_drains_during_unloading_and_new_request_is_rejected(tmp_path: Path):
    started = Event()
    release = Event()
    finished = Event()

    def handler(method, payload, *, session_id, turn_id):
        started.set()
        assert release.wait(5)
        finished.set()
        return {"method": method, "payload": payload, "session_id": session_id, "turn_id": turn_id}

    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    old_task = asyncio.create_task(
        provider.query("mobile", revision, "old", {}, session_id=None, turn_id=None)
    )
    dispose_task = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        dispose_task = asyncio.create_task(owner.dispose())
        await state["context"]._fiber._admission_closed.wait()  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(MobileUiPluginUnavailable):
            await provider.query("mobile", revision, "new", {}, session_id=None, turn_id=None)
        assert not finished.is_set()
        assert state["cleanup_count"] == 0
        release.set()
        assert (await old_task)["method"] == "old"
        await dispose_task
        assert owner.state.value == "disposed"
        assert state["cleanup_count"] == 1
    finally:
        release.set()
        if not old_task.done():
            old_task.cancel()
        await asyncio.gather(old_task, return_exceptions=True)
        if dispose_task is not None:
            await asyncio.gather(dispose_task, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_caller_cancel_keeps_target_permit_and_physical_slot_until_thread_finishes(
    tmp_path: Path,
    monkeypatch,
):
    started = Event()
    release = Event()
    finished = Event()

    def handler(method, payload, *, session_id, turn_id):
        _ = method, payload, session_id, turn_id
        started.set()
        assert release.wait(5)
        finished.set()
        return {"ok": True}

    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    original_async_wait = asyncio.wait
    first_wait = asyncio.Event()
    second_wait = asyncio.Event()
    wait_count = 0

    async def observe_wait(awaitables, *args, **kwargs):
        nonlocal wait_count
        awaitables = tuple(awaitables)
        wait_count += 1
        if wait_count == 1:
            first_wait.set()
        elif wait_count == 2:
            second_wait.set()
        return await original_async_wait(awaitables, *args, **kwargs)

    monkeypatch.setattr(asyncio, "wait", observe_wait)
    caller = asyncio.create_task(
        provider.query("mobile", revision, "cancel", {}, session_id=None, turn_id=None)
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        await first_wait.wait()
        child = next(iter(provider._draining_queries))  # pyright: ignore[reportPrivateUsage]
        child.cancel()
        await second_wait.wait()
        assert not finished.is_set()
        assert provider._admitted_queries == 1  # pyright: ignore[reportPrivateUsage]
        assert state["context"]._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
        child.cancel()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert provider._admitted_queries == 1  # pyright: ignore[reportPrivateUsage]
        assert state["context"]._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
        assert not finished.is_set()
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await provider.aclose()
        assert provider._admitted_queries == 0  # pyright: ignore[reportPrivateUsage]
        assert not state["context"]._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
    finally:
        release.set()
        monkeypatch.setattr(asyncio, "wait", original_async_wait)
        if not caller.done():
            caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_query_timeout_reschedule_keeps_physical_query_until_settlement(
    tmp_path: Path, monkeypatch,
):
    started = Event()
    release = Event()
    finished = Event()

    def handler(method, payload, *, session_id, turn_id):
        _ = method, payload, session_id, turn_id
        started.set()
        assert release.wait(5)
        finished.set()
        return {"ok": True}

    timeout_box: list[asyncio.Timeout] = []
    original_timeout = asyncio.timeout

    def capture_timeout(delay):
        timeout = original_timeout(delay)
        timeout_box.append(timeout)
        return timeout

    monkeypatch.setattr(asyncio, "timeout", capture_timeout)
    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    query_task = asyncio.create_task(
        provider.query("mobile", revision, "timeout", {}, session_id=None, turn_id=None)
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        assert timeout_box
        timeout_box[0].reschedule(asyncio.get_running_loop().time() + 0.001)
        with pytest.raises(MobileUiQueryTimeout):
            await query_task
        assert provider._admitted_queries == 1  # pyright: ignore[reportPrivateUsage]
        assert state["context"]._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
        assert not finished.is_set()
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await provider.aclose()
        assert provider._admitted_queries == 0  # pyright: ignore[reportPrivateUsage]
        assert not state["context"]._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
        assert state["cleanup_count"] == 0
        await owner.dispose()
        assert state["cleanup_count"] == 1
    finally:
        release.set()
        await asyncio.gather(query_task, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_capture_create_prestart_and_submit_failures_release_every_owner(tmp_path: Path, monkeypatch):
    started = Event()

    def handler(method, payload, *, session_id, turn_id):
        _ = method, payload, session_id, turn_id
        started.set()
        return {"ok": True}

    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    context = state["context"]
    original_capture = context.capture_runtime_scope
    try:

        def fail_capture():
            raise RuntimeError("capture failed")

        monkeypatch.setattr(context, "capture_runtime_scope", fail_capture)
        with pytest.raises(RuntimeError, match="capture failed"):
            await provider.query("mobile", revision, "capture", {}, session_id=None, turn_id=None)
        assert provider._admitted_queries == 0  # pyright: ignore[reportPrivateUsage]
        assert not state["context"]._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
    finally:
        monkeypatch.setattr(context, "capture_runtime_scope", original_capture)
        await provider.aclose()
        await root.dispose()

    # Rebuild a real provider for the create/pre-start/submit seams.
    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    original_create_task = asyncio.create_task
    try:
        created: list[object] = []

        def fail_create_task(coroutine, *args, **kwargs):
            created.append(coroutine)
            raise RuntimeError("create task failed")

        monkeypatch.setattr(asyncio, "create_task", fail_create_task)
        with pytest.raises(RuntimeError, match="create task failed"):
            await provider.query("mobile", revision, "create", {}, session_id=None, turn_id=None)
        assert created
        assert inspect.getcoroutinestate(created[0]) is inspect.CORO_CLOSED
        assert not started.is_set()
        assert provider._admitted_queries == 0  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(asyncio, "create_task", original_create_task)

        original_publish = provider._publish_query

        async def cancel_before_first_line(coroutine):
            task = await original_publish(coroutine)
            task.cancel()
            return task

        monkeypatch.setattr(provider, "_publish_query", cancel_before_first_line)
        with pytest.raises(asyncio.CancelledError):
            await provider.query("mobile", revision, "prestart", {}, session_id=None, turn_id=None)
        assert provider._admitted_queries == 0  # pyright: ignore[reportPrivateUsage]
        assert not state["context"]._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
    finally:
        monkeypatch.setattr(asyncio, "create_task", original_create_task)
        await provider.aclose()
        await root.dispose()

    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    try:
        def fail_submit(*args, **kwargs):
            raise RuntimeError("submit failed")

        monkeypatch.setattr(provider._executor, "submit", fail_submit)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(MobileUiRpcExecutionError, match="执行失败"):
            await provider.query("mobile", revision, "submit", {}, session_id=None, turn_id=None)
        assert provider._admitted_queries == 0  # pyright: ignore[reportPrivateUsage]
        assert not state["context"]._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
    finally:
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_provider_close_rejects_new_admission_waits_physical_work_and_retries(
    tmp_path: Path, monkeypatch,
):
    started = Event()
    release = Event()

    def handler(method, payload, *, session_id, turn_id):
        _ = method, payload, session_id, turn_id
        started.set()
        assert release.wait(5)
        return {"ok": True}

    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    query_task = asyncio.create_task(
        provider.query("mobile", revision, "close", {}, session_id=None, turn_id=None)
    )
    close_task = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        admission_started = asyncio.Event()
        original_provider_wait = provider._wait_for_queries

        async def observe_wait():
            admission_started.set()
            await original_provider_wait()

        monkeypatch.setattr(provider, "_wait_for_queries", observe_wait)
        close_task = asyncio.create_task(provider.aclose())
        await admission_started.wait()
        with pytest.raises(MobileUiPluginUnavailable):
            await provider.query("mobile", revision, "late", {}, session_id=None, turn_id=None)
        assert not close_task.done()
        release.set()
        await query_task
        await close_task
    finally:
        release.set()
        await asyncio.gather(query_task, return_exceptions=True)
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_provider_close_accumulates_cancellation_until_worker_and_shutdown_finish(
    tmp_path: Path, monkeypatch,
):
    started = Event()
    release = Event()
    finished = Event()
    shutdown_started = Event()
    shutdown_release = Event()

    def handler(method, payload, *, session_id, turn_id):
        _ = method, payload, session_id, turn_id
        started.set()
        assert release.wait(5)
        finished.set()
        return {"ok": True}

    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    query_task = asyncio.create_task(
        provider.query("mobile", revision, "close-cancel", {}, session_id=None, turn_id=None)
    )
    close_task = None
    original_async_wait_close = asyncio.wait
    drain_second_wait = asyncio.Event()
    shutdown_second_wait = asyncio.Event()
    drain_waits = 0
    shutdown_waits = 0

    async def observe_wait(awaitables, *args, **kwargs):
        nonlocal drain_waits, shutdown_waits
        awaitables = tuple(awaitables)
        phase = None
        for awaitable in awaitables:
            get_coro = getattr(awaitable, "get_coro", None)
            if get_coro is None:
                continue
            coroutine = get_coro()
            name = getattr(coroutine, "__name__", "")
            if name == "observe_wait":
                phase = "drain"
                break
            if name == "to_thread":
                phase = "shutdown"
                break
        if phase == "drain":
            drain_waits += 1
            if drain_waits == 2:
                drain_second_wait.set()
        elif phase == "shutdown":
            shutdown_waits += 1
            if shutdown_waits == 2:
                shutdown_second_wait.set()
        return await original_async_wait_close(awaitables, *args, **kwargs)

    monkeypatch.setattr(asyncio, "wait", observe_wait)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        admission_started = asyncio.Event()
        original_provider_wait_close = provider._wait_for_queries

        async def observe_wait():
            admission_started.set()
            await original_provider_wait_close()

        monkeypatch.setattr(provider, "_wait_for_queries", observe_wait)
        original_shutdown = provider._executor.shutdown  # pyright: ignore[reportPrivateUsage]

        def blocked_shutdown(*args, **kwargs):
            shutdown_started.set()
            assert shutdown_release.wait(5)
            return original_shutdown(*args, **kwargs)

        monkeypatch.setattr(provider._executor, "shutdown", blocked_shutdown)  # pyright: ignore[reportPrivateUsage]
        close_task = asyncio.create_task(provider.aclose())
        await admission_started.wait()
        close_task.cancel()
        await drain_second_wait.wait()
        assert not close_task.done()
        assert not finished.is_set()
        close_task.cancel()
        release.set()
        await query_task
        assert await asyncio.to_thread(shutdown_started.wait, 5)
        close_task.cancel()
        await shutdown_second_wait.wait()
        assert not close_task.done()
        close_task.cancel()
        shutdown_release.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert provider._executor_closed  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(MobileUiPluginUnavailable):
            await provider.query("mobile", revision, "late", {}, session_id=None, turn_id=None)
    finally:
        release.set()
        shutdown_release.set()
        monkeypatch.setattr(asyncio, "wait", original_async_wait_close)
        await asyncio.gather(query_task, return_exceptions=True)
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_cancelled_drain_and_shutdown_failure_preserve_both_errors(
    tmp_path: Path, monkeypatch,
):
    started = Event()
    release = Event()

    def handler(method, payload, *, session_id, turn_id):
        _ = method, payload, session_id, turn_id
        started.set()
        assert release.wait(5)
        return {"ok": True}

    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    root._defer_internal_cleanup(  # pyright: ignore[reportPrivateUsage]
        "test:mobile-ui-provider-combined", provider.aclose,
    )
    query_task = asyncio.create_task(
        provider.query("mobile", revision, "combined", {}, session_id=None, turn_id=None)
    )
    close_task = None
    close_started = asyncio.Event()
    drain_second_wait = asyncio.Event()
    drain_waits = 0
    original_async_wait = asyncio.wait
    original_provider_wait = provider._wait_for_queries
    original_shutdown = provider._executor.shutdown  # pyright: ignore[reportPrivateUsage]
    shutdown_failed = True

    async def observe_async_wait(awaitables, *args, **kwargs):
        nonlocal drain_waits
        awaitables = tuple(awaitables)
        is_close_drain = any(
            getattr(getattr(awaitable, "get_coro", lambda: None)(), "__name__", "")
            == "provider_wait"
            for awaitable in awaitables
        )
        if is_close_drain:
            drain_waits += 1
            if drain_waits == 2:
                drain_second_wait.set()
        return await original_async_wait(awaitables, *args, **kwargs)

    async def provider_wait():
        close_started.set()
        await original_provider_wait()

    def fail_shutdown_once(*args, **kwargs):
        nonlocal shutdown_failed
        if shutdown_failed:
            shutdown_failed = False
            raise RuntimeError("combined shutdown failed")
        return original_shutdown(*args, **kwargs)

    monkeypatch.setattr(asyncio, "wait", observe_async_wait)
    monkeypatch.setattr(provider, "_wait_for_queries", provider_wait)
    monkeypatch.setattr(provider._executor, "shutdown", fail_shutdown_once)  # pyright: ignore[reportPrivateUsage]
    try:
        assert await asyncio.to_thread(started.wait, 5)
        close_task = asyncio.create_task(provider.aclose())
        await close_started.wait()
        close_task.cancel()
        await drain_second_wait.wait()
        assert not close_task.done()
        release.set()
        await query_task
        with pytest.raises(BaseExceptionGroup) as caught:
            await close_task
        error_group = caught.value
        assert error_group.subgroup(asyncio.CancelledError) is not None
        assert error_group.subgroup(RuntimeError) is not None
        assert not provider._executor_closed  # pyright: ignore[reportPrivateUsage]
        assert any(
            resource == "test:mobile-ui-provider-combined"
            for resource, _cleanup in root._internal_cleanups  # pyright: ignore[reportPrivateUsage]
        )
        monkeypatch.setattr(asyncio, "wait", original_async_wait)
        monkeypatch.setattr(provider._executor, "shutdown", original_shutdown)  # pyright: ignore[reportPrivateUsage]
        await root.dispose()
        assert provider._executor_closed  # pyright: ignore[reportPrivateUsage]
        assert not root._internal_cleanups  # pyright: ignore[reportPrivateUsage]
    finally:
        release.set()
        monkeypatch.setattr(asyncio, "wait", original_async_wait)
        monkeypatch.setattr(provider._executor, "shutdown", original_shutdown)  # pyright: ignore[reportPrivateUsage]
        await asyncio.gather(query_task, return_exceptions=True)
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
async def test_root_internal_mobile_ui_cleanup_retries_same_provider_after_failure(
    tmp_path: Path, monkeypatch,
):
    def handler(method, payload, *, session_id, turn_id):
        _ = method, payload, session_id, turn_id
        return {"ok": True}

    root, provider, owner, state, revision = await _mobile_fixture(tmp_path, handler)
    original_shutdown = provider._executor.shutdown  # pyright: ignore[reportPrivateUsage]
    failed = True

    def fail_once(*args, **kwargs):
        nonlocal failed
        if failed:
            failed = False
            raise RuntimeError("shutdown failed")
        return original_shutdown(*args, **kwargs)

    monkeypatch.setattr(provider._executor, "shutdown", fail_once)  # pyright: ignore[reportPrivateUsage]
    root._defer_internal_cleanup("test:mobile-ui-provider", provider.aclose)  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(RuntimeError, match="shutdown failed"):
            await root.dispose()
        assert not provider._executor_closed  # pyright: ignore[reportPrivateUsage]
        assert any(
            resource == "test:mobile-ui-provider"
            for resource, _cleanup in root._internal_cleanups  # pyright: ignore[reportPrivateUsage]
        )
        await root.dispose()
        assert provider._executor_closed  # pyright: ignore[reportPrivateUsage]
        assert not root._internal_cleanups  # pyright: ignore[reportPrivateUsage]
    finally:
        monkeypatch.setattr(provider._executor, "shutdown", original_shutdown)  # pyright: ignore[reportPrivateUsage]
        await root.dispose()
