import asyncio
from collections.abc import Mapping
from contextlib import aclosing, asynccontextmanager, closing, suppress
from typing import cast

import pytest
from fastapi.testclient import TestClient

from agent.config_models import Config
from agent.plugin_composition import CompositionRoot, Context, FiberState, RUNTIME_STARTED, ServiceKey
from agent.plugin_composition.tasks import Tasks
from bootstrap.app_server import build_control_service
from bootstrap import tools as bootstrap
from bootstrap.reply_status import RuntimeReplyStatus
from core.net.http import SharedHttpResources
from agent.plugin_composition.message_view import follow_messages, message_rows
from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.akashic_clients.services import MessageCatalogPort
from plugins.reply.status import REPLY_STATUS, ReplyState
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input, Control, Output
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def page_data(value: object) -> Mapping[str, object]:
    """Narrow one follow frame before inspecting its JSON fields."""
    assert isinstance(value, Mapping)
    return value


def page_items(page: Mapping[str, object]) -> list[Mapping[str, object]]:
    items = page.get("items")
    assert isinstance(items, list)
    assert all(isinstance(item, Mapping) for item in items)
    return items


def _catalog_port(log: MessageLog) -> MessageCatalogPort:
    return cast(MessageCatalogPort, log.catalog())


@pytest.mark.asyncio
async def test_follow_pages_fixed_prefix_then_reconnect_from_last_seq(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        writer = log.writer('s', author='u', source='conversation', body_types=(Input, Control),
                            content={'text': lambda p: ContentReferences()})
        for i in range(106):
            writer.append(str(i), Input((ContentPart('text', str(i)),)))
        before = log.reader('s').snapshot()
        async with aclosing(follow_messages(log.reader('s'), after_seq=2)) as stream:
            first = page_data(await anext(stream))
            writer.append('late', Control('pause', 105, 'reason'))
            second, third, fourth = (page_data(await anext(stream)), page_data(await anext(stream)), page_data(await anext(stream)))
            assert [p['through_seq'] for p in (first, second, third, fourth)] == [105, 105, 105, 106]
            rows = [row for page in (first, second, third, fourth) for row in page_items(page)]
            assert rows == message_rows(log.reader('s').read_page(after_seq=2, limit=200))
            assert [p['next_after_seq'] for p in (first, second, third, fourth)] == [52, 102, 105, 106]
        assert log.reader('s').snapshot()[:-1] == before
        async with aclosing(follow_messages(log.reader('s'), after_seq=106)) as stream:
            pending = asyncio.create_task(anext(stream))
            writer.append('reconnected', Input(()))
            page = page_data(await asyncio.wait_for(pending, 3))
            assert [row['id'] for row in page_items(page)] == ['reconnected']
        assert not log._listeners


async def mount_status(root, state, name='reply'):
    async def plugin(ctx):
        await ctx.provide(REPLY_STATUS, state.read)
    return await root.context.mount(plugin, name=name)


async def status_root(name, state):
    root = CompositionRoot(name)
    fiber = None if state is None else await mount_status(root, state)
    return root, fiber


@asynccontextmanager
async def real_control_runtime_before_start(tmp_path, monkeypatch):
    """Build the real empty CoreRuntime, then add only the status provider."""
    workspace = tmp_path / "control-workspace"
    initialize_plugin_workspace(workspace)
    isolated_plugin_home = tmp_path / "isolated-plugin-home"
    isolated_plugin_home.mkdir()
    monkeypatch.setattr(bootstrap, "plugins_root", lambda: isolated_plugin_home)
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http, plugin_dirs=[])
    service = build_control_service(core)
    provider = None
    try:
        await core.start()
        root = core.plugin_manager.live_root
        assert root is not None
        state = ReplyState()

        async def provider_plugin(ctx):
            await ctx.provide(REPLY_STATUS, state.read)
            await ctx.effect(lambda: state.close, label="reply-status-owner-close")

        provider = await root.context.mount(
            provider_plugin, name="reply-status-control-provider",
        )
        yield core, service, provider
    finally:
        await service.shutdown()
        await core.bus.aclose()
        await core.stop()
        await http.aclose()


class _FailingReader:
    """Raise at one concrete reader lifecycle boundary."""

    def __init__(self, mode: str):
        self.mode = mode
        self.boundary_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.error_raised = asyncio.Event()

    def follow(self, _session_id):
        if self.mode == "acquire":
            raise RuntimeError("reader acquisition failed")
        return _FailingStream(self)


class _FailingStream:
    def __init__(self, reader: _FailingReader):
        self.reader = reader
        self.yielded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.yielded:
            self.yielded = True
            return ()
        self.reader.boundary_entered.set()
        await self.reader.release.wait()
        if self.reader.mode == "iteration":
            self.reader.error_raised.set()
            raise RuntimeError("reader iteration failed")
        raise StopAsyncIteration

    async def aclose(self):
        if self.reader.mode == "close":
            self.reader.boundary_entered.set()
            await self.reader.release.wait()
            self.reader.error_raised.set()
            raise RuntimeError("reader close failed")


class _NormalEndReader:
    """Expose a controlled normal-end wait and its close boundary."""

    def __init__(self):
        self.allow_end = asyncio.Event()
        self.yielded = False
        self.ended = asyncio.Event()
        self.closed = asyncio.Event()

    def follow(self, _session_id):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.yielded:
            self.yielded = True
            return ()
        await self.allow_end.wait()
        self.ended.set()
        raise StopAsyncIteration

    async def aclose(self):
        self.closed.set()


@pytest.mark.asyncio
async def test_status_switch_and_absence_never_pin_old_generation():
    old, new = ReplyState(), ReplyState()
    root, _ = await status_root('reply-status', None)
    tasks = Tasks()
    entered, release = asyncio.Event(), asyncio.Event()
    async def operation(task):
        with old.open(task, 's', 'conversation') as preview:
            with preview('draft') as delta:
                await delta({'content_delta': 'old preview'})
                entered.set()
                await release.wait()
    task = await tasks.admit('s', lambda slot: slot.start(operation))
    await asyncio.wait_for(entered.wait(), 3)
    try:
        async with aclosing(RuntimeReplyStatus(root).follow('s')) as stream:
            assert not (await anext(stream))['available']
            old_fiber = await mount_status(root, old, name='reply-old')
            frame = page_data(await anext(stream))
            items = page_items(frame)
            preview = items[0].get('preview')
            assert isinstance(preview, Mapping)
            assert preview.get('text') == 'old preview'
            await old_fiber.dispose()
            frame = page_data(await asyncio.wait_for(anext(stream), 3))
            assert not frame['available'] and frame['items'] == []
            release.set()
            await task.join()
            new_fiber = await mount_status(root, new, name='reply-new')
            frame = page_data(await asyncio.wait_for(anext(stream), 3))
            assert frame['available'] and frame['items'] == []
            assert frame['snapshot_id'] is not None
            await new_fiber.dispose()
    finally:
        release.set()
        await task.join()
        await tasks.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_closed_status_and_cancelled_wait_release_subscriptions():
    state = ReplyState()
    root, _ = await status_root('closed', state)
    stream = RuntimeReplyStatus(root).follow('s')
    try:
        assert (await anext(stream))['available']
        state.close()
        closed_frame = await asyncio.wait_for(anext(stream), 3)
        assert not closed_frame['available'] and closed_frame['items'] == []
        pending = asyncio.create_task(anext(stream))
        asyncio.get_running_loop().call_soon(pending.cancel)
        asyncio.get_running_loop().call_soon(pending.cancel)
        with suppress(asyncio.CancelledError):
            await pending
        await stream.aclose()
    finally:
        await stream.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_status_normal_end_wait_and_repeated_cancel_settle_pump():
    """Normal-end waiting is entered before cancellation and closes its reader."""
    root = CompositionRoot("reply-status-normal-end")
    reader = _NormalEndReader()

    async def provider(ctx):
        await ctx.provide(REPLY_STATUS, reader)

    fiber = await root.context.mount(provider, name="reply-status-normal-end-provider")
    stream = RuntimeReplyStatus(root).follow("normal-end")
    pending = None
    try:
        assert (await anext(stream))["available"]
        reader.allow_end.set()
        await asyncio.wait_for(reader.ended.wait(), 3)
        unavailable = await asyncio.wait_for(anext(stream), 3)
        assert not unavailable["available"] and unavailable["items"] == []
        pending = asyncio.create_task(anext(stream), name="reply-status-normal-end-wait")
        asyncio.get_running_loop().call_soon(pending.cancel)
        asyncio.get_running_loop().call_soon(pending.cancel)
        with suppress(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(reader.closed.wait(), 3)
        assert not any(fiber.name.startswith("reply-status:") for fiber in root._fibers.values())
        await fiber.dispose()
    finally:
        if pending is not None:
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
        await stream.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_build_control_service_reply_status_uses_live_root_not_snapshot(tmp_path, monkeypatch):
    """Exercise public ControlService.follow on a real CoreRuntime/Manager."""
    async with real_control_runtime_before_start(tmp_path, monkeypatch) as (core, service, provider):
        manager = core.plugin_manager
        root = manager.live_root
        assert root is not None

        async def reject_snapshot_entry(*args, **kwargs):
            raise AssertionError("reply follow must not acquire the old snapshot")

        def reject_snapshot_compile(*args, **kwargs):
            raise AssertionError("reply follow must not compile the old snapshot")

        # 真实 service 已在 core.start 前构造；负控覆盖 cold-start 后仍不能走旧入口。
        monkeypatch.setattr(manager.snapshot_store, "acquire", reject_snapshot_entry)
        monkeypatch.setattr(
            manager.snapshot_store, "wait_for_stable_change", reject_snapshot_entry,
        )
        monkeypatch.setattr(manager._snapshot_compiler, "compile", reject_snapshot_compile)
        async with aclosing(service.follow("s", -1)) as stream:
            frame = await anext(stream)
            if frame["type"] != "reply.status":
                frame = await anext(stream)
            assert frame["type"] == "reply.status"
            assert frame["available"] and frame["snapshot_id"] is not None
        assert provider.state is FiberState.ACTIVE


@pytest.mark.asyncio
async def test_status_subscriptions_are_independent_and_root_close_wakes_pending():
    root, provider = await status_root('reply-status-lifecycle', ReplyState())
    first = RuntimeReplyStatus(root).follow('first')
    second = RuntimeReplyStatus(root).follow('second')
    try:
        assert (await anext(first))['available']
        assert (await anext(second))['available']
        await first.aclose()
        assert provider is not None
        await provider.dispose()
        unavailable = await asyncio.wait_for(anext(second), 3)
        assert not unavailable['available'] and unavailable['items'] == []
        await second.aclose()

        pending_root, _ = await status_root('reply-status-pending', None)
        pending_stream = RuntimeReplyStatus(pending_root).follow('pending')
        assert not (await anext(pending_stream))['available']
        cancelled = asyncio.create_task(anext(pending_stream))
        assert any(
            fiber.name.startswith("reply-status:")
            and fiber.state is FiberState.PENDING
            for fiber in pending_root._fibers.values()
        )
        asyncio.get_running_loop().call_soon(cancelled.cancel)
        asyncio.get_running_loop().call_soon(cancelled.cancel)
        with suppress(asyncio.CancelledError):
            await cancelled
        await pending_stream.aclose()
        assert not any(
            fiber.name.startswith("reply-status:")
            for fiber in pending_root._fibers.values()
        )
        root_pending_stream = RuntimeReplyStatus(pending_root).follow('pending-root-close')
        assert not (await anext(root_pending_stream))['available']
        pending = asyncio.create_task(anext(root_pending_stream))
        assert any(
            fiber.name.startswith("reply-status:")
            and fiber.state is FiberState.PENDING
            for fiber in pending_root._fibers.values()
        )
        await pending_root.dispose()
        with suppress(StopAsyncIteration):
            await pending
        assert not any(
            fiber.name.startswith("reply-status:")
            for fiber in pending_root._fibers.values()
        )
        await root_pending_stream.aclose()
    finally:
        await first.aclose()
        await second.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_status_loading_activation_delivers_after_provider_started_gate():
    """Keep LOADING unavailable until the provider activation really becomes ACTIVE."""
    root = CompositionRoot("reply-status-loading")
    state = ReplyState()
    started, release = asyncio.Event(), asyncio.Event()

    async def provider(ctx):
        await ctx.provide(REPLY_STATUS, state.read)

        async def block(_payload):
            started.set()
            await release.wait()

        await ctx.on(RUNTIME_STARTED, block)

    mount_task = asyncio.create_task(root.mount(provider, name="reply-loading"))
    stream = RuntimeReplyStatus(root).follow("loading")
    try:
        await started.wait()
        assert not (await anext(stream))["available"]
        release.set()
        provider_fiber = await mount_task
        frame = await asyncio.wait_for(anext(stream), 3)
        assert frame["available"] and frame["items"] == []
        await provider_fiber.dispose()
    finally:
        release.set()
        if not mount_task.done():
            await mount_task
        await stream.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_status_identity_is_frozen_per_provider_registration():
    """Only the reply provider registration changes the visible snapshot identity."""
    root, provider = await status_root("reply-status-identity", ReplyState())
    first = RuntimeReplyStatus(root).follow("first")
    second = RuntimeReplyStatus(root).follow("second")
    third = None
    unrelated_key = ServiceKey[object]("reply-status-unrelated")
    try:
        first_frame = await anext(first)
        second_frame = await anext(second)
        old_id = first_frame["snapshot_id"]
        assert old_id == second_frame["snapshot_id"]

        async def unrelated(ctx):
            await ctx.provide(unrelated_key, object())

        unrelated_fiber = await root.mount(unrelated, name="reply-status-unrelated")
        third = RuntimeReplyStatus(root).follow("third")
        assert (await anext(third))["snapshot_id"] == old_id
        await third.aclose()
        await unrelated_fiber.dispose()

        assert provider is not None
        await provider.dispose()
        assert not (await anext(first))["available"]
        assert not (await anext(second))["available"]
        replacement = await mount_status(root, ReplyState(), name="reply-status-replacement")
        new_first = await anext(first)
        new_second = await anext(second)
        assert new_first["available"] and new_second["available"]
        assert new_first["snapshot_id"] == new_second["snapshot_id"] != old_id
        await replacement.dispose()
    finally:
        await first.aclose()
        await second.aclose()
        if third is not None:
            await third.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_status_same_fiber_reactivation_changes_registration_identity():
    """A hard dependency replacement reactivates one Fiber with new identity."""
    root = CompositionRoot("reply-status-reactivation")
    first_state, second_state = ReplyState(), ReplyState()
    dependency_key = ServiceKey[object]("reply-status-reactivation-dependency")
    provider_context: Context | None = None
    dependency_value = None
    activation_contexts = []

    async def provider_plugin(ctx):
        nonlocal provider_context, dependency_value
        provider_context = ctx
        activation_contexts.append(ctx)
        dependency_value = ctx.require(dependency_key)
        state = first_state if dependency_value == "first" else second_state
        await ctx.provide(REPLY_STATUS, state.read)

    async def dependency_plugin(value):
        async def plugin(ctx):
            await ctx.provide(dependency_key, value)
        return plugin

    dependency = await root.context.mount(
        await dependency_plugin("first"), name="reply-status-reactivation-dependency-first",
    )
    provider = await root.context.mount(
        provider_plugin,
        name="reply-status-reactivation-provider",
        inject=(dependency_key,),
    )
    stream = RuntimeReplyStatus(root).follow("reprovide")
    try:
        old_frame = await anext(stream)
        old_id = old_frame["snapshot_id"]
        old_token = provider.activation_token
        assert old_frame["available"]
        old_context = provider_context
        assert old_context is not None
        assert dependency_value == "first"

        await dependency.dispose()
        unavailable = await anext(stream)
        assert not unavailable["available"]
        assert provider.state is FiberState.PENDING
        replacement_dependency = await root.context.mount(
            await dependency_plugin("second"),
            name="reply-status-reactivation-dependency-second",
        )
        new_frame = await asyncio.wait_for(anext(stream), 3)
        assert new_frame["available"]
        assert new_frame["snapshot_id"] != old_id
        assert provider.activation_token is not old_token
        assert provider_context is not old_context
        assert len(activation_contexts) == 2
        assert dependency_value == "second"

        unrelated_key = ServiceKey[object]("reply-status-reactivation-unrelated")
        async def unrelated(ctx):
            await ctx.provide(unrelated_key, object())

        unrelated_fiber = await root.context.mount(unrelated, name="reply-status-reactivation-unrelated")
        third = RuntimeReplyStatus(root).follow("reprovide-unrelated")
        assert (await anext(third))["snapshot_id"] == new_frame["snapshot_id"]
        await third.aclose()
        await unrelated_fiber.dispose()
        await replacement_dependency.dispose()
    finally:
        await stream.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_status_provider_disposal_drains_real_reply_scope_before_close():
    """Provider cleanup waits for OwnerCall, not for an external subscriber."""
    root = CompositionRoot("reply-status-drain")
    state = ReplyState()
    tasks = Tasks()
    provider_context: Context | None = None
    peer_context: Context | None = None
    close_started, hard_consumer_closed = asyncio.Event(), asyncio.Event()
    work_entered, work_release = asyncio.Event(), asyncio.Event()
    close_count = 0
    peer_activation_count = 0
    peer_close_count = 0
    work = None
    peer_task = None
    dispose_task = None
    provider = None
    peer = None
    hard_consumer = None

    async def provider_plugin(ctx):
        nonlocal provider_context, close_count
        provider_context = ctx
        await ctx.provide(REPLY_STATUS, state.read)

        async def close():
            nonlocal close_count
            close_count += 1
            close_started.set()
            state.close()

        await ctx.effect(lambda: close, label="reply-status-owner-close")

    provider = await root.context.mount(provider_plugin, name="reply-status-drain-provider")
    assert provider_context is not None
    stream = RuntimeReplyStatus(root).follow("drain")
    try:
        assert (await anext(stream))["available"]

        async def hard_plugin(ctx):
            ctx.require(REPLY_STATUS)

            async def close():
                hard_consumer_closed.set()

            await ctx.effect(lambda: close, label="reply-status-hard-consumer-close")

        hard_consumer = await root.context.mount(
            hard_plugin,
            name="reply-status-hard-consumer",
            inject=(REPLY_STATUS,),
        )
        assert hard_consumer.state is FiberState.ACTIVE

        async def operation(task):
            with state.open(task, "drain", "conversation") as preview:
                with preview("answer") as delta:
                    await delta({"content_delta": "running"})
                    work_entered.set()
                    await work_release.wait()

        async with provider_context.runtime_scope():
            work = await tasks.admit("drain", lambda slot: slot.start(operation))
        await work_entered.wait()

        async def unrelated_plugin(ctx):
            nonlocal peer_context, peer_activation_count, peer_close_count
            peer_context = ctx
            peer_activation_count += 1
            await ctx.provide(ServiceKey[object]("reply-status-peer"), object())
            async def close():
                nonlocal peer_close_count
                peer_close_count += 1
            await ctx.effect(lambda: close, label="reply-status-peer-close")

        # The unrelated Fiber exists before A starts unloading.
        peer = await root.context.mount(unrelated_plugin, name="reply-status-peer")
        peer_token = peer.activation_token
        peer_entered, peer_release = asyncio.Event(), asyncio.Event()

        # The provider has a real hard consumer and a real in-flight Task.
        dispose_task = asyncio.create_task(provider.dispose())
        await asyncio.wait_for(hard_consumer_closed.wait(), 3)
        assert provider.state is FiberState.UNLOADING
        assert not dispose_task.done() and not close_started.is_set()
        assert peer.state is FiberState.ACTIVE
        assert peer.activation_token is peer_token
        assert peer_activation_count == 1 and peer_close_count == 0

        # Enter a fresh unrelated runtime scope only after A is visibly unloading.
        peer_exited = asyncio.Event()
        async def peer_probe_after_unload():
            assert peer_context is not None
            async with peer_context.runtime_scope():
                peer_entered.set()
                await peer_release.wait()
            peer_exited.set()

        peer_task = asyncio.create_task(
            peer_probe_after_unload(), name="reply-status-peer-probe",
        )
        await peer_entered.wait()
        assert peer.state is FiberState.ACTIVE
        assert peer.activation_token is peer_token
        assert peer_activation_count == 1 and peer_close_count == 0
        peer_release.set()
        await peer_task
        await peer_exited.wait()
        assert peer_task.done()
        assert peer._fiber._calls_idle.is_set()
        assert peer.state is FiberState.ACTIVE
        assert peer_close_count == 0

        # A running preview/boundary may be queued; external anext is still paused.
        assert not work.done
        assert close_count == 0 and not state.closed
        work_release.set()
        await work.join()
        await dispose_task
        assert close_started.is_set() and close_count == 1 and state.closed
        assert peer.state is FiberState.ACTIVE
        assert peer.activation_token is peer_token
        assert peer_activation_count == 1 and peer_close_count == 0
        await peer.dispose()
    finally:
        work_release.set()
        if work is not None:
            await work.join()
        if dispose_task is not None:
            await asyncio.gather(dispose_task, return_exceptions=True)
        if peer_task is not None and not peer_task.done():
            peer_release.set()
            await peer_task
        await tasks.close()
        await stream.aclose()
        if peer is not None:
            await asyncio.gather(peer.dispose(), return_exceptions=True)
        if hard_consumer is not None:
            await asyncio.gather(hard_consumer.dispose(), return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["acquire", "iteration", "close"])
async def test_status_reader_errors_reach_follow_without_boundary_overwrite(mode):
    """Reader errors survive provider replacement until the consumer observes them."""
    root = CompositionRoot("reply-status-errors")
    reader = _FailingReader(mode)

    async def provider(ctx):
        await ctx.provide(REPLY_STATUS, reader)

    fiber = await root.context.mount(provider, name="reply-status-error-provider")
    stream = RuntimeReplyStatus(root).follow("errors")
    dispose_task = None
    replacement = None
    try:
        if mode == "acquire":
            with pytest.raises(RuntimeError, match="reader acquisition failed"):
                await anext(stream)
        else:
            assert (await anext(stream))["available"]
            await asyncio.wait_for(reader.boundary_entered.wait(), 3)
            reader.release.set()
            await asyncio.wait_for(reader.error_raised.wait(), 3)
            dispose_task = asyncio.create_task(fiber.dispose())
            await dispose_task
            replacement = await mount_status(
                root, ReplyState(), name="reply-status-error-replacement",
            )
            error_text = "reader close failed" if mode == "close" else "reader iteration failed"
            with pytest.raises(RuntimeError, match=error_text):
                await anext(stream)
    finally:
        reader.release.set()
        if dispose_task is not None:
            await asyncio.gather(dispose_task, return_exceptions=True)
        if replacement is not None:
            await asyncio.gather(replacement.dispose(), return_exceptions=True)
        await stream.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_status_initial_provider_spawn_failure_is_real_error(monkeypatch):
    """An existing provider's first pump failure has no synthetic unavailable frame."""
    root, provider = await status_root("reply-status-initial-failure", ReplyState())
    stream = RuntimeReplyStatus(root).follow("initial-failure")
    original_create_task = asyncio.create_task

    def fail_pump(coroutine, *args, **kwargs):
        if str(kwargs.get("name", "")).startswith("plugin-task:reply-status-pump"):
            coroutine.close()
            raise RuntimeError("controlled initial pump task creation failure")
        return original_create_task(coroutine, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_pump)
    try:
        with pytest.raises(RuntimeError, match="controlled initial pump task creation failure"):
            await anext(stream)
        assert provider is not None
    finally:
        await stream.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_status_pending_activation_spawn_failure_is_joined(monkeypatch):
    """A real PENDING subscriber fails when its first provider activation cannot spawn."""
    root, _ = await status_root("reply-status-pending-failure", None)
    stream = RuntimeReplyStatus(root).follow("pending-failure")
    original_create_task = asyncio.create_task
    fail_next_pump = True

    def fail_pump(coroutine, *args, **kwargs):
        nonlocal fail_next_pump
        if fail_next_pump and str(kwargs.get("name", "")).startswith("plugin-task:reply-status-pump"):
            fail_next_pump = False
            coroutine.close()
            raise RuntimeError("controlled pending pump task creation failure")
        return original_create_task(coroutine, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_pump)
    provider = None
    try:
        assert not (await anext(stream))["available"]
        provider = await mount_status(root, ReplyState(), name="reply-status-pending-failure-provider")
        assert any(
            fiber.name.startswith("reply-status:") and fiber.state is FiberState.FAILED
            for fiber in root._fibers.values()
        )
        with pytest.raises(RuntimeError, match="controlled pending pump task creation failure"):
            await anext(stream)
        assert not any(fiber.name.startswith("reply-status:") for fiber in root._fibers.values())
    finally:
        await stream.aclose()
        if provider is not None:
            await asyncio.gather(provider.dispose(), return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
async def test_status_later_activation_failure_is_observable_and_joined(monkeypatch):
    """A later PENDING-to-ACTIVE spawn failure fails the public subscription."""
    root, provider = await status_root("reply-status-later-failure", ReplyState())
    stream = RuntimeReplyStatus(root).follow("later-failure")
    original_create_task = asyncio.create_task
    fail_next_pump = False

    def fail_pump(coroutine, *args, **kwargs):
        nonlocal fail_next_pump
        if fail_next_pump and str(kwargs.get("name", "")).startswith("plugin-task:reply-status-pump"):
            fail_next_pump = False
            coroutine.close()
            raise RuntimeError("controlled pump task creation failure")
        return original_create_task(coroutine, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_pump)
    try:
        assert (await anext(stream))["available"]
        fail_next_pump = True
        assert provider is not None
        await provider.dispose()
        assert not (await anext(stream))["available"]
        replacement = await mount_status(root, ReplyState(), name="reply-status-failed-replacement")
        assert any(
            fiber.name.startswith("reply-status:") and fiber.state is FiberState.FAILED
            for fiber in root._fibers.values()
        )
        with pytest.raises(RuntimeError, match="controlled pump task creation failure"):
            await asyncio.wait_for(anext(stream), 3)
        assert not any(
            fiber.name.startswith("reply-status:") for fiber in root._fibers.values()
        )
        assert not any(
            effect.startswith("root:reply-status-root-close:")
            or ":reply-status-pump:" in effect
            for effect in root.receipt().effects
        )
        await replacement.dispose()
    finally:
        await stream.aclose()
        await root.dispose()


def test_websocket_follow_reads_real_messages_switches_and_disconnects(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        def append(session, identity):
            return log.writer(session, author='真实作者', source='source', body_types=(Input,),
                              content={}).append(identity, Input(()))
        append('akashic:a', 'a0')
        append('akashic:a', 'a1')
        append('akashic:b', 'b0')
        channel = WebChatChannel()
        app = create_chat_app(workspace=tmp_path, channel=channel, messages=_catalog_port(log))
        def follow(ws, session, seq):
            ws.send_json({'type': 'session.follow', 'version': 2, 'session_id': session,
                          'after_seq': seq, 'request_id': 'follow'})
            frame = ws.receive_json()
            assert frame['type'] == 'session.following' and frame['session_id'] == session
        def message(ws, session):
            while True:
                frame = ws.receive_json()
                assert frame['session_id'] == session
                if frame['type'] == 'messages.appended':
                    return frame
                assert frame['type'] == 'reply.status' and not frame['available']
        with TestClient(app) as client:
            with client.websocket_connect('/ws') as ws:
                follow(ws, 'akashic:a', 0)
                page = message(ws, 'akashic:a')
                assert [row['id'] for row in page['items']] == ['a1']
                append('akashic:a', 'a2')
                assert [row['id'] for row in message(ws, 'akashic:a')['items']] == ['a2']
                follow(ws, 'akashic:b', -1)
                append('akashic:a', 'old-session')
                assert [row['id'] for row in message(ws, 'akashic:b')['items']] == ['b0']
            with client.websocket_connect('/ws') as ws:
                follow(ws, 'akashic:a', 2)
                assert [row['id'] for row in message(ws, 'akashic:a')['items']] == ['old-session']
        assert not log._listeners and not channel._followers and not channel._connections


def test_websocket_preview_is_separate_until_same_id_commits(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        state = ReplyState()
        tasks = Tasks()
        channel = WebChatChannel()
        roots = []
        app = create_chat_app(workspace=tmp_path, channel=channel, messages=_catalog_port(log),
                              reply_status=lambda session_id: RuntimeReplyStatus(roots[0]).follow(session_id))
        with TestClient(app) as client:
            root, _ = client.portal.call(status_root, 'socket-reply', state)
            roots.append(root)
            entered, release = asyncio.Event(), asyncio.Event()
            async def operation(task):
                with state.open(task, 'akashic:s', 'conversation') as preview:
                    with preview('answer') as delta:
                        await delta({'call_record_id': 'model-call'})
                        await delta({'content_delta': '正在生成', 'thinking_delta': '思考'})
                        entered.set()
                        await release.wait()
                        return log.writer('akashic:s', author='assistant', source='conversation',
                            body_types=(Output,), content={'text': lambda p: ContentReferences()}
                        ).append('answer', Output((ContentPart('text', '完整回答'),), 'complete'))
            async def start():
                task = await tasks.admit('s', lambda slot: slot.start(operation))
                await entered.wait()
                return task
            try:
                with client.websocket_connect('/ws') as ws:
                    ws.send_json({'type': 'session.follow', 'version': 2, 'session_id': 'akashic:s',
                                  'after_seq': -1, 'request_id': 'follow'})
                    assert ws.receive_json()['type'] == 'session.following'
                    assert ws.receive_json()['items'] == []
                    task = client.portal.call(start)
                    frame = ws.receive_json()
                    assert frame['type'] == 'reply.status'
                    assert frame['items'][0]['preview'] == {
                        'message_id': 'answer', 'text': '正在生成', 'thinking': '思考', 'call_record_id': 'model-call'}
                    assert log.reader('akashic:s').get('answer') is None
                    client.portal.call(release.set)
                    saved = client.portal.call(task.join)
                    committed, cleared = False, False
                    while not committed or not cleared:
                        frame = ws.receive_json()
                        if frame['type'] == 'messages.appended':
                            assert frame['items'][0]['id'] == saved.message_id == 'answer'
                            assert frame['items'][0]['body']['parts'][0]['value'] == '完整回答'
                            committed = True
                        elif frame['type'] == 'reply.status' and frame['items'] == []:
                            cleared = True
                assert not channel._followers and not log._listeners
            finally:
                client.portal.call(release.set)
                client.portal.call(tasks.close)
                client.portal.call(root.dispose)


def test_channel_stop_waits_for_active_subscriptions_to_close(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        state = ReplyState()
        roots = []
        finalizing, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def status(session_id):
            try:
                async with aclosing(RuntimeReplyStatus(roots[0]).follow(session_id)) as frames:
                    async for frame in frames:
                        yield frame
            finally:
                finalizing.set()
                await release.wait()
                finished.set()
        channel = WebChatChannel()
        app = create_chat_app(workspace=tmp_path, channel=channel, messages=_catalog_port(log), reply_status=status)
        with TestClient(app) as client:
            root, _ = client.portal.call(status_root, 'stop', state)
            roots.append(root)
            try:
                with client.websocket_connect('/ws') as ws:
                    ws.send_json({'type': 'session.follow', 'version': 2, 'session_id': 'akashic:s',
                                  'after_seq': -1, 'request_id': 'follow'})
                    assert ws.receive_json()['type'] == 'session.following'
                    assert ws.receive_json()['type'] == 'reply.status'
                    stopping = client.portal.start_task_soon(channel.stop)
                    client.portal.call(finalizing.wait)
                    assert not stopping.done()
                    client.portal.call(release.set)
                    stopping.result(timeout=3)
                    assert finished.is_set() and not channel._followers and not log._listeners
            finally:
                client.portal.call(release.set)
                client.portal.call(root.dispose)


@pytest.mark.parametrize('changes', [{'version': 1}, {'version': True}, {'after_seq': True},
                                      {'after_seq': -2}, {'after_seq': 2}, {'session_id': 'qq:a'}])
def test_websocket_follow_rejects_invalid_boundary(tmp_path, changes):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        channel = WebChatChannel()
        with TestClient(create_chat_app(workspace=tmp_path, channel=channel, messages=_catalog_port(log))) as client:
            with client.websocket_connect('/ws') as ws:
                ws.send_json({'type': 'session.follow', 'version': 2, 'session_id': 'akashic:new',
                              'after_seq': -1, 'request_id': 'bad', **changes})
                assert ws.receive_json()['type'] == 'error'
                ws.send_json({'type': 'ping', 'request_id': 'alive'})
                assert ws.receive_json()['type'] == 'pong'
        assert not log._listeners and not channel._followers
