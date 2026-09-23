import asyncio
from functools import partial
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
import shutil

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.context import CompositionRoot
from agent.plugin_composition.model import FiberState, PluginRuntime, ServiceKey
from agent.plugin_contracts import Message
from agent.plugin_composition.tasks import Tasks
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.delivery.api import Receipt, Sink
from plugins.delivery.execution import Deliveries
from plugins.delivery.records import DeliveryRecords
from plugins.delivery.senders import DELIVERY_SENDERS, Senders, open_sender
from session.log import MessageLog, OwnerTransaction
from session.message import ContentPart, ContentReferences, Output

SENDERS = ServiceKey("delivery.senders.v1")
DELIVERY = ServiceKey("delivery.v1")


def sources(path):
    shutil.copytree(Path(__file__).parents[1] / "plugins/delivery", path / "delivery",
                    ignore=shutil.ignore_patterns("__pycache__"))
    target = path / "test_sender"
    target.mkdir()
    (target / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from typing import Literal
from agent.plugin_composition import RUNTIME_STARTED, ServiceKey
api_version = 3
name = "test_sender"
version = "1.0.0"
inject = (ServiceKey("delivery.senders.v1"), ServiceKey("delivery.v1"))

@dataclass(frozen=True)
class SendResult:
    status: Literal["delivered", "rejected", "failed"]
    provider_ids: tuple[str, ...] = ()
    error: str | None = None

async def apply(ctx):
    async def start(_event):
        path = ctx.data_root / "receiver-starts"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as file:
            file.write("started\\n")
    await ctx.on(RUNTIME_STARTED, start)
    class Sender:
        idempotent = True
        async def send(self, key, address, message):
            path = ctx.data_root / "sent.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as file:
                file.write(json.dumps([key, address, message.message_id, "original-A"]) + "\\n")
            return SendResult(status="delivered", provider_ids=("original-A",))
        async def query(self, key, address):
            path = ctx.data_root / "sent.jsonl"
            if not path.exists():
                return None
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                if entry[0] == key and entry[1] == address:
                    return SendResult(status="delivered", provider_ids=("original-A",))
            return None
    @asynccontextmanager
    async def open():
        if (ctx.data_root / "credential-revoked").exists():
            raise PermissionError("original credential revoked")
        yield Sender()
    await ctx.require(inject[0]).register(ctx, name="test", idempotent=True, open=open)
    await ctx.provide(ServiceKey("fixture.delivery"), lambda: ctx.require(ServiceKey("delivery.v1")).open(ctx))
''')


def manager(tmp_path, plugins, log):
    return PluginManager(plugins, event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)


@asynccontextmanager
async def local_senders_root(tmp_path, *, opener_mode="normal"):
    """Build a live Senders provider, target owner, consumer, and unrelated Fiber."""
    root = CompositionRoot("delivery-local")
    target_key = ServiceKey("fixture.delivery.target")
    metadata = {"name": "local", "owner": "sender-target", "idempotent": True}
    state = {
        "metadata": metadata,
        "provider_cleanup_calls": 0,
        "target_cleanup_calls": 0,
        "consumer_cleanup_calls": 0,
        "opener_cleanup_calls": 0,
        "opener_entered": asyncio.Event(),
        "opener_release": asyncio.Event(),
        "consumer_released": asyncio.Event(),
        "continue_work": asyncio.Event(),
        "sequence": [],
        "callback_scope_counts": [],
        "opener_scope_counts": [],
        "unrelated_events": [],
        "body_ran": False,
    }

    async def consumer(ctx):
        _ = ctx.require(target_key)

        def setup():
            def cleanup():
                state["consumer_cleanup_calls"] += 1
                state["sequence"].append("consumer-cleanup")
                state["consumer_released"].set()

            return cleanup

        await ctx.effect(setup, label="delivery-local-consumer")

    await root.mount(
        consumer,
        name="delivery-local-consumer",
        inject=(target_key,),
        runtime=PluginRuntime(
            "delivery-local-consumer", "generation-local", tmp_path / "consumer-data",
            tmp_path / "consumer-workspace", tmp_path / "consumer-plugin", {},
        ),
    )

    async def provider(ctx):
        state["senders_context"] = ctx
        senders = Senders(ctx)

        def setup():
            def cleanup():
                state["provider_cleanup_calls"] += 1

            return cleanup

        await ctx.effect(setup, label="delivery-local-provider")
        await ctx.provide(DELIVERY_SENDERS, senders)
        state["senders"] = senders

    await root.mount(
        provider,
        name="delivery-senders",
        runtime=PluginRuntime(
            "delivery-senders", "generation-local", tmp_path / "data",
            tmp_path / "workspace", tmp_path / "plugin", {},
        ),
    )

    async def target(ctx):
        senders = ctx.require(DELIVERY_SENDERS)

        class LocalTarget:
            idempotent = True

            async def send(self, key, address, message):
                async with ctx.runtime_scope():
                    state["callback_scope_counts"].append(
                        ("send", ctx.fiber.state, len(ctx._fiber._in_flight_calls))
                    )
                    return Receipt(status="delivered", provider_ids=("local",))

            async def query(self, key, address):
                async with ctx.runtime_scope():
                    state["callback_scope_counts"].append(
                        ("query", ctx.fiber.state, len(ctx._fiber._in_flight_calls))
                    )
                    return Receipt(status="delivered", provider_ids=("local",))

        target_value = LocalTarget()

        def setup():
            def cleanup():
                state["target_cleanup_calls"] += 1
                state["sequence"].append("target-cleanup")

            return cleanup

        await ctx.effect(setup, label="delivery-local-target")
        await ctx.provide(target_key, target_value)

        @asynccontextmanager
        async def open_target():
            try:
                async with ctx.runtime_scope():
                    state["opener_entered"].set()
                    state["opener_scope_counts"].append(len(ctx._fiber._in_flight_calls))
                    if opener_mode == "error":
                        raise OSError("sender opener failed")
                    if opener_mode == "cancel":
                        await state["opener_release"].wait()
                    yield target_value
            finally:
                state["opener_cleanup_calls"] += 1
                state["sequence"].append("opener-exit")

        await senders.register(ctx, name="local", idempotent=True, open=open_target)

    target_fiber = await root.mount(
        target,
        name="sender-target",
        inject=(DELIVERY_SENDERS,),
        runtime=PluginRuntime(
            "sender-target", "generation-local", tmp_path / "target-data",
            tmp_path / "target-workspace", tmp_path / "target-plugin", {},
        ),
    )
    state["target_context"] = target_fiber.context

    async def unrelated(ctx):
        await ctx.on(RUNTIME_STARTED, lambda _event: state["unrelated_events"].append("started"))
        await ctx.on(RUNTIME_STOPPING, lambda _event: state["unrelated_events"].append("stopping"))

    unrelated_fiber = await root.mount(
        unrelated,
        name="delivery-unrelated",
        runtime=PluginRuntime(
            "delivery-unrelated", "generation-local", tmp_path / "unrelated-data",
            tmp_path / "unrelated-workspace", tmp_path / "unrelated-plugin", {},
        ),
    )
    state["unrelated_context"] = unrelated_fiber.context
    try:
        yield root, state
    finally:
        state["continue_work"].set()
        state["opener_release"].set()
        await root.dispose()


def local_message():
    """Build one real Message value for the Sender protocol boundary."""
    return Message(
        "delivery-message",
        "delivery-session",
        0,
        datetime.now(UTC),
        "tester",
        "conversation",
        Output((ContentPart("text", "hello"),), "complete"),
    )


@pytest.mark.asyncio
async def test_senders_open_keeps_target_scope_during_unload(tmp_path):
    """既有 sender view 在目标 owner 排空期间仍完成真实 callback。"""
    async with local_senders_root(tmp_path) as (root, state):
        senders = state["senders"]
        provider_context = state["senders_context"]
        target_context = state["target_context"]
        unrelated_context = state["unrelated_context"]
        provider_state = provider_context.fiber.state
        provider_activation = provider_context.fiber.activation_token
        unrelated_state = unrelated_context.fiber.state
        unrelated_activation = unrelated_context.fiber.activation_token
        unrelated_events = tuple(state["unrelated_events"])
        root_identity = root.instance_token
        view_holder = {}
        message = local_message()
        worker = None
        new_open_task = None
        target_dispose = None

        async def use_sender():
            async with senders.open(state["metadata"]) as view:
                view_holder["view"] = view
                await state["continue_work"].wait()
                assert view.idempotent
                view_holder["send"] = await view.send("key", "address", message)
                view_holder["query"] = await view.query("key", "address")

        try:
            worker = asyncio.create_task(use_sender())
            await state["opener_entered"].wait()
            target_dispose = asyncio.create_task(target_context.fiber.dispose())
            await state["consumer_released"].wait()
            assert target_context.fiber.state is FiberState.UNLOADING
            assert state["consumer_cleanup_calls"] == 1
            assert state["target_cleanup_calls"] == 0
            assert root.instance_token is root_identity
            assert provider_context.fiber.state is provider_state
            assert provider_context.fiber.activation_token is provider_activation
            assert unrelated_context.fiber.state is unrelated_state
            assert unrelated_context.fiber.activation_token is unrelated_activation
            assert tuple(state["unrelated_events"]) == unrelated_events

            async with unrelated_context.runtime_scope():
                state.setdefault("unrelated_scope_work", []).append(
                    unrelated_context.fiber.state
                )

            new_open_result = {"body_ran": False, "error": ""}

            async def open_from_new_task():
                try:
                    async with senders.open(state["metadata"]):
                        new_open_result["body_ran"] = True
                except RuntimeError as error:
                    new_open_result["error"] = str(error)

            new_open_task = asyncio.create_task(open_from_new_task())
            await new_open_task
            assert not new_open_result["body_ran"]
            assert "不接纳新调用" in new_open_result["error"]
            state["continue_work"].set()
            await worker
            await target_dispose

            assert target_context.fiber.state is FiberState.DISPOSED
            assert state["provider_cleanup_calls"] == 0
            assert state["target_cleanup_calls"] == 1
            assert state["opener_cleanup_calls"] == 1
            assert state["sequence"].index("opener-exit") < state["sequence"].index(
                "target-cleanup"
            )
            assert state["unrelated_scope_work"] == [FiberState.ACTIVE]
            assert not provider_context._fiber._in_flight_calls
            assert not target_context._fiber._in_flight_calls
            assert all(
                name in {"send", "query"}
                and fiber_state is FiberState.UNLOADING
                and calls > 0
                for name, fiber_state, calls in state["callback_scope_counts"]
            )
            assert state["opener_scope_counts"]
            assert all(calls > 0 for calls in state["opener_scope_counts"])
            assert view_holder["send"].status == "delivered"
            assert view_holder["send"].provider_ids == ("local",)
            assert view_holder["query"].status == "delivered"
            assert view_holder["query"].provider_ids == ("local",)
            with pytest.raises(RuntimeError, match="释放"):
                await view_holder["view"].send("closed", "address", message)
        finally:
            state["continue_work"].set()
            state["opener_release"].set()
            if worker is not None:
                try:
                    await worker
                finally:
                    if new_open_task is not None:
                        await new_open_task
                    if target_dispose is not None:
                        await target_dispose
            else:
                if new_open_task is not None:
                    await new_open_task
                if target_dispose is not None:
                    await target_dispose


@pytest.mark.asyncio
@pytest.mark.parametrize("opener_mode", ["error", "cancel"])
async def test_senders_open_releases_scopes_on_opener_failure_or_cancel(
    tmp_path, opener_mode,
):
    """opener 异常或取消都释放目标、provider permit，并只清理一次。"""
    async with local_senders_root(tmp_path, opener_mode=opener_mode) as (root, state):
        provider_context = state["senders_context"]
        target_context = state["target_context"]
        task = None
        target_dispose = None

        async def use_sender():
            async with state["senders"].open(state["metadata"]):
                state["body_ran"] = True

        try:
            task = asyncio.create_task(use_sender())
            await state["opener_entered"].wait()
            if opener_mode == "error":
                with pytest.raises(OSError, match="sender opener failed"):
                    await task
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert not state["body_ran"]
            assert state["opener_cleanup_calls"] == 1
            assert not provider_context._fiber._in_flight_calls
            assert not target_context._fiber._in_flight_calls

            target_dispose = asyncio.create_task(target_context.fiber.dispose())
            await state["consumer_released"].wait()
            await target_dispose
            assert target_context.fiber.state is FiberState.DISPOSED
            assert state["target_cleanup_calls"] == 1
        finally:
            state["opener_release"].set()
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if target_dispose is not None:
                await target_dispose


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_write_fails", [False, True])
async def test_sender_survives_restart_without_repeating_a_delivery(
    tmp_path, monkeypatch, receipt_write_fails,
):
    source = tmp_path / "plugins"
    sources(source)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = manager(tmp_path, [source], log)
    tasks = Tasks()
    restored = None
    try:
        await host.load_all()
        await host.start_runtime()
        message = log.writer("chat", author="reply", source="conversation", body_types=(Output,),
                             content={"text": lambda part: ContentReferences()}).append(
            "answer", Output((ContentPart("text", "original body"),), "complete"))
        root = host.live_root
        sender = host.generation("test_sender")
        assert root is not None and sender is not None and sender.fiber is not None
        bindings = root.service_value(BINDINGS)
        assert bindings is not None
        async with sender.fiber.context.runtime_scope():
            binding = sender.fiber.context.require(SENDERS).bind("test", bindings)
            execution = sender.fiber.context.require(ServiceKey("fixture.delivery"))()
            sink = Sink(name="phone", binding_id=binding, address="original-room")
            execution.prepare(log.reader("chat"), message, (sink,))
            if receipt_write_fails:
                save = OwnerTransaction.save

                def fail_delivered(self, key, value, *, expected_version):
                    if key.startswith("delivery:") and value.get("phase") == "delivered":
                        raise OSError("receipt disk unavailable")
                    return save(self, key, value, expected_version=expected_version)

                with monkeypatch.context() as patch:
                    patch.setattr(OwnerTransaction, "save", fail_delivered)
                    with pytest.raises(OSError, match="receipt disk unavailable"):
                        await execution.send(message.message_id, sink.name)
                records = DeliveryRecords(log.owner("plugin:delivery"), "test_sender")
                assert records.read(message.message_id, sink.name)[1].phase == "started"
                effect = next((tmp_path / "workspace").rglob("sent.jsonl"))
                assert len(effect.read_text().splitlines()) == 1
        starts = next((tmp_path / "workspace").rglob("receiver-starts"))
        assert starts.read_text().splitlines() == ["started"]
        await host.terminate_all()
        log.close()
        log = MessageLog(tmp_path / "sessions.db")
        restored = manager(tmp_path, [source], log)
        await restored.load_all()
        await restored.start_runtime()
        root = restored.live_root
        assert root is not None
        bindings = root.service_value(BINDINGS)
        assert bindings is not None
        records = DeliveryRecords(log.owner("plugin:delivery"), "test_sender")
        execution = Deliveries(
            records, log.catalog(), tasks, partial(open_sender, bindings), task_key="delivery"
        )
        result = await execution.send(message.message_id, sink.name)
        assert result.provider_ids == ("original-A",)
        assert (await execution.send(message.message_id, sink.name)).provider_ids == ("original-A",)
        effect = next((tmp_path / "workspace").rglob("sent.jsonl"))
        assert len(effect.read_text().splitlines()) == 1
        assert "original-room" in effect.read_text()
        assert starts.read_text().splitlines() == ["started", "started"]
        async with open_sender(bindings, binding) as closed:
            assert closed.idempotent
        with pytest.raises(RuntimeError, match="释放"):
            await closed.send("escaped", "wrong-room", message)
        assert len(effect.read_text().splitlines()) == 1
    finally:
        await tasks.close()
        if restored is not None:
            await restored.terminate_all()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_selected_and_formal_delivery_share_target_coordination(tmp_path):
    """正式服务与当前 binding 共用 Delivery owner 的短命活动。"""
    import asyncio

    source = tmp_path / "plugins"
    sources(source)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = manager(tmp_path, [source], log)
    try:
        await host.load_all()
        await host.start_runtime()
        root = host.live_root
        sender = host.generation("test_sender")
        assert root is not None and sender is not None and sender.fiber is not None
        bindings = root.service_value(BINDINGS)
        assert bindings is not None
        async with sender.fiber.context.runtime_scope():
            service = ServiceKey("fixture.delivery")
            binding = bindings.bind(service, {})
            formal = sender.fiber.context.require(service)()
            async with bindings.open(binding, service) as (factory, _):
                selected = factory()
                waiting = asyncio.Event()

                async def check():
                    waiting.set()
                    await selected.wait_idle("test", "room")

                with formal.activity("test", "room"):
                    pending = asyncio.create_task(check())
                    await waiting.wait()
                    assert not pending.done()
                await pending
                # 反方向也走相同 owner；测试不依赖两个 Root 内的 Python 类身份。
                waiting.clear()
                with selected.activity("test", "room"):
                    async def reverse():
                        waiting.set()
                        await formal.wait_idle("test", "room")
                    pending = asyncio.create_task(reverse())
                    await waiting.wait()
                    assert not pending.done()
                await pending
    finally:
        await host.terminate_all()
        log.close()
