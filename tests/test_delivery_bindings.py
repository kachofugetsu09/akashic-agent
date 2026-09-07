from functools import partial
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.tasks import Tasks
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.delivery.api import Sink
from plugins.delivery.execution import Deliveries
from plugins.delivery.records import DeliveryRecords
from plugins.delivery.senders import open_sender
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
import json
from agent.plugin_composition import RUNTIME_STARTED, ServiceKey
from plugins.delivery.api import Receipt
api_version = 3
name = "test_sender"
version = "1.0.0"
inject = (ServiceKey("delivery.senders.v1"),)
async def apply(ctx, config):
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
            return Receipt(status="delivered", provider_ids=("original-A",))
        async def query(self, key, address):
            path = ctx.data_root / "sent.jsonl"
            if not path.exists():
                return None
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                if entry[0] == key and entry[1] == address:
                    return Receipt(status="delivered", provider_ids=("original-A",))
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


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_write_fails", [False, True])
async def test_archived_sender_survives_removed_source_without_starting_a_receiver(
    tmp_path, monkeypatch, receipt_write_fails,
):
    source = tmp_path / "plugins"
    sources(source)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [source], log)
    tasks = Tasks()
    restored = None
    try:
        await host.load_all()
        await host.start_runtime()
        message = log.writer("chat", author="reply", source="conversation", body_types=(Output,),
                             content={"text": lambda part: ContentReferences()}).append(
            "answer", Output((ContentPart("text", "original body"),), "complete"))
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            binding = root.require(SENDERS).bind("test", bindings)
            execution = root.require(ServiceKey("fixture.delivery"))()
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
        shutil.rmtree(source)
        log.close()
        log = MessageLog(tmp_path / "sessions.db")
        restored = manager(tmp_path, [], log)
        bindings = Bindings(log, restored._archive, restored.open_binding)
        records = DeliveryRecords(log.owner("plugin:delivery"), "test_sender")
        execution = Deliveries(records, log.catalog(), tasks, partial(open_sender, bindings), task_key="delivery")
        result = await execution.send(message.message_id, sink.name)
        assert result.provider_ids == ("original-A",)
        assert (await execution.send(message.message_id, sink.name)).provider_ids == ("original-A",)
        effect = next((tmp_path / "workspace").rglob("sent.jsonl"))
        assert len(effect.read_text().splitlines()) == 1
        assert "original-room" in effect.read_text()
        assert starts.read_text().splitlines() == ["started"]
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
async def test_formal_and_archived_delivery_share_target_coordination(tmp_path):
    """两个真实 Root 共用 Delivery owner 的短命活动，归档不能绕过正式忙闲状态。"""
    import asyncio

    source = tmp_path / "plugins"
    sources(source)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [source], log)
    try:
        await host.load_all()
        await host.start_runtime()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            bindings = Bindings(log, host._archive, host.open_binding)
            service = ServiceKey("fixture.delivery")
            binding = bindings.bind(service, {})
            formal = root.require(service)()
            async with bindings.open(binding, service) as (factory, _):
                archived = factory()
                waiting = asyncio.Event()

                async def check():
                    waiting.set()
                    await archived.wait_idle("test", "room")

                with formal.activity("test", "room"):
                    pending = asyncio.create_task(check())
                    await waiting.wait()
                    assert not pending.done()
                await pending
                # 反方向也走相同 owner；测试不依赖两个 Root 内的 Python 类身份。
                waiting.clear()
                with archived.activity("test", "room"):
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
