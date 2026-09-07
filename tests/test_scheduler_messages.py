import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.scheduler.schedule import ScheduledJob
from plugins.scheduler.store import JobStore, fire_key
from session.log import OwnerTransaction
from session.message import Input, Output, ToolResult
from tests.test_default_reply import application
from tests.test_delivery_bindings import sources


def install(tmp_path):
    root = tmp_path / "plugins"
    sources(root)
    for name in ("scheduler", "delivery_policy"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, root / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (root / "scheduler/akashic.plugin.toml").write_text(
        'schema_version = 1\nname = "scheduler"\nversion = "4.0.0"\napi_version = 3\nentrypoint = "message_plugin.py"\n')


async def settled(store, key):
    # 文件提交是真实完成边界；此 helper 仅用于外部观察，不协调竞态。
    async with asyncio.timeout(10):
        while store.read().fires.get(key) is None or store.read().fires[key].status == "pending":
            await asyncio.sleep(.01)
    return store.read().fires[key]


@pytest.mark.asyncio
async def test_actual_scheduler_uses_isolated_sessions_and_only_publishes_final_notification(tmp_path):
    install(tmp_path)
    async with application(tmp_path, replying=True, start=False) as (log, host):
        store = JobStore(tmp_path / "workspace/schedules.json")
        job = ScheduledJob(trigger="after", tier="soft", fire_at=datetime.now(UTC) + timedelta(seconds=1),
                           channel="test", chat_id="room", timezone="UTC", prompt="do the work")
        store.add("schedule", job, "created")
        await host.start_runtime()
        fire = await settled(store, fire_key(job))
        assert fire.status == "delivered"
        internal = log.reader(fire.session_id)
        assert internal.attributes.visibility == "internal"
        assert internal.attributes.learning == "excluded"
        assert [type(message.body) for message in internal.snapshot()] == [Input, Output, ToolResult, Output]
        target = log.reader("test:room").snapshot()
        assert len(target) == 1 and target[0].message_id == fire.notification_id
        assert target[0].body.parts[0].value == "finished"
        assert store.read().jobs[job.id].run_count == 1
        assert (tmp_path / "effect.txt").read_text() == "once\n"
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(ServiceKey("fixture.calls"))
            assert len(calls) == 2
        # 第二次触发只读自己的输入，第一轮工具轨迹不进入下一次模型请求。
        from dataclasses import replace
        second = replace(job, id="second", fire_at=datetime.now(UTC) + timedelta(seconds=1))
        store.add("schedule-2", second, "created")
        second_fire = await settled(store, fire_key(second))
        assert second_fire.session_id != fire.session_id
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(ServiceKey("fixture.calls"))
            assert len(calls) == 3
            assert "written" not in str(calls[-1].messages)
        assert len(next((tmp_path / "workspace").rglob("sent.jsonl")).read_text().splitlines()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["notification", "selection", "receipt", "settlement"])
async def test_actual_scheduler_restart_resumes_after_saved_output_without_repeating_work(tmp_path, monkeypatch, fault):
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus
    from session.log import MessageLog, MessageWriter
    import infra.persistence.json_store as json_store

    install(tmp_path)
    failed = asyncio.Event()
    async with application(tmp_path, replying=False, start=False) as (log, host):
        store = JobStore(tmp_path / "workspace/schedules.json")
        job = ScheduledJob(trigger="after", tier="soft", fire_at=datetime.now(UTC) + timedelta(seconds=1),
                           channel="test", chat_id="room", timezone="UTC", prompt="do the work")
        store.add("schedule", job, "created")
        append, save, atomic_save = OwnerTransaction.append, OwnerTransaction.save, json_store.atomic_save_json

        def fail_notification(self, writer, message_id, body, **kwargs):
            if fault == "notification" and message_id.startswith("scheduler-notification:"):
                failed.set()
                raise OSError("notification disk unavailable")
            return append(self, writer, message_id, body, **kwargs)

        def fail_receipt(self, key, value, **kwargs):
            if fault == "selection" and key.startswith("selection:scheduler-notification:"):
                failed.set()
                raise OSError("selection disk unavailable")
            if fault == "receipt" and key.startswith("delivery:") and value.get("phase") == "delivered":
                failed.set()
                raise OSError("receipt disk unavailable")
            return save(self, key, value, **kwargs)

        def fail_settlement(path, value, **kwargs):
            if fault == "settlement" and str(path).endswith("schedules.json") and any(
                    fire["status"] == "delivered" for fire in value.get("fires", {}).values()):
                failed.set()
                raise OSError("settlement disk unavailable")
            return atomic_save(path, value, **kwargs)

        # 归档模块装配前替换共享边界，真实 Scheduler/Tool/Delivery 代码保持原样。
        with monkeypatch.context() as patch:
            patch.setattr(OwnerTransaction, "append", fail_notification)
            patch.setattr(OwnerTransaction, "save", fail_receipt)
            # store.py 在 load_all 时已导入，沿实际归档模块绑定本次磁盘故障。
            import sys
            for module in tuple(sys.modules.values()):
                if module is not None and getattr(module, "__file__", "") and str(module.__file__).endswith("/store.py"):
                    if hasattr(module, "atomic_save_json") and hasattr(module, "JobStore"):
                        patch.setattr(module, "atomic_save_json", fail_settlement)
            await host.start_runtime()
            async with asyncio.timeout(10):
                await failed.wait()
            await host.terminate_all()
        if fault in {"notification", "selection"}:
            assert log.reader("test:room").snapshot() == ()
        assert store.read().fires[fire_key(job)].status == "pending"
        assert (tmp_path / "effect.txt").read_text() == "once\n"
        # 关闭整个 Host 和数据库后再启动，恢复没有旧 watcher/model/Task 指针。
        log.close()
        reopened = MessageLog(tmp_path / "sessions.db")
        restarted = PluginManager([tmp_path / "plugins"], event_bus=EventBus(), workspace=tmp_path / "workspace",
                                  installed_cache_root=tmp_path / "home", message_log=reopened)
        try:
            await restarted.load_all()
            await restarted.start_runtime()
            fire = await settled(store, fire_key(job))
            assert fire.status == "delivered"
            assert store.read().jobs[job.id].run_count == 1
            async with lease_runtime_snapshot(restarted.snapshot_store) as snapshot:
                assert snapshot.composition_root.context.require(ServiceKey("fixture.calls")) == []
            assert (tmp_path / "effect.txt").read_text() == "once\n"
            assert len(reopened.reader("test:room").snapshot()) == 1
            assert len(next((tmp_path / "workspace").rglob("sent.jsonl")).read_text().splitlines()) == 1
        finally:
            await restarted.terminate_all()
            reopened.close()


@pytest.mark.asyncio
async def test_archived_schedule_tool_recovers_original_operation_after_source_removal(tmp_path):
    from agent.plugin_composition.bindings import Bindings
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus
    from plugins.tools.plugin import TOOLS, open_tool

    install(tmp_path)
    async with application(tmp_path, replying=False, start=False) as (log, host):
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            tool = snapshot.composition_root.context.require(TOOLS).bind("schedule", bindings)
            async with open_tool(bindings, tool) as bound:
                prepared = await bound.prepare({"tier": "instant", "trigger": "after", "when": "1h",
                    "channel": "test", "chat_id": "room", "timezone": "UTC", "message": "original"})
                result = await bound.invoke("original-schedule", prepared)
        store = JobStore(tmp_path / "workspace/schedules.json")
        original = store.load()[0]
        await host.terminate_all()
        shutil.rmtree(tmp_path / "plugins")
        restarted = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "workspace",
                                  installed_cache_root=tmp_path / "home", message_log=log)
        try:
            bindings = Bindings(log, restarted._archive, restarted.open_binding)
            async with open_tool(bindings, tool) as bound:
                assert await bound.query("original-schedule") == result
                assert await bound.invoke("original-schedule", prepared) == result
            assert store.load() == [original]
            assert store.read().fires == {}
            assert log.catalog().snapshot_heads() == {}
        finally:
            await restarted.terminate_all()


@pytest.mark.asyncio
async def test_cancelling_a_failed_prepared_fire_revisits_cleanup_in_same_runtime(tmp_path, monkeypatch):
    """第一次触发已退出且保留 prepared，后来的取消仍被当前 watcher 恢复。"""
    install(tmp_path)
    prepared, rejected = asyncio.Event(), asyncio.Event()
    save = OwnerTransaction.save

    def observe(self, key, value, **kwargs):
        result = save(self, key, value, **kwargs)
        if key.startswith("delivery:"):
            if value["phase"] == "prepared":
                prepared.set()
            elif value["phase"] == "rejected":
                rejected.set()
        return result

    monkeypatch.setattr(OwnerTransaction, "save", observe)
    async with application(tmp_path, replying=False, start=False) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            data = snapshot.generations["test_sender"].data_dir
            data.mkdir(parents=True, exist_ok=True)
            (data / "credential-revoked").touch()
        store = JobStore(tmp_path / "workspace/schedules.json")
        job = ScheduledJob(trigger="after", tier="instant", fire_at=datetime.now(UTC),
                           channel="test", chat_id="room", timezone="UTC", message="reminder")
        store.add("schedule", job, "created")
        await host.start_runtime()
        async with asyncio.timeout(5):
            await prepared.wait()
        store.cancel("cancel", (job.id,))
        async with asyncio.timeout(5):
            await rejected.wait()
        assert store.read().fires[fire_key(job)].status == "cancelled"
        assert not list((tmp_path / "workspace").rglob("sent.jsonl"))


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [True, False])
async def test_restart_preclaims_passive_effect_before_scheduler_or_archive_can_start(tmp_path, monkeypatch, confirmed):
    """原回复丢失本地回执后重启，启动顺序与归档 Root 都不能绕过首次恢复查询。"""
    import sys
    from types import ModuleType
    from agent.plugin_composition import RUNTIME_STARTED
    from agent.plugin_composition.bindings import Bindings
    from agent.plugin_composition.tasks import Tasks
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus
    from plugins.delivery.records import DeliveryRecords
    from session.log import MessageLog
    from session.message import ContentPart, ContentReferences

    fixture = ModuleType("scheduler_recovery_fixture")
    probing, release, failed, waiting = (asyncio.Event() for _ in range(4))
    state = {"pause": False, "queries": 0, "confirmed": True}

    async def query():
        if state["pause"]:
            state["queries"] += 1
            probing.set()
            await release.wait()
        return state["confirmed"]

    fixture.query = query
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
    install(tmp_path)
    sender = tmp_path / "plugins/test_sender/plugin.py"
    sender.write_text(sender.read_text().replace('idempotent = True', 'idempotent = False')
                      .replace('idempotent=True', 'idempotent=False').replace(
        'async def query(self, key, address):',
        'async def query(self, key, address):\n            from scheduler_recovery_fixture import query\n            if not await query():\n                return None'))
    save = OwnerTransaction.save

    def fail_receipt(self, key, value, **kwargs):
        if key.startswith("delivery:") and value.get("phase") == "delivered":
            failed.set()
            raise OSError("lost local delivery receipt")
        return save(self, key, value, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(OwnerTransaction, "save", fail_receipt)
        async with application(tmp_path, replying=False) as (log, host):
            writer = log.writer("test:room", author="reply", source="conversation", body_types=(Input, Output),
                                content={"text": lambda _: ContentReferences(), "channel.origin": lambda _: ContentReferences()})
            writer.append("original-input", Input((ContentPart("channel.origin", {
                "channel": "test", "chat_id": "room", "sender": "user"}),)))
            writer.append("original-output", Output((ContentPart("text", "original reply"),), "complete"))
            async with asyncio.timeout(5):
                await failed.wait()
    assert len(next((tmp_path / "workspace").rglob("sent.jsonl")).read_text().splitlines()) == 1

    store = JobStore(tmp_path / "workspace/schedules.json")
    job = ScheduledJob(trigger="after", tier="soft", fire_at=datetime.now(UTC) - timedelta(seconds=1),
                       channel="test", chat_id="room", timezone="UTC", prompt="scheduled work")
    store.add("new-schedule", job, "created")
    state.update(pause=True, queries=0, confirmed=confirmed)
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([tmp_path / "plugins"], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    wait_idle = Tasks.wait_idle

    async def observe_idle(self, key):
        waiting.set()
        await wait_idle(self, key)

    monkeypatch.setattr(Tasks, "wait_idle", observe_idle)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root
            bindings = Bindings(log, host._archive, host.open_binding)
            service = ServiceKey("fixture.delivery")
            binding = bindings.bind(service, {})
            async with bindings.open(binding, service) as (factory, _):
                archived = factory()
                archive_waiter = asyncio.create_task(archived.wait_idle("test", "room"))
                await waiting.wait()
                assert not archive_waiter.done()
                waiting.clear()
                # 受控顺序：真正 Scheduler 先启动并抵达 idle wait，策略此时还没启动 follower。
                listeners = root._events._listeners[RUNTIME_STARTED]
                scheduler = next(item for item in listeners if item.owner.runtime.plugin_id == "scheduler")
                listeners.remove(scheduler)
                listeners.insert(0, scheduler)
                start = scheduler.callback

                async def start_first(event):
                    await start(event)
                    await waiting.wait()

                from dataclasses import replace
                listeners[0] = replace(scheduler, callback=start_first)
                try:
                    await host.start_runtime()
                finally:
                    listeners[0] = scheduler
                async with asyncio.timeout(5):
                    await probing.wait()
                assert root.context.require(ServiceKey("fixture.calls")) == []
                assert not archive_waiter.done()
                assert not (tmp_path / "effect.txt").exists()
                release.set()
                async with asyncio.timeout(5):
                    await archive_waiter
                fire = await settled(store, fire_key(job))
                assert fire.status == "delivered"
                records = DeliveryRecords(log.owner("plugin:delivery"), "delivery_policy")
                assert records.read("original-output", "test")[1].phase == ("delivered" if confirmed else "unknown")
                assert state["queries"] == 1
                assert len(next((tmp_path / "workspace").rglob("sent.jsonl")).read_text().splitlines()) == 2
    finally:
        release.set()
        await host.terminate_all()
        log.close()
