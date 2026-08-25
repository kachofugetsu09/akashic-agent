from __future__ import annotations

import asyncio
import ast
import shutil
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from typing import Any, cast

import pytest

import agent.plugins.manager as plugin_manager_module
from agent.control.timer import TimerReceipt, TimerStatus
from agent.turn_effects import PostCommitEffect, TurnStorage
from agent.control.scoped_turn import TurnAdmissionRetiredError
from agent.plugin_composition.channels import ChannelDeliveryReceipt, DeliveryStatus
from agent.plugin_composition.deliveries import PluginDeliveries
from agent.plugin_composition.scoped_turns import PluginScopedTurns
from agent.plugin_composition.timers import PluginTimers
from agent.scheduler import (
    JobStore,
    SCHEDULE_MAX_ACTIVE_JOBS,
    ScheduleCapacityError,
    ScheduledJob,
)
from agent.control.models import TurnRequest
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from session.store import SessionStore
from plugins.scheduler import plugin as scheduler_plugin
from plugins.scheduler.plugin import SchedulerRuntime


class _TimerHandle:
    def __init__(self, timer_id: str, deadline: datetime, now: datetime) -> None:
        self._id = timer_id
        self.deadline = deadline
        self.now = now
        self.future: asyncio.Future[TimerReceipt] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def id(self) -> str:
        return self._id

    async def result(self) -> TimerReceipt:
        return await asyncio.shield(self.future)

    async def cancel(self) -> TimerReceipt:
        if not self.future.done():
            self.future.set_result(self._receipt(TimerStatus.CANCELLED))
        return await self.future

    async def cleanup(self) -> None:
        _ = await self.cancel()

    def fire(self) -> None:
        self.future.set_result(self._receipt(TimerStatus.FIRED))

    def _receipt(self, status: TimerStatus) -> TimerReceipt:
        return TimerReceipt(self.id, self.deadline, self.now, status)


class _Timer:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.handles: list[_TimerHandle] = []

    def schedule(self, deadline: datetime) -> _TimerHandle:
        handle = _TimerHandle(f"timer:{len(self.handles)}", deadline, self.now)
        self.handles.append(handle)
        return handle


class _TurnHandle:
    def __init__(self, content: str | None, status: str = "completed") -> None:
        self._result = SimpleNamespace(
            status=SimpleNamespace(value=status),
            final_response=content,
        )

    async def result(self) -> object:
        return self._result

    async def cleanup(self) -> None:
        return None


class _Turns:
    def __init__(self, response: str | None = "soft result") -> None:
        self.response = response
        self.sessions: list[tuple[str, dict[str, object]]] = []
        self.starts: list[dict[str, object]] = []

    async def ensure_session(self, key: str, *, metadata: dict[str, object]) -> str:
        self.sessions.append((key, metadata))
        return key

    async def start(
        self, session_id: str, content: str, **kwargs: object
    ) -> _TurnHandle:
        self.starts.append({"session_id": session_id, "content": content, **kwargs})
        return _TurnHandle(self.response)


class _RetiredTurns(_Turns):
    async def start(
        self, session_id: str, content: str, **kwargs: object
    ) -> _TurnHandle:
        _ = session_id, content, kwargs
        raise TurnAdmissionRetiredError("fixture Root retired before admission")


async def _settled() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def _eventually(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


def _job(
    now: datetime,
    *,
    tier: str = "instant",
    trigger: str = "after",
    job_id: str = "weather-d494",
    fire_at: datetime | None = None,
) -> ScheduledJob:
    return ScheduledJob(
        id=job_id,
        trigger=trigger,
        tier=tier,
        fire_at=fire_at or now + timedelta(seconds=30),
        channel="fixture",
        chat_id="chat",
        interval_seconds=60 if trigger == "every" else None,
        message="drink" if tier == "instant" else None,
        prompt="weather" if tier == "soft" else None,
        timezone="UTC",
    )


def _runtime(tmp_path, now: datetime, timer: _Timer, turns: _Turns, deliveries):
    return SchedulerRuntime(
        tmp_path / "schedules.json",
        PluginTimers(timer),
        cast(PluginScopedTurns, turns),
        PluginDeliveries(deliveries),
        now=lambda: now,
    )


@pytest.mark.asyncio
async def test_scheduler_shadow_one_shot_fires_delivers_and_disables(tmp_path) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timer = _Timer(now)
    delivered = []

    async def send(message):
        delivered.append(message)
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    runtime = _runtime(tmp_path, now, timer, _Turns(), send)
    await runtime.start()
    await runtime.add_job(_job(now))
    timer.handles[0].fire()
    await _settled()

    stored = runtime.store.load()
    assert [message.content for message in delivered] == ["drink"]
    assert runtime.wait_count == 0
    assert stored[0].enabled is False
    assert stored[0].run_count == 1


@pytest.mark.asyncio
async def test_scheduler_shadow_every_settles_then_arms_exactly_one_next_wait(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timer = _Timer(now)

    async def send(_message):
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    runtime = _runtime(tmp_path, now, timer, _Turns(), send)
    await runtime.start()
    await runtime.add_job(_job(now, trigger="every"))
    timer.handles[0].fire()
    await _settled()

    stored = runtime.store.load()[0]
    assert len(timer.handles) == 2
    assert runtime.wait_count == 1
    assert stored.run_count == 1
    assert stored.fire_at > now + timedelta(seconds=30)
    await runtime.close()
    assert runtime.wait_count == 0


@pytest.mark.asyncio
async def test_scheduler_shadow_delivery_rejection_does_not_count_success(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timer = _Timer(now)

    async def reject(_message):
        return ChannelDeliveryReceipt(
            "delivery:1", DeliveryStatus.REJECTED, error="offline"
        )

    runtime = _runtime(tmp_path, now, timer, _Turns(), reject)
    await runtime.start()
    await runtime.add_job(_job(now))
    timer.handles[0].fire()
    await _settled()

    stored = runtime.store.load()[0]
    assert stored.enabled is False
    assert stored.run_count == 0


@pytest.mark.asyncio
async def test_scheduler_shadow_soft_uses_stateless_memoryless_scoped_turn(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timer = _Timer(now)
    turns = _Turns("weather result")
    delivered = []

    async def send(message):
        delivered.append(message)
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    runtime = _runtime(tmp_path, now, timer, turns, send)
    await runtime.start()
    await runtime.add_job(_job(now, tier="soft"))
    timer.handles[0].fire()
    await _settled()

    scope = turns.starts[0]["scope"]
    assert turns.sessions[0][0] == "scheduler:weather-d494"
    assert scope.storage is TurnStorage.IN_MEMORY
    assert scope.post_commit_effect is PostCommitEffect.SUPPRESS
    assert scope.disabled_prompt_sections == frozenset({"memory"})
    assert scope.tool_grant.allows("web_search") is True
    assert scope.tool_grant.allows("message_push") is False
    assert [message.content for message in delivered] == ["weather result"]


@pytest.mark.asyncio
async def test_scheduler_hands_unaccepted_soft_job_to_new_root(tmp_path) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    old_timer = _Timer(now)

    async def send(_message):
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    old = _runtime(tmp_path, now, old_timer, _RetiredTurns(), send)
    await old.start()
    await old.add_job(_job(now, tier="soft"))
    old_timer.handles[0].fire()
    await _settled()

    stored = old.store.load()[0]
    assert stored.enabled is True
    assert stored.run_count == 0
    assert old.wait_count == 0

    new_timer = _Timer(now)
    new = _runtime(tmp_path, now, new_timer, _Turns(), send)
    await new.start()
    assert new.wait_count == 1
    await old.close()
    await new.close()


@pytest.mark.asyncio
async def test_scheduler_shadow_cancel_and_dispose_leave_no_wait_or_delivery(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timer = _Timer(now)
    delivered = []

    async def send(message):
        delivered.append(message)
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    runtime = _runtime(tmp_path, now, timer, _Turns(), send)
    await runtime.start()
    await runtime.add_job(_job(now))

    assert await runtime.cancel_job("weather-d494") is True
    await runtime.close()

    assert runtime.wait_count == 0
    assert runtime.store.load() == []
    assert delivered == []


@pytest.mark.asyncio
async def test_scheduler_shadow_capacity_rejects_before_write_or_wait(tmp_path) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timer = _Timer(now)

    async def send(_message):
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    runtime = _runtime(tmp_path, now, timer, _Turns(), send)
    await runtime.start()
    for index in range(SCHEDULE_MAX_ACTIVE_JOBS):
        await runtime.add_job(_job(now, job_id=f"job-{index}"))

    before = (len(runtime.store.load()), len(timer.handles), runtime.wait_count)
    with pytest.raises(ScheduleCapacityError):
        await runtime.add_job(_job(now, job_id="overflow"))

    assert before == (
        SCHEDULE_MAX_ACTIVE_JOBS,
        SCHEDULE_MAX_ACTIVE_JOBS,
        SCHEDULE_MAX_ACTIVE_JOBS,
    )
    assert (len(runtime.store.load()), len(timer.handles), runtime.wait_count) == before
    await runtime.close()


@pytest.mark.asyncio
async def test_scheduler_shadow_recovers_grace_expired_every_and_disabled(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    path = tmp_path / "schedules.json"
    within = _job(
        now,
        job_id="within",
        fire_at=now - timedelta(seconds=100),
    )
    expired = _job(
        now,
        job_id="expired",
        fire_at=now - timedelta(seconds=301),
    )
    every = _job(
        now,
        trigger="every",
        job_id="every",
        fire_at=now - timedelta(hours=3),
    )
    disabled = replace(_job(now, job_id="disabled"), enabled=False)
    JobStore(path).save({job.id: job for job in (within, expired, every, disabled)})
    timer = _Timer(now)

    async def send(_message):
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    runtime = _runtime(tmp_path, now, timer, _Turns(), send)
    await runtime.start()

    stored = {job.id: job for job in runtime.store.load()}
    assert {wait.job.id for wait in runtime._waits.values()} == {"within", "every"}
    assert stored["expired"].enabled is False
    assert stored["every"].fire_at > now
    assert stored["disabled"].enabled is False
    await runtime.close()


@pytest.mark.asyncio
async def test_scheduler_shadow_no_work_arms_nothing(tmp_path) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    disabled = replace(_job(now), enabled=False)
    JobStore(tmp_path / "schedules.json").save({disabled.id: disabled})
    timer = _Timer(now)

    async def send(_message):
        raise AssertionError("disabled job must not deliver")

    runtime = _runtime(tmp_path, now, timer, _Turns(), send)
    await runtime.start()
    assert runtime.wait_count == 0
    assert timer.handles == []
    await runtime.close()


@pytest.mark.asyncio
async def test_scheduler_shadow_restart_arms_one_wait_and_cron_advances(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)

    async def send(_message):
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    first_timer = _Timer(now)
    first = _runtime(tmp_path, now, first_timer, _Turns(), send)
    cron = replace(
        _job(now, trigger="every", job_id="cron"),
        interval_seconds=None,
        cron_expr="0 9 * * *",
        timezone="Asia/Shanghai",
    )
    await first.add_job(cron)
    await first.close()

    second_timer = _Timer(now)
    restarted = _runtime(tmp_path, now, second_timer, _Turns(), send)
    await restarted.start()
    assert restarted.wait_count == 1
    assert len(second_timer.handles) == 1
    second_timer.handles[0].fire()
    await _settled()
    stored = restarted.store.load()[0]
    assert stored.run_count == 1
    assert stored.fire_at > cron.fire_at
    assert restarted.wait_count == 1
    await restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("soft_response", [None, ""])
async def test_scheduler_shadow_soft_terminal_without_content_is_failure(
    tmp_path,
    soft_response,
) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timer = _Timer(now)
    delivered = []

    async def send(message):
        delivered.append(message)
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    runtime = _runtime(tmp_path, now, timer, _Turns(soft_response), send)
    await runtime.start()
    await runtime.add_job(_job(now, tier="soft"))
    timer.handles[0].fire()
    await _settled()

    stored = runtime.store.load()[0]
    assert stored.enabled is False
    assert stored.run_count == 0
    assert delivered == []


@pytest.mark.asyncio
async def test_scheduler_plugin_tools_keep_schema_and_drive_private_runtime(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    timer = _Timer(now)

    async def send(_message):
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    runtime = _runtime(tmp_path, now, timer, _Turns(), send)
    try:
        definitions = {item.name: item for item in scheduler_plugin._tool_definitions()}
        assert set(definitions) == {"schedule", "list_schedules", "cancel_schedule"}
        assert all(
            item.parameters["additionalProperties"] is False
            for item in definitions.values()
        )
        result = await scheduler_plugin._schedule(
            runtime,
            object(),
            {
                "tier": "instant",
                "trigger": "after",
                "when": "5m",
                "message": "drink",
                "channel": "fixture",
                "chat_id": "chat",
                "request_time": now.isoformat(),
                "name": "water",
            },
        )
        assert result.startswith("已注册定时任务 「water」")
        assert "water" in await scheduler_plugin._list_schedules(runtime, object(), {})
        assert (
            await scheduler_plugin._cancel_schedule(
                runtime, object(), {"name": "water"}
            )
            == "已取消 1 个名为 'water' 的任务"
        )
        assert runtime.wait_count == 0
    finally:
        await runtime.close()


def test_scheduler_plugin_imports_only_public_composition_and_domain_ports() -> None:
    source = Path(scheduler_plugin.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden = {
        "agent.plugins.manager",
        "agent.looping.core",
        "agent.tools.registry",
        "session.store",
        "agent.tools.message_push",
    }
    assert imported.isdisjoint(forbidden)
    assert "SchedulerService" not in source


@pytest.mark.asyncio
async def test_scheduler_v3_loader_mounts_dormant_and_candidate_never_reads_store(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    schedules = workspace / "schedules.json"
    schedules.write_text("not-json", encoding="utf-8")
    store = SessionStore(workspace / "sessions.db")

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        raise AssertionError("dormant scheduler must not start a Turn")

    async def deliver(_message):
        raise AssertionError("dormant scheduler must not deliver")

    conversation = ConversationRuntime(store, execute)
    manager = PluginManager(
        plugin_dirs=[Path(__file__).resolve().parents[1] / "plugins" / "scheduler"],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
        programmatic_session_reader=store.get_session_meta,
    )
    manager.bind_delivery_sender(deliver)
    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None
    generation = snapshot.generations["scheduler"]
    plugin = cast(ComposablePlugin, generation.instance)
    assert plugin.workspace_files == ("schedules.json",)
    assert snapshot.tool_registry is not None
    assert snapshot.tool_registry.get_document("schedule").risk == "write"
    assert schedules.read_text(encoding="utf-8") == "not-json"

    candidate = await manager.prepare_candidate("scheduler")
    assert candidate is not None
    assert schedules.read_text(encoding="utf-8") == "not-json"
    await manager.discard_prepared("scheduler")
    if snapshot.composition_root is not None:
        await snapshot.composition_root.dispose()
    await conversation.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_scheduler_runtime_lifecycle_follows_hot_reloaded_stable_root(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = Path(scheduler_plugin.__file__)
    plugin_dir = tmp_path / "plugins" / "scheduler"
    plugin_dir.mkdir(parents=True)
    shutil.copy2(source, plugin_dir / "plugin.py")
    now = datetime.now(UTC)
    job = _job(now, fire_at=now + timedelta(hours=1))
    JobStore(workspace / "schedules.json").save({job.id: job})
    store = SessionStore(workspace / "sessions.db")

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        raise AssertionError("future fixture must not start a Turn")

    async def deliver(_message):
        raise AssertionError("future fixture must not deliver")

    conversation = ConversationRuntime(store, execute)
    manager = PluginManager(
        plugin_dirs=[plugin_dir],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
        programmatic_session_reader=store.get_session_meta,
    )
    manager.bind_delivery_sender(deliver)
    await manager.load_all()
    lifecycle = asyncio.create_task(manager.run_runtime_services())

    def active_waits() -> int:
        return sum(
            task.get_name() == f"scheduler:{job.id}" and not task.done()
            for task in asyncio.all_tasks()
        )

    try:
        await _eventually(lambda: active_waits() == 1)
        assert manager.current_snapshot.lease_count == 0

        with (plugin_dir / "plugin.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# fixture revision\n")
        assert await manager.prepare_candidate("scheduler") is not None
        result = await manager.publish_prepared("scheduler")
        assert result["publication_state"] == "committed"

        await _eventually(lambda: active_waits() == 1)
    finally:
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await _eventually(lambda: active_waits() == 0)
        await manager.terminate_all()
        await conversation.shutdown()
        store.close()


@pytest.mark.asyncio
async def test_scheduler_hot_reload_hands_unaccepted_job_to_new_root(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove an old fire cannot mutate the ledger after its Root retires."""

    # 1. Mount the real plugin with controllable Core Timer implementations.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = Path(scheduler_plugin.__file__)
    plugin_dir = tmp_path / "plugins" / "scheduler"
    plugin_dir.mkdir(parents=True)
    shutil.copy2(source, plugin_dir / "plugin.py")
    now = datetime.now(UTC)
    job = _job(now, tier="soft", fire_at=now + timedelta(hours=1))
    JobStore(workspace / "schedules.json").save({job.id: job})
    timers: list[_Timer] = []

    def timer_factory() -> _Timer:
        timer = _Timer(now)
        timers.append(timer)
        return timer

    monkeypatch.setattr(plugin_manager_module, "AsyncioOneShotTimer", timer_factory)
    executions: list[TurnRequest] = []
    delivered = []
    store = SessionStore(workspace / "sessions.db")

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        executions.append(request)
        return ControlExecutionResult(response="new root result")

    async def deliver(message):
        delivered.append(message)
        return ChannelDeliveryReceipt("delivery:1", DeliveryStatus.DELIVERED)

    conversation = ConversationRuntime(store, execute)
    manager = PluginManager(
        plugin_dirs=[plugin_dir],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
        programmatic_session_reader=store.get_session_meta,
    )
    manager.bind_delivery_sender(deliver)
    await manager.load_all()
    lifecycle = asyncio.create_task(manager.run_runtime_services())
    release_stop = asyncio.Event()
    stop_entered = asyncio.Event()
    try:
        await _eventually(lambda: sum(bool(timer.handles) for timer in timers) == 1)
        old_timer = next(timer for timer in timers if timer.handles)
        old_snapshot = manager.current_snapshot
        original_stop = cast(Any, manager)._stop_runtime_snapshot

        async def gated_stop(snapshot) -> None:
            if snapshot is old_snapshot:
                stop_entered.set()
                await release_stop.wait()
            await original_stop(snapshot)

        cast(Any, manager)._stop_runtime_snapshot = gated_stop
        with (plugin_dir / "plugin.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# handoff fixture revision\n")
        assert await manager.prepare_candidate("scheduler") is not None
        result = await manager.publish_prepared("scheduler")
        assert result["publication_state"] == "committed"
        await asyncio.wait_for(stop_entered.wait(), timeout=5)

        # 2. Fire the retired Root while stopping is gated; it must not settle state.
        old_timer.handles[0].fire()
        await _settled()
        retained = JobStore(workspace / "schedules.json").load()[0]
        assert retained.enabled is True
        assert retained.run_count == 0
        assert executions == []
        assert delivered == []

        # 3. Let the old Root settle; the new Root re-arms and completes once.
        release_stop.set()
        await _eventually(lambda: sum(bool(timer.handles) for timer in timers) == 2)
        new_timer = next(
            timer for timer in timers if timer is not old_timer and timer.handles
        )
        new_timer.handles[0].fire()
        await _eventually(lambda: len(delivered) == 1)
        settled = JobStore(workspace / "schedules.json").load()[0]
        assert len(executions) == 1
        assert delivered[0].content == "new root result"
        assert settled.enabled is False
        assert settled.run_count == 1
    finally:
        release_stop.set()
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()
        await conversation.shutdown()
        store.close()
