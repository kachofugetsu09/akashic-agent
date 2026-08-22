from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from pathlib import Path
from typing import cast

import pytest

from agent.control.timer import TimerReceipt, TimerStatus
from agent.plugin_composition.channels import ChannelDeliveryReceipt, DeliveryStatus
from agent.plugin_composition.deliveries import PluginDeliveries
from agent.plugin_composition.timers import PluginTimers
from agent.scheduler import ScheduledJob
from agent.control.models import TurnRequest
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.store import SessionStore
from plugins.scheduler.plugin import SchedulerShadowRuntime


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
    def __init__(self, content: str, status: str = "completed") -> None:
        self._result = SimpleNamespace(
            status=SimpleNamespace(value=status),
            final_response=content,
        )

    async def result(self) -> object:
        return self._result

    async def cleanup(self) -> None:
        return None


class _Turns:
    def __init__(self, response: str = "soft result") -> None:
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


async def _settled() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def _job(
    now: datetime, *, tier: str = "instant", trigger: str = "after"
) -> ScheduledJob:
    return ScheduledJob(
        id="weather-d494",
        trigger=trigger,
        tier=tier,
        fire_at=now + timedelta(seconds=30),
        channel="fixture",
        chat_id="chat",
        interval_seconds=60 if trigger == "every" else None,
        message="drink" if tier == "instant" else None,
        prompt="weather" if tier == "soft" else None,
        timezone="UTC",
    )


def _runtime(tmp_path, now: datetime, timer: _Timer, turns: _Turns, deliveries):
    return SchedulerShadowRuntime(
        tmp_path / "schedules.json",
        PluginTimers(timer),
        turns,
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
    await runtime.add_job(_job(now, tier="soft"))
    timer.handles[0].fire()
    await _settled()

    scope = turns.starts[0]["scope"]
    assert turns.sessions[0][0] == "scheduler:weather-d494"
    assert scope.stateless is True
    assert scope.memory_read is False
    assert scope.memory_write is False
    assert scope.tool_grant.allows("web_search") is True
    assert scope.tool_grant.allows("message_push") is False
    assert [message.content for message in delivered] == ["weather result"]


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
    await runtime.add_job(_job(now))

    assert await runtime.cancel_job("weather-d494") is True
    await runtime.close()

    assert runtime.wait_count == 0
    assert runtime.store.load() == []
    assert delivered == []


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
    assert plugin.module.runtime is not None
    assert plugin.module.runtime.wait_count == 0
    assert schedules.read_text(encoding="utf-8") == "not-json"

    candidate = await manager.prepare_candidate("scheduler")
    assert candidate is not None
    assert schedules.read_text(encoding="utf-8") == "not-json"
    await manager.discard_prepared("scheduler")
    if snapshot.composition_root is not None:
        await snapshot.composition_root.dispose()
    await conversation.shutdown()
    store.close()
