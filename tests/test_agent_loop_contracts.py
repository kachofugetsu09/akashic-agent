from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from agent.control.context import running_turn_id
from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    DeliveryStatus as ChannelDeliveryStatus,
)
from agent.looping.core import AgentLoop
from agent.looping.ports import LLMConfig
from agent.looping.session_lane import SessionLaneRegistry
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from bus.event_bus import EventBus
from bus.events import (
    InboundItem,
    InboundMessage,
    OutboundMessage,
    TurnTerminalStatus,
)
from bus.events_lifecycle import TurnStarted
from bus.queue import MessageBus
from core.error_context import (
    current_client_message_id,
    current_session_key,
)
from session.store import SessionStore
from tests.model_plugin_fakes import build_test_model_store


@pytest.mark.asyncio
async def test_runtime_admission_reuses_exact_task_bound_snapshot() -> None:
    old_snapshot = SimpleNamespace(snapshot_id="old", tool_registry=None)
    lease = SimpleNamespace(
        active=True,
        snapshot=old_snapshot,
        validation_candidate_plugin_ids=frozenset(),
    )
    store = SimpleNamespace(
        current=SimpleNamespace(snapshot_id="new"),
        acquire=AsyncMock(side_effect=AssertionError("must not reacquire current")),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = store

    async def process(_item: object, **_kwargs: object) -> str:
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        assert snapshot is old_snapshot
        return snapshot.snapshot_id

    loop._process = process
    item = SimpleNamespace(session_key="feishu:chat")
    token = bind_runtime_snapshot(cast(Any, lease))
    try:
        result = await loop._process_with_runtime_admission(cast(Any, item))
    finally:
        reset_runtime_snapshot(token)

    assert result == "old"
    store.acquire.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_cleans_active_state_before_inbound_completion_failure() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    bus = SimpleNamespace(
        consume_inbound=AsyncMock(return_value=item),
        complete_inbound=AsyncMock(side_effect=RuntimeError("ack failed")),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = bus
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = AsyncMock(
        return_value=OutboundMessage(channel="cli", chat_id="1", content="ok")
    )

    with pytest.raises(RuntimeError, match="ack failed"):
        await loop.run()

    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}


@pytest.mark.asyncio
async def test_run_propagates_runtime_cancellation_after_ack() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    consumed = False
    started = asyncio.Event()

    async def consume_inbound() -> InboundMessage:
        nonlocal consumed
        if consumed:
            raise AssertionError("运行器取消后不应继续消费消息")
        consumed = True
        return item

    async def process(
        _item: InboundMessage,
        *,
        execution_turn_id: str | None = None,
    ) -> OutboundMessage:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    complete_inbound = AsyncMock()
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = SimpleNamespace(
        consume_inbound=consume_inbound,
        complete_inbound=complete_inbound,
    )
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = process

    run_task = asyncio.create_task(loop.run())
    await started.wait()
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=0.5)

    complete_inbound.assert_awaited_once_with(item)
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}
    assert loop._running is False


@pytest.mark.asyncio
async def test_run_waits_for_ack_before_propagating_runtime_cancellation() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    consumed = False
    started = asyncio.Event()
    ack_started = asyncio.Event()
    release_ack = asyncio.Event()

    async def consume_inbound() -> InboundMessage:
        nonlocal consumed
        if consumed:
            raise AssertionError("运行器取消后不应继续消费消息")
        consumed = True
        return item

    async def process(
        _item: InboundMessage,
        *,
        execution_turn_id: str | None = None,
    ) -> OutboundMessage:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def complete_inbound(_item: InboundMessage) -> None:
        ack_started.set()
        await release_ack.wait()

    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = SimpleNamespace(
        consume_inbound=consume_inbound,
        complete_inbound=complete_inbound,
    )
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = process

    run_task = asyncio.create_task(loop.run())
    await started.wait()
    run_task.cancel()
    await ack_started.wait()
    assert run_task.done() is False

    release_ack.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=0.5)

    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}


@pytest.mark.asyncio
async def test_stop_cancels_active_turn_and_acknowledges_inbound() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def process(
        _item: InboundMessage,
        *,
        execution_turn_id: str | None = None,
    ) -> OutboundMessage:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    bus = SimpleNamespace(
        consume_inbound=AsyncMock(return_value=item),
        complete_inbound=AsyncMock(),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = bus
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = process

    run_task = asyncio.create_task(loop.run())
    await started.wait()
    loop.stop()
    await asyncio.wait_for(run_task, timeout=0.5)

    assert cancelled.is_set()
    bus.complete_inbound.assert_awaited_once_with(item)
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}


def _real_path_loop(
    bus: Any,
    core_process: object,
) -> AgentLoop:
    """最小化真实执行链脚手架：真实 _run_inbound_turn / _process_with_runtime_admission / _process，
    只替换 _react 与总线/事件观察点。"""
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop._session_services = SimpleNamespace(
        session_manager=SimpleNamespace(
            get_or_create=lambda _key: SimpleNamespace(metadata={}),
        )
    )
    loop.bus = bus
    loop._event_bus = EventBus()
    loop.tools = SimpleNamespace(get_tool=lambda _name: None)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = build_test_model_store(object())
    loop._passive_pipeline = SimpleNamespace(
        run_command=AsyncMock(return_value=None),
    )

    async def react(
        message: InboundMessage,
        key: str,
        *,
        chat_models: object,
        model_id: str | None,
        reasoning_effort: str | None,
        dispatch_outbound: bool,
        command_admitted: bool,
    ) -> OutboundMessage:
        _ = command_admitted, chat_models, model_id, reasoning_effort
        return await core_process(  # type: ignore[operator]
            message,
            key,
            dispatch_outbound=dispatch_outbound,
        )

    loop._react = react
    loop._outbound_port = SimpleNamespace(
        dispatch=AsyncMock(
            return_value=ChannelDeliveryReceipt(
                delivery_id="test-delivery",
                status=ChannelDeliveryStatus.DELIVERED,
            )
        )
    )
    loop._processing_state = None
    loop._active_tasks = {}
    loop._active_turn_states = {}
    return loop


@pytest.mark.asyncio
async def test_error_final_carries_authoritative_execution_turn_id() -> None:
    """真实 child 链：child 捕获 execution ID 后抛错，parent 错误 final 同源发布。"""
    item = InboundMessage(
        channel="mobile",
        sender="user",
        chat_id="chat-a",
        content="hello",
        metadata={"display_content": "继续说明"},
    )
    bus = SimpleNamespace(
        complete_inbound=AsyncMock(),
    )
    observed_child_turn_ids: list[str] = []
    started_events: list[TurnStarted] = []

    async def core_process(
        _msg: InboundMessage,
        _key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        observed_child_turn_ids.append(running_turn_id.get())
        assert running_turn_id.get().startswith("turn:")
        raise RuntimeError("boom")

    loop = _real_path_loop(bus, core_process)
    loop._event_bus.on(TurnStarted, started_events.append)

    await loop._run_inbound_turn(item)

    (outbound,) = loop._outbound_port.dispatch.call_args.args
    assert outbound.content == "出错：boom"
    assert outbound.control_turn_id == observed_child_turn_ids[0]
    assert outbound.control_turn_id.startswith("turn:")
    assert started_events[0].turn_id == observed_child_turn_ids[0]
    assert started_events[0].content == "继续说明"
    bus.complete_inbound.assert_awaited_once_with(item)
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}


@pytest.mark.asyncio
async def test_error_final_preserves_preprovided_execution_turn_id() -> None:
    """预提供 execution ID 原样贯通；与 control_turn_id 分叉时 execution 恒为 owner。"""
    item = InboundMessage(
        channel="mobile",
        sender="user",
        chat_id="chat-a",
        content="hello",
        metadata={
            "_control_execution_turn_id": "turn:pre",
            "control_turn_id": "interaction:1",
        },
    )
    bus = SimpleNamespace(
        complete_inbound=AsyncMock(),
    )
    started_events: list[TurnStarted] = []

    async def core_process(
        _msg: InboundMessage,
        _key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        assert running_turn_id.get() == "turn:pre"
        raise RuntimeError("boom")

    loop = _real_path_loop(bus, core_process)
    loop._event_bus.on(TurnStarted, started_events.append)

    await loop._run_inbound_turn(item)

    (outbound,) = loop._outbound_port.dispatch.call_args.args
    assert outbound.control_turn_id == "interaction:1"
    assert outbound.execution_attempt_id == "turn:pre"
    assert outbound.terminal_status is TurnTerminalStatus.FAILED
    assert started_events[0].turn_id == "turn:pre"
    bus.complete_inbound.assert_awaited_once_with(item)
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}


@pytest.mark.asyncio
async def test_error_final_preserves_control_turn_id_only_metadata() -> None:
    """只有 control_turn_id 的 direct-call/恢复合同：原样保留为该轮 owner。"""
    item = InboundMessage(
        channel="mobile",
        sender="user",
        chat_id="chat-a",
        content="hello",
        metadata={"control_turn_id": "turn:ctrl"},
    )
    bus = SimpleNamespace(
        complete_inbound=AsyncMock(),
    )
    started_events: list[TurnStarted] = []

    async def core_process(
        _msg: InboundMessage,
        _key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        assert running_turn_id.get() == "turn:ctrl"
        raise RuntimeError("boom")

    loop = _real_path_loop(bus, core_process)
    loop._event_bus.on(TurnStarted, started_events.append)

    await loop._run_inbound_turn(item)

    (outbound,) = loop._outbound_port.dispatch.call_args.args
    assert outbound.control_turn_id == "turn:ctrl"
    assert started_events[0].turn_id == "turn:ctrl"
    bus.complete_inbound.assert_awaited_once_with(item)
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}


@pytest.mark.asyncio
async def test_non_str_control_turn_id_fails_loud_without_polluting_owner_maps() -> (
    None
):
    """非字符串 control_turn_id 在入站边界 fail-loud：TypeError 原样抛出、
    active maps 不被污染、无 TurnStarted、无 error final；execution owner
    尚未建立，绝不确认——durable handoff 保留供下一次恢复。"""
    item = InboundMessage(
        channel="mobile",
        sender="user",
        chat_id="chat-a",
        content="hello",
        metadata={"control_turn_id": 7},
    )
    bus = SimpleNamespace(
        complete_inbound=AsyncMock(),
    )
    started_events: list[TurnStarted] = []

    async def core_process(
        _msg: InboundMessage,
        _key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        raise AssertionError("边界校验失败不应进入核心处理")

    loop = _real_path_loop(bus, core_process)
    loop._event_bus.on(TurnStarted, started_events.append)

    with pytest.raises(TypeError):
        await loop._run_inbound_turn(item)

    assert started_events == []
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}
    loop._outbound_port.dispatch.assert_not_awaited()
    bus.complete_inbound.assert_not_awaited()


@pytest.mark.asyncio
async def test_durable_poison_inbound_not_acked_and_offered_to_next_recovery(
    tmp_path: Path,
) -> None:
    """真实 MessageBus + SessionStore：边界失败绝不确认删除 durable handoff，
    记录仍存在、仍被 durable owner 持有。同进程恢复不复制；模拟重启后由同一
    DB 新建的 SessionStore/MessageBus 重放同一 handoff 再次处理，recover 两次
    仍只入队一次、唯一 accepted owner、handoff id 相同，第二次边界失败后
    row 仍存在。"""
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        channel="akashic",
        sender="device:1",
        chat_id="chat-a",
        content="hello",
        metadata={"client_message_id": "client-poison", "control_turn_id": 7},
        handoff_id="handoff-client-poison",
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    assert consumed.handoff_id is not None
    started_events: list[TurnStarted] = []

    async def core_process(
        _msg: InboundMessage,
        _key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        raise AssertionError("边界校验失败不应进入核心处理")

    loop = _real_path_loop(bus, core_process)
    loop._event_bus.on(TurnStarted, started_events.append)

    with pytest.raises(TypeError):
        await loop._run_inbound_turn(cast(InboundItem, consumed))

    assert started_events == []
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}
    rows = store.list_inbound_handoffs()
    assert [row["handoff_id"] for row in rows] == [consumed.handoff_id]
    assert bus.has_pending_mobile_handoff(
        session_key=consumed.session_key,
        client_message_id="client-poison",
    )
    assert id(consumed) in bus._inbound_accepted

    # 同进程恢复不得复制：poison 仍被 accepted owner 持有，整页重放直接跳过。
    await bus.recover_durable_inbounds()
    assert bus.inbound_size == 0
    assert [row["handoff_id"] for row in store.list_inbound_handoffs()] == [
        consumed.handoff_id
    ]

    # 模拟重启：遗弃旧 bus/store，用同一 DB 新建 SessionStore 与 MessageBus。
    await bus.aclose()
    store.close()
    restarted_store = SessionStore(db_path)
    restarted = MessageBus()
    restarted.bind_durable_inbound_store(restarted_store)
    await restarted.recover_durable_inbounds()
    await restarted.recover_durable_inbounds()
    assert restarted.inbound_size == 1
    owners = [
        owner.item
        for owner in restarted._inbound_accepted.values()
        if isinstance(owner.item, InboundMessage) and owner.item.handoff_id is not None
    ]
    assert [owner.handoff_id for owner in owners] == [consumed.handoff_id]

    recovered = await restarted.consume_inbound()
    assert recovered.handoff_id == consumed.handoff_id
    assert recovered.metadata["control_turn_id"] == 7

    restarted_loop = _real_path_loop(restarted, core_process)
    restarted_loop._event_bus.on(TurnStarted, started_events.append)
    with pytest.raises(TypeError):
        await restarted_loop._run_inbound_turn(cast(InboundItem, recovered))
    assert started_events == []
    assert restarted_loop._active_tasks == {}
    assert restarted_loop._active_turn_states == {}
    assert [row["handoff_id"] for row in restarted_store.list_inbound_handoffs()] == [
        consumed.handoff_id
    ]
    assert id(recovered) in restarted._inbound_accepted
    await restarted.aclose()
    restarted_store.close()


@pytest.mark.asyncio
async def test_owner_task_creation_failure_never_acks_and_leaves_no_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """child task 建立失败：绝不确认、不发布 outbound、不残留任何 active map。"""
    item = InboundMessage(
        channel="mobile",
        sender="user",
        chat_id="chat-a",
        content="hello",
    )
    bus = SimpleNamespace(
        complete_inbound=AsyncMock(),
    )
    real_create_task = asyncio.create_task

    def failing_create(
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
    ):
        if name is not None and name.startswith("agent-turn:"):
            coro.close()
            raise RuntimeError("task create failed")
        return real_create_task(coro, name=name)

    monkeypatch.setattr(asyncio, "create_task", failing_create)
    loop = _real_path_loop(bus, object())

    with pytest.raises(RuntimeError, match="task create failed"):
        await loop._run_inbound_turn(item)

    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}
    loop._outbound_port.dispatch.assert_not_awaited()
    bus.complete_inbound.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_direct_message_real_entry_execution_owner() -> None:
    """公开入口 process_direct_message：execution turn id 恒为 owner，与
    interaction 分组 id 分叉时 execution 胜出，child 与 TurnStarted 同源。"""
    bus = SimpleNamespace()
    observed_child_turn_ids: list[str] = []
    started_events: list[TurnStarted] = []

    async def core_process(
        _msg: InboundMessage,
        _key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        observed_child_turn_ids.append(running_turn_id.get())
        raise RuntimeError("boom")

    loop = _real_path_loop(bus, core_process)
    loop._event_bus.on(TurnStarted, started_events.append)

    with pytest.raises(RuntimeError, match="boom"):
        await loop.process_direct_message(
            "hello",
            turn_id="turn:owner",
            interaction_id="interaction:1",
        )

    assert observed_child_turn_ids == ["turn:owner"]
    assert started_events[0].turn_id == "turn:owner"


@pytest.mark.asyncio
async def test_process_direct_message_non_str_metadata_fails_loud_clean() -> None:
    """公开入口非字符串 metadata：TypeError 原样抛出，不发布 outbound，
    不泄漏任何 contextvar（零副作用）。"""
    bus = SimpleNamespace()
    loop = _real_path_loop(bus, object())

    with pytest.raises(TypeError):
        await loop.process_direct_message(
            "hello",
            metadata={"control_turn_id": 7},
        )

    loop._outbound_port.dispatch.assert_not_awaited()
    assert current_session_key.get() is None
    assert running_turn_id.get() == ""
    assert current_client_message_id.get() == ""
