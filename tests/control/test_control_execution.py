from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from agent.control.models import TurnItemKind, TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.looping.core import AgentLoop
from agent.looping.session_lane import SessionLaneRegistry
from agent.tools.registry import ToolRegistry
from agent.tools.shell import ShellTool
from agent.tools.unified_exec import ShellProcessManager
from bootstrap.control_execution import execute_control_turn
from bus.event_bus import EventBus
from bus.events import OutboundMessage, TurnDisposition
from bus.events_lifecycle import ToolCallCompleted, ToolCallStarted, TurnCommitted
from session.store import SessionStore


@pytest.mark.asyncio
async def test_committed_control_turn_survives_shell_cleanup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    tools = ToolRegistry()
    tools.register(shell)
    loop = AgentLoop.__new__(AgentLoop)
    loop.tools = tools
    loop._event_bus = bus
    loop._processing_state = None
    loop._interrupt_states = {}
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = None
    loop._resume_interrupted_message = AsyncMock(
        side_effect=lambda message, _key: (message, False)
    )
    loop._observe_turn_started = AsyncMock()
    executions = 0

    async def process(message: object, key: str, **_kwargs: object) -> OutboundMessage:
        nonlocal executions
        executions += 1
        metadata = cast(Any, message).metadata
        turn_id = str(metadata["control_turn_id"])
        await bus.fanout(
            TurnCommitted(
                session_key=key,
                channel="mobile",
                chat_id="cleanup",
                input_message="update",
                persisted_user_message="update",
                assistant_response="all updated",
                tools_used=["shell"],
                turn_id=turn_id,
            )
        )
        return OutboundMessage(
            "mobile",
            "cleanup",
            "all updated",
            session_message_id="mobile:cleanup:1",
        )

    loop._core_runner = SimpleNamespace(process=process)

    async def fail_cleanup(_owner_session_key: str) -> None:
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(shell, "terminate_owner", fail_cleanup)
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest):
        return await execute_control_turn(loop, bus, request)

    runtime = ConversationRuntime(store, execute)
    try:
        with caplog.at_level("ERROR", logger="agent.loop"):
            handle = await runtime.start_turn(
                TurnRequest(
                    "mobile:cleanup",
                    "update",
                    {"channel": "mobile", "chatId": "cleanup"},
                )
            )
            result = await handle.result()

        persisted = runtime.read_turn(handle.thread_id, handle.id)
        assert result.status.value == "completed"
        assert result.final_response == "all updated"
        assert persisted.status.value == "completed"
        assert persisted.final_response == "all updated"
        assert persisted.error is None
        assert executions == 1
        assert "event=cleanup_degraded" in caplog.text
    finally:
        await runtime.shutdown()
        await bus.aclose()
        store.close()


@pytest.mark.asyncio
async def test_tool_started_is_published_before_core_execution_finishes(tmp_path: Path) -> None:
    bus = EventBus()
    release = asyncio.Event()

    class _Loop:
        async def process_direct_message(
            self,
            _content: str,
            **kwargs: object,
        ) -> OutboundMessage:
            turn_id = str(kwargs["turn_id"])
            await bus.observe(
                ToolCallStarted(
                    session_key="programmatic:live",
                    channel="programmatic",
                    chat_id="programmatic:live",
                    iteration=1,
                    call_id="call-live",
                    tool_name="lookup",
                    arguments={"query": "now"},
                    turn_id=turn_id,
                )
            )
            await release.wait()
            await bus.observe(
                ToolCallCompleted(
                    session_key="programmatic:live",
                    channel="programmatic",
                    chat_id="programmatic:live",
                    iteration=1,
                    call_id="call-live",
                    tool_name="lookup",
                    arguments={"query": "now"},
                    final_arguments={"query": "now"},
                    status="completed",
                    result_preview="found",
                    turn_id=turn_id,
                )
            )
            await bus.fanout(
                TurnCommitted(
                    session_key="programmatic:live",
                    channel="programmatic",
                    chat_id="programmatic:live",
                    input_message="hello",
                    persisted_user_message="hello",
                    assistant_response="done",
                    tools_used=["lookup"],
                    turn_id=turn_id,
                )
            )
            return OutboundMessage(
                "programmatic",
                "programmatic:live",
                "done",
                session_message_id="programmatic:live:1",
            )

    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest):
        return await execute_control_turn(cast(Any, _Loop()), bus, request)

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("programmatic:live", "hello"))
    events = handle.events().__aiter__()
    live_started = None
    while live_started is None:
        event = await asyncio.wait_for(events.__anext__(), 1)
        item = event.data.get("item")
        if event.method == "item/started" and isinstance(item, dict):
            data = item.get("data")
            if isinstance(data, dict) and data.get("callId") == "call-live":
                live_started = event

    assert runtime.read_turn(handle.thread_id, handle.id).status.value == "in_progress"
    release.set()
    result = await handle.result()
    assert [item.kind for item in result.items] == [
        TurnItemKind.USER_MESSAGE,
        TurnItemKind.TOOL_CALL,
        TurnItemKind.ASSISTANT_MESSAGE,
    ]
    assert result.items[-1].data["sessionMessageId"] == "programmatic:live:1"
    await runtime.shutdown()
    await bus.aclose()
    store.close()


@pytest.mark.asyncio
async def test_short_circuited_turn_completes_without_turn_committed(tmp_path: Path) -> None:
    bus = EventBus()

    class _Loop:
        async def process_direct_message(
            self,
            _content: str,
            **_kwargs: object,
        ) -> OutboundMessage:
            return OutboundMessage(
                "telegram",
                "123",
                "memory status",
                turn_disposition=TurnDisposition.SHORT_CIRCUITED,
            )

    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest):
        return await execute_control_turn(cast(Any, _Loop()), bus, request)

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("telegram:123", "/memorystatus"))

    result = await handle.result()

    assert result.status.value == "completed"
    assert result.final_response == "memory status"
    assert result.usage is None
    await runtime.shutdown()
    await bus.aclose()
    store.close()


@pytest.mark.asyncio
async def test_control_execution_preserves_inbound_metadata(tmp_path: Path) -> None:
    bus = EventBus()

    class _Loop:
        async def process_direct_message(
            self,
            _content: str,
            **kwargs: object,
        ) -> OutboundMessage:
            assert kwargs["metadata"] == {
                "client_message_id": "client-1",
                "reply_to_message_id": "mobile:one:0",
            }
            assert kwargs["runtime_selector"] == "latest"
            turn_id = str(kwargs["turn_id"])
            await bus.fanout(
                TurnCommitted(
                    session_key="mobile:one",
                    channel="mobile",
                    chat_id="one",
                    input_message="reply",
                    persisted_user_message="reply",
                    assistant_response="done",
                    tools_used=[],
                    turn_id=turn_id,
                )
            )
            return OutboundMessage("mobile", "one", "done")

    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest):
        return await execute_control_turn(cast(Any, _Loop()), bus, request)

    runtime = ConversationRuntime(store, execute)
    request = TurnRequest(
        "mobile:one",
        "reply",
        {
            "channel": "mobile",
            "chatId": "one",
            "runtime": "latest",
            "inboundMetadata": {
                "client_message_id": "client-1",
                "reply_to_message_id": "mobile:one:0",
            },
        },
    )
    result = await (await runtime.start_turn(request)).result()

    assert result.status.value == "completed"
    await runtime.shutdown()
    await bus.aclose()
    store.close()


@pytest.mark.asyncio
async def test_regular_turn_without_turn_committed_still_fails(tmp_path: Path) -> None:
    bus = EventBus()

    class _Loop:
        async def process_direct_message(
            self,
            _content: str,
            **_kwargs: object,
        ) -> OutboundMessage:
            return OutboundMessage("programmatic", "regular", "incomplete")

    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest):
        return await execute_control_turn(cast(Any, _Loop()), bus, request)

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("programmatic:regular", "hello"))

    result = await handle.result()

    assert result.status.value == "failed"
    assert result.error is not None
    assert result.error.message.startswith("turn 缺少 TurnCommitted 事件")
    await runtime.shutdown()
    await bus.aclose()
    store.close()
