from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from agent.control.models import TurnItemKind, TurnRequest
from agent.control.runtime import ConversationRuntime
from bootstrap.control_execution import execute_control_turn
from bus.event_bus import EventBus
from bus.events import OutboundMessage, TurnDisposition
from bus.events_lifecycle import ToolCallCompleted, ToolCallStarted, TurnCommitted
from session.store import SessionStore


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
