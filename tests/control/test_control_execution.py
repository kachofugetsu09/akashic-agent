from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from agent.control.models import TurnItemKind, TurnRequest
from agent.control.runtime import ConversationRuntime
from bootstrap.control_execution import execute_control_turn
from bus.event_bus import EventBus
from bus.events import OutboundMessage
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
            return OutboundMessage("programmatic", "programmatic:live", "done")

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
    await runtime.shutdown()
    await bus.aclose()
    store.close()
