from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.runner import CoreRunner, CoreRunnerDeps
from bus.events import InboundMessage, OutboundMessage


@pytest.mark.asyncio
async def test_core_runner_routes_passive_message_to_agent_core():
    runner = CoreRunner(
        CoreRunnerDeps(
            agent_core=SimpleNamespace(
                process=AsyncMock(
                    return_value=OutboundMessage(
                        channel="cli",
                        chat_id="1",
                        content="final",
                    )
                )
            ),
        )
    )
    msg = InboundMessage(channel="cli", sender="hua", chat_id="1", content="hi")

    out = await runner.process(msg, "cli:1")

    assert out.content == "final"
    runner._agent_core.process.assert_awaited_once_with(
        msg,
        "cli:1",
        dispatch_outbound=True,
    )


@pytest.mark.asyncio
async def test_core_runner_handles_spawn_completion_via_direct_helper_deps():
    session = MagicMock()
    session.get_history.return_value = [{"role": "user", "content": "old"}]
    session_svc = SimpleNamespace(
        session_manager=SimpleNamespace(get_or_create=MagicMock(return_value=session))
    )
    context = SimpleNamespace(
        build_messages=MagicMock(return_value=[{"role": "system", "content": "prompt"}])
    )
    context_store = SimpleNamespace(
        commit=AsyncMock(
            return_value=OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="spawn done",
            )
        )
    )
    tools = SimpleNamespace(set_context=MagicMock())
    run_agent_loop_fn = AsyncMock(
        return_value=("done", ["spawn"], [{"name": "spawn"}], None, None)
    )
    runner = CoreRunner(
        CoreRunnerDeps(
            agent_core=SimpleNamespace(process=AsyncMock()),
            session=session_svc,
            context=context,
            context_store=context_store,
            tools=tools,
            memory_window=12,
            run_agent_loop_fn=run_agent_loop_fn,
        )
    )
    msg = InboundMessage(
        channel="telegram",
        sender="spawn",
        chat_id="123",
        content="[internal spawn completed]",
        metadata={
            "internal_event": "spawn_completed",
            "spawn": {
                "label": "任务",
                "task": "总结结果",
                "status": "completed",
                "result": "ok",
                "exit_reason": "completed",
                "retry_count": 0,
            },
        },
    )

    out = await runner.process(msg, "scheduler:job-1", dispatch_outbound=False)

    assert out.content == "spawn done"
    session_svc.session_manager.get_or_create.assert_called_once_with("scheduler:job-1")
    tools.set_context.assert_called_once_with(channel="telegram", chat_id="123")
    context.build_messages.assert_called_once()
    run_agent_loop_fn.assert_awaited_once()
    context_store.commit.assert_awaited_once()
    runner._agent_core.process.assert_not_awaited()
