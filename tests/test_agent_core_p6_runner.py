from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from typing import Any, cast

import pytest

from agent.looping.ports import SessionServices
from agent.core.runner import CoreRunner, CoreRunnerDeps
from bus.events import InboundMessage, OutboundMessage, SpawnCompletionItem
from bus.internal_events import SpawnCompletionEvent


@pytest.mark.asyncio
async def test_core_runner_routes_passive_message_to_agent_core():
    runner = CoreRunner(
        CoreRunnerDeps(
            agent_core=cast(
                Any,
                SimpleNamespace(
                    process=AsyncMock(
                        return_value=OutboundMessage(
                            channel="cli",
                            chat_id="1",
                            content="final",
                        )
                    ),
                    pipeline=SimpleNamespace(),
                ),
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
async def test_core_runner_handles_spawn_completion_via_session_pipeline():
    session_svc = SimpleNamespace(
        session_manager=SimpleNamespace()
    )
    pipeline_mock = SimpleNamespace(
        run=AsyncMock(
            return_value=OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="spawn done",
            )
        )
    )
    runner = CoreRunner(
        CoreRunnerDeps(
            agent_core=cast(
                Any,
                SimpleNamespace(
                    process=AsyncMock(),
                    pipeline=pipeline_mock,
                ),
            ),
            session=cast(SessionServices, session_svc),
        )
    )
    item = SpawnCompletionItem(
        channel="telegram",
        chat_id="123",
        event=SpawnCompletionEvent(
            job_id="",
            label="任务",
            task="总结结果",
            status="completed",
            result="ok",
            exit_reason="completed",
            retry_count=0,
        ),
    )

    out = await runner.process(item, "scheduler:job-1", dispatch_outbound=False)

    assert out.content == "spawn done"
    pipeline_mock.run.assert_awaited_once()
    run_args = pipeline_mock.run.await_args
    assert run_args.args[1] == "scheduler:job-1"
    assert run_args.kwargs["dispatch_outbound"] is False
    pseudo_msg = run_args.args[0]
    assert "后台任务回传" in pseudo_msg.content
    assert pseudo_msg.metadata == {
        "skip_post_memory": True,
        "omit_user_turn": True,
        "skip_memory_retrieval": True,
    }
    runner._agent_core.process.assert_not_awaited()
