from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from typing import Any, cast

import pytest

from agent.context import ContextBuilder
from agent.lifecycle.types import PromptRenderResult
from agent.looping.ports import SessionServices
from agent.tools.registry import ToolRegistry
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
async def test_core_runner_handles_spawn_completion_via_direct_helper_deps():
    session = MagicMock()
    session.get_history.return_value = [{"role": "user", "content": "old"}]
    session_svc = SimpleNamespace(
        session_manager=SimpleNamespace(get_or_create=MagicMock(return_value=session))
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(messages=[{"role": "system", "content": "prompt"}]))
    )
    pipeline_mock = SimpleNamespace(
        post_reasoning=AsyncMock(
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
    prompt_render_fn = AsyncMock(
        return_value=PromptRenderResult(
            messages=[{"role": "system", "content": "prompt"}]
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
            context=cast(ContextBuilder, context),
            tools=cast(ToolRegistry, tools),
            memory_window=12,
            run_agent_loop_fn=run_agent_loop_fn,
            prompt_render_fn=prompt_render_fn,
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
    session_svc.session_manager.get_or_create.assert_called_once_with("scheduler:job-1")
    tools.set_context.assert_called_once_with(channel="telegram", chat_id="123")
    prompt_render_fn.assert_awaited_once()
    render_input = prompt_render_fn.await_args.args[0]
    assert render_input.session_key == "scheduler:job-1"
    assert "后台任务回传" in render_input.content
    run_agent_loop_fn.assert_awaited_once()
    pipeline_mock.post_reasoning.assert_awaited_once()
    pr_kwargs = pipeline_mock.post_reasoning.await_args.kwargs
    assert pr_kwargs["dispatch_outbound"] is False
    runner._agent_core.process.assert_not_awaited()
