import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps
from agent.looping.turn_types import ToolCall, ToolCallGroup
from agent.memory import MemoryStore
from agent.postturn.default_pipeline import DefaultPostTurnPipeline
from agent.postturn.protocol import PostTurnEvent, PostTurnPipeline
from agent.provider import LLMResponse
from agent.retrieval.protocol import (
    MemoryRetrievalPipeline,
    RetrievalRequest,
    RetrievalResult,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.events import InboundMessage
from core.memory.port import DefaultMemoryPort


class _NoopTool(Tool):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> str:
        return "ok"


class _Provider:
    async def chat(self, **kwargs):
        return LLMResponse(content="ok", tool_calls=[])


class _CustomRetrieval(MemoryRetrievalPipeline):
    def __init__(self, block: str) -> None:
        self._block = block
        self.requests: list[RetrievalRequest] = []

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        return RetrievalResult(block=self._block)


class _CustomPostTurn(PostTurnPipeline):
    def __init__(self) -> None:
        self.events: list[PostTurnEvent] = []

    def schedule(self, event: PostTurnEvent) -> None:
        self.events.append(event)


def _make_loop(
    tmp_path: Path,
    *,
    retrieval_pipeline: MemoryRetrievalPipeline | None = None,
    post_turn_pipeline: PostTurnPipeline | None = None,
) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(_NoopTool())
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, _Provider()),
            light_provider=cast(Any, _Provider()),
            tools=tools,
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
            retrieval_pipeline=retrieval_pipeline,
            post_turn_pipeline=post_turn_pipeline,
        ),
        AgentLoopConfig(),
    )


@pytest.mark.asyncio
async def test_default_post_turn_pipeline_uses_scheduler_post_mem_callback():
    scheduler = MagicMock()
    worker = MagicMock()
    worker.run = AsyncMock(return_value=None)
    pipeline = DefaultPostTurnPipeline(scheduler=scheduler, post_mem_worker=worker)

    event = PostTurnEvent(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        user_message="hello",
        assistant_response="ok",
        tools_used=["tool_a"],
        tool_chain=[
            ToolCallGroup(
                text="t",
                calls=[
                    ToolCall(
                        call_id="c1",
                        name="tool_a",
                        arguments={"x": 1},
                        result="done",
                    )
                ],
            )
        ],
        session=MagicMock(),
        timestamp=datetime.now(),
    )
    pipeline.schedule(event)
    scheduler.schedule_consolidation.assert_called_once()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    worker.run.assert_awaited_once()
    assert pipeline._failures == 0


def test_agent_loop_uses_custom_pipelines(tmp_path: Path):
    custom_retrieval = _CustomRetrieval(block="MEM_BLOCK")
    custom_post_turn = _CustomPostTurn()
    loop = _make_loop(
        tmp_path,
        retrieval_pipeline=custom_retrieval,
        post_turn_pipeline=custom_post_turn,
    )
    session = MagicMock()
    session.key = "cli:1"
    session.messages = []
    session.metadata = {}
    session.get_history = MagicMock(
        return_value=[{"role": "user", "content": f"m{i}"} for i in range(200)]
    )
    session.add_message = MagicMock()
    loop.session_manager.get_or_create.return_value = session
    loop.session_manager.append_messages = AsyncMock(return_value=None)
    loop._safety_retry.run = AsyncMock(return_value=("ok", [], [], None))

    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")
    asyncio.run(loop._process_inner(msg, msg.session_key))

    assert custom_retrieval.requests
    assert custom_retrieval.requests[0].message == "hello"
    assert custom_post_turn.events
    assert custom_post_turn.events[0].assistant_response == "ok"
    run_kwargs = loop._safety_retry.run.await_args.kwargs
    assert "base_history" in run_kwargs
    assert run_kwargs["base_history"] is None
