from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.agent_core import AgentCore, AgentCoreDeps
from agent.core.types import ContextBundle
from bus.events import InboundMessage, OutboundMessage


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500) -> list[dict]:
        return self.messages[-max_messages:]


@pytest.mark.asyncio
async def test_agent_core_process_runs_prepare_prompt_run_commit_in_order():
    order: list[str] = []
    session = _DummySession("telegram:123")
    context_store = SimpleNamespace(
        prepare=AsyncMock(
            side_effect=lambda **kwargs: order.append("prepare")
            or ContextBundle(
                skill_mentions=["refactor"],
                retrieved_memory_block="remembered",
                retrieval_trace_raw={"route": "RETRIEVE"},
            )
        )
    )
    context = SimpleNamespace(
        build_system_prompt=MagicMock(
            side_effect=lambda **kwargs: order.append("build_prompt") or "system prompt"
        )
    )
    tools = SimpleNamespace(
        set_context=MagicMock(side_effect=lambda **kwargs: order.append("tool_context"))
    )
    turn_runner = SimpleNamespace(
        run=AsyncMock(
            side_effect=lambda *args, **kwargs: order.append("run")
            or ("final", ["shell"], [{"text": "done", "calls": []}], "think")
        ),
        last_retry_trace={"selected_plan": "full"},
    )
    context_store.commit = AsyncMock(
        side_effect=lambda **kwargs: order.append("commit")
        or OutboundMessage(channel="telegram", chat_id="123", content="final")
    )
    agent_core = AgentCore(
        AgentCoreDeps(
            session=SimpleNamespace(
                session_manager=SimpleNamespace(
                    get_or_create=MagicMock(return_value=session)
                )
            ),
            context_store=context_store,
            context=context,
            tools=tools,
            turn_runner=turn_runner,
        )
    )
    msg = InboundMessage(
        channel="telegram",
        sender="hua",
        chat_id="123",
        content="你好",
        timestamp=datetime(2026, 4, 4, 22, 0, 0),
    )

    out = await agent_core.process(msg, "telegram:123")

    assert out.content == "final"
    assert order == ["prepare", "build_prompt", "tool_context", "run", "commit"]
    assert context_store.prepare.await_args.kwargs["session_key"] == "telegram:123"
    tools.set_context.assert_called_once_with(channel="telegram", chat_id="123")
    assert turn_runner.run.await_args.kwargs["skill_names"] == ["refactor"]
    assert turn_runner.run.await_args.kwargs["retrieved_memory_block"] == "remembered"
    assert context_store.commit.await_args.kwargs["retrieval_raw"] == {"route": "RETRIEVE"}
    assert context_store.commit.await_args.kwargs["thinking"] == "think"


@pytest.mark.asyncio
async def test_agent_core_process_coerces_empty_reply_before_commit():
    session = _DummySession("cli:1")
    context_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle()),
        commit=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="fallback")
        )
    )
    agent_core = AgentCore(
        AgentCoreDeps(
            session=SimpleNamespace(
                session_manager=SimpleNamespace(
                    get_or_create=MagicMock(return_value=session)
                )
            ),
            context_store=context_store,
            context=SimpleNamespace(build_system_prompt=MagicMock(return_value="prompt")),
            tools=SimpleNamespace(set_context=MagicMock()),
            turn_runner=SimpleNamespace(
                run=AsyncMock(return_value=(None, [], [], None)),
                last_retry_trace={},
            ),
        )
    )
    msg = InboundMessage(channel="cli", sender="hua", chat_id="1", content="hi")

    await agent_core.process(msg, "cli:1")

    assert "no response to give" in context_store.commit.await_args.kwargs["reply"]
