from unittest.mock import AsyncMock

import pytest

from agent.delegation_policy import SpawnDecision, SpawnDecisionMeta
from agent.tools.registry import ToolRegistry
from agent.tools.spawn import SpawnTool


@pytest.mark.asyncio
async def test_spawn_tool_uses_registry_context():
    registry = ToolRegistry()
    manager = AsyncMock()
    manager.spawn = AsyncMock(return_value="started")
    tool = SpawnTool(manager, registry)
    registry.set_context(channel="telegram", chat_id="123")

    result = await tool.execute(task="do work", label="job")

    assert result == "started"
    manager.spawn.assert_awaited_once_with(
        task="do work",
        label="job",
        origin_channel="telegram",
        origin_chat_id="123",
        decision=SpawnDecision(
            should_spawn=False,
            label="job",
            meta=SpawnDecisionMeta(
                source="heuristic",
                confidence="medium",
                reason_code="stay_inline",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_spawn_tool_returns_error_when_context_missing():
    registry = ToolRegistry()
    manager = AsyncMock()
    tool = SpawnTool(manager, registry)

    result = await tool.execute(task="do work")

    assert "上下文缺失" in result
    manager.spawn.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_tool_keeps_spawning_even_when_policy_prefers_inline():
    registry = ToolRegistry()
    manager = AsyncMock()
    manager.spawn = AsyncMock(return_value="started")
    tool = SpawnTool(manager, registry)
    registry.set_context(channel="telegram", chat_id="123")

    result = await tool.execute(task="帮我看一下这个函数名是不是合适", label="small")

    assert result == "started"
    kwargs = manager.spawn.await_args.kwargs
    assert kwargs["decision"].should_spawn is False
    assert kwargs["decision"].meta.reason_code == "stay_inline"
