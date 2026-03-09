from unittest.mock import AsyncMock

import pytest

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
    )


@pytest.mark.asyncio
async def test_spawn_tool_returns_error_when_context_missing():
    registry = ToolRegistry()
    manager = AsyncMock()
    tool = SpawnTool(manager, registry)

    result = await tool.execute(task="do work")

    assert "上下文缺失" in result
    manager.spawn.assert_not_called()
