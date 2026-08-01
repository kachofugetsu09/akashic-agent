from unittest.mock import AsyncMock, Mock

import pytest

from agent.policies.delegation import SpawnDecision, SpawnDecisionMeta
from agent.tools.registry import ToolRegistry
from agent.tools.spawn import SpawnManageTool, SpawnTool


def _make_manager(spawn_return="started", spawn_sync_return="sync-result"):
    manager = AsyncMock()
    manager.spawn = AsyncMock(return_value=spawn_return)
    manager.spawn_sync = AsyncMock(return_value=spawn_sync_return)
    manager.get_running_count = Mock(return_value=0)
    manager.list_running_jobs = Mock(return_value=[])
    manager.cancel = AsyncMock(return_value=False)
    return manager


@pytest.mark.asyncio
async def test_spawn_tool_uses_registry_context():
    """Background mode (run_in_background=True) routes through manager.spawn with context."""
    registry = ToolRegistry()
    manager = _make_manager()
    tool = SpawnTool(manager, registry)
    registry.set_context(channel="telegram", chat_id="123")

    result = await tool.execute(task="do work", label="job", run_in_background=True)

    assert result == "started"
    manager.spawn.assert_awaited_once_with(
        task="do work",
        label="job",
        origin_channel="telegram",
        origin_chat_id="123",
        decision=SpawnDecision(
            should_spawn=True,
            label="job",
            meta=SpawnDecisionMeta(
                source="llm",
                confidence="high",
                reason_code="tool_chain_heavy",
            ),
        ),
        profile="research",
        retry_count=0,
    )


@pytest.mark.asyncio
async def test_spawn_tool_sync_mode_skips_context_check():
    """Sync mode (default) routes through manager.spawn_sync without needing context."""
    registry = ToolRegistry()
    manager = _make_manager()
    tool = SpawnTool(manager, registry)
    # No context set

    result = await tool.execute(task="do work", label="job")

    assert result == "sync-result"
    manager.spawn_sync.assert_awaited_once_with(
        task="do work",
        label="job",
        profile="research",
    )
    manager.spawn.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_tool_returns_error_when_context_missing():
    """Background mode requires session context; returns error if missing."""
    registry = ToolRegistry()
    manager = _make_manager()
    tool = SpawnTool(manager, registry)

    result = await tool.execute(task="do work", run_in_background=True)

    assert "上下文缺失" in result
    manager.spawn.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_tool_keeps_spawning_even_when_policy_prefers_inline():
    """DelegationPolicy allows all tasks under concurrent limit; small tasks still spawn."""
    registry = ToolRegistry()
    manager = _make_manager()
    tool = SpawnTool(manager, registry)
    registry.set_context(channel="telegram", chat_id="123")

    result = await tool.execute(
        task="帮我看一下这个函数名是不是合适", label="small", run_in_background=True
    )

    assert result == "started"
    kwargs = manager.spawn.await_args.kwargs
    assert kwargs["decision"].should_spawn is True
    assert kwargs["decision"].meta.reason_code == "tool_chain_heavy"
    assert kwargs["decision"].meta.source == "llm"


@pytest.mark.asyncio
async def test_spawn_tool_blocks_when_concurrent_limit_reached():
    """DelegationPolicy returns should_spawn=False when concurrent limit exceeded."""
    registry = ToolRegistry()
    manager = AsyncMock()
    manager.get_running_count = Mock(return_value=3)  # at limit
    tool = SpawnTool(manager, registry)
    registry.set_context(channel="telegram", chat_id="123")

    result = await tool.execute(task="another task", run_in_background=True)

    assert "任务被拦截" in result
    manager.spawn.assert_not_called()
    manager.spawn_sync.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_tool_sync_mode_uses_shared_concurrency_limit():
    registry = ToolRegistry()
    manager = _make_manager()
    manager.get_running_count = Mock(return_value=3)
    tool = SpawnTool(manager, registry)

    result = await tool.execute(task="inline task")

    assert "任务被拦截" in result
    manager.spawn_sync.assert_not_awaited()
    manager.spawn.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_tool_forwards_retry_count_in_background_mode():
    registry = ToolRegistry()
    manager = _make_manager()
    tool = SpawnTool(manager, registry)
    registry.set_context(channel="telegram", chat_id="123")

    await tool.execute(
        task="retry task auto_promote=false",
        label="job",
        run_in_background=True,
        retry_count=1,
    )

    kwargs = manager.spawn.await_args.kwargs
    assert kwargs["retry_count"] == 1
    assert kwargs["task"] == "retry task auto_promote=false"


@pytest.mark.asyncio
async def test_spawn_manage_lists_running_jobs():
    manager = _make_manager()
    manager.get_running_count = Mock(return_value=1)
    manager.list_running_jobs = Mock(
        return_value=[{"job_id": "abc123", "label": "job", "status": "running"}]
    )
    tool = SpawnManageTool(manager)

    result = await tool.execute(action="list")

    assert '"running_count": 1' in result
    assert "abc123" in result


@pytest.mark.asyncio
async def test_spawn_manage_cancels_by_job_id():
    manager = _make_manager()
    manager.cancel = AsyncMock(return_value=True)
    tool = SpawnManageTool(manager)

    result = await tool.execute(action="cancel", job_id="abc123")

    assert "cancel_requested" in result
    manager.cancel.assert_awaited_once_with("abc123")
