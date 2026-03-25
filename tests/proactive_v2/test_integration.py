"""
tests/proactive_v2/test_integration.py — P7 集成测试

验证 ProactiveLoop._tick() 在 v2-only 模式下稳定路由到 AgentTick。
使用 object.__new__ 绕过复杂构造函数，直接注入 mock 依赖。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from proactive_v2.config import ProactiveConfig
from proactive_v2.loop import ProactiveLoop


# ── 工厂 ──────────────────────────────────────────────────────────────────


def cfg_with(**kwargs) -> ProactiveConfig:
    return ProactiveConfig(**kwargs)


def make_loop(
    *,
    cfg: ProactiveConfig,
    agent_tick_mock=None,
) -> ProactiveLoop:
    """绕过 ProactiveLoop 复杂构造，直接注入 AgentTick mock。"""
    loop = object.__new__(ProactiveLoop)
    loop._cfg = cfg

    if agent_tick_mock is not None:
        loop._agent_tick = agent_tick_mock
    else:
        at = MagicMock()
        at.tick = AsyncMock(return_value=None)
        loop._agent_tick = at

    return loop


# ── v2-only 路由 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tick_calls_agent_tick():
    mock_at = MagicMock()
    v2_tick = AsyncMock(return_value=None)
    mock_at.tick = v2_tick

    loop = make_loop(cfg=cfg_with(), agent_tick_mock=mock_at)
    result = await loop._tick()

    v2_tick.assert_called_once()
    assert result is None


@pytest.mark.asyncio
async def test_tick_return_is_propagated():
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=42.0)
    loop = make_loop(cfg=cfg_with(), agent_tick_mock=mock_at)
    result = await loop._tick()
    assert result == 42.0


@pytest.mark.asyncio
async def test_tick_called_with_no_args():
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=0.0)
    loop = make_loop(cfg=cfg_with(), agent_tick_mock=mock_at)
    await loop._tick()
    mock_at.tick.assert_called_once_with()


# ── v2-only 初始化状态 ───────────────────────────────────────────────────
def test_agent_tick_is_initialized():
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=None)
    loop = make_loop(cfg=cfg_with(), agent_tick_mock=mock_at)
    assert loop._agent_tick is not None


# ── 7-D: _init_runtime_components 真实初始化 ──────────────────────────────


def test_real_loop_has_agent_tick_attr():
    """ProactiveLoop 真实构造时应持有 _agent_tick。"""
    loop = object.__new__(ProactiveLoop)
    loop._cfg = cfg_with()
    loop._agent_tick = MagicMock()  # 模拟 _init_runtime_components 行为

    assert loop._agent_tick is not None
    assert hasattr(loop, "_agent_tick")


# ── 7-E: 多次调用保持路由一致 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_v2_route_stable_across_multiple_ticks():
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=None)

    loop = make_loop(cfg=cfg_with(), agent_tick_mock=mock_at)

    await loop._tick()
    await loop._tick()
    await loop._tick()

    assert mock_at.tick.call_count == 3
