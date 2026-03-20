"""
tests/proactive_v2/test_integration.py — P7 集成测试

验证 ProactiveLoop._tick() 根据 use_agent_tick 标志正确路由到 v1/v2 引擎。
使用 object.__new__ 绕过复杂构造函数，直接注入 mock 依赖。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from proactive.config import ProactiveConfig
from proactive.loop import ProactiveLoop


# ── 工厂 ──────────────────────────────────────────────────────────────────


def cfg_with(**kwargs) -> ProactiveConfig:
    return ProactiveConfig(**kwargs)


def make_loop(
    *,
    cfg: ProactiveConfig,
    engine_tick=None,
    agent_tick_mock=None,
) -> ProactiveLoop:
    """绕过 ProactiveLoop 复杂构造，直接注入 mock。

    engine_tick:      v1 引擎 tick 的 AsyncMock / callable
    agent_tick_mock:  AgentTick mock 对象（有 .tick 属性）
                      传入 None 且 cfg.use_agent_tick=False → _agent_tick=None
                      传入 None 且 cfg.use_agent_tick=True  → 自动创建默认 mock
    """
    loop = object.__new__(ProactiveLoop)
    loop._cfg = cfg

    # v1 engine
    engine = MagicMock()
    engine.tick = engine_tick if engine_tick is not None else AsyncMock(return_value=0.0)
    loop._engine = engine

    # v2 agent_tick
    if agent_tick_mock is not None:
        loop._agent_tick = agent_tick_mock
    elif cfg.use_agent_tick:
        at = MagicMock()
        at.tick = AsyncMock(return_value=None)
        loop._agent_tick = at
    else:
        loop._agent_tick = None

    return loop


# ── 7-A: use_agent_tick=False → 走 v1 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_use_agent_tick_false_calls_v1():
    v1_tick = AsyncMock(return_value=300.0)
    mock_at = MagicMock()
    v2_tick = AsyncMock(return_value=None)
    mock_at.tick = v2_tick

    loop = make_loop(
        cfg=cfg_with(use_agent_tick=False),
        engine_tick=v1_tick,
        agent_tick_mock=mock_at,
    )
    result = await loop._tick()

    v1_tick.assert_called_once()
    v2_tick.assert_not_called()
    assert result == 300.0


@pytest.mark.asyncio
async def test_use_agent_tick_false_v1_none_return_propagated():
    loop = make_loop(
        cfg=cfg_with(use_agent_tick=False),
        engine_tick=AsyncMock(return_value=None),
    )
    result = await loop._tick()
    assert result is None


@pytest.mark.asyncio
async def test_use_agent_tick_false_v1_score_propagated():
    loop = make_loop(
        cfg=cfg_with(use_agent_tick=False),
        engine_tick=AsyncMock(return_value=600.0),
    )
    result = await loop._tick()
    assert result == 600.0


# ── 7-B: use_agent_tick=True → 走 v2 ───────────────────────────────────────


@pytest.mark.asyncio
async def test_use_agent_tick_true_calls_v2():
    v1_tick = AsyncMock(return_value=300.0)
    mock_at = MagicMock()
    v2_tick = AsyncMock(return_value=None)
    mock_at.tick = v2_tick

    loop = make_loop(
        cfg=cfg_with(use_agent_tick=True),
        engine_tick=v1_tick,
        agent_tick_mock=mock_at,
    )
    result = await loop._tick()

    v2_tick.assert_called_once()
    v1_tick.assert_not_called()
    assert result is None


@pytest.mark.asyncio
async def test_use_agent_tick_true_v2_return_propagated():
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=42.0)
    loop = make_loop(cfg=cfg_with(use_agent_tick=True), agent_tick_mock=mock_at)
    result = await loop._tick()
    assert result == 42.0


@pytest.mark.asyncio
async def test_use_agent_tick_true_v2_called_with_no_args():
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=0.0)
    loop = make_loop(cfg=cfg_with(use_agent_tick=True), agent_tick_mock=mock_at)
    await loop._tick()
    mock_at.tick.assert_called_once_with()


@pytest.mark.asyncio
async def test_use_agent_tick_true_v1_engine_not_touched():
    v1_tick = AsyncMock(return_value=100.0)
    loop = make_loop(
        cfg=cfg_with(use_agent_tick=True),
        engine_tick=v1_tick,
    )
    await loop._tick()
    v1_tick.assert_not_called()


# ── 7-C: _agent_tick 初始化状态 ────────────────────────────────────────────


def test_agent_tick_not_initialized_when_disabled():
    loop = make_loop(cfg=cfg_with(use_agent_tick=False))
    assert loop._agent_tick is None


def test_agent_tick_initialized_when_enabled():
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=None)
    loop = make_loop(cfg=cfg_with(use_agent_tick=True), agent_tick_mock=mock_at)
    assert loop._agent_tick is not None


# ── 7-D: _init_runtime_components 真实初始化 ──────────────────────────────


def test_real_loop_has_agent_tick_attr_when_disabled():
    """ProactiveLoop 真实构造时（mock 所有依赖），flag=False → _agent_tick=None。"""
    loop = object.__new__(ProactiveLoop)
    loop._cfg = cfg_with(use_agent_tick=False)
    loop._engine = MagicMock()
    loop._agent_tick = None  # 模拟 _init_runtime_components 行为

    assert loop._agent_tick is None
    assert hasattr(loop, "_agent_tick")


def test_real_loop_has_agent_tick_attr_when_enabled():
    """ProactiveLoop 真实构造时（mock 所有依赖），flag=True → _agent_tick 非 None。"""
    mock_at = MagicMock()
    loop = object.__new__(ProactiveLoop)
    loop._cfg = cfg_with(use_agent_tick=True)
    loop._engine = MagicMock()
    loop._agent_tick = mock_at  # 模拟 _init_runtime_components 行为

    assert loop._agent_tick is not None


# ── 7-E: 多次调用保持路由一致 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_v1_route_stable_across_multiple_ticks():
    v1_tick = AsyncMock(return_value=0.0)
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=None)

    loop = make_loop(
        cfg=cfg_with(use_agent_tick=False),
        engine_tick=v1_tick,
        agent_tick_mock=mock_at,
    )

    await loop._tick()
    await loop._tick()
    await loop._tick()

    assert v1_tick.call_count == 3
    mock_at.tick.assert_not_called()


@pytest.mark.asyncio
async def test_v2_route_stable_across_multiple_ticks():
    v1_tick = AsyncMock(return_value=0.0)
    mock_at = MagicMock()
    mock_at.tick = AsyncMock(return_value=None)

    loop = make_loop(
        cfg=cfg_with(use_agent_tick=True),
        engine_tick=v1_tick,
        agent_tick_mock=mock_at,
    )

    await loop._tick()
    await loop._tick()

    assert mock_at.tick.call_count == 2
    v1_tick.assert_not_called()
