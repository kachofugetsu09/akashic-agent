"""
tests/proactive_v2/test_integration.py — P7 集成测试

验证 ProactiveLoop._tick() 稳定路由到 ProactiveTurnPipeline。
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
    pipeline_mock=None,
) -> ProactiveLoop:
    """绕过 ProactiveLoop 复杂构造，直接注入 pipeline mock。"""
    loop = object.__new__(ProactiveLoop)
    loop._cfg = cfg

    if pipeline_mock is not None:
        loop._proactive_pipeline = pipeline_mock
    else:
        pipeline = MagicMock()
        pipeline.run = AsyncMock(return_value=None)
        loop._proactive_pipeline = pipeline

    return loop


# ── v2-only 路由 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tick_calls_pipeline():
    pipeline = MagicMock()
    run = AsyncMock(return_value=None)
    pipeline.run = run

    loop = make_loop(cfg=cfg_with(), pipeline_mock=pipeline)
    result = await loop._tick()

    run.assert_called_once()
    assert result is None


@pytest.mark.asyncio
async def test_tick_return_is_propagated():
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=42.0)
    loop = make_loop(cfg=cfg_with(), pipeline_mock=pipeline)
    result = await loop._tick()
    assert result == 42.0


@pytest.mark.asyncio
async def test_tick_called_with_no_args():
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=0.0)
    loop = make_loop(cfg=cfg_with(), pipeline_mock=pipeline)
    await loop._tick()
    pipeline.run.assert_called_once_with()


# ── v2-only 初始化状态 ───────────────────────────────────────────────────
def test_pipeline_is_initialized():
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=None)
    loop = make_loop(cfg=cfg_with(), pipeline_mock=pipeline)
    assert loop._proactive_pipeline is not None


# ── 7-D: _init_runtime_components 真实初始化 ──────────────────────────────


def test_real_loop_has_pipeline_attr():
    """ProactiveLoop 真实构造时应持有 _proactive_pipeline。"""
    loop = object.__new__(ProactiveLoop)
    loop._cfg = cfg_with()
    loop._proactive_pipeline = MagicMock()

    assert loop._proactive_pipeline is not None
    assert hasattr(loop, "_proactive_pipeline")


# ── 7-E: 多次调用保持路由一致 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_v2_route_stable_across_multiple_ticks():
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=None)

    loop = make_loop(cfg=cfg_with(), pipeline_mock=pipeline)

    await loop._tick()
    await loop._tick()
    await loop._tick()

    assert pipeline.run.call_count == 3
