from __future__ import annotations

import os

import pytest

from tests_scenarios.fixtures import (
    build_rag_with_noise_scenario,
    build_smalltalk_no_retrieve_scenario,
    build_tool_search_schedule_scenario,
)
from tests_scenarios.scenario_runner import ScenarioRunner

_RUN_SCENARIOS = os.getenv("AKASIC_RUN_SCENARIOS") == "1"


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_tool_search_and_schedule_flow() -> None:
    """验证真实工具注册逻辑下，tool_search 能解锁 schedule 并完成一次真实工具执行。"""
    # 1. 构造场景：主模型响应脚本化，但工具注册、tool_search、schedule 都走真实实现。
    spec = build_tool_search_schedule_scenario()
    runner = ScenarioRunner()
    # 2. 运行场景：在隔离 workspace 中执行一次完整 AgentLoop 主路径。
    result = await runner.run(spec)
    # 3. 校验结果：若失败，直接输出 artifact 对应的错误摘要，方便回看现场。
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_smalltalk_does_not_trigger_retrieve() -> None:
    """验证正常闲聊在真实模型和真实记忆链路下不会误触发 retrieve。"""
    # 1. 构造场景：history、memory、memory2 都使用测试专用数据，但消息本身只是普通闲聊。
    spec = build_smalltalk_no_retrieve_scenario()
    runner = ScenarioRunner()
    # 2. 运行场景：真实执行 route gate、memory retrieve 和回答生成。
    result = await runner.run(spec)
    # 3. 校验结果：要求 route decision 为 NO_RETRIEVE，并且没有 history 命中。
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_rag_with_related_and_irrelevant_noise() -> None:
    """验证带相关干扰和不相关噪音时，真实 RAG 仍能召回并使用正确记忆。"""
    # 1. 构造场景：写入正确记忆、相似错误记忆和不相关噪音，模拟更脏的真实记忆库。
    spec = build_rag_with_noise_scenario()
    runner = ScenarioRunner()
    # 2. 运行场景：真实执行 route gate、history retrieve、memory injection 和回答生成。
    result = await runner.run(spec)
    # 3. 校验结果：要求命中 history 检索，并且最终回答稳定落到正确答案而不是噪音。
    assert result.passed, result.failure_message()
