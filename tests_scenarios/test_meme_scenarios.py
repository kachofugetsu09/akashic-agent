from __future__ import annotations

import os

import pytest

from tests_scenarios.fixtures import (
    build_meme_direct_affection_scenario,
    build_meme_direct_affection_after_long_technical_context_scenario,
    build_meme_explicit_sticker_request_scenario,
    build_meme_explicit_sticker_request_with_noisy_memory_scenario,
    build_meme_recent_trace_replay_scenario,
    build_meme_regression_implicit_with_session_history,
    build_meme_regression_real_memory_explicit_request,
    build_meme_regression_real_memory_simple_praise,
    build_meme_simple_praise_scenario,
    build_meme_technical_question_scenario,
)
from tests_scenarios.scenario_runner import ScenarioRunner

_RUN_SCENARIOS = os.getenv("AKASIC_RUN_SCENARIOS") == "1"


def _last_llm_output(result) -> str:
    for call in reversed(result.llm_calls):
        content = str(call.get("response_content", "") or "")
        if content.strip():
            return content
    return ""


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_direct_affection_should_emit_shy_meme_tag() -> None:
    """验证真实模型在直球喜欢场景下，会自己产出 <meme:shy>。"""
    spec = build_meme_direct_affection_scenario()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:shy>" in llm_output, llm_output or "最终 LLM 输出为空"


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_direct_affection_after_long_technical_context_should_emit_shy_meme_tag() -> None:
    """验证长技术上下文后，真实模型仍会自己产出 <meme:shy>。"""
    spec = build_meme_direct_affection_after_long_technical_context_scenario()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:shy>" in llm_output, llm_output or "最终 LLM 输出为空"


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_recent_trace_replay_should_emit_shy_meme_tag() -> None:
    """验证按最近真实失败会话复刻后，真实模型是否会自己产出 <meme:shy>。"""
    spec = build_meme_recent_trace_replay_scenario()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:shy>" in llm_output, llm_output or "最终 LLM 输出为空"


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_technical_question_should_not_emit_meme_tag() -> None:
    """验证真实模型在技术问答场景下，不会乱产出 meme tag。"""
    spec = build_meme_technical_question_scenario()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:" not in llm_output, llm_output


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_explicit_sticker_request_should_emit_meme_tag_without_tool_search() -> None:
    """验证用户明确要表情时，真实模型直接输出 meme tag，而不是去找工具。"""
    spec = build_meme_explicit_sticker_request_scenario()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:" in llm_output, llm_output or "最终 LLM 输出为空"


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_simple_praise_should_emit_shy_meme_tag() -> None:
    """验证轻度夸赞场景也会稳定触发 shy meme。"""
    spec = build_meme_simple_praise_scenario()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:shy>" in llm_output, llm_output or "最终 LLM 输出为空"


@pytest.mark.asyncio
@pytest.mark.scenario_regression
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_regression_implicit_with_session_history_should_emit_meme_tag() -> None:
    """复现线上核心 bug：session 历史里全是颜文字回复，导致隐式情感场景不触发 meme。

    此测试预期在修复前 FAIL，修复后 PASS。
    """
    spec = build_meme_regression_implicit_with_session_history()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:" in llm_output, (
        f"预期输出包含 <meme:> tag，但实际输出：\n{llm_output or '（空）'}"
    )


@pytest.mark.asyncio
@pytest.mark.scenario_regression
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_regression_real_memory_simple_praise_should_emit_meme_tag() -> None:
    """复现线上 bug：MEMORY.md + SELF.md 里的多处"禁止 emoji"压制了 meme 协议。

    此测试预期在修复前 FAIL，修复后 PASS。
    """
    spec = build_meme_regression_real_memory_simple_praise()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:" in llm_output, (
        f"预期输出包含 <meme:> tag，但实际输出：\n{llm_output or '（空）'}"
    )


@pytest.mark.asyncio
@pytest.mark.scenario_regression
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_regression_real_memory_explicit_request_should_emit_meme_tag() -> None:
    """复现线上 bug：用户明确要表情，但真实 workspace 记忆里有多处"禁止 emoji"。

    此测试预期在修复前 FAIL，修复后 PASS。
    """
    spec = build_meme_regression_real_memory_explicit_request()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:" in llm_output, (
        f"预期输出包含 <meme:> tag，但实际输出：\n{llm_output or '（空）'}"
    )


@pytest.mark.asyncio
@pytest.mark.scenario_mvp
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_real_explicit_sticker_request_with_noisy_memory_should_emit_meme_tag() -> None:
    """验证带线上类似噪音记忆时，表情请求仍直接输出 meme tag。"""
    spec = build_meme_explicit_sticker_request_with_noisy_memory_scenario()
    runner = ScenarioRunner()
    result = await runner.run(spec)

    assert result.passed, result.failure_message()
    llm_output = _last_llm_output(result)
    assert "<meme:" in llm_output, llm_output or "最终 LLM 输出为空"
