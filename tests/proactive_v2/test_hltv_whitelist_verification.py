"""
tests/proactive_v2/test_hltv_whitelist_verification.py

验证 proactive 系统提示包含时效性数据必须查询的约束。
"""
from __future__ import annotations

import pytest

from tests.proactive_v2.conftest import make_agent_tick


# ── 确定性单元测试（不调用真实 LLM）─────────────────────────────────────────

@pytest.mark.asyncio
async def test_system_prompt_contains_training_data_warning():
    """
    验证系统提示里包含"训练数据记忆"不能替代 web_fetch 的规则。
    这是一个确定性测试，不依赖真实 LLM。
    """
    # 构建一个最小化的 AgentTick 来获取系统提示
    tick = make_agent_tick(llm_fn=None)

    prompt = tick._build_system_prompt()

    assert "训练数据" in prompt, (
        "系统提示应包含'训练数据'相关的规则，防止 LLM 用训练记忆跳过 web_fetch 验证"
    )
    assert "web_fetch" in prompt.lower() or "web_fetch" in prompt, (
        "系统提示应包含 web_fetch 工具相关说明"
    )


@pytest.mark.asyncio
async def test_system_prompt_rule8_covers_ranking_verification():
    """
    验证规则第8条明确说明排名/赛况等时效性数据必须 web_fetch，不能用训练记忆。
    """
    tick = make_agent_tick(llm_fn=None)
    prompt = tick._build_system_prompt()

    # 规则8的核心断言
    assert "训练数据记忆" in prompt, "规则8应明确提到训练数据记忆不等于常识"
    assert "时效性数据" in prompt, "规则8应涵盖排名等时效性数据的处理要求"
    assert "必须步骤" in prompt, "规则8应明确 web_fetch 是必须步骤而非可选"
