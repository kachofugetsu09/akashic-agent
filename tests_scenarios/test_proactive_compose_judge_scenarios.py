"""
test_proactive_compose_judge_scenarios.py — ProactiveLoop 行为契约真实测试。

每个场景描述一个"像人"与"不像人"的决策边界，用真实 LLM 验证 compose + judge 链路的行为。

运行方式：
    AKASIC_RUN_PROACTIVE=1 pytest tests_scenarios/test_proactive_compose_judge_scenarios.py -v
"""
from __future__ import annotations

import os

import pytest

from tests_scenarios.proactive_fixtures import (
    build_dont_repeat_empathy_scenario,
    build_health_alert_suppresses_entertainment_scenario,
    build_heavy_emotional_context_no_cheerful_send_scenario,
    build_interest_mismatch_no_send_scenario,
    build_long_unreplied_is_reason_to_send_scenario,
    build_no_feed_background_context_scenario,
    build_no_question_ending_scenario,
    build_short_unreplied_lightweight_scenario,
    build_sleeping_high_interest_scenario,
    build_topic_discontinuity_still_sends_scenario,
    build_truly_nothing_no_send_scenario,
)
from tests_scenarios.proactive_runner import ProactiveScenarioRunner

_RUN = os.getenv("AKASIC_RUN_PROACTIVE") == "1"
_skip = pytest.mark.skipif(not _RUN, reason="设置 AKASIC_RUN_PROACTIVE=1 后再执行")


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_sleeping_but_high_interest_should_send() -> None:
    """
    用户睡着 + CS2 决赛结果 → should_send=True。
    验证：sleeping 状态不是拒绝发送的理由，有实质内容就应发。
    """
    spec = build_sleeping_high_interest_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_long_unreplied_is_reason_to_send() -> None:
    """
    用户 6 小时未回复 + 有兴趣内容 → should_send=True。
    验证：长时间未回复是主动联系的理由，不是压制理由。
    """
    spec = build_long_unreplied_is_reason_to_send_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_short_unreplied_message_is_lightweight() -> None:
    """
    上条主动消息发出 8 分钟 + 用户未回 → 若发送，消息应轻量不重复话题。
    """
    spec = build_short_unreplied_lightweight_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_topic_discontinuity_does_not_block_send() -> None:
    """
    近期对话聊咖啡，feed 是 CS2 赛事 → should_send=True。
    验证：话题连续性不是硬门槛，有兴趣内容就可以开新话题。
    """
    spec = build_topic_discontinuity_still_sends_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_no_feed_background_context_can_trigger_send() -> None:
    """
    无 feed + Steam 活动数据 → should_send=True。
    验证：没有订阅内容时，background_context 可以作为主动搭话的依据。
    """
    spec = build_no_feed_background_context_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_truly_nothing_should_not_send() -> None:
    """
    完全没有内容，刚对话完 → should_send=False。
    验证：prompt 不会无中生有强行发消息。
    """
    spec = build_truly_nothing_no_send_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_message_does_not_end_with_question() -> None:
    """
    有内容 + 应该发 → message 不以"你怎么看/你觉得呢"等反问句结尾。
    验证：prompt 的"禁止反问句结尾"规则实际生效。
    """
    spec = build_no_question_ending_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_dont_repeat_empathy_when_new_content_available() -> None:
    """
    上条主动消息已安慰用户 + 新游戏资讯 → 直接说新内容，不重复安慰。
    """
    spec = build_dont_repeat_empathy_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_interest_mismatch_should_not_send() -> None:
    """
    用户明确只关心 NiKo/Falcons，feed 是 Astralis vs Complexity → should_send=False。
    验证：兴趣明确不匹配时不应发送，不能无视用户偏好强行推送。
    """
    spec = build_interest_mismatch_no_send_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_heavy_emotional_context_no_cheerful_send() -> None:
    """
    用户家人刚去世 + 欢快的 Steam 打折资讯 → 不发，或绝不以欢快语气发送。
    验证：情绪上下文感知，重大悲伤事件期间不能发欢快娱乐消息。
    """
    spec = build_heavy_emotional_context_no_cheerful_send_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.proactive_scenario
@_skip
async def test_health_alert_takes_priority_over_entertainment() -> None:
    """
    心率异常告警 + 艾尔登法环 DLC 资讯 → 告警优先，不应以欢快语气推送娱乐内容。
    验证：alert_events 存在时，LLM 应感知到健康异常而非忽视。
    """
    spec = build_health_alert_suppresses_entertainment_scenario()
    runner = ProactiveScenarioRunner()
    result = await runner.run(spec)
    assert result.passed, result.failure_message()
