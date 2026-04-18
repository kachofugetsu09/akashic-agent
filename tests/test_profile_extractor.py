from unittest.mock import AsyncMock, MagicMock

import pytest

from memory2.profile_extractor import ProfileFact, ProfileFactExtractor


def _make_extractor(llm_response: str) -> ProfileFactExtractor:
    client = MagicMock()
    client.chat = AsyncMock(return_value=llm_response)
    return ProfileFactExtractor(llm_client=client)


def test_profile_fact_dataclass_fields():
    fact = ProfileFact(
        summary="用户2026-03-12购买了Zigbee网关SNZB-02D和加湿器",
        category="purchase",
        happened_at="2026-03-12",
    )
    assert fact.summary.startswith("用户2026-03-12购买了")
    assert fact.category == "purchase"
    assert fact.happened_at == "2026-03-12"


@pytest.mark.asyncio
async def test_extract_purchase_fact_from_zigbee_conversation():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户购买了Zigbee网关SNZB-02D和加湿器</summary><category>purchase</category><happened_at>2026-03-12</happened_at></fact>
</facts>
"""
    )
    facts = await extractor.extract("我买了 Zigbee 网关和加湿器")
    assert any(f.category == "purchase" and ("Zigbee" in f.summary or "加湿器" in f.summary) for f in facts)


@pytest.mark.asyncio
async def test_extract_decision_fact_from_solution_discussion():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户决定采用SNZB-02D + BroadLink方案</summary><category>decision</category><happened_at>2026-03-12</happened_at></fact>
</facts>
"""
    )
    facts = await extractor.extract("决定用SNZB-02D + BroadLink方案")
    assert any(f.category == "decision" for f in facts)


@pytest.mark.asyncio
async def test_pure_technical_discussion_returns_empty():
    extractor = _make_extractor("<facts></facts>")
    facts = await extractor.extract("这里讨论某个算法原理和时间复杂度")
    assert facts == []


@pytest.mark.asyncio
async def test_greeting_conversation_returns_empty():
    extractor = _make_extractor("<facts></facts>")
    facts = await extractor.extract("你好呀，今天天气不错")
    assert facts == []


@pytest.mark.asyncio
async def test_extract_fails_open_on_malformed_output():
    extractor = _make_extractor("这是乱码")
    facts = await extractor.extract("我买了 Zigbee 网关")
    assert facts == []


@pytest.mark.asyncio
async def test_extract_fails_open_on_llm_exception():
    client = MagicMock()
    client.chat = AsyncMock(side_effect=RuntimeError("timeout"))
    extractor = ProfileFactExtractor(llm_client=client)
    facts = await extractor.extract("我买了 Zigbee 网关")
    assert facts == []


@pytest.mark.asyncio
async def test_conversation_appears_in_prompt():
    client = MagicMock()
    captured: list[str] = []

    async def _cap(*, messages, **kwargs):
        captured.append(messages[0]["content"])
        return "<facts></facts>"

    client.chat = AsyncMock(side_effect=_cap)
    extractor = ProfileFactExtractor(llm_client=client)
    await extractor.extract("我买了 Zigbee 网关")
    assert captured and "Zigbee" in captured[0]


@pytest.mark.asyncio
async def test_existing_profile_appears_in_prompt():
    client = MagicMock()
    captured: list[str] = []

    async def _cap(*, messages, **kwargs):
        captured.append(messages[0]["content"])
        return "<facts></facts>"

    client.chat = AsyncMock(side_effect=_cap)
    extractor = ProfileFactExtractor(llm_client=client)
    await extractor.extract("我买了 Zigbee 网关", existing_profile="用户长期偏好本地控制")
    assert captured and "用户长期偏好本地控制" in captured[0]


@pytest.mark.asyncio
async def test_happened_at_parsed_when_provided():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户购买了加湿器</summary><category>purchase</category><happened_at>2026-03-12</happened_at></fact>
</facts>
"""
    )
    facts = await extractor.extract("买了加湿器")
    assert facts[0].happened_at == "2026-03-12"


@pytest.mark.asyncio
async def test_happened_at_is_none_when_not_provided():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户正在等待Zigbee网关到货</summary><category>status</category><happened_at></happened_at></fact>
</facts>
"""
    )
    facts = await extractor.extract("在等 Zigbee 网关到货")
    assert facts[0].happened_at is None


@pytest.mark.asyncio
async def test_personal_fact_discards_happened_at_even_if_model_outputs_it():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户有一块 Fitbit 手表</summary><category>personal_fact</category><happened_at>2026-04-06</happened_at></fact>
</facts>
"""
    )
    facts = await extractor.extract("我有一块 Fitbit 手表")
    assert facts[0].happened_at is None


@pytest.mark.asyncio
async def test_duplicate_facts_within_one_extraction_are_deduplicated():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户购买了Zigbee网关</summary><category>purchase</category><happened_at>2026-03-12</happened_at></fact>
<fact><summary>用户购买了Zigbee网关</summary><category>purchase</category><happened_at>2026-03-12</happened_at></fact>
</facts>
"""
    )
    facts = await extractor.extract("买了 Zigbee 网关")
    assert len(facts) == 1


@pytest.mark.asyncio
async def test_extract_from_exchange_returns_only_targeted_categories():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户刚买了一个新键盘</summary><category>purchase</category><happened_at>2026-03-15</happened_at></fact>
<fact><summary>用户决定采用新方案</summary><category>decision</category><happened_at>2026-03-15</happened_at></fact>
<fact><summary>用户正在等待键盘到货</summary><category>status</category><happened_at></happened_at></fact>
</facts>
"""
    )
    facts = await extractor.extract_from_exchange("我买了键盘", "记住了")
    assert [fact.category for fact in facts] == ["purchase", "status"]


@pytest.mark.asyncio
async def test_extract_from_exchange_empty_for_chitchat():
    extractor = _make_extractor("<facts></facts>")
    facts = await extractor.extract_from_exchange("你好", "你好呀")
    assert facts == []


@pytest.mark.asyncio
async def test_extract_from_exchange_includes_both_user_and_agent_in_prompt():
    client = MagicMock()
    captured: list[str] = []

    async def _cap(*, messages, **kwargs):
        captured.append(messages[0]["content"])
        return "<facts></facts>"

    client.chat = AsyncMock(side_effect=_cap)
    extractor = ProfileFactExtractor(llm_client=client)
    await extractor.extract_from_exchange("我刚买了一个新键盘", "记住了")
    assert captured
    assert "我刚买了一个新键盘" in captured[0]
    assert "记住了" in captured[0]


# ---------------------------------------------------------------------------
# USER-first 证据源规则 — prompt 结构验证
# ---------------------------------------------------------------------------


def _capture_prompt(method_name: str = "extract") -> tuple["ProfileFactExtractor", list[str]]:
    """返回一个会记录 prompt 的 extractor 和 captured 列表。"""
    client = MagicMock()
    captured: list[str] = []

    async def _cap(*, messages, **kwargs):
        captured.append(messages[0]["content"])
        return "<facts></facts>"

    client.chat = AsyncMock(side_effect=_cap)
    return ProfileFactExtractor(llm_client=client), captured


@pytest.mark.asyncio
async def test_build_prompt_includes_user_first_evidence_rule():
    """_build_prompt() 必须包含 USER-first 证据源声明。"""
    extractor, captured = _capture_prompt()
    await extractor.extract("测试对话内容")
    assert captured
    assert "证据源规则" in captured[0]
    assert "ASSISTANT" in captured[0]
    assert "只有 USER 原话中明确陈述的事实才允许提取" in captured[0]
    assert "这是 preference，不是 profile" in captured[0]
    assert "<category>purchase|decision|status|personal_fact</category>" in captured[0]


@pytest.mark.asyncio
async def test_build_prompt_forbids_engineering_process():
    """_build_prompt() 必须禁止把工程操作提取为 profile。"""
    extractor, captured = _capture_prompt()
    await extractor.extract("测试对话内容")
    assert captured
    assert "工程操作" in captured[0]
    assert "安装依赖" in captured[0]


@pytest.mark.asyncio
async def test_build_prompt_forbids_hypothetical_first_person_examples():
    extractor, captured = _capture_prompt()
    await extractor.extract("测试对话内容")
    assert captured
    assert "举例 / 假设 / 如果 / 比如" in captured[0]
    assert "如果我有一家咖啡店，最近在亏损" in captured[0]


@pytest.mark.asyncio
async def test_build_prompt_distinguishes_profile_from_preference():
    extractor, captured = _capture_prompt()
    await extractor.extract("测试对话内容")
    assert captured
    assert "我有一块 Fitbit 手表" in captured[0]
    assert "讲内容时最好附带一个很棒的例子并贯穿始终" in captured[0]


@pytest.mark.asyncio
async def test_extract_ignores_preference_category_even_if_model_outputs_it():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>讲内容时最好附带一个很棒的例子并贯穿始终</summary><category>preference</category><happened_at></happened_at></fact>
</facts>
"""
    )
    facts = await extractor.extract("你给我讲内容的时候最好附带一个很棒的例子，并且最好贯穿始终。")
    assert facts == []


@pytest.mark.asyncio
async def test_build_exchange_prompt_assistant_not_evidence():
    """_build_exchange_prompt() 必须声明 ASSISTANT 内容不作为证据。"""
    extractor, captured = _capture_prompt()
    await extractor.extract_from_exchange("用户消息", "assistant 回复")
    assert captured
    assert "ASSISTANT" in captured[0]
    # 单轮 prompt 里的 ASSISTANT 不作为证据说明
    assert any(
        kw in captured[0]
        for kw in ["不算用户陈述", "不得提取", "不能作为"]
    )


@pytest.mark.asyncio
async def test_build_exchange_prompt_forbids_engineering_status():
    """_build_exchange_prompt() 必须禁止把工程操作写成 status。"""
    extractor, captured = _capture_prompt()
    await extractor.extract_from_exchange("用户消息", "assistant 回复")
    assert captured
    assert "工程操作" in captured[0] or "安装" in captured[0]


# ---------------------------------------------------------------------------
# PE1 — ASSISTANT 复述不产生幻觉 purchase profile（合成对话）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_rejects_assistant_restatement_as_purchase():
    """ASSISTANT 复述了购买事实，但 USER 只是提问，不应提取为 purchase。

    模拟 f4de91df8d02 类型污染：LLM 被告知 USER-first 规则后，应返回空列表。
    """
    # 合成对话：USER 只是记忆测试提问，ASSISTANT 复述了购买事实
    conversation = (
        "USER: 你还记得我之前买的那个鼠标吗？\n"
        "ASSISTANT: 你之前买了一款罗技 MX Master 3 鼠标。"
    )
    # mock LLM 正确遵从 USER-first 规则，返回空
    extractor = _make_extractor("<facts></facts>")
    facts = await extractor.extract(conversation)
    purchase_facts = [f for f in facts if f.category == "purchase"]
    assert not purchase_facts


@pytest.mark.asyncio
async def test_extract_rejects_engineering_operations_as_profile():
    """工程操作（升级工具、安装依赖）不应被提取为 profile。

    模拟 d6ab2c80fc4a 类型污染：LLM 遵从规则后应返回空列表。
    """
    conversation = (
        "USER: 我刚把 node 版本升到了 22，顺便装了一下 pnpm，配置文件也更新了。\n"
        "ASSISTANT: 好的，已记录。"
    )
    extractor = _make_extractor("<facts></facts>")
    facts = await extractor.extract(conversation)
    assert facts == [], f"不应提取工程操作为 profile，但得到: {facts}"
