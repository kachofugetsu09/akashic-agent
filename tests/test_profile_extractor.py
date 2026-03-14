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
