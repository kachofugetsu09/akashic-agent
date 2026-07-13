from unittest.mock import AsyncMock, MagicMock

import pytest

from memory2.profile_extractor import ProfileFactExtractor


def _make_extractor(llm_response: str) -> ProfileFactExtractor:
    client = MagicMock()
    client.chat = AsyncMock(return_value=llm_response)
    return ProfileFactExtractor(llm_client=client)


@pytest.mark.asyncio
async def test_game_completion_captured_as_status():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户和朋友通关了仁王3 A10</summary><category>status</category><happened_at></happened_at></fact>
</facts>
"""
    )
    facts = (await extractor.extract("我和朋友通关了仁王3 A10")).facts
    assert any(f.category == "status" and "通关" in f.summary for f in facts)


@pytest.mark.asyncio
async def test_project_launch_captured_as_decision_or_status():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户将仓库脱敏后公开发布</summary><category>decision</category><happened_at></happened_at></fact>
</facts>
"""
    )
    facts = (await extractor.extract("我把仓库脱敏后公开了")).facts
    assert facts
    assert facts[0].category in {"decision", "status"}


@pytest.mark.asyncio
async def test_routine_chat_not_captured():
    extractor = _make_extractor("<facts></facts>")
    result = await extractor.extract("今天天气不错")
    assert result.facts == ()


@pytest.mark.asyncio
async def test_exchange_milestone_captured():
    extractor = _make_extractor(
        """
<facts>
<fact><summary>用户和朋友通关了仁王3 A10</summary><category>status</category><happened_at></happened_at></fact>
</facts>
"""
    )
    facts = (
        await extractor.extract_from_exchange(
            "我和朋友通关了仁王3 A10",
            "记住了",
        )
    ).facts
    assert any(f.category == "status" and "通关" in f.summary for f in facts)
