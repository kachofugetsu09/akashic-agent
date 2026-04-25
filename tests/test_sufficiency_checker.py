from unittest.mock import AsyncMock, MagicMock

import pytest

from memory2.sufficiency_checker import SufficiencyChecker, SufficiencyResult


def _make_checker(llm_response: str) -> SufficiencyChecker:
    client = MagicMock()
    client.chat = AsyncMock(return_value=llm_response)
    return SufficiencyChecker(llm_client=client)


def _item(memory_type: str, score: float, summary: str) -> dict:
    return {"id": summary[:8], "memory_type": memory_type, "score": score, "summary": summary}


def test_sufficiency_result_fields():
    r = SufficiencyResult(
        is_sufficient=False,
        reason="irrelevant",
        refined_query="用户与仁王游戏相关的讨论记录",
        latency_ms=55,
    )
    assert r.is_sufficient is False
    assert r.reason == "irrelevant"
    assert r.refined_query == "用户与仁王游戏相关的讨论记录"


def test_sufficiency_result_sufficient_has_no_refined_query():
    """sufficient 时 refined_query 应为 None，不需要重查。"""
    r = SufficiencyResult(
        is_sufficient=True,
        reason="sufficient",
        refined_query=None,
        latency_ms=30,
    )
    assert r.refined_query is None


def test_should_check_when_no_items():
    from memory2.sufficiency_checker import should_check_sufficiency

    assert should_check_sufficiency([]) is True


def test_should_not_check_when_items_present():
    """有结果时不触发——分数过滤已在注入筛选阶段完成。"""
    from memory2.sufficiency_checker import should_check_sufficiency

    items = [
        _item("procedure", 0.479, "西历2236读书进度"),
        _item("procedure", 0.461, "其他规则"),
    ]
    assert should_check_sufficiency(items) is False


def test_should_not_check_when_single_item_present():
    from memory2.sufficiency_checker import should_check_sufficiency

    items = [_item("procedure", 0.538, "天气查询强制走 weather 技能")]
    assert should_check_sufficiency(items) is False


def test_should_not_check_when_forced_procedure_present():
    """有 tool_requirement 的 forced procedure 存在时，不触发 sufficiency check。"""
    from memory2.sufficiency_checker import should_check_sufficiency

    items = [
        {
            "id": "p1",
            "memory_type": "procedure",
            "score": 0.42,
            "summary": "必须先查工具状态",
            "extra_json": {"tool_requirement": "tool_search"},
        },
    ]
    assert should_check_sufficiency(items) is False


@pytest.mark.asyncio
async def test_check_returns_sufficient_when_llm_says_so():
    """真实场景：天气查询命中 weather 技能 procedure，LLM 判断相关。"""
    checker = _make_checker(
        """
<sufficient>yes</sufficient>
<refined_query></refined_query>
"""
    )
    items = [
        _item(
            "procedure",
            0.538,
            "天气查询强制走 weather 技能：询问天气时必须先读取并执行 weather 技能的 SKILL.md",
        )
    ]
    result = await checker.check(query="北京今天天气怎么样", items=items)
    assert result.is_sufficient is True
    assert result.refined_query is None


@pytest.mark.asyncio
async def test_check_returns_insufficient_with_refined_query():
    """真实场景：问仁王的内容，命中了西历2236读书进度，LLM 判断无关。"""
    checker = _make_checker(
        """
<sufficient>no</sufficient>
<refined_query>用户与仁王游戏相关的讨论历史</refined_query>
"""
    )
    items = [
        _item(
            "procedure",
            0.479,
            "分享《西历 2236》阅读进度前，必须先回源确认阅读记录",
        )
    ]
    result = await checker.check(
        query="我之前和你聊过什么有关仁王的内容吗",
        items=items,
    )
    assert result.is_sufficient is False
    assert result.refined_query == "用户与仁王游戏相关的讨论历史"


@pytest.mark.asyncio
async def test_check_returns_partial_keeps_existing_items():
    """partial：有些相关有些不相关，不触发重查，保留现有结果。"""
    checker = _make_checker(
        """
<sufficient>partial</sufficient>
<refined_query></refined_query>
"""
    )
    items = [
        _item("event", 0.51, "用户讨论了仁王的机制"),
        _item("procedure", 0.48, "西历2236读书进度规则"),
    ]
    result = await checker.check(query="仁王的内容", items=items)
    assert result.is_sufficient is True
    assert result.reason == "partial"


@pytest.mark.asyncio
async def test_check_empty_items_returns_insufficient():
    """无结果时，LLM 应判断 no，refined_query 应为改写后的 query。"""
    checker = _make_checker(
        """
<sufficient>no</sufficient>
<refined_query>用户过去讨论仁王游戏的相关记忆</refined_query>
"""
    )
    result = await checker.check(query="我之前和你聊过什么有关仁王的内容吗", items=[])
    assert result.is_sufficient is False
    assert result.refined_query


@pytest.mark.asyncio
async def test_check_fails_open_on_malformed_output():
    """LLM 返回乱码时，视为 sufficient，不触发重查。"""
    checker = _make_checker("这是乱码，没有 XML 标签")
    items = [_item("procedure", 0.48, "某个规则")]
    result = await checker.check(query="随便", items=items)
    assert result.is_sufficient is True
    assert result.reason == "parse_error"


@pytest.mark.asyncio
async def test_check_fails_open_on_llm_exception():
    client = MagicMock()
    client.chat = AsyncMock(side_effect=RuntimeError("timeout"))
    checker = SufficiencyChecker(llm_client=client)
    result = await checker.check(query="test", items=[])
    assert result.is_sufficient is True
    assert result.reason == "checker_error"


@pytest.mark.asyncio
async def test_item_summaries_appear_in_llm_prompt():
    """checker 把 item summaries 格式化后传给 LLM，LLM 才能判断相关性。"""
    client = MagicMock()
    captured: list[str] = []

    async def _cap(*, messages, **kwargs):
        captured.append(messages[0]["content"])
        return "<sufficient>yes</sufficient><refined_query></refined_query>"

    client.chat = AsyncMock(side_effect=_cap)
    checker = SufficiencyChecker(llm_client=client)
    items = [_item("procedure", 0.48, "西历2236读书进度规则")]

    await checker.check(query="仁王内容", items=items)
    assert captured
    assert "西历2236读书进度规则" in captured[0]
    assert "仁王内容" in captured[0]


@pytest.mark.asyncio
async def test_latency_ms_is_non_negative_int():
    checker = _make_checker("<sufficient>yes</sufficient><refined_query></refined_query>")
    result = await checker.check(query="test", items=[])
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0
