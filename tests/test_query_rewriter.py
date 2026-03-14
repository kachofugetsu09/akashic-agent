from unittest.mock import AsyncMock, MagicMock

import pytest

from memory2.query_rewriter import QueryRewriter, RewriteDecision


def _make_rewriter(llm_response: str) -> QueryRewriter:
    """用给定的 LLM 输出字符串构造一个 QueryRewriter。"""
    client = MagicMock()
    client.chat = AsyncMock(return_value=llm_response)
    return QueryRewriter(llm_client=client)


def test_rewrite_decision_is_dataclass_with_required_fields():
    d = RewriteDecision(
        needs_retrieval=True,
        procedure_query="B站下载",
        history_query="用户关于B站下载的历史偏好",
        memory_types_hint=["procedure", "preference"],
        latency_ms=42,
    )
    assert d.needs_retrieval is True
    assert d.procedure_query == "B站下载"
    assert d.history_query == "用户关于B站下载的历史偏好"
    assert d.memory_types_hint == ["procedure", "preference"]
    assert d.latency_ms == 42


@pytest.mark.asyncio
async def test_decide_retrieve_when_llm_says_retrieve():
    rewriter = _make_rewriter(
        """
<decision>RETRIEVE</decision>
<procedure_query>B站视频下载流程</procedure_query>
<history_query>用户的B站下载偏好历史</history_query>
<memory_types>procedure,preference</memory_types>
"""
    )
    result = await rewriter.decide(user_msg="把这个B站视频下载下来", recent_history="")
    assert result.needs_retrieval is True
    assert result.procedure_query == "B站视频下载流程"
    assert result.history_query == "用户的B站下载偏好历史"
    assert "procedure" in result.memory_types_hint


@pytest.mark.asyncio
async def test_decide_no_retrieve_for_greeting():
    rewriter = _make_rewriter(
        """
<decision>NO_RETRIEVE</decision>
<procedure_query>你好</procedure_query>
<history_query>你好</history_query>
<memory_types></memory_types>
"""
    )
    result = await rewriter.decide(user_msg="你好", recent_history="")
    assert result.needs_retrieval is False


@pytest.mark.asyncio
async def test_decide_fails_open_on_malformed_llm_output():
    """LLM 返回垃圾内容时，不应抛异常，应 fail-open 并使用原始消息作为 query。"""
    rewriter = _make_rewriter("这是乱码输出，没有任何 XML 标签")
    result = await rewriter.decide(user_msg="帮我搜代码", recent_history="")
    assert result.needs_retrieval is True
    assert result.procedure_query
    assert result.history_query


@pytest.mark.asyncio
async def test_decide_fails_open_on_llm_exception():
    """LLM 调用本身抛异常时，也要 fail-open。"""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=RuntimeError("timeout"))
    rewriter = QueryRewriter(llm_client=client)
    result = await rewriter.decide(user_msg="帮我搜代码", recent_history="")
    assert result.needs_retrieval is True
    assert result.procedure_query == "帮我搜代码"


@pytest.mark.asyncio
async def test_procedure_query_and_history_query_never_empty_on_retrieve():
    """needs_retrieval=True 时，两个 query 字段都不能是空串。"""
    rewriter = _make_rewriter(
        """
<decision>RETRIEVE</decision>
<procedure_query></procedure_query>
<history_query></history_query>
<memory_types>event</memory_types>
"""
    )
    result = await rewriter.decide(user_msg="帮我查一下", recent_history="")
    assert result.needs_retrieval is True
    assert result.procedure_query
    assert result.history_query


@pytest.mark.asyncio
async def test_memory_types_hint_parses_comma_separated():
    rewriter = _make_rewriter(
        """
<decision>RETRIEVE</decision>
<procedure_query>安装包</procedure_query>
<history_query>用户安装包的历史</history_query>
<memory_types>procedure, preference, event</memory_types>
"""
    )
    result = await rewriter.decide(user_msg="pacman 安装这个包", recent_history="")
    assert set(result.memory_types_hint) == {"procedure", "preference", "event"}


@pytest.mark.asyncio
async def test_memory_types_hint_empty_tag_returns_empty_list():
    rewriter = _make_rewriter(
        """
<decision>NO_RETRIEVE</decision>
<procedure_query>x</procedure_query>
<history_query>x</history_query>
<memory_types></memory_types>
"""
    )
    result = await rewriter.decide(user_msg="你好", recent_history="")
    assert result.memory_types_hint == []


@pytest.mark.asyncio
async def test_recent_history_injected_into_llm_prompt():
    """recent_history 不为空时，应出现在发给 LLM 的 prompt 里。"""
    client = MagicMock()
    captured_prompt: list[str] = []

    async def _capture(*, messages, **kwargs):
        captured_prompt.append(messages[0]["content"])
        return (
            "<decision>RETRIEVE</decision>"
            "<procedure_query>q</procedure_query>"
            "<history_query>q</history_query>"
            "<memory_types></memory_types>"
        )

    client.chat = AsyncMock(side_effect=_capture)
    rewriter = QueryRewriter(llm_client=client)

    history = "[user] 上一条消息\n[assistant] 上一条回复"
    await rewriter.decide(user_msg="现在呢", recent_history=history)

    assert captured_prompt
    assert "上一条消息" in captured_prompt[0]


@pytest.mark.asyncio
async def test_latency_ms_is_non_negative_int():
    rewriter = _make_rewriter(
        """
<decision>RETRIEVE</decision>
<procedure_query>q</procedure_query>
<history_query>q</history_query>
<memory_types></memory_types>
"""
    )
    result = await rewriter.decide(user_msg="test", recent_history="")
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0
