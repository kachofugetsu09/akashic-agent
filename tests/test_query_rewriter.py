from unittest.mock import AsyncMock, MagicMock

import pytest

from memory2.query_rewriter import GateDecision, QueryRewriter


def _make_rewriter(llm_response: str) -> QueryRewriter:
    """用给定的 LLM 输出字符串构造一个 QueryRewriter。"""
    client = MagicMock()
    client.chat = AsyncMock(return_value=llm_response)
    return QueryRewriter(llm_client=client)


def test_gate_decision_is_dataclass_with_required_fields():
    d = GateDecision(
        needs_episodic=True,
        episodic_query="用户关于B站下载的历史偏好",
        latency_ms=42,
        procedure_query="B站视频下载 SOP",
    )
    assert d.needs_episodic is True
    assert d.episodic_query == "用户关于B站下载的历史偏好"
    assert d.latency_ms == 42
    assert d.procedure_query == "B站视频下载 SOP"


@pytest.mark.asyncio
async def test_decide_retrieve_when_llm_says_retrieve():
    rewriter = _make_rewriter(
        """
<decision>RETRIEVE</decision>
<history_query>用户的B站下载偏好历史</history_query>
"""
    )
    result = await rewriter.decide(user_msg="把这个B站视频下载下来", recent_history="")
    assert result.needs_episodic is True
    assert result.episodic_query == "用户的B站下载偏好历史"


@pytest.mark.asyncio
async def test_decide_no_retrieve_for_greeting():
    rewriter = _make_rewriter(
        """
<decision>NO_RETRIEVE</decision>
<history_query>你好</history_query>
"""
    )
    result = await rewriter.decide(user_msg="你好", recent_history="")
    assert result.needs_episodic is False


@pytest.mark.asyncio
async def test_decide_fails_open_on_malformed_llm_output():
    """LLM 返回垃圾内容时，不应抛异常，应 fail-open 并使用原始消息作为 query。"""
    rewriter = _make_rewriter("这是乱码输出，没有任何 XML 标签")
    result = await rewriter.decide(user_msg="帮我搜代码", recent_history="")
    assert result.needs_episodic is True
    assert result.episodic_query


@pytest.mark.asyncio
async def test_decide_fails_open_on_llm_exception():
    """LLM 调用本身抛异常时，也要 fail-open。"""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=RuntimeError("timeout"))
    rewriter = QueryRewriter(llm_client=client)
    result = await rewriter.decide(user_msg="帮我搜代码", recent_history="")
    assert result.needs_episodic is True
    assert result.episodic_query == "帮我搜代码"
    assert result.procedure_query == ""


@pytest.mark.asyncio
async def test_decide_preserves_procedure_query_when_history_call_fails():
    client = MagicMock()
    calls = 0

    async def _chat(*, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("history failed")
        return "用户发送哔哩哔哩视频链接时 agent 应如何处理"

    client.chat = AsyncMock(side_effect=_chat)
    rewriter = QueryRewriter(llm_client=client)

    result = await rewriter.decide(
        user_msg="【视频-哔哩哔哩】 https://example.test/item",
        recent_history="",
    )

    assert result.needs_episodic is True
    assert result.episodic_query.startswith("【视频")
    assert result.procedure_query == "用户发送哔哩哔哩视频链接时 agent 应如何处理"


@pytest.mark.asyncio
async def test_decide_cleans_empty_procedure_query_sentinels():
    client = MagicMock()

    async def _chat(*, messages, **kwargs):
        prompt = messages[0]["content"]
        if "只输出一行检索 query" in prompt:
            return "None."
        return """
<decision>NO_RETRIEVE</decision>
<history_query></history_query>
"""

    client.chat = AsyncMock(side_effect=_chat)
    rewriter = QueryRewriter(llm_client=client)

    result = await rewriter.decide(user_msg="你好", recent_history="")

    assert result.needs_episodic is False
    assert result.procedure_query == ""


@pytest.mark.asyncio
async def test_episodic_query_never_empty_on_retrieve():
    """needs_episodic=True 时，episodic_query 不能是空串。"""
    rewriter = _make_rewriter(
        """
<decision>RETRIEVE</decision>
<history_query></history_query>
"""
    )
    result = await rewriter.decide(user_msg="帮我查一下", recent_history="")
    assert result.needs_episodic is True
    assert result.episodic_query


@pytest.mark.asyncio
async def test_recent_history_injected_into_llm_prompt():
    """recent_history 不为空时，应出现在发给 LLM 的 prompt 里。"""
    client = MagicMock()
    captured_prompt: list[str] = []

    async def _capture(*, messages, **kwargs):
        captured_prompt.append(messages[0]["content"])
        return (
            "<decision>RETRIEVE</decision>"
            "<history_query>q</history_query>"
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
<history_query>q</history_query>
"""
    )
    result = await rewriter.decide(user_msg="test", recent_history="")
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0
