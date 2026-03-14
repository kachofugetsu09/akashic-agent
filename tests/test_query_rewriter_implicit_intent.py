from unittest.mock import AsyncMock, MagicMock

import pytest

from memory2.query_rewriter import QueryRewriter


def _make_rewriter(llm_response: str) -> QueryRewriter:
    client = MagicMock()
    client.chat = AsyncMock(return_value=llm_response)
    return QueryRewriter(llm_client=client)


@pytest.mark.asyncio
async def test_logistics_message_implies_purchase_history_retrieval():
    rewriter = _make_rewriter(
        """
<thinking>用户提到申通快递，隐含是在问最近买了什么。</thinking>
<decision>RETRIEVE</decision>
<procedure_query>最近购买记录</procedure_query>
<history_query>用户最近购买了什么商品</history_query>
<memory_types>event,profile</memory_types>
"""
    )
    result = await rewriter.decide(user_msg="这是个申通快递，你猜最近买了什么？", recent_history="")
    assert result.needs_retrieval is True
    assert "购买" in result.history_query or "购物" in result.history_query
    assert "物流" not in result.history_query


@pytest.mark.asyncio
async def test_package_arrival_message_implies_purchase_retrieval():
    rewriter = _make_rewriter(
        """
<thinking>包裹到达是在追问最近下单的东西。</thinking>
<decision>RETRIEVE</decision>
<procedure_query>最近购买记录</procedure_query>
<history_query>用户最近购买和到货的商品记录</history_query>
<memory_types>event,profile</memory_types>
"""
    )
    result = await rewriter.decide(user_msg="我的包裹到了吗", recent_history="")
    assert result.needs_retrieval is True
    assert "购买" in result.history_query or "商品" in result.history_query


@pytest.mark.asyncio
async def test_health_symptom_implies_user_health_profile():
    rewriter = _make_rewriter(
        """
<thinking>用户描述近期症状，隐含需要结合健康档案。</thinking>
<decision>RETRIEVE</decision>
<procedure_query>健康情况</procedure_query>
<history_query>用户最近头疼相关的健康档案和历史记录</history_query>
<memory_types>profile,event</memory_types>
"""
    )
    result = await rewriter.decide(user_msg="最近头总是疼", recent_history="")
    assert result.needs_retrieval is True
    assert "健康" in result.history_query or "档案" in result.history_query


@pytest.mark.asyncio
async def test_pure_logistics_query_without_personal_context_may_skip():
    rewriter = _make_rewriter(
        """
<thinking>这是工具型物流查询，不需要查个人记忆。</thinking>
<decision>NO_RETRIEVE</decision>
<procedure_query>快递状态查询</procedure_query>
<history_query>顺丰单号查询</history_query>
<memory_types></memory_types>
"""
    )
    result = await rewriter.decide(user_msg="帮我查一下顺丰1234567的快递状态", recent_history="")
    assert result.history_query != "快递状态"


@pytest.mark.asyncio
async def test_thinking_block_not_in_xml_output():
    rewriter = _make_rewriter(
        """
<thinking>这里是内部推理，不应污染最终字段。</thinking>
<decision>RETRIEVE</decision>
<procedure_query>最近购买记录</procedure_query>
<history_query>用户最近购买了什么</history_query>
<memory_types>profile</memory_types>
"""
    )
    result = await rewriter.decide(user_msg="这是个申通快递，你看看", recent_history="")
    assert "<thinking>" not in result.procedure_query
    assert "<thinking>" not in result.history_query


@pytest.mark.asyncio
async def test_explicit_history_question_with_implicit_context():
    rewriter = _make_rewriter(
        """
<thinking>显式问最近买了什么，快递只是补充上下文。</thinking>
<decision>RETRIEVE</decision>
<procedure_query>最近购买记录</procedure_query>
<history_query>用户最近购买了什么商品和相关快递记录</history_query>
<memory_types>profile,event</memory_types>
"""
    )
    result = await rewriter.decide(user_msg="我最近买了什么 那个快递是什么", recent_history="")
    assert result.needs_retrieval is True
    assert "购买" in result.history_query
