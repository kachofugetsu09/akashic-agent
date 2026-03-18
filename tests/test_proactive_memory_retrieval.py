from __future__ import annotations

import pytest

from feeds.base import FeedItem
from proactive.config import ProactiveConfig
from proactive.components import build_proactive_preference_hyde_prompt
from proactive.tick import ProactiveEngine
from proactive.ports import DefaultMemoryRetrievalPort, ProactiveRetrievedMemory
from proactive.state import ProactiveStateStore
from proactive.components import build_proactive_memory_query


def _item() -> FeedItem:
    return FeedItem(
        source_name="TestFeed",
        source_type="rss",
        title="Elden Ring DLC",
        content="Trailer and release window update.",
        url="https://example.com/post",
        author=None,
        published_at=None,
    )


def _item_with(*, source_name: str, title: str, url: str) -> FeedItem:
    return FeedItem(
        source_name=source_name,
        source_type="rss",
        title=title,
        content=title,
        url=url,
        author=None,
        published_at=None,
    )


def test_build_proactive_memory_query_contains_source_labels():
    q = build_proactive_memory_query(
        items=[_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )
    assert "来源标签: rss:testfeed" in q
    assert "来源域名: example.com" in q


def test_build_proactive_preference_hyde_prompt_matches_preference_style():
    prompt = build_proactive_preference_hyde_prompt(
        query="用户对 HLTV News 的偏好和态度；相关话题：HooXi on people writing him off after G2",
        context="候选内容：HooXi on people writing him off after G2\n来源：HLTV News",
    )
    assert "偏好记忆系统" in prompt
    assert "用户明确" in prompt
    assert "长期偏好" in prompt
    assert "不要总结新闻事实本身" in prompt
    assert "负向偏好" in prompt or "反感" in prompt


@pytest.mark.asyncio
async def test_default_memory_retrieval_port_uses_event_only_history_channel():
    calls: list[dict] = []

    class _Memory:
        async def retrieve_related(self, query, **kwargs):
            calls.append({"query": query, **kwargs})
            if kwargs.get("memory_types") == ["procedure", "preference"]:
                return [{"id": "p1", "memory_type": "procedure", "summary": "先查证"}]
            if kwargs.get("memory_types") == ["event"]:
                return [{"id": "e1", "memory_type": "event", "summary": "聊过DLC"}]
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "## block", [str(i.get("id")) for i in items if i.get("id")]

    cfg = ProactiveConfig()
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_item()],
        recent=[{"role": "user", "content": "之前聊过DLC"}],
        decision_signals={},
        is_crisis=False,
    )

    assert result.fallback_reason == ""
    history_calls = [c for c in calls if c.get("memory_types") == ["event"]]
    assert history_calls, "H 通道应检索 event"
    assert not any(c.get("memory_types") == ["profile"] for c in calls)


@pytest.mark.asyncio
async def test_history_channel_scoped_first_without_fallback():
    calls: list[dict] = []

    class _Memory:
        async def retrieve_related(self, query, **kwargs):
            calls.append({"query": query, **kwargs})
            if kwargs.get("memory_types") == ["procedure", "preference"]:
                return []
            if kwargs.get("memory_types") == ["event"] and kwargs.get(
                "require_scope_match"
            ):
                return []
            if kwargs.get("memory_types") == ["event"]:
                return [{"id": "e-global", "memory_type": "event", "summary": "global"}]
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "", []

    cfg = ProactiveConfig(memory_scope_fallback_to_global=False)
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    event_calls = [c for c in calls if c.get("memory_types") == ["event"]]
    assert len(event_calls) == 1
    assert event_calls[0].get("require_scope_match") is True
    assert result.history_scope_mode == "disabled"
    assert result.history_hits == 0


@pytest.mark.asyncio
async def test_history_channel_scoped_first_with_global_fallback():
    calls: list[dict] = []

    class _Memory:
        async def retrieve_related(self, query, **kwargs):
            calls.append({"query": query, **kwargs})
            if kwargs.get("memory_types") == ["procedure", "preference"]:
                return []
            if kwargs.get("memory_types") == ["event"] and kwargs.get(
                "require_scope_match"
            ):
                return []
            if kwargs.get("memory_types") == ["event"]:
                return [{"id": "e-global", "memory_type": "event", "summary": "global"}]
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "## block", [str(i.get("id")) for i in items if i.get("id")]

    cfg = ProactiveConfig(memory_scope_fallback_to_global=True)
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    event_calls = [c for c in calls if c.get("memory_types") == ["event"]]
    assert len(event_calls) == 2
    assert event_calls[0].get("require_scope_match") is True
    assert event_calls[1].get("require_scope_match") is False
    assert result.history_scope_mode == "global-fallback"
    assert result.history_hits == 1


@pytest.mark.asyncio
async def test_default_memory_retrieval_port_fail_open_on_exception():
    class _BrokenMemory:
        async def retrieve_related(self, query, **kwargs):
            raise RuntimeError("boom")

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "", []

    cfg = ProactiveConfig()
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_BrokenMemory(),
        item_id_fn=lambda _: "item1",
    )

    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )
    assert result.block == ""
    assert result.fallback_reason == "retrieve_exception"


@pytest.mark.asyncio
async def test_history_channel_skips_scoped_when_channel_or_chat_missing():
    calls: list[dict] = []

    class _Memory:
        async def retrieve_related(self, query, **kwargs):
            calls.append({"query": query, **kwargs})
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "", []

    cfg = ProactiveConfig(memory_scope_fallback_to_global=False)
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    await port.retrieve_proactive_context(
        session_key="",
        channel="",
        chat_id="",
        items=[_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    event_calls = [c for c in calls if c.get("memory_types") == ["event"]]
    assert len(event_calls) == 0


@pytest.mark.asyncio
async def test_preference_retrieval_queries_same_source_items_separately():
    calls: list[dict] = []

    class _Memory:
        async def retrieve_related(self, query, **kwargs):
            calls.append({"query": query, **kwargs})
            if kwargs.get("memory_types") == ["procedure", "preference"]:
                return []
            if kwargs.get("memory_types") == ["event"]:
                return []
            if kwargs.get("memory_types") == ["preference", "profile"]:
                return [{"id": query, "memory_type": "preference", "summary": query}]
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "## block", [str(i.get("id")) for i in items if i.get("id")]

    cfg = ProactiveConfig(preference_retrieval_enabled=True, preference_per_source_top_k=2)
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[
            _item_with(
                source_name="HLTV News",
                title="w0nderful shines as NAVI beat Aurora to lift EPL",
                url="https://www.hltv.org/news/1",
            ),
            _item_with(
                source_name="HLTV News",
                title='HooXi on people writing him off after G2: "It feels good"',
                url="https://www.hltv.org/news/2",
            ),
        ],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    pref_calls = [c for c in calls if c.get("memory_types") == ["preference", "profile"]]
    assert len(pref_calls) == 2
    assert "w0nderful shines as NAVI beat Aurora to lift EPL" in pref_calls[0]["query"]
    assert 'HooXi on people writing him off after G2: "It feels good"' in pref_calls[1]["query"]


@pytest.mark.asyncio
async def test_preference_retrieval_uses_hyde_query_when_enabled():
    calls: list[str] = []
    prompts: list[str] = []

    class _Provider:
        async def chat(self, **kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            from agent.provider import LLMResponse

            return LLMResponse(
                content="用户明确关注 HooXi 与 NiKo、G2 旧阵容相关的人物动态。",
                tool_calls=[],
            )

    class _Memory:
        async def retrieve_related(self, query, **kwargs):
            if kwargs.get("memory_types") == ["preference", "profile"]:
                calls.append(query)
                return [{"id": query, "memory_type": "preference", "summary": query}]
            if kwargs.get("memory_types") == ["procedure", "preference"]:
                return []
            if kwargs.get("memory_types") == ["event"]:
                return []
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            summaries = [str(i.get("summary", "")) for i in items]
            ids = [str(i.get("id")) for i in items if i.get("id")]
            return "\n".join(summaries), ids

    cfg = ProactiveConfig(
        preference_retrieval_enabled=True,
        preference_hyde_enabled=True,
        preference_per_source_top_k=2,
    )
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
        light_provider=_Provider(),
        light_model="qwen-flash",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[
            _item_with(
                source_name="HLTV News",
                title='HooXi on people writing him off after G2: "It feels good"',
                url="https://www.hltv.org/news/2",
            )
        ],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert len(calls) == 2
    assert "相关话题" in calls[0]
    assert "用户明确关注 HooXi 与 NiKo、G2 旧阵容相关的人物动态。" == calls[1]
    assert prompts and "不要总结新闻事实本身" in prompts[0]
    assert "用户明确关注 HooXi" in result.preference_block


@pytest.mark.asyncio
async def test_preference_retrieval_hyde_falls_back_to_raw_on_failure():
    calls: list[str] = []

    class _BrokenProvider:
        async def chat(self, **kwargs):
            raise RuntimeError("hyde failed")

    class _Memory:
        async def retrieve_related(self, query, **kwargs):
            if kwargs.get("memory_types") == ["preference", "profile"]:
                calls.append(query)
                return [{"id": query, "memory_type": "preference", "summary": query}]
            if kwargs.get("memory_types") == ["procedure", "preference"]:
                return []
            if kwargs.get("memory_types") == ["event"]:
                return []
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            summaries = [str(i.get("summary", "")) for i in items]
            ids = [str(i.get("id")) for i in items if i.get("id")]
            return "\n".join(summaries), ids

    cfg = ProactiveConfig(
        preference_retrieval_enabled=True,
        preference_hyde_enabled=True,
        preference_per_source_top_k=2,
    )
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
        light_provider=_BrokenProvider(),
        light_model="qwen-flash",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert len(calls) == 1
    assert "Elden Ring DLC" in calls[0]
    assert "Elden Ring DLC" in result.preference_block
