from __future__ import annotations

import pytest

from feeds.base import FeedItem
from proactive.config import ProactiveConfig
from proactive.engine import ProactiveEngine
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


def test_build_proactive_memory_query_contains_source_labels():
    q = build_proactive_memory_query(
        items=[_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )
    assert "来源标签: rss:testfeed" in q
    assert "来源域名: example.com" in q


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
async def test_engine_feature_scoring_receives_retrieved_memory_block(tmp_path):
    class _Sense:
        def compute_energy(self):
            return 0.2

        def collect_recent(self):
            return [{"role": "user", "content": "之前说过喜欢魂类"}]

        def collect_recent_proactive(self, n=5):
            return []

        def compute_interruptibility(self, **kw):
            return 1.0, {
                "f_reply": 1.0,
                "f_activity": 1.0,
                "f_fatigue": 1.0,
                "random_delta": 0.0,
            }

        async def fetch_items(self, n):
            return [_item()]

        def filter_new_items(self, items):
            return items, [("rss:test", "item1")], []

        def read_memory_text(self):
            return ""

        def has_global_memory(self):
            return False

        def last_user_at(self):
            return None

        def refresh_sleep_context(self):
            return False

        def target_session_key(self):
            return "telegram:123"

    captured: dict[str, str] = {}

    class _Decide:
        async def score_features(self, **kw):
            captured["retrieved_memory_block"] = kw.get("retrieved_memory_block", "")
            return {
                "topic_continuity": 0.9,
                "interest_match": 0.9,
                "content_novelty": 0.7,
                "reconnect_value": 0.8,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            return "ping"

        async def reflect(self, *a, **kw):
            raise AssertionError("feature_scoring_enabled=true 时不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return []

        def build_delivery_key(self, ids, msg):
            return "k"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return "item1"

    class _Act:
        async def send(self, message, meta=None):
            return False

    class _Retrieval:
        async def retrieve_proactive_context(self, **kwargs):
            return ProactiveRetrievedMemory(
                query="q",
                block="## 相关记忆（本次触达召回）\n- 用户偏好魂类",
                item_ids=["p1"],
            )

    cfg = ProactiveConfig(
        enabled=True,
        feature_scoring_enabled=True,
        feature_send_threshold=0.0,
        score_llm_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
    )
    state = ProactiveStateStore(tmp_path / "state.json")
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_Sense(),
        decide=_Decide(),
        act=_Act(),
        memory_retrieval=_Retrieval(),
    )

    await engine.tick()
    assert "相关记忆" in captured.get("retrieved_memory_block", "")
