from __future__ import annotations

import json

import pytest

from agent.memory import MemoryStore
from core.memory.port import DefaultMemoryPort
from feeds.base import FeedItem
from memory2.memorizer import Memorizer
from memory2.retriever import Retriever
from memory2.store import MemoryStore2
from proactive.components import build_proactive_preference_query
from proactive.config import ProactiveConfig
from proactive.ports import DefaultMemoryRetrievalPort, ProactiveRetrievedMemory


def _orbit_item() -> FeedItem:
    return FeedItem(
        source_name="HLTV",
        source_type="rss",
        title="TeamOrbit beat TeamForge to march to EPL playoffs",
        content="TeamOrbit defeated TeamForge in a convincing match.",
        url="https://www.hltv.org/news/44042/teamorbit-beat-teamforge",
        author=None,
        published_at=None,
    )


def _atlas_item() -> FeedItem:
    return FeedItem(
        source_name="HLTV",
        source_type="rss",
        title="TeamAtlas win ESL Pro League",
        content="TeamAtlas secured the trophy with PlayerNova at the helm.",
        url="https://www.hltv.org/news/00001/teamatlas-win-epl",
        author=None,
        published_at=None,
    )


def _hltv_major_race_item() -> FeedItem:
    return FeedItem(
        source_name="HLTV",
        source_type="rss",
        title="科隆 Major 名额冲刺分析：TeamComet 势头很猛，TeamDelta 基本稳了",
        content=(
            "刚看到 HLTV 的科隆 Major 名额冲刺分析，TeamComet 势头很猛，TeamDelta 基本稳了。"
            "虽然没直接提到 PlayerNova 和 TeamAtlas 的战况，但大赛前的格局变动总是值得留意。"
        ),
        url=(
            "https://www.hltv.org/news/44056/"
            "cologne-major-race-update-teamcomet-surge-teamdelta-all-but-confirm-spot-after-epl-run"
        ),
        author=None,
        published_at=None,
    )


class _FakePreferenceEmbedder:
    _KEYWORDS = (
        ("hltv", "cs", "counter-strike"),
        ("teamatlas", "playernova"),
        ("major", "名额", "冲刺", "科隆", "teamcomet", "teamdelta", "race", "格局"),
        ("只想看", "只关注", "不想看", "不关心", "偏好"),
    )

    async def embed(self, text: str) -> list[float]:
        raw = (text or "").lower()
        vector = [
            float(sum(1 for kw in group if kw in raw)) for group in self._KEYWORDS
        ]
        norm = sum(v * v for v in vector) ** 0.5
        if norm <= 0:
            return [0.0 for _ in vector]
        return [v / norm for v in vector]


def _build_real_preference_memory_port(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    embedder = _FakePreferenceEmbedder()
    memorizer = Memorizer(store, embedder)
    retriever = Retriever(
        store=store,
        embedder=embedder,
        score_threshold=0.2,
        score_thresholds={
            "procedure": 0.2,
            "preference": 0.2,
            "event": 0.2,
            "profile": 0.2,
        },
        relative_delta=0.2,
    )
    return DefaultMemoryPort(
        MemoryStore(tmp_path), memorizer=memorizer, retriever=retriever
    )


def test_preference_query_includes_item_source_name():
    query = build_proactive_preference_query(items=[_orbit_item()], max_items=3)
    assert "hltv" in query.lower()
    assert "teamorbit" in query.lower()


def test_preference_query_includes_multiple_sources():
    query = build_proactive_preference_query(
        items=[_orbit_item(), _atlas_item()],
        max_items=3,
    )
    query_lower = query.lower()
    assert "hltv" in query_lower
    assert "teamatlas" in query_lower or "teamorbit" in query_lower


def test_preference_query_contains_preference_signal_words():
    query = build_proactive_preference_query(items=[_orbit_item()], max_items=3)
    assert any(
        word in query
        for word in ["偏好", "兴趣", "关注", "喜欢", "不关心", "preference", "interest"]
    )


@pytest.mark.asyncio
async def test_memory_port_preference_query_includes_title_and_fallback_source():
    calls: list[str] = []

    class _Memory:
        async def retrieve_related(self, query: str, **kwargs):
            if set(kwargs.get("memory_types") or []) == {"preference", "profile"}:
                calls.append(query)
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "", []

    item = FeedItem(
        source_name="",
        source_type="",
        title="这是一条没有来源标记但有标题的话题偏好测试",
        content="",
        url=None,
        author=None,
        published_at=None,
    )
    port = DefaultMemoryRetrievalPort(
        cfg=ProactiveConfig(preference_retrieval_enabled=True),
        memory=_Memory(),
        item_id_fn=lambda _: "x",
    )
    await port.retrieve_proactive_context(
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        items=[item],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )
    assert calls
    assert "相关话题" in calls[0]
    assert "话题偏好测试" in calls[0]


def test_proactive_retrieved_memory_has_preference_block():
    result = ProactiveRetrievedMemory()
    assert hasattr(result, "preference_block")
    assert result.preference_block == ""


def test_proactive_retrieved_memory_empty_has_preference_block():
    result = ProactiveRetrievedMemory.empty("test")
    assert result.preference_block == ""


@pytest.mark.asyncio
async def test_memory_port_sends_preference_specific_query():
    pref_calls: list[dict] = []

    class _Memory:
        async def retrieve_related(self, query: str, **kwargs):
            if set(kwargs.get("memory_types") or []) == {"preference", "profile"}:
                pref_calls.append({"query": query, **kwargs})
                return [
                    {
                        "id": "p1",
                        "memory_type": "preference",
                        "summary": "只关注 TeamAtlas 和 PlayerNova，不关心其他战队",
                    }
                ]
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            if not items:
                return "", []
            return "## block\n- 偏好", [str(i.get("id")) for i in items if i.get("id")]

    port = DefaultMemoryRetrievalPort(
        cfg=ProactiveConfig(),
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_orbit_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert pref_calls
    assert "hltv" in pref_calls[0]["query"].lower() or "teamorbit" in pref_calls[0][
        "query"
    ].lower()
    assert result.preference_block


@pytest.mark.asyncio
async def test_memory_port_preference_block_empty_when_no_preference_hits():
    class _Memory:
        async def retrieve_related(self, query: str, **kwargs):
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "", []

    port = DefaultMemoryRetrievalPort(
        cfg=ProactiveConfig(),
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_orbit_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert result.preference_block == ""


def test_config_loader_parses_preference_fields(tmp_path):
    from agent.config import load_config

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude-test",
                "api_key": "test",
                "proactive": {
                    "enabled": False,
                    "default_chat_id": "123",
                    "compose_judge_enabled": False,
                    "preference_retrieval_enabled": False,
                    "preference_top_k": 8,
                    "preference_hyde_enabled": True,
                    "preference_hyde_timeout_ms": 3200,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match="compose_judge_enabled"):
        cfg = load_config(str(cfg_file))
    p = cfg.proactive

    assert not hasattr(p, "compose_judge_enabled")
    assert p.preference_retrieval_enabled is False
    assert p.preference_top_k == 8
    assert p.preference_hyde_enabled is True
    assert p.preference_hyde_timeout_ms == 3200


def test_config_loader_parses_interest_filter_fields(tmp_path):
    from agent.config import load_config

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude-test",
                "api_key": "test",
                "proactive": {
                    "enabled": False,
                    "default_chat_id": "123",
                    "interest_filter": {
                        "enabled": True,
                        "memory_max_chars": 5000,
                        "keyword_max_count": 120,
                        "min_token_len": 3,
                        "min_score": 0.22,
                        "top_k": 6,
                        "exploration_ratio": 0.1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))
    interest_filter = getattr(cfg.proactive, "interest_filter", None)

    assert interest_filter is not None
    assert interest_filter.enabled is True
    assert interest_filter.memory_max_chars == 5000
    assert interest_filter.keyword_max_count == 120
    assert interest_filter.min_token_len == 3
    assert interest_filter.min_score == 0.22
    assert interest_filter.top_k == 6
    assert interest_filter.exploration_ratio == 0.1


@pytest.mark.asyncio
async def test_real_vector_retrieval_hits_hltv_major_race_preference(tmp_path):
    port = _build_real_preference_memory_port(tmp_path)
    await port.save_item(
        summary=(
            "HLTV 的 CS 资讯里，我只想看 PlayerNova、TeamAtlas 相关消息；"
            "不想看 TeamComet、TeamDelta 这种 Major 名额分析。"
        ),
        memory_type="preference",
        extra={},
        source_ref="pref-major-race",
    )
    await port.save_item(
        summary="更关心 CS2 枪皮和 V 社更新。",
        memory_type="preference",
        extra={},
        source_ref="pref-skins",
    )
    retrieval = DefaultMemoryRetrievalPort(
        cfg=ProactiveConfig(preference_retrieval_enabled=True, preference_top_k=4),
        memory=port,
        item_id_fn=lambda _: "item1",
    )

    result = await retrieval.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_hltv_major_race_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert "PlayerNova" in result.preference_block
    assert "TeamAtlas" in result.preference_block
    assert "TeamComet" in result.preference_block
    assert "TeamDelta" in result.preference_block
