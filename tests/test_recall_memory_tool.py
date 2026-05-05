# pyright: reportPrivateUsage=false, reportUnusedCallResult=false

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias, cast
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

import agent.tools.recall_memory as recall_memory_module
import memory2.retriever as retriever_module
from agent.provider import LLMProvider, LLMResponse
from agent.tools.recall_memory import RecallMemoryTool
from core.memory.default_engine import DefaultMemoryEngine
from memory2.embedder import Embedder
from memory2.retriever import Retriever
from memory2.store import MemoryStore2

_MemoryHit: TypeAlias = dict[str, object]
_EmbeddingRow: TypeAlias = tuple[
    str,
    str,
    str,
    list[float] | None,
    dict[str, object],
    str | None,
    str | None,
]


def _recall_tool(store: object, embedder: object, provider: object) -> RecallMemoryTool:
    retriever = Retriever(cast(MemoryStore2, store), cast(Embedder, embedder))
    facade = DefaultMemoryEngine.__new__(DefaultMemoryEngine)
    facade._config = None
    facade._workspace = Path(".")
    facade._provider = cast(LLMProvider, provider)
    facade._light_provider = cast(LLMProvider, provider)
    facade._light_model = "test-model"
    facade._v1_store = None
    facade._v2_store = store
    facade._embedder = embedder
    facade._memorizer = None
    facade._retriever = retriever
    facade._tagger = None
    facade._post_response_worker = None
    facade._event_bus = None
    facade._consolidation = None
    facade.closeables = []
    return RecallMemoryTool(
        facade=cast(Any, facade),
    )


class _FakeProvider:
    chat: AsyncMock

    def __init__(self, content: str = "") -> None:
        self.chat = AsyncMock(return_value=LLMResponse(content=content))


class _FailingEmbedder:
    async def embed(self, text: str) -> list[float]:
        raise RuntimeError(f"embed failed: {text}")


class _HangingEmbedder:
    async def embed(self, text: str) -> list[float]:
        await asyncio.Event().wait()
        return []


class _StaticEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _KeywordOnlyStore:
    def __init__(self) -> None:
        self.vector_search_called = False

    def vector_search(self, *_args: object, **_kwargs: object) -> list[_MemoryHit]:
        self.vector_search_called = True
        return []

    def keyword_search_summary(
        self,
        terms: list[str],
        memory_types: list[str] | None = None,
        limit: int = 20,
        **_kwargs: object,
    ) -> list[_MemoryHit]:
        assert "支付" in terms
        assert memory_types is None
        assert limit == 30
        return [
            {
                "id": "mem:1",
                "memory_type": "event",
                "summary": "用户处理过支付相关问题",
                "source_ref": "tg:1:2",
                "happened_at": "2026-01-01T00:00:00+00:00",
                "keyword_score": 1.0,
            }
        ]


class _TimelineStore:
    def __init__(self) -> None:
        self.vector_search_called = False
        self.keyword_search_called = False
        self.time_start: datetime | None = None
        self.time_end: datetime | None = None

    def vector_search(self, *_args: object, **_kwargs: object) -> list[_MemoryHit]:
        self.vector_search_called = True
        return []

    def keyword_search_summary(
        self, *_args: object, **_kwargs: object
    ) -> list[_MemoryHit]:
        self.keyword_search_called = True
        return []

    def list_events_by_time_range(
        self, time_start: datetime, time_end: datetime, limit: int = 200
    ) -> list[_MemoryHit]:
        self.time_start = time_start
        self.time_end = time_end
        assert limit == 80
        hits: list[_MemoryHit] = [
            {
                "id": "e1",
                "memory_type": "event",
                "summary": "[2026-04-25 09:00] 用户调试缓存",
                "source_ref": "tg:1",
                "happened_at": "2026-04-25T09:00:00",
            },
            {
                "id": "e2",
                "memory_type": "event",
                "summary": "[2026-04-25 11:00] 用户讨论 DeepSeek",
                "source_ref": "tg:2",
                "happened_at": "2026-04-25T11:00:00",
            },
        ]
        return hits[:limit]


class _TimedSemanticStore:
    def __init__(self) -> None:
        self.vector_kwargs: list[dict[str, object]] = []
        self.vector_batch_kwargs: list[dict[str, object]] = []
        self.vector_batch_vec_count = 0
        self.keyword_kwargs: list[dict[str, object]] = []

    def vector_search(
        self, *_args: object, **kwargs: object
    ) -> list[_MemoryHit]:
        self.vector_kwargs.append(kwargs)
        raise AssertionError("带 time_filter 的 semantic 模式应复用 batch 候选")

    def vector_search_batch(
        self, query_vecs: list[list[float]], **kwargs: object
    ) -> list[list[_MemoryHit]]:
        self.vector_batch_kwargs.append(kwargs)
        self.vector_batch_vec_count = len(query_vecs)
        return [
            [
                {
                    "id": "deepseek",
                    "memory_type": "event",
                    "summary": "[2026-04-25 03:19] 用户调试 DeepSeek 缓存命中率",
                    "source_ref": "tg:1",
                    "happened_at": "2026-04-25T03:19:00",
                    "score": 0.9,
                }
            ]
            for _query_vec in query_vecs
        ]

    def keyword_search_summary(
        self, _terms: list[str], **kwargs: object
    ) -> list[_MemoryHit]:
        self.keyword_kwargs.append(kwargs)
        return []

    def list_events_by_time_range(
        self, *_args: object, **_kwargs: object
    ) -> list[_MemoryHit]:
        raise AssertionError("semantic 模式不应直接走 grep 列表")


class _FusionStore:
    def __init__(
        self,
        vector_groups: list[list[_MemoryHit]],
        keyword_hits: list[_MemoryHit],
    ) -> None:
        self.vector_groups = vector_groups
        self.keyword_hits = keyword_hits
        self.vector_kwargs: list[dict[str, object]] = []
        self.keyword_kwargs: list[dict[str, object]] = []

    def vector_search_batch(
        self,
        _query_vecs: list[list[float]],
        **kwargs: object,
    ) -> list[list[_MemoryHit]]:
        self.vector_kwargs.append(kwargs)
        return self.vector_groups

    def keyword_search_summary(
        self,
        _terms: list[str],
        **kwargs: object,
    ) -> list[_MemoryHit]:
        self.keyword_kwargs.append(kwargs)
        return self.keyword_hits


def test_parse_time_filter_supports_presets_and_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tz = ZoneInfo("Asia/Shanghai")
    monkeypatch.setattr(
        recall_memory_module,
        "_now_local",
        lambda: datetime(2026, 4, 25, 15, 30, tzinfo=tz),
    )

    today = recall_memory_module._parse_time_filter("today")
    assert today is not None
    assert today[0] == datetime(2026, 4, 25, 0, 0, tzinfo=tz)
    assert today[1] == datetime(2026, 4, 26, 0, 0, tzinfo=tz)

    recent = recall_memory_module._parse_time_filter("recent_3d")
    assert recent is not None
    assert recent[0] == datetime(2026, 4, 22, 15, 30, tzinfo=tz)
    assert recent[1] == datetime(2026, 4, 25, 15, 30, tzinfo=tz)

    one_day = recall_memory_module._parse_time_filter("2026-04-20")
    assert one_day is not None
    assert one_day[0] == datetime(2026, 4, 20, 0, 0, tzinfo=tz)
    assert one_day[1] == datetime(2026, 4, 21, 0, 0, tzinfo=tz)

    date_range = recall_memory_module._parse_time_filter("2026-04-20~2026-04-25")
    assert date_range is not None
    assert date_range[0] == datetime(2026, 4, 20, 0, 0, tzinfo=tz)
    assert date_range[1] == datetime(2026, 4, 26, 0, 0, tzinfo=tz)


def test_store_time_range_filters_mixed_timezone_happened_at(tmp_path: Path) -> None:
    store = MemoryStore2(tmp_path / "memory2.db")
    tz = ZoneInfo("Asia/Shanghai")
    store.upsert_item(
        "event",
        "[2026-04-25 01:00] 本地凌晨事件",
        [1.0, 0.0],
        happened_at="2026-04-25T01:00:00",
    )
    store.upsert_item(
        "event",
        "[2026-04-25 01:30] UTC 存储的凌晨事件",
        [1.0, 0.0],
        happened_at="2026-04-24T17:30:00+00:00",
    )
    store.upsert_item(
        "event",
        "[2026-04-24 23:00] 前一天本地事件",
        [1.0, 0.0],
        happened_at="2026-04-24T23:00:00",
    )

    hits = store.list_events_by_time_range(
        datetime(2026, 4, 25, 0, 0, tzinfo=tz),
        datetime(2026, 4, 26, 0, 0, tzinfo=tz),
        limit=20,
    )

    summaries = [str(item["summary"]) for item in hits]
    assert summaries == [
        "[2026-04-25 01:00] 本地凌晨事件",
        "[2026-04-25 01:30] UTC 存储的凌晨事件",
    ]


def test_store_time_range_limit_keeps_latest_events_in_chronological_order(
    tmp_path: Path,
) -> None:
    store = MemoryStore2(tmp_path / "memory2.db")
    tz = ZoneInfo("Asia/Shanghai")
    for hour in (9, 10, 11):
        store.upsert_item(
            "event",
            f"[2026-04-25 {hour:02d}:00] 用户处理第 {hour} 点事件",
            [1.0, 0.0],
            happened_at=f"2026-04-25T{hour:02d}:00:00",
        )

    hits = store.list_events_by_time_range(
        datetime(2026, 4, 25, 0, 0, tzinfo=tz),
        datetime(2026, 4, 26, 0, 0, tzinfo=tz),
        limit=2,
    )

    assert [item["summary"] for item in hits] == [
        "[2026-04-25 10:00] 用户处理第 10 点事件",
        "[2026-04-25 11:00] 用户处理第 11 点事件",
    ]


def test_store_semantic_searches_respect_time_range(tmp_path: Path) -> None:
    store = MemoryStore2(tmp_path / "memory2.db")
    tz = ZoneInfo("Asia/Shanghai")
    store.upsert_item(
        "event",
        "[2026-04-25 01:30] DeepSeek 今日事件",
        [1.0, 0.0],
        happened_at="2026-04-24T17:30:00+00:00",
    )
    store.upsert_item(
        "event",
        "[2026-04-24 23:00] DeepSeek 昨日事件",
        [1.0, 0.0],
        happened_at="2026-04-24T23:00:00",
    )

    start = datetime(2026, 4, 25, 0, 0, tzinfo=tz)
    end = datetime(2026, 4, 26, 0, 0, tzinfo=tz)
    vector_hits = store.vector_search(
        [1.0, 0.0],
        top_k=5,
        memory_types=["event"],
        score_threshold=0.0,
        time_start=start,
        time_end=end,
    )
    keyword_hits = store.keyword_search_summary(
        ["DeepSeek"],
        memory_types=["event"],
        limit=5,
        time_start=start,
        time_end=end,
    )

    assert [item["summary"] for item in vector_hits] == [
        "[2026-04-25 01:30] DeepSeek 今日事件"
    ]
    assert [item["summary"] for item in keyword_hits] == [
        "[2026-04-25 01:30] DeepSeek 今日事件"
    ]


def test_store_keyword_search_respects_required_scope(tmp_path: Path) -> None:
    store = MemoryStore2(tmp_path / "memory2.db")
    store.upsert_item(
        "event",
        "用户在当前会话讨论支付问题",
        None,
        extra={"scope_channel": "telegram", "scope_chat_id": "chat-a"},
    )
    store.upsert_item(
        "event",
        "用户在其他会话讨论支付问题",
        None,
        extra={"scope_channel": "telegram", "scope_chat_id": "chat-b"},
    )

    hits = store.keyword_search_summary(
        ["支付"],
        limit=10,
        scope_channel="telegram",
        scope_chat_id="chat-a",
        require_scope_match=True,
    )

    assert [item["summary"] for item in hits] == ["用户在当前会话讨论支付问题"]


@pytest.mark.asyncio
async def test_retriever_returns_keyword_hits_when_vector_empty() -> None:
    store = _FusionStore(
        vector_groups=[[]],
        keyword_hits=[
            {
                "id": "kw1",
                "memory_type": "event",
                "summary": "用户处理过支付问题",
                "keyword_score": 1.0,
            }
        ],
    )
    retriever = Retriever(cast(MemoryStore2, store), cast(Embedder, _StaticEmbedder()))

    hits = await retriever.retrieve("支付", top_k=5)

    assert [item["id"] for item in hits] == ["kw1"]
    assert hits[0]["score"] == 1.0


@pytest.mark.asyncio
async def test_retriever_keeps_strong_vector_order_when_keyword_hits_are_low_rank() -> None:
    vector_hits: list[_MemoryHit] = [
        {
            "id": "vec1",
            "memory_type": "event",
            "summary": "高质量向量命中 1",
            "score": 0.95,
        },
        {
            "id": "vec2",
            "memory_type": "event",
            "summary": "高质量向量命中 2",
            "score": 0.9,
        },
    ]
    keyword_hits: list[_MemoryHit] = [
        {
            "id": f"kw{i}",
            "memory_type": "event",
            "summary": f"低排名关键词命中 {i}",
            "keyword_score": 1.0,
        }
        for i in range(12)
    ]
    store = _FusionStore(vector_groups=[vector_hits], keyword_hits=keyword_hits)
    retriever = Retriever(cast(MemoryStore2, store), cast(Embedder, _StaticEmbedder()))

    hits = await retriever.retrieve("支付", top_k=2)

    assert [item["id"] for item in hits] == ["vec1", "vec2"]


def test_store_vector_batch_reuses_time_filtered_embedding_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore2(tmp_path / "memory2.db")
    tz = ZoneInfo("Asia/Shanghai")
    store.upsert_item(
        "event",
        "[2026-04-25 09:00] DeepSeek 今日事件",
        [1.0, 0.0],
        happened_at="2026-04-25T09:00:00",
    )
    store.upsert_item(
        "event",
        "[2026-04-25 10:00] 重构今日事件",
        [0.0, 1.0],
        happened_at="2026-04-25T10:00:00",
    )
    calls = 0
    original = store._get_embedding_rows_by_time_filter

    def counted_get_embedding_rows_by_time_filter(
        *,
        memory_types: list[str] | None,
        include_superseded: bool,
        scope_channel: str | None,
        scope_chat_id: str | None,
        require_scope_match: bool,
        time_start: datetime | None,
        time_end: datetime | None,
    ) -> list[_EmbeddingRow]:
        nonlocal calls
        calls += 1
        return original(
            memory_types=memory_types,
            include_superseded=include_superseded,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=require_scope_match,
            time_start=time_start,
            time_end=time_end,
        )

    monkeypatch.setattr(
        store,
        "_get_embedding_rows_by_time_filter",
        counted_get_embedding_rows_by_time_filter,
    )

    results = store.vector_search_batch(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        top_k=2,
        score_threshold=0.0,
        time_start=datetime(2026, 4, 25, 0, 0, tzinfo=tz),
        time_end=datetime(2026, 4, 26, 0, 0, tzinfo=tz),
    )

    assert calls == 1
    assert len(results) == 3
    assert results[0][0]["summary"] == "[2026-04-25 09:00] DeepSeek 今日事件"
    assert results[1][0]["summary"] == "[2026-04-25 10:00] 重构今日事件"


def test_store_keyword_time_filter_prefilters_before_candidate_limit(
    tmp_path: Path,
) -> None:
    store = MemoryStore2(tmp_path / "memory2.db")
    tz = ZoneInfo("Asia/Shanghai")
    for index in range(1005):
        store.upsert_item(
            "event",
            f"[2026-04-24 10:00] DeepSeek margin 内旧事件 {index}",
            None,
            happened_at="2026-04-24T10:00:00",
        )
    store._db.execute(
        "UPDATE memory_items SET reinforcement=20 WHERE happened_at=?",
        ("2026-04-24T10:00:00",),
    )
    store._db.commit()
    store.upsert_item(
        "event",
        "[2026-04-25 09:00] DeepSeek 今日事件",
        None,
        happened_at="2026-04-25T09:00:00",
    )

    hits = store.keyword_search_summary(
        ["DeepSeek"],
        limit=5,
        time_start=datetime(2026, 4, 25, 0, 0, tzinfo=tz),
        time_end=datetime(2026, 4, 26, 0, 0, tzinfo=tz),
    )

    assert [item["summary"] for item in hits] == [
        "[2026-04-25 09:00] DeepSeek 今日事件"
    ]


@pytest.mark.asyncio
async def test_recall_memory_grep_mode_lists_events_without_embedding() -> None:
    store = _TimelineStore()
    provider = _FakeProvider()
    tool = _recall_tool(store, _FailingEmbedder(), provider)

    payload = json.loads(
        await tool.execute(
            query="今天我都做了什么",
            search_mode="grep",
            time_filter="2026-04-25",
            limit=80,
        )
    )

    assert payload["count"] == 2
    assert [item["id"] for item in payload["items"]] == ["e1", "e2"]
    assert store.vector_search_called is False
    assert store.keyword_search_called is False
    provider.chat.assert_not_called()
    assert store.time_start is not None
    assert store.time_end is not None


@pytest.mark.asyncio
async def test_recall_memory_semantic_mode_passes_time_range_to_searches() -> None:
    store = _TimedSemanticStore()
    provider = _FakeProvider("用户讨论 DeepSeek 缓存")
    tool = _recall_tool(store, _StaticEmbedder(), provider)

    payload = json.loads(
        await tool.execute(
            query="DeepSeek 缓存命中率",
            search_mode="semantic",
            time_filter="2026-04-25",
        )
    )

    assert payload["items"][0]["id"] == "deepseek"
    assert store.vector_kwargs == []
    assert store.vector_batch_kwargs
    assert store.vector_batch_vec_count == 2
    assert store.keyword_kwargs
    assert store.vector_batch_kwargs[0]["memory_types"] is None
    assert store.vector_batch_kwargs[0]["time_start"] is not None
    assert store.vector_batch_kwargs[0]["time_end"] is not None
    assert store.keyword_kwargs[0]["memory_types"] is None
    assert store.keyword_kwargs[0]["time_start"] is not None
    assert store.keyword_kwargs[0]["time_end"] is not None


@pytest.mark.asyncio
async def test_recall_memory_falls_back_to_keyword_when_query_embed_fails() -> None:
    store = _KeywordOnlyStore()
    provider = _FakeProvider("用户处理过支付相关问题")
    tool = _recall_tool(store, _FailingEmbedder(), provider)

    payload = json.loads(await tool.execute(query="phase 支付"))

    assert payload["count"] == 1
    assert payload["items"][0]["id"] == "mem:1"
    assert payload["items"][0]["source_ref"] == "tg:1:2"
    assert payload["citation_required"] is True
    assert payload["citation_format"] == "§cited:[id1,id2,...]§"
    assert payload["cited_item_ids"] == ["mem:1"]
    assert "§cited:[" in payload["citation_rule"]
    assert store.vector_search_called is False


@pytest.mark.asyncio
async def test_recall_memory_falls_back_to_keyword_when_query_embed_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retriever_module, "_EMBED_TIMEOUT_S", 0.01)
    store = _KeywordOnlyStore()
    provider = _FakeProvider("用户处理过支付相关问题")
    tool = _recall_tool(store, _HangingEmbedder(), provider)

    payload = json.loads(
        await asyncio.wait_for(tool.execute(query="phase 支付"), timeout=0.5)
    )

    assert payload["count"] == 1
    assert payload["items"][0]["id"] == "mem:1"
    assert store.vector_search_called is False


def test_recall_memory_description_emphasizes_mandatory_citation() -> None:
    assert "只要最终回复使用了本工具返回的任何记忆条目" in RecallMemoryTool.description
    assert "cited_item_ids / citation_required / citation_format" in RecallMemoryTool.description
