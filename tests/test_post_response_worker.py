import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from core.memory.events import MemoryWritten, TurnIngested
from memory2.memorizer import Memorizer
from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.rule_schema import (
    build_procedure_rule_schema,
    resolve_procedure_rule_schema,
)
from memory2.store import MemoryHit, MemoryStore2


class _DummyProvider:
    def __init__(self):
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        raise AssertionError("provider.chat should not be called in this test")


class _DummyRetriever:
    def __init__(self, results):
        self._results = results
        self.calls = []

    async def retrieve(self, query: str, memory_types=None):
        self.calls.append((query, tuple(memory_types or [])))
        return list(self._results)


class _DummyMemorizer:
    def __init__(self, store=None):
        from unittest.mock import AsyncMock, MagicMock
        self.save_item = AsyncMock(return_value="new:testid")
        self.supersede_batch = MagicMock()
        self.merge_item = AsyncMock()
        self._store = store


class _InterleavingProvider:
    def __init__(self) -> None:
        self.a_started = asyncio.Event()
        self.b_started = asyncio.Event()
        self.b_release = asyncio.Event()

    async def chat(self, **kwargs: Any) -> SimpleNamespace:
        prompt = kwargs["messages"][0]["content"]
        if "session-a" in prompt:
            self.a_started.set()
            await self.b_started.wait()
            return SimpleNamespace(content='["topic-a"]')
        if "session-b" in prompt:
            self.b_started.set()
            await self.b_release.wait()
            return SimpleNamespace(content='["topic-b"]')
        if "topic-a" in prompt:
            return SimpleNamespace(content='["id-a"]')
        if "topic-b" in prompt:
            return SimpleNamespace(content='["id-b"]')
        raise AssertionError(f"unexpected prompt: {prompt}")


class _InterleavingRetriever:
    async def retrieve(self, query: str, memory_types: Any = None) -> list[dict[str, Any]]:
        return [{"id": f"id-{query[-1]}", "score": 0.9, "summary": query}]


class _RecordingPublisher:
    def __init__(self, trace: list[tuple[str, str]]) -> None:
        self.events: list[MemoryWritten] = []
        self.trace = trace
        self.first_event = asyncio.Event()

    async def fanout(self, event: object) -> None:
        assert isinstance(event, MemoryWritten)
        self.events.append(event)
        self.trace.append(("event", event.session_key))
        self.first_event.set()


class _RecordingSuperseder:
    def __init__(self, trace: list[tuple[str, str]]) -> None:
        self.trace = trace

    def supersede_batch(self, item_ids: list[str]) -> None:
        self.trace.append(("supersede", item_ids[0]))


class _StaticEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return list(self._mapping.get(text, [0.0, 0.0]))


def test_post_worker_run_only_handles_invalidations_no_implicit_save():
    """per-turn run() 只做 invalidation 处理，不再做隐式 procedure/preference/profile 提取。
    隐式提取已移至 consolidation 窗口期（与 event 提取并行，用主模型处理）。
    """
    from unittest.mock import AsyncMock, MagicMock
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever([])
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    worker._handle_invalidations = AsyncMock(
        side_effect=lambda *args, **kwargs: args[-1] if args else 0
    )

    asyncio.run(
        worker.run(
            user_msg="你以后多问我一句",
            agent_response="好的",
            tool_chain=[],
            source_ref="test@post_response",
        )
    )

    # run() 不再写入任何隐式记忆
    memorizer.save_item.assert_not_called()
    # 但 invalidation 检查仍然运行
    worker._handle_invalidations.assert_awaited_once()


def test_post_worker_handle_delegates_turn_ingested_event():
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    worker.run = AsyncMock()

    asyncio.run(
        worker.handle(
            TurnIngested(
                session_key="cli:1",
                channel="cli",
                chat_id="1",
                user_message="以后用中文",
                assistant_response="好的",
                tool_chain=[{"text": "memo", "calls": []}],
                source_ref="cli:1@post_response",
            )
        )
    )

    worker.run.assert_awaited_once_with(
        user_msg="以后用中文",
        agent_response="好的",
        tool_chain=[{"text": "memo", "calls": []}],
        source_ref="cli:1@post_response",
        session_key="cli:1",
        channel="cli",
        chat_id="1",
    )


@pytest.mark.asyncio
async def test_post_worker_keeps_memory_written_scope_per_interleaved_run():
    trace: list[tuple[str, str]] = []
    provider = _InterleavingProvider()
    publisher = _RecordingPublisher(trace)
    memorizer = _RecordingSuperseder(trace)
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, _InterleavingRetriever()),
        light_provider=cast(Any, provider),
        light_model="test",
        event_publisher=cast(Any, publisher),
    )

    task_a = asyncio.create_task(
        worker.run(
            user_msg="session-a 旧流程错了",
            agent_response="收到",
            tool_chain=[],
            source_ref="source-a",
            session_key="session-a",
            channel="telegram",
            chat_id="chat-a",
        )
    )
    task_b = asyncio.create_task(
        worker.run(
            user_msg="session-b 旧流程错了",
            agent_response="收到",
            tool_chain=[],
            source_ref="source-b",
            session_key="session-b",
            channel="cli",
            chat_id="chat-b",
        )
    )

    await provider.b_started.wait()
    await publisher.first_event.wait()
    assert publisher.events[0] == MemoryWritten(
        session_key="session-a",
        channel="telegram",
        chat_id="chat-a",
        source_ref="source-a",
        action="supersede",
        superseded_ids=["id-a"],
    )
    assert trace[:2] == [("supersede", "id-a"), ("event", "session-a")]

    provider.b_release.set()
    await asyncio.gather(task_a, task_b)

    assert publisher.events[1] == MemoryWritten(
        session_key="session-b",
        channel="cli",
        chat_id="chat-b",
        source_ref="source-b",
        action="supersede",
        superseded_ids=["id-b"],
    )
    assert trace == [
        ("supersede", "id-a"),
        ("event", "session-a"),
        ("supersede", "id-b"),
        ("event", "session-b"),
    ]


@pytest.mark.asyncio
async def test_post_worker_exposes_unexpected_storage_failure():
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    worker._handle_invalidations = AsyncMock(
        side_effect=RuntimeError("memory database unavailable")
    )

    with pytest.raises(RuntimeError, match="memory database unavailable"):
        await worker.run(
            user_msg="旧流程不对",
            agent_response="收到",
            tool_chain=[],
            source_ref="test@post_response",
        )


def test_build_procedure_rule_schema_prefers_explicit_rule_schema():
    schema = build_procedure_rule_schema(
        "查 Steam 信息时不要直接用 web_search，必须先使用 steam MCP。",
        tool_requirement="steam_mcp",
        rule_schema={
            "required_tools": ["steam_mcp"],
            "forbidden_tools": ["web_search"],
            "mentioned_tools": ["steam", "web_search"],
        },
    )

    assert "web_search" in schema["forbidden_tools"]
    assert schema["required_tools"] == ["steam_mcp"]
    assert "steam" in schema["mentioned_tools"]


def test_build_procedure_rule_schema_fills_missing_slot_from_summary():
    schema = build_procedure_rule_schema(
        "查 Steam 信息时必须先使用 steam MCP，不能直接使用 web_search。",
        rule_schema={"required_tools": ["steam_mcp"]},
    )

    assert schema["required_tools"] == ["steam_mcp"]
    assert schema["forbidden_tools"] == ["web_search"]


def test_build_procedure_rule_schema_infers_constraints_without_explicit_schema():
    schema = build_procedure_rule_schema(
        "查 Steam 信息时不要直接用 web_search，必须先使用 steam MCP。"
    )

    assert "steam_mcp" in schema["required_tools"]
    assert "web_search" in schema["forbidden_tools"]
    assert "steam" in schema["mentioned_tools"]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"steps": "不是列表"}, "steps"),
        ({"steps": ["正常步骤", 1]}, r"steps\[1\]"),
        ({"tool_requirement": 1}, "tool_requirement"),
        ({"rule_schema": {"required_tools": "steam_mcp"}}, "required_tools"),
    ],
)
def test_resolve_procedure_rule_schema_rejects_invalid_metadata(
    extra: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        resolve_procedure_rule_schema("规则摘要", extra)


def test_merge_item_rejects_corrupt_persisted_procedure_steps() -> None:
    embedder = _StaticEmbedder(
        {
            "旧规则": [1.0, 0.0],
            "合并规则": [0.9, 0.1],
        }
    )
    store = MemoryStore2(":memory:")
    memorizer = Memorizer(store, cast(Any, embedder))
    row_ref = store.upsert_item(
        memory_type="procedure",
        summary="旧规则",
        embedding=[1.0, 0.0],
        extra={"steps": "损坏的步骤"},
    )

    with pytest.raises(TypeError, match="steps"):
        asyncio.run(memorizer.merge_item(row_ref.split(":", 1)[1], "合并规则"))


def test_merge_item_rolls_back_content_hash_conflict() -> None:
    store = MemoryStore2(":memory:")
    first_ref = store.upsert_item("procedure", "规则 A", [1.0, 0.0], extra={})
    _ = store.upsert_item("procedure", "规则 B", [0.0, 1.0], extra={})
    memorizer = Memorizer(
        store,
        cast(Any, _StaticEmbedder({"规则 B": [0.0, 1.0]})),
    )

    with pytest.raises(RuntimeError, match="content_hash 冲突"):
        asyncio.run(
            memorizer.merge_item(first_ref.split(":", 1)[1], "规则 B")
        )

    summaries = {item["summary"] for item in store.list_by_type("procedure")}
    assert summaries == {"规则 A", "规则 B"}


def test_collect_protected_memory_ids_accepts_long_mixed_id():
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    tool_chain = [
        {
            "calls": [
                {
                    "name": "memorize",
                    "arguments": {"summary": "规则A"},
                    "result": "已记住（new:AbCDef12_34567890）：规则A",
                }
            ]
        }
    ]
    protected = worker._collect_protected_memory_ids(tool_chain)
    assert "AbCDef12_34567890" in protected


def test_collect_protected_memory_ids_accepts_item_id_format():
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    tool_chain = [
        {
            "calls": [
                {
                    "name": "memorize",
                    "arguments": {"summary": "规则B"},
                    "result": "已记住（item_id=memu_12345）：规则B",
                }
            ]
        }
    ]
    protected = worker._collect_protected_memory_ids(tool_chain)
    assert "memu_12345" in protected


@pytest.mark.parametrize("calls", [object(), [object()]])
def test_collect_protected_memory_ids_rejects_invalid_call_shape(calls: object):
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )

    with pytest.raises(TypeError, match=r"tool_chain\[\]\.calls"):
        worker._collect_protected_memory_ids([{"calls": calls}])


def test_extract_invalidation_topics_skips_when_token_budget_exhausted():
    provider = _DummyProvider()
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, provider),
        light_model="test",
    )
    topics, remain = asyncio.run(
        worker._extract_invalidation_topics("也许这个流程不对", token_budget=0)
    )
    assert topics == []
    assert remain == 0
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_extract_invalidation_topics_exposes_provider_failure():
    provider = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("provider down")))
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, provider),
        light_model="test",
    )

    with pytest.raises(RuntimeError, match="provider down"):
        await worker._extract_invalidation_topics("旧流程错了", token_budget=100)


@pytest.mark.asyncio
async def test_extract_invalidation_topics_exposes_invalid_json_schema():
    provider = SimpleNamespace(
        chat=AsyncMock(return_value=SimpleNamespace(content='{"topic": "流程"}'))
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, provider),
        light_model="test",
    )

    with pytest.raises(ValueError, match="JSON 数组"):
        await worker._extract_invalidation_topics("旧流程错了", token_budget=100)


@pytest.mark.asyncio
async def test_check_invalidate_exposes_unknown_candidate_id():
    provider = SimpleNamespace(
        chat=AsyncMock(return_value=SimpleNamespace(content='["unknown"]'))
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, provider),
        light_model="test",
    )

    with pytest.raises(ValueError, match="未知候选 ID"):
        await worker._check_invalidate(
            "流程",
            [
                {
                    "id": "known",
                    "memory_type": "procedure",
                    "summary": "旧流程",
                    "source_ref": "turn:old",
                    "happened_at": "2025-01-01T00:00:00+00:00",
                    "score": 0.9,
                }
            ],
            token_budget=100,
        )


def test_merge_item_should_keep_procedure_metadata_consistent():
    embedder = _StaticEmbedder(
        {
            "查 Steam 必须先用 steam_mcp，不能直接使用 web_search": [1.0, 0.0],
            "合并后的 Steam 查询规则：先用 steam_mcp，再补充区服确认": [0.9, 0.1],
        }
    )
    store = MemoryStore2(":memory:")
    memorizer = Memorizer(store, cast(Any, embedder))

    row_ref = store.upsert_item(
        memory_type="procedure",
        summary="查 Steam 必须先用 steam_mcp，不能直接使用 web_search",
        embedding=[1.0, 0.0],
        extra={
            "tool_requirement": "steam_mcp",
            "steps": [],
            "rule_schema": {
                "required_tools": ["steam_mcp"],
                "forbidden_tools": ["web_search"],
                "mentioned_tools": ["steam_mcp", "web_search"],
            },
        },
    )
    item_id = row_ref.split(":", 1)[1]

    asyncio.run(
        memorizer.merge_item(
            item_id,
            "合并后的 Steam 查询规则：先用 steam_mcp，再补充区服确认",
        )
    )

    row = store._db.execute(
        "SELECT summary, extra_json FROM memory_items WHERE id=?",
        (item_id,),
    ).fetchone()
    assert row is not None
    summary, extra_json = row
    assert "补充区服确认" in summary
    assert extra_json is not None

    import json

    extra = json.loads(extra_json)
    assert extra["tool_requirement"] == "steam_mcp"
    assert "区服确认" in str(extra), "merge 后的 extra_json 应与新摘要保持一致"


def test_merge_item_should_refresh_trigger_tags_for_procedure():
    embedder = _StaticEmbedder(
        {
            "查 Steam 必须直接使用 web_search": [1.0, 0.0],
            "查 Steam 必须先使用 steam_mcp": [0.9, 0.1],
        }
    )
    store = MemoryStore2(":memory:")
    memorizer = Memorizer(store, cast(Any, embedder))

    row_ref = store.upsert_item(
        memory_type="procedure",
        summary="查 Steam 必须直接使用 web_search",
        embedding=[1.0, 0.0],
        extra={
            "tool_requirement": "web_search",
            "steps": [],
            "rule_schema": {
                "required_tools": ["web_search"],
                "forbidden_tools": [],
                "mentioned_tools": ["web_search"],
            },
            "trigger_tags": {
                "tools": ["web_search"],
                "skills": [],
                "keywords": ["web_search"],
                "scope": "tool_triggered",
            },
        },
    )
    item_id = row_ref.split(":", 1)[1]

    asyncio.run(
        memorizer.merge_item(
            item_id,
            "查 Steam 必须先使用 steam_mcp",
        )
    )

    row = store._db.execute(
        "SELECT extra_json FROM memory_items WHERE id=?",
        (item_id,),
    ).fetchone()
    assert row is not None and row[0] is not None

    import json

    extra = json.loads(row[0])
    tags = extra.get("trigger_tags") or {}
    assert "web_search" not in (tags.get("keywords") or []), "merge 后不应保留旧关键词"
