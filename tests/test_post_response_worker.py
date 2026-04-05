import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory2.dedup_decider import DedupDecision, DedupResult, ExistingAction, MemoryAction
from memory2.memorizer import Memorizer
from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.rule_schema import build_procedure_rule_schema
from memory2.store import MemoryStore2


class _DummyProvider:
    def __init__(self):
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        raise AssertionError("provider.chat should not be called in this test")


class _PromptCaptureProvider:
    def __init__(self, response_text: str = "[]"):
        self.calls = 0
        self.last_prompt = ""
        self.prompts: list[str] = []
        self._response_text = response_text

    async def chat(self, **kwargs):
        self.calls += 1
        messages = kwargs.get("messages") or []
        if messages:
            self.last_prompt = str(messages[-1].get("content", ""))
            self.prompts.append(self.last_prompt)

        class _Resp:
            def __init__(self, content: str):
                self.content = content

        return _Resp(self._response_text)


class _DummyRetriever:
    def __init__(self, results):
        self._results = results
        self.calls = []

    async def retrieve(self, query: str, memory_types=None):
        self.calls.append((query, tuple(memory_types or [])))
        return list(self._results)


class _DummyMemorizer:
    def __init__(self, store=None):
        self.save_item = AsyncMock(return_value="new:testid")
        self.supersede_batch = MagicMock()
        self.merge_item = AsyncMock()
        self._store = store


class _FixedDedupDecider:
    def __init__(self, result: DedupResult):
        self._result = result

    async def decide(self, candidate: dict, *, batch_vecs=None) -> DedupResult:
        return self._result


class _StaticEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return list(self._mapping.get(text, [0.0, 0.0]))


def test_post_worker_skips_implicit_when_semantic_dup_to_explicit():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever(
        [
            {
                "id": "2b1ba4e802bf",
                "memory_type": "procedure",
                "score": 0.93,
                "summary": "记忆冲突时必须实时验证",
            }
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )

    worker._collect_explicit_memorized = lambda tool_chain: (
        ["记忆冲突时必须实时验证，禁止推测，必须外部工具验证"],
        {"2b1ba4e802bf"},
    )
    worker._handle_invalidations = AsyncMock(
        side_effect=lambda *args, **kwargs: args[-1] if args else 0
    )
    worker._extract_implicit = AsyncMock(
        return_value=(
            [
                {
                    "summary": "在记忆冲突情况下应优先外部工具验证，不依赖内部推测",
                    "memory_type": "procedure",
                    "tool_requirement": None,
                    "steps": [],
                }
            ],
            256,
        )
    )

    asyncio.run(
        worker.run(
            user_msg="以后遇到冲突不要猜",
            agent_response="已记住",
            tool_chain=[],
            source_ref="test@post_response",
        )
    )

    memorizer.save_item.assert_not_called()


def test_post_worker_saves_implicit_when_not_duplicate_to_explicit():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever([])
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )

    worker._collect_explicit_memorized = lambda tool_chain: (
        ["记得后续查 Steam 要走 MCP"],
        {"abcdef123456"},
    )
    worker._handle_invalidations = AsyncMock(
        side_effect=lambda *args, **kwargs: args[-1] if args else 0
    )
    worker._extract_implicit = AsyncMock(
        return_value=(
            [
                {
                    "summary": "回复结尾要主动追问用户最关心的点",
                    "memory_type": "preference",
                    "tool_requirement": None,
                    "steps": [],
                }
            ],
            256,
        )
    )

    asyncio.run(
        worker.run(
            user_msg="你以后多问我一句",
            agent_response="好的",
            tool_chain=[],
            source_ref="test@post_response",
        )
    )

    memorizer.save_item.assert_called_once()


def test_post_worker_deterministic_supersede_on_explicit_rule_conflict():
    store = MagicMock()
    store.get_items_by_ids.side_effect = lambda ids: [
        {
            "id": item_id,
            "memory_type": "procedure",
            "summary": "查 Steam 信息时必须直接使用 web_search，不能先用 steam MCP。",
            "extra_json": {"tool_requirement": "web_search"},
            "source_ref": "old@source",
            "happened_at": None,
        }
        if item_id == "old-rule-1"
        else {
            "id": "testid",
            "memory_type": "procedure",
            "summary": "用户明确纠正 agent 的操作流程：查 Steam 信息时必须先使用 steam MCP，不能直接使用 web_search",
            "extra_json": {"tool_requirement": "steam_mcp"},
            "source_ref": "test@post_response",
            "happened_at": None,
        }
        for item_id in ids
    ]
    memorizer = _DummyMemorizer(store=store)
    retriever = _DummyRetriever(
        [
            {
                "id": "old-rule-1",
                "memory_type": "procedure",
                "score": 0.91,
                "summary": "查 Steam 信息时必须直接使用 web_search，不能先用 steam MCP。",
                "extra_json": {
                    "tool_requirement": "web_search",
                    "rule_schema": {
                        "required_tools": ["web_search"],
                        "forbidden_tools": ["steam_mcp"],
                        "mentioned_tools": ["steam", "web_search", "steam_mcp"],
                    },
                },
            }
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )

    asyncio.run(
        worker._save_with_supersede(
            {
                "summary": "用户明确纠正 agent 的操作流程：查 Steam 信息时必须先使用 steam MCP，不能直接使用 web_search",
                "memory_type": "procedure",
                "tool_requirement": "steam_mcp",
                "steps": [],
                "rule_schema": {
                    "required_tools": ["steam_mcp"],
                    "forbidden_tools": ["web_search"],
                    "mentioned_tools": ["steam", "steam_mcp", "web_search"],
                },
            },
            "test@post_response",
            token_budget=256,
        )
    )

    memorizer.supersede_batch.assert_called_once_with(["old-rule-1"])
    memorizer.save_item.assert_called_once()
    store.record_replacements.assert_called_once()
    record_call = store.record_replacements.call_args.kwargs
    assert record_call["relation_type"] == "supersede"
    assert record_call["old_items"][0]["id"] == "old-rule-1"
    assert record_call["new_item"]["id"] == "testid"


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


def test_post_worker_merges_deterministic_and_llm_supersede_candidates():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever(
        [
            {
                "id": "old-rule-1",
                "memory_type": "procedure",
                "score": 0.91,
                "summary": "查 Steam 信息时必须直接使用 web_search，不能先用 steam MCP。",
                "extra_json": {
                    "tool_requirement": "web_search",
                    "rule_schema": {
                        "required_tools": ["web_search"],
                        "forbidden_tools": ["steam_mcp"],
                        "mentioned_tools": ["steam", "web_search", "steam_mcp"],
                    },
                },
            },
            {
                "id": "old-rule-2",
                "memory_type": "procedure",
                "score": 0.9,
                "summary": "这是 Steam 查询的旧版流程，需要按旧版执行。",
                "extra_json": {},
            },
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    worker._check_supersede = AsyncMock(return_value=(["old-rule-2"], 128))

    asyncio.run(
        worker._save_with_supersede(
            {
                "summary": "用户明确纠正 agent 的操作流程：查 Steam 信息时必须先使用 steam MCP，不能直接使用 web_search",
                "memory_type": "procedure",
                "tool_requirement": "steam_mcp",
                "steps": [],
                "rule_schema": {
                    "required_tools": ["steam_mcp"],
                    "forbidden_tools": ["web_search"],
                    "mentioned_tools": ["steam", "steam_mcp", "web_search"],
                },
            },
            "test@post_response",
            token_budget=256,
        )
    )

    worker._check_supersede.assert_awaited_once()
    memorizer.supersede_batch.assert_called_once_with(["old-rule-1", "old-rule-2"])
    memorizer.save_item.assert_called_once()


def test_post_worker_deterministic_supersede_without_explicit_rule_schema():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever(
        [
            {
                "id": "old-rule-1",
                "memory_type": "procedure",
                "score": 0.91,
                "summary": "查 Steam 信息时必须直接使用 web_search，不能先用 steam MCP。",
                "extra_json": {},
            }
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )

    asyncio.run(
        worker._save_with_supersede(
            {
                "summary": "用户明确纠正 agent 的操作流程：查 Steam 信息时必须先使用 steam MCP，不能直接使用 web_search",
                "memory_type": "procedure",
                "tool_requirement": None,
                "steps": [],
            },
            "test@post_response",
            token_budget=256,
        )
    )

    memorizer.supersede_batch.assert_called_once_with(["old-rule-1"])
    memorizer.save_item.assert_called_once()


def test_post_worker_deterministic_supersede_with_partial_rule_schema():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever(
        [
            {
                "id": "old-rule-1",
                "memory_type": "procedure",
                "score": 0.91,
                "summary": "查 Steam 信息时必须直接使用 web_search。",
                "extra_json": {"rule_schema": {"required_tools": ["web_search"]}},
            }
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )

    asyncio.run(
        worker._save_with_supersede(
            {
                "summary": "查 Steam 信息时必须先使用 steam MCP，不能直接使用 web_search。",
                "memory_type": "procedure",
                "tool_requirement": None,
                "steps": [],
                "rule_schema": {"required_tools": ["steam_mcp"]},
            },
            "test@post_response",
            token_budget=256,
        )
    )

    memorizer.supersede_batch.assert_called_once_with(["old-rule-1"])
    memorizer.save_item.assert_called_once()


def test_collect_explicit_memorized_accepts_long_mixed_id():
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
    summaries, protected = worker._collect_explicit_memorized(tool_chain)
    assert summaries == ["规则A"]
    assert "AbCDef12_34567890" in protected


def test_collect_explicit_memorized_accepts_item_id_format():
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
    summaries, protected = worker._collect_explicit_memorized(tool_chain)
    assert summaries == ["规则B"]
    assert "memu_12345" in protected


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


def test_dedup_none_should_not_delete_when_merge_cannot_complete():
    memorizer = _DummyMemorizer()
    dedup_result = DedupResult(
        decision=DedupDecision.NONE,
        candidate_summary="新的 Steam 查询规则，补充了必须先确认区服",
        candidate_type="procedure",
        similar_items=[
            {"id": "old-merge-target", "summary": "Steam 查询规则：先用 steam_mcp"},
            {"id": "old-delete-target", "summary": "Steam 查询废弃旧流程"},
        ],
        actions=[
            ExistingAction(
                item_id="old-merge-target",
                summary="Steam 查询规则：先用 steam_mcp",
                action=MemoryAction.MERGE,
                reason="same topic partial update",
            ),
            ExistingAction(
                item_id="old-delete-target",
                summary="Steam 查询废弃旧流程",
                action=MemoryAction.DELETE,
                reason="obsolete",
            ),
        ],
        reason="merge one + delete one",
        query_vector=[1.0, 0.0],
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
        dedup_decider=cast(Any, _FixedDedupDecider(dedup_result)),
    )

    asyncio.run(
        worker._save_with_dedup(
            {
                "summary": "新的 Steam 查询规则，补充了必须先确认区服",
                "memory_type": "procedure",
                "tool_requirement": "steam_mcp",
                "steps": [],
            },
            "test@post_response",
            protected_ids=None,
            token_budget=0,  # 强制 merge 摘要生成失败
            batch_vecs=[],
        )
    )

    memorizer.merge_item.assert_not_called()
    memorizer.supersede_batch.assert_not_called()


def test_dedup_none_with_delete_only_should_still_delete():
    memorizer = _DummyMemorizer()
    dedup_result = DedupResult(
        decision=DedupDecision.NONE,
        candidate_summary="用户明确废弃旧的 Steam 查询流程",
        candidate_type="procedure",
        similar_items=[
            {"id": "old-delete-target", "summary": "Steam 查询废弃旧流程"},
        ],
        actions=[
            ExistingAction(
                item_id="old-delete-target",
                summary="Steam 查询废弃旧流程",
                action=MemoryAction.DELETE,
                reason="fully obsolete",
            ),
        ],
        reason="delete obsolete memory only",
        query_vector=[1.0, 0.0],
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
        dedup_decider=cast(Any, _FixedDedupDecider(dedup_result)),
    )

    asyncio.run(
        worker._save_with_dedup(
            {
                "summary": "用户明确废弃旧的 Steam 查询流程",
                "memory_type": "procedure",
                "tool_requirement": None,
                "steps": [],
            },
            "test@post_response",
            protected_ids=None,
            token_budget=256,
            batch_vecs=[],
        )
    )

    memorizer.merge_item.assert_not_called()
    memorizer.supersede_batch.assert_called_once_with(["old-delete-target"])


def test_dedup_create_records_replacement_snapshot():
    store = MagicMock()
    store.get_items_by_ids.side_effect = lambda ids: [
        {
            "id": item_id,
            "memory_type": "procedure",
            "summary": "Steam 查询废弃旧流程",
            "extra_json": {"tool_requirement": "web_search"},
            "source_ref": "old@source",
            "happened_at": None,
        }
        if item_id == "old-delete-target"
        else {
            "id": "testid",
            "memory_type": "procedure",
            "summary": "新的 Steam 查询规则，必须先确认区服并使用 steam_mcp",
            "extra_json": {"tool_requirement": "steam_mcp"},
            "source_ref": "test@post_response",
            "happened_at": None,
        }
        for item_id in ids
    ]
    memorizer = _DummyMemorizer(store=store)
    dedup_result = DedupResult(
        decision=DedupDecision.CREATE,
        candidate_summary="新的 Steam 查询规则，必须先确认区服并使用 steam_mcp",
        candidate_type="procedure",
        similar_items=[
            {"id": "old-delete-target", "summary": "Steam 查询废弃旧流程"},
        ],
        actions=[
            ExistingAction(
                item_id="old-delete-target",
                summary="Steam 查询废弃旧流程",
                action=MemoryAction.DELETE,
                reason="fully obsolete",
            ),
        ],
        reason="replace old flow",
        query_vector=[1.0, 0.0],
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
        dedup_decider=cast(Any, _FixedDedupDecider(dedup_result)),
    )

    asyncio.run(
        worker._save_with_dedup(
            {
                "summary": "新的 Steam 查询规则，必须先确认区服并使用 steam_mcp",
                "memory_type": "procedure",
                "tool_requirement": "steam_mcp",
                "steps": [],
            },
            "test@post_response",
            protected_ids=None,
            token_budget=256,
            batch_vecs=[],
        )
    )

    memorizer.supersede_batch.assert_called_once_with(["old-delete-target"])
    store.record_replacements.assert_called_once()
    record_call = store.record_replacements.call_args.kwargs
    assert record_call["old_items"][0]["summary"] == "Steam 查询废弃旧流程"
    assert record_call["new_item"]["summary"] == "新的 Steam 查询规则，必须先确认区服并使用 steam_mcp"


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


def test_implicit_language_reply_rule_should_not_stay_procedure():
    item = PostResponseMemoryWorker._normalize_extracted_item(
        {
            "summary": "之后跟我说话只用中文，不要夹杂英文，专有名词也尽量翻译。",
            "memory_type": "procedure",
            "tool_requirement": None,
            "steps": [],
        }
    )

    assert item["memory_type"] == "preference"


def test_extract_obvious_preferences_catches_language_reply_rule():
    items = PostResponseMemoryWorker._extract_obvious_preferences(
        "之后跟我说话只用中文，不要夹杂英文，哪怕专有名词也尽量翻译。"
    )

    assert len(items) == 1
    assert items[0]["memory_type"] == "preference"
    assert "中文" in items[0]["summary"]


def test_extract_implicit_prompt_is_conservative_for_preference_upgrade():
    provider = _PromptCaptureProvider(response_text="[]")
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, provider),
        light_model="test",
    )

    items, remain = asyncio.run(
        worker._extract_implicit(
            user_msg="没买呢 感觉不太玩的下去慢节奏的游戏",
            agent_response="《红色沙漠》可能确实偏慢，你可以先别买。",
            already_memorized=[],
            token_budget=worker.TOKENS_EXTRACT_IMPLICIT,
        )
    )

    assert items == []
    assert remain == 0
    extract_prompt = provider.prompts[0]
    assert "宁可不提取" in extract_prompt
    assert "summary" in extract_prompt
    assert "USER" in extract_prompt
    # 三项检查均存在
    assert "检查 A" in extract_prompt
    assert "检查 B" in extract_prompt
    assert "检查 C" in extract_prompt


def test_build_implicit_prompt_uses_four_memory_classes():
    prompt = PostResponseMemoryWorker._build_implicit_prompt(
        user_msg="这个任务你不该派给他的 你做一下简单研究就能回答的",
        agent_response="好的，我下次会注意。",
    )

    # 核心类型定义仍存在
    assert 'procedure' in prompt
    assert 'preference' in prompt
    assert 'event' in prompt
    assert 'profile' in prompt
    # 三项检查存在
    assert '检查 A' in prompt
    assert '检查 B' in prompt
    assert '检查 C' in prompt
    # 示例存在
    assert '<example' in prompt
    # 防止输出 event/profile 的说明存在
    assert '绝对不输出' in prompt


def test_build_finalize_prompt_is_user_first_and_temporal_conservative():
    prompt = PostResponseMemoryWorker._build_finalize_prompt(
        user_msg="可惜今天不能看 falcons 比赛了，明天早上有个美团笔试",
        agent_response="那今晚先早点休息，比赛回头再补。",
        candidates=[
            {
                "summary": "为了确保明天笔试状态，应避免熬夜并优先保证睡眠",
                "memory_type": "procedure",
            }
        ],
    )

    assert "长期记忆入库决策" in prompt
    # 资格审查存在（情境性丢弃、来源方向丢弃）
    assert "ASSISTANT" in prompt
    assert "丢弃" in prompt
    # 忠实度核查存在
    assert "忠实度" in prompt
    # 示例存在
    assert '<example' in prompt


def test_extract_implicit_runs_finalize_stage_before_return():
    provider = _PromptCaptureProvider(
        response_text='[{"summary":"规则A","memory_type":"procedure","tool_requirement":null,"steps":[]}]'
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, _DummyMemorizer()),
        retriever=cast(Any, _DummyRetriever([])),
        light_provider=cast(Any, provider),
        light_model="test",
    )

    async def _run():
        return await worker._extract_implicit(
            user_msg="以后这样做",
            agent_response="好的",
            already_memorized=[],
            token_budget=worker.TOKEN_BUDGET_PER_RUN,
        )

    items, _ = asyncio.run(_run())
    assert items == [{"summary": "规则A", "memory_type": "procedure", "tool_requirement": None, "steps": [], "rule_schema": build_procedure_rule_schema(summary="规则A", tool_requirement=None, steps=[], rule_schema=None)}]
    assert provider.calls == 2
    assert "记忆提取专家" in provider.prompts[0]
    assert "长期记忆入库决策" in provider.prompts[1]


# ---------------------------------------------------------------------------
# _should_drop_by_heuristic 单测
# ---------------------------------------------------------------------------


def test_heuristic_drops_too_short_summary():
    # 2 个字符或以下属于碎片
    assert PostResponseMemoryWorker._should_drop_by_heuristic({"summary": "短", "memory_type": "procedure"}) is True
    assert PostResponseMemoryWorker._should_drop_by_heuristic({"summary": "ok", "memory_type": "procedure"}) is True
    assert PostResponseMemoryWorker._should_drop_by_heuristic({"summary": "", "memory_type": "procedure"}) is True


def test_heuristic_drops_knowledge_point_procedure_with_signal():
    signals = ["是指", "即为", "原理是", "的概念是", "协议规定", "定义为", "实现原理"]
    for signal in signals:
        item = {"summary": f"TCP 三次握手{signal}建立连接的过程", "memory_type": "procedure"}
        assert PostResponseMemoryWorker._should_drop_by_heuristic(item) is True, signal


def test_heuristic_keeps_normal_procedure():
    item = {"summary": "查询 Steam 游戏信息时必须先走 steam MCP，不能直接网页搜索", "memory_type": "procedure"}
    assert PostResponseMemoryWorker._should_drop_by_heuristic(item) is False


def test_heuristic_keeps_preference_even_with_knowledge_signal():
    # 知识点信号只过滤 procedure，不过滤 preference
    item = {"summary": "用户偏好直接回答，不需要知识点原理铺垫", "memory_type": "preference"}
    assert PostResponseMemoryWorker._should_drop_by_heuristic(item) is False


def test_heuristic_keeps_empty_type_as_default_procedure_passthrough():
    # type 缺失时视为 procedure，但 summary 正常则不 drop
    item = {"summary": "以后提醒用户先备份再操作", "memory_type": "procedure"}
    assert PostResponseMemoryWorker._should_drop_by_heuristic(item) is False


def test_heuristic_drops_architecture_discussion_with_two_signals():
    # 同时出现 2+ 个架构讨论信号 → drop
    item = {
        "summary": "未来在设计主动推送架构时，应采用插件化模式，由每个 channel 注册 tick handler",
        "memory_type": "procedure",
    }
    assert PostResponseMemoryWorker._should_drop_by_heuristic(item) is True


def test_heuristic_keeps_procedure_with_only_one_arch_signal():
    # 只有一个架构信号，不 drop（避免误伤）
    item = {
        "summary": "查询工具时应采用 tool_search 先搜索再调用",
        "memory_type": "procedure",
    }
    assert PostResponseMemoryWorker._should_drop_by_heuristic(item) is False
