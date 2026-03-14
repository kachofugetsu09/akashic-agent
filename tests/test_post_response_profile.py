import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.profile_extractor import ProfileFact


class _DummyProvider:
    async def chat(self, **kwargs):
        raise AssertionError("provider.chat should not be called in this test")


class _DummyRetriever:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def retrieve(self, query: str, memory_types=None, top_k=None):
        self.calls.append((query, tuple(memory_types or []), top_k))
        return list(self._results)


class _DummyMemorizer:
    def __init__(self):
        self.save_item = AsyncMock(return_value="new:testid")
        self.supersede_batch = MagicMock()


def _fact(summary: str, category: str, happened_at: str | None = None) -> ProfileFact:
    return ProfileFact(summary=summary, category=category, happened_at=happened_at)


def test_worker_calls_profile_extractor_with_exchange_content():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever([])
    extractor = MagicMock()
    extractor.extract_from_exchange = AsyncMock(return_value=[])
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
        profile_extractor=extractor,
    )
    worker._collect_explicit_memorized = lambda tool_chain: ([], set())
    worker._handle_invalidations = AsyncMock(side_effect=lambda *args, **kwargs: args[-1])
    worker._extract_implicit = AsyncMock(return_value=([], 500))

    asyncio.run(
        worker.run(
            user_msg="我刚买了一个新键盘",
            agent_response="记住了",
            tool_chain=[],
            source_ref="test@post_response",
        )
    )

    extract_call = extractor.extract_from_exchange.await_args
    assert extract_call.args[0] == "我刚买了一个新键盘"
    assert extract_call.args[1] == "记住了"


def test_worker_runs_without_profile_extractor():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever([])
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
        profile_extractor=None,
    )
    worker._collect_explicit_memorized = lambda tool_chain: ([], set())
    worker._handle_invalidations = AsyncMock(side_effect=lambda *args, **kwargs: args[-1])
    worker._extract_implicit = AsyncMock(return_value=([], 500))

    asyncio.run(
        worker.run(
            user_msg="我刚买了一个新键盘",
            agent_response="记住了",
            tool_chain=[],
            source_ref="test@post_response",
        )
    )


def test_worker_saves_purchase_fact_to_memory():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever([])
    extractor = MagicMock()
    extractor.extract_from_exchange = AsyncMock(
        return_value=[_fact("用户刚买了一个新键盘", "purchase", "2026-03-15")]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
        profile_extractor=extractor,
    )
    worker._collect_explicit_memorized = lambda tool_chain: ([], set())
    worker._handle_invalidations = AsyncMock(side_effect=lambda *args, **kwargs: args[-1])
    worker._extract_implicit = AsyncMock(return_value=([], 500))

    asyncio.run(
        worker.run(
            user_msg="我刚买了一个新键盘",
            agent_response="记住了",
            tool_chain=[],
            source_ref="test@post_response",
        )
    )

    memorizer.save_item.assert_awaited_once_with(
        summary="用户刚买了一个新键盘",
        memory_type="profile",
        extra={"category": "purchase"},
        source_ref="test@post_response",
        happened_at="2026-03-15",
    )


def test_worker_skips_profile_extraction_when_budget_exhausted():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever([])
    extractor = MagicMock()
    extractor.extract_from_exchange = AsyncMock(return_value=[])
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
        profile_extractor=extractor,
    )
    worker._collect_explicit_memorized = lambda tool_chain: ([], set())
    worker._handle_invalidations = AsyncMock(side_effect=lambda *args, **kwargs: 0)
    worker._extract_implicit = AsyncMock(return_value=([], 0))

    asyncio.run(
        worker.run(
            user_msg="我刚买了一个新键盘",
            agent_response="记住了",
            tool_chain=[],
            source_ref="test@post_response",
        )
    )

    extractor.extract_from_exchange.assert_not_awaited()


def test_status_fact_supersedes_stale_status_on_same_subject():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever(
        [
            {
                "id": "old-status",
                "memory_type": "profile",
                "score": 0.82,
                "summary": "用户正在等待键盘到货",
                "extra_json": {"category": "status"},
            }
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    worker._check_supersede = AsyncMock(return_value=(["old-status"], 400))

    asyncio.run(
        worker._save_profile_with_supersede(
            _fact("用户的键盘今天到了", "status"),
            "test@post_response",
            500,
        )
    )

    memorizer.supersede_batch.assert_called_once_with(["old-status"])


def test_different_category_profile_not_superseded():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever(
        [
            {
                "id": "old-status",
                "memory_type": "profile",
                "score": 0.90,
                "summary": "用户正在等待键盘到货",
                "extra_json": {"category": "status"},
            }
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    worker._check_supersede = AsyncMock(return_value=(["old-status"], 400))

    asyncio.run(
        worker._save_profile_with_supersede(
            _fact("用户买了一个新键盘", "purchase"),
            "test@post_response",
            500,
        )
    )

    worker._check_supersede.assert_not_awaited()
    memorizer.supersede_batch.assert_not_called()


def test_low_similarity_profile_skips_supersede_check():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever(
        [
            {
                "id": "old-status",
                "memory_type": "profile",
                "score": 0.60,
                "summary": "用户正在等待键盘到货",
                "extra_json": {"category": "status"},
            }
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    worker._check_supersede = AsyncMock(return_value=(["old-status"], 400))

    asyncio.run(
        worker._save_profile_with_supersede(
            _fact("用户的键盘今天到了", "status"),
            "test@post_response",
            500,
        )
    )

    worker._check_supersede.assert_not_awaited()


def test_profile_supersede_check_fails_open_on_llm_exception():
    memorizer = _DummyMemorizer()
    retriever = _DummyRetriever(
        [
            {
                "id": "old-status",
                "memory_type": "profile",
                "score": 0.82,
                "summary": "用户正在等待键盘到货",
                "extra_json": {"category": "status"},
            }
        ]
    )
    worker = PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, _DummyProvider()),
        light_model="test",
    )
    worker._check_supersede = AsyncMock(side_effect=RuntimeError("timeout"))

    asyncio.run(
        worker._save_profile_with_supersede(
            _fact("用户的键盘今天到了", "status"),
            "test@post_response",
            500,
        )
    )

    memorizer.save_item.assert_awaited_once()
    memorizer.supersede_batch.assert_not_called()
