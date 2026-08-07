from __future__ import annotations
from typing import Any, cast

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.store import MemoryHit


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.mark.asyncio
async def test_post_response_worker_invalidation_paths():
    memorizer = SimpleNamespace(
        save_item=AsyncMock(return_value="new:1"),
        supersede_batch=MagicMock(),
    )
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            side_effect=[
                [{"id": "x1", "score": 0.9, "summary": "旧规则"}],
                [{"id": "x1", "score": 0.9, "summary": "旧规则"}],
            ]
        )
    )
    provider = SimpleNamespace(chat=AsyncMock(return_value=_Resp('["topic"]')))
    worker = PostResponseMemoryWorker(cast(Any, memorizer), cast(Any, retriever), cast(Any, provider), "lm")

    assert worker._consume_budget(10, 3) == (True, 7)
    assert worker._collect_protected_memory_ids(
        [{"calls": [{"name": "memorize", "arguments": {"summary": "规则A"}, "result": "已记住（new:AbCDef12_34567890）：规则A"}]}]
    ) == {"AbCDef12_34567890"}

    topics, remain = await worker._extract_invalidation_topics("你之前这个流程错了", 700)
    assert topics == ["topic"]

    provider.chat = AsyncMock(return_value=_Resp('["x1"]'))
    candidates: list[MemoryHit] = [
        {
            "id": "x1",
            "memory_type": "procedure",
            "summary": "旧规则",
            "source_ref": "turn:old",
            "happened_at": "2025-01-01T00:00:00+00:00",
            "score": 0.9,
        }
    ]
    ids, remain = await worker._check_invalidate("topic", candidates, remain)
    assert ids == ["x1"]


@pytest.mark.asyncio
async def test_post_response_worker_budget_exhausted_skips_invalidation():
    memorizer = SimpleNamespace(save_item=AsyncMock(return_value="new:2"), supersede_batch=MagicMock())
    retriever = SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("boom")))
    provider = SimpleNamespace(chat=AsyncMock(return_value=_Resp("bad json")))
    worker = PostResponseMemoryWorker(cast(Any, memorizer), cast(Any, retriever), cast(Any, provider), "lm")

    topics, remain = await worker._extract_invalidation_topics("也许这个流程不对", 0)
    assert topics == []
    assert remain == 0
