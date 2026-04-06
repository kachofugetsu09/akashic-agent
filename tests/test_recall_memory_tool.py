import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.tools.recall_memory import RecallMemoryTool


class _FailingEmbedder:
    async def embed(self, text: str) -> list[float]:
        raise RuntimeError(f"embed failed: {text}")


class _KeywordOnlyStore:
    def __init__(self) -> None:
        self.vector_search_called = False

    def vector_search(self, *_args, **_kwargs):
        self.vector_search_called = True
        return []

    def keyword_search_summary(self, terms, memory_types=None, limit=20):
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


@pytest.mark.asyncio
async def test_recall_memory_falls_back_to_keyword_when_query_embed_fails():
    store = _KeywordOnlyStore()
    provider = SimpleNamespace(
        chat=AsyncMock(return_value=SimpleNamespace(content="用户处理过支付相关问题"))
    )
    tool = RecallMemoryTool(
        store=store,
        embedder=_FailingEmbedder(),
        provider=provider,
        model="test-model",
    )

    payload = json.loads(await tool.execute(query="phase 支付"))

    assert payload["count"] == 1
    assert payload["items"][0]["id"] == "mem:1"
    assert payload["items"][0]["source_ref"] == "tg:1:2"
    assert store.vector_search_called is False
    assert payload["hypotheses_used"] == ["用户处理过支付相关问题", "用户处理过支付相关问题"]
