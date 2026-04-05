from __future__ import annotations

from unittest.mock import AsyncMock

from core.memory.engine import MemoryEngineRetrieveRequest, MemoryIngestRequest, MemoryScope
from core.memory.memu_engine import MemUMemoryEngine


async def test_memu_engine_retrieve_maps_response_to_hits():
    service = AsyncMock()
    service.retrieve.return_value = {
        "needs_retrieval": True,
        "rewritten_query": "fitbit charge 6",
        "items": [
            {
                "id": "item-1",
                "summary": "用户最近在看 FitBit Charge 6",
                "score": 0.82,
                "category": "device_preference",
            }
        ],
        "resources": [
            {
                "id": "res-1",
                "name": "FitBit 对比文档",
                "content": "Charge 6 和 Inspire 3 对比",
                "score": 0.61,
            }
        ],
    }
    engine = MemUMemoryEngine(service=service)

    result = await engine.retrieve(
        MemoryEngineRetrieveRequest(
            query="FitBit 型号",
            context={"recent_turns": "用户昨天提到手环"},
            scope=MemoryScope(
                session_key="telegram:1",
                channel="telegram",
                chat_id="1",
            ),
            mode="episodic",
        )
    )

    service.retrieve.assert_awaited_once()
    kwargs = service.retrieve.await_args.kwargs
    assert kwargs["queries"] == [
        {"role": "system", "content": "用户昨天提到手环"},
        {"role": "user", "content": "FitBit 型号"},
    ]
    assert kwargs["where"] == {
        "channel": "telegram",
        "chat_id": "1",
        "session_key": "telegram:1",
    }
    assert result.text_block == ""
    assert [hit.id for hit in result.hits] == ["item-1", "res-1"]
    assert result.hits[0].engine_kind == "item"
    assert result.hits[1].engine_kind == "resource"
    assert result.trace["rewritten_query"] == "fitbit charge 6"


async def test_memu_engine_ingest_is_disabled_for_now():
    engine = MemUMemoryEngine(service=AsyncMock())

    result = await engine.ingest(
        MemoryIngestRequest(
            content={"user_message": "hi"},
            source_kind="conversation_turn",
            scope=MemoryScope(session_key="cli:1"),
        )
    )

    assert result.accepted is False
    assert result.raw["reason"] == "not_implemented"
