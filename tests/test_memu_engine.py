from __future__ import annotations
from typing import Any, cast

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent.looping.ports import LLMServices, MemoryConfig, MemoryServices
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import RetrievalRequest
from core.memory.engine import (
    MemoryEngineRetrieveRequest,
    MemoryIngestRequest,
    MemoryScope,
    RememberRequest,
)
from memory2.query_rewriter import GateDecision
from core.memory.memu_engine import MemUMemoryEngine


async def test_memu_engine_retrieve_maps_response_to_injectable_hits():
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
    engine = MemUMemoryEngine(service=service, input_dir=Path("/tmp/memu-test"))

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
    assert [hit.id for hit in result.hits] == ["item-1", "res-1"]
    assert result.hits[0].engine_kind == "item"
    assert result.hits[1].engine_kind == "resource"
    assert result.trace["rewritten_query"] == "fitbit charge 6"
    assert result.text_block.startswith("## 【相关记忆】")
    assert [hit.id for hit in result.hits if hit.injected] == ["item-1", "res-1"]


async def test_memu_engine_ingest_memorizes_text_payload(tmp_path: Path):
    service = AsyncMock()
    service.memorize.return_value = {
        "items": [{"id": "m1"}, {"id": "m2"}],
    }
    engine = MemUMemoryEngine(service=service, input_dir=tmp_path)

    result = await engine.ingest(
        MemoryIngestRequest(
            content={
                "user_message": "hi",
                "assistant_response": "ok",
                "tool_chain": [{"name": "a"}],
            },
            source_kind="conversation_turn",
            scope=MemoryScope(session_key="cli:1", channel="cli", chat_id="1"),
        )
    )

    assert result.accepted is True
    assert result.created_ids == ["m1", "m2"]
    kwargs = service.memorize.await_args.kwargs
    assert kwargs["modality"] == "text"
    assert kwargs["user"] == {
        "session_key": "cli:1",
        "channel": "cli",
        "chat_id": "1",
    }
    resource_path = Path(kwargs["resource_url"])
    assert resource_path.exists()
    text = resource_path.read_text(encoding="utf-8")
    assert "user: hi" in text
    assert "assistant: ok" in text
    assert "tool_chain:" in text


async def test_memu_engine_ingest_rejects_unsupported_source_kind(tmp_path: Path):
    engine = MemUMemoryEngine(service=AsyncMock(), input_dir=tmp_path)

    result = await engine.ingest(
        MemoryIngestRequest(
            content={"user_message": "hi"},
            source_kind="resource",
            scope=MemoryScope(session_key="cli:1"),
        )
    )

    assert result.accepted is False
    assert result.raw["reason"] == "unsupported_source_kind"


async def test_memu_engine_remember_uses_engine_scope(tmp_path: Path):
    service = AsyncMock()
    service.memorize.return_value = {"items": [{"id": "memu-1"}]}
    engine = MemUMemoryEngine(service=service, input_dir=tmp_path)

    result = await engine.remember(
        RememberRequest(
            summary="以后用中文回复",
            memory_type="preference",
            scope=MemoryScope(session_key="cli:1", channel="cli", chat_id="1"),
        )
    )

    assert result.item_id == "memu-1"
    kwargs = service.memorize.await_args.kwargs
    assert kwargs["user"] == {
        "session_key": "cli:1",
        "channel": "cli",
        "chat_id": "1",
    }
    assert result.write_status == "new"


async def test_memu_engine_retrieval_pipeline_injects_memory_block(tmp_path: Path):
    service = AsyncMock()
    service.retrieve.side_effect = [
        {
            "needs_retrieval": True,
            "rewritten_query": "fitbit charge 6",
            "items": [],
        },
        {
            "needs_retrieval": True,
            "rewritten_query": "fitbit charge 6",
            "items": [
                {
                    "id": "item-1",
                    "summary": "用户最近在看 FitBit Charge 6",
                    "score": 0.82,
                    "memory_type": "event",
                }
            ],
        },
    ]
    engine = MemUMemoryEngine(service=service, input_dir=tmp_path)
    pipeline = DefaultMemoryRetrievalPipeline(
        memory=MemoryServices(
            engine=engine,
            query_rewriter=SimpleNamespace(
                decide=AsyncMock(
                    return_value=GateDecision(
                        needs_episodic=True,
                        episodic_query="FitBit 型号",
                        latency_ms=1,
                    )
                )
            ),
        ),
        memory_config=MemoryConfig(),
        llm=LLMServices(provider=cast(Any, object()), light_provider=cast(Any, object())),
        workspace=tmp_path,
        light_model="test-light",
    )

    result = await pipeline.retrieve(
        RetrievalRequest(
            message="我之前看过什么 FitBit？",
            session_key="telegram:1",
            channel="telegram",
            chat_id="1",
            history=[],
            session_metadata={},
            timestamp=datetime.now(),
        )
    )

    assert "用户最近在看 FitBit Charge 6" in result.block
    assert result.trace is not None
    assert result.trace.injected_count == 1
