from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent.config_models import Config, MemoryV2Config
from agent.tools.registry import ToolRegistry
from bootstrap.memory import build_memory_runtime
from core.memory.default_engine import DefaultMemoryEngine
from core.memory.engine import (
    EngineProfile,
    MemoryCapability,
    MemoryEngineRetrieveRequest,
    MemoryEngineRetrieveResult,
    MemoryIngestRequest,
    MemoryHit,
    MemoryScope,
    RememberRequest,
    RememberResult,
)


async def test_default_memory_engine_retrieve_maps_hits_and_text_block():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "m1",
                    "summary": "记住用户偏好中文回复",
                    "score": 0.88,
                    "source_ref": "cli:1@seed",
                    "memory_type": "preference",
                    "extra_json": {"origin": "test"},
                }
            ]
        ),
        build_injection_block=lambda items: ("注入块", ["m1"]),
    )
    engine = DefaultMemoryEngine(retriever=retriever)

    result = await engine.retrieve(
        MemoryEngineRetrieveRequest(
            query="中文回复",
            scope=MemoryScope(channel="cli", chat_id="1"),
            hints={"memory_types": ["preference"], "require_scope_match": True},
            top_k=3,
        )
    )

    assert result.text_block == "注入块"
    assert len(result.hits) == 1
    assert result.hits[0].id == "m1"
    assert result.hits[0].injected is True
    assert result.hits[0].engine_kind == "default"
    assert result.hits[0].metadata["memory_type"] == "preference"
    assert result.trace["profile"] == EngineProfile.RICH_MEMORY_ENGINE.value


async def test_default_memory_engine_retrieve_keeps_raw_items_and_mode_trace():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "e1",
                    "summary": "用户昨天提过 FitBit",
                    "score": 0.81,
                    "source_ref": "telegram:1@seed",
                    "memory_type": "event",
                    "extra_json": {"origin": "test"},
                }
            ]
        ),
        build_injection_block=lambda items: ("历史块", ["e1"]),
    )
    engine = DefaultMemoryEngine(retriever=retriever)

    result = await engine.retrieve(
        MemoryEngineRetrieveRequest(
            query="Fitbit 型号",
            scope=MemoryScope(session_key="telegram:1"),
            mode="episodic",
            hints={"memory_types": ["event"], "require_scope_match": True},
            top_k=2,
        )
    )

    assert result.text_block == "历史块"
    assert result.trace["mode"] == "episodic"
    assert result.raw["items"][0]["id"] == "e1"
    assert result.hits[0].id == "e1"
    assert result.hits[0].injected is True


async def test_default_memory_engine_retrieve_falls_back_to_session_scope():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=[]),
        build_injection_block=lambda items: ("", []),
    )
    engine = DefaultMemoryEngine(retriever=retriever)

    await engine.retrieve(
        MemoryEngineRetrieveRequest(
            query="作用域测试",
            scope=MemoryScope(session_key="telegram:test_user"),
            hints={"require_scope_match": True},
        )
    )

    kwargs = retriever.retrieve.await_args.kwargs
    assert kwargs["scope_channel"] == "telegram"
    assert kwargs["scope_chat_id"] == "test_user"
    assert kwargs["require_scope_match"] is True


async def test_default_engine_keeps_history_injected_ids():
    from agent.retrieval.default_pipeline import _build_injection_payload

    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "e1",
                    "summary": "用户昨天提过 FitBit",
                    "score": 0.81,
                    "source_ref": "telegram:1@seed",
                    "memory_type": "event",
                    "extra_json": {"origin": "engine"},
                }
            ]
        ),
        build_injection_block=lambda items: ("## 【相关历史】\n- 用户昨天提过 FitBit", ["e1"]),
    )
    engine = DefaultMemoryEngine(retriever=retriever)

    history_result = await engine.retrieve(
        MemoryEngineRetrieveRequest(
            query="Fitbit 型号",
            scope=MemoryScope(session_key="telegram:1", channel="telegram", chat_id="1"),
            mode="episodic",
            hints={"memory_types": ["event"], "require_scope_match": True},
            top_k=8,
        )
    )

    selected_items, block, injected_ids = _build_injection_payload(
        procedure_items=[],
        procedure_result=None,
        history_items=[
            {
                "id": "e1",
                "memory_type": "event",
                "summary": "用户昨天提过 FitBit",
                "score": 0.81,
                "source_ref": "telegram:1@seed",
                "extra_json": {"origin": "engine"},
            }
        ],
        history_result=history_result,
    )

    assert "用户昨天提过 FitBit" in block
    assert [item["id"] for item in selected_items] == ["e1"]
    assert injected_ids == ["e1"]


async def test_default_memory_engine_ingest_delegates_to_post_worker():
    worker = SimpleNamespace(run=AsyncMock())
    engine = DefaultMemoryEngine(
        retriever=SimpleNamespace(),
        post_response_worker=worker,
    )

    result = await engine.ingest(
        MemoryIngestRequest(
            content={
                "user_message": "以后用中文",
                "assistant_response": "好的",
                "tool_chain": [{"text": "memo", "calls": []}],
            },
            source_kind="conversation_turn",
            scope=MemoryScope(session_key="cli:1"),
        )
    )

    assert result.accepted is True
    assert result.raw["engine"] == "default"
    worker.run.assert_awaited_once()


async def test_default_memory_engine_remember_uses_memorizer():
    memorizer = SimpleNamespace(
        save_item_with_supersede=AsyncMock(return_value="new:memu-1")
    )
    engine = DefaultMemoryEngine(
        retriever=SimpleNamespace(),
        memorizer=memorizer,
    )

    result = await engine.remember(
        RememberRequest(
            summary="以后用中文回复",
            memory_type="preference",
            scope=MemoryScope(session_key="cli:1", channel="cli", chat_id="1"),
        )
    )

    assert result.item_id == "memu-1"
    assert result.write_status == "new"
    memorizer.save_item_with_supersede.assert_awaited_once()


async def test_default_memory_engine_remember_merged_keeps_target_id_alive():
    memorizer = SimpleNamespace(
        save_item_with_supersede=AsyncMock(return_value="merged:memu-1")
    )
    engine = DefaultMemoryEngine(
        retriever=SimpleNamespace(),
        memorizer=memorizer,
    )

    result = await engine.remember(
        RememberRequest(
            summary="以后用中文回复",
            memory_type="preference",
            scope=MemoryScope(session_key="cli:1", channel="cli", chat_id="1"),
        )
    )

    assert result.item_id == "memu-1"
    assert result.write_status == "merged"
    assert result.superseded_ids == []


async def test_default_memory_engine_ingest_accepts_conversation_batch_messages():
    worker = SimpleNamespace(run=AsyncMock())
    engine = DefaultMemoryEngine(
        retriever=SimpleNamespace(),
        post_response_worker=worker,
    )

    result = await engine.ingest(
        MemoryIngestRequest(
            content=[
                {"role": "user", "content": "以后用中文"},
                {
                    "role": "assistant",
                    "content": "好的",
                    "tool_chain": [{"text": "memo", "calls": []}],
                },
            ],
            source_kind="conversation_batch",
            scope=MemoryScope(session_key="cli:1"),
        )
    )

    assert result.accepted is True
    kwargs = worker.run.await_args.kwargs
    assert kwargs["user_msg"] == "以后用中文"
    assert kwargs["agent_response"] == "好的"
    assert kwargs["tool_chain"] == [{"text": "memo", "calls": []}]
    assert kwargs["session_key"] == "cli:1"


async def test_default_memory_engine_ingest_falls_back_to_post_response_source_ref():
    worker = SimpleNamespace(run=AsyncMock())
    engine = DefaultMemoryEngine(
        retriever=SimpleNamespace(),
        post_response_worker=worker,
    )

    result = await engine.ingest(
        MemoryIngestRequest(
            content={
                "user_message": "以后用中文",
                "assistant_response": "好的",
            },
            source_kind="conversation_turn",
            scope=MemoryScope(session_key="cli:1"),
        )
    )

    assert result.accepted is True
    kwargs = worker.run.await_args.kwargs
    assert kwargs["source_ref"] == "cli:1@post_response"
    assert kwargs["session_key"] == "cli:1"


async def test_default_memory_engine_ingest_rejects_unsupported_source_kind():
    worker = SimpleNamespace(run=AsyncMock())
    engine = DefaultMemoryEngine(
        retriever=SimpleNamespace(),
        post_response_worker=worker,
    )

    result = await engine.ingest(
        MemoryIngestRequest(
            content="以后用中文",
            source_kind="text",
            scope=MemoryScope(session_key="cli:1"),
        )
    )

    assert result.accepted is False
    assert result.raw["reason"] == "unsupported_source_kind"
    worker.run.assert_not_awaited()


async def test_default_memory_engine_ingest_rejects_when_worker_missing():
    engine = DefaultMemoryEngine(
        retriever=SimpleNamespace(),
        post_response_worker=None,
    )

    result = await engine.ingest(
        MemoryIngestRequest(
            content={
                "user_message": "以后用中文",
                "assistant_response": "好的",
            },
            source_kind="conversation_turn",
            scope=MemoryScope(session_key="cli:1"),
        )
    )

    assert result.accepted is False
    assert result.raw["reason"] == "worker_unavailable"


def test_default_memory_engine_descriptor_keeps_messages_capability_only():
    descriptor = DefaultMemoryEngine.DESCRIPTOR

    assert descriptor.profile == EngineProfile.RICH_MEMORY_ENGINE
    assert MemoryCapability.INGEST_MESSAGES in descriptor.capabilities
    assert MemoryCapability.INGEST_TEXT not in descriptor.capabilities


def test_build_memory_runtime_uses_memory_engine_factory(monkeypatch, tmp_path: Path):
    import bootstrap.memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "register_memory_meta_tools",
        lambda *args, **kwargs: None,
    )

    class _MemoryStore:
        def __init__(self, workspace):
            self.workspace = workspace

    class _SkillsLoader:
        def __init__(self, workspace):
            self.workspace = workspace

        def list_skills(self, filter_unavailable=False):
            return [{"name": "demo"}]

    class _WriteFileTool:
        pass

    class _EditFileTool:
        pass

    class _MemorizeTool:
        def __init__(self, engine):
            self.engine = engine

    class _DefaultMemoryPort:
        def __init__(self, store, memorizer=None, retriever=None):
            self.store = store
            self.memorizer = memorizer
            self.retriever = retriever

    class _Store2:
        def __init__(self, db_path):
            self.db_path = db_path

        def close(self):
            return None

    class _Embedder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            return None

    class _Memorizer:
        def __init__(self, store, embedder):
            self.store = store
            self.embedder = embedder

    class _Retriever:
        def __init__(self, store, embedder, **kwargs):
            self.store = store
            self.embedder = embedder
            self.kwargs = kwargs

    class _ProcedureTagger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _ProfileFactExtractor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _PostResponseMemoryWorker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    captured: dict[str, object] = {}

    class _CustomEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def describe(self):
            return SimpleNamespace(name="custom")

    monkeypatch.setattr("agent.memory.MemoryStore", _MemoryStore)
    monkeypatch.setattr("agent.skills.SkillsLoader", _SkillsLoader)
    monkeypatch.setattr("agent.tools.memorize.MemorizeTool", _MemorizeTool)
    monkeypatch.setattr("agent.tools.filesystem.WriteFileTool", _WriteFileTool)
    monkeypatch.setattr("agent.tools.filesystem.EditFileTool", _EditFileTool)
    monkeypatch.setattr("core.memory.port.DefaultMemoryPort", _DefaultMemoryPort)
    monkeypatch.setattr("memory2.store.MemoryStore2", _Store2)
    monkeypatch.setattr("memory2.embedder.Embedder", _Embedder)
    monkeypatch.setattr("memory2.memorizer.Memorizer", _Memorizer)
    monkeypatch.setattr("memory2.retriever.Retriever", _Retriever)
    monkeypatch.setattr("memory2.procedure_tagger.ProcedureTagger", _ProcedureTagger)
    monkeypatch.setattr(
        "memory2.profile_extractor.ProfileFactExtractor",
        _ProfileFactExtractor,
    )
    monkeypatch.setattr(
        memory_module,
        "PostResponseMemoryWorker",
        _PostResponseMemoryWorker,
    )
    monkeypatch.setattr(
        "bootstrap.wiring.resolve_memory_engine_builder",
        lambda name: (lambda deps: _CustomEngine(
            retriever=deps.retriever,
            memorizer=deps.memorizer,
            tagger=deps.tagger,
            post_response_worker=deps.post_response_worker,
        )),
    )

    runtime = build_memory_runtime(
        config=Config(
            provider="test",
            model="gpt-test",
            api_key="k",
            system_prompt="hi",
            memory_v2=MemoryV2Config(enabled=True),
        ),
        workspace=tmp_path,
        tools=ToolRegistry(),
        provider=SimpleNamespace(),
        light_provider=None,
        http_resources=SimpleNamespace(external_default=SimpleNamespace()),
    )

    assert runtime.engine is not None
    assert runtime.facade is not None
    assert runtime.engine.describe().name == "custom"
    assert "retriever" in captured
    assert "memorizer" in captured
    assert "tagger" in captured
    assert "post_response_worker" in captured


def test_build_memory_runtime_exposes_default_memory_engine(
    monkeypatch,
    tmp_path: Path,
):
    import bootstrap.memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "register_memory_meta_tools",
        lambda *args, **kwargs: None,
    )

    class _MemoryStore:
        def __init__(self, workspace):
            self.workspace = workspace

    class _SkillsLoader:
        def __init__(self, workspace):
            self.workspace = workspace

        def list_skills(self, filter_unavailable=False):
            return [{"name": "demo"}]

    class _WriteFileTool:
        pass

    class _EditFileTool:
        pass

    class _MemorizeTool:
        def __init__(self, engine):
            self.engine = engine

    class _DefaultMemoryPort:
        def __init__(self, store, memorizer=None, retriever=None):
            self.store = store
            self.memorizer = memorizer
            self.retriever = retriever

    class _Store2:
        def __init__(self, db_path):
            self.db_path = db_path

        def close(self):
            return None

    class _Embedder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            return None

    class _Memorizer:
        def __init__(self, store, embedder):
            self.store = store
            self.embedder = embedder

    class _Retriever:
        def __init__(self, store, embedder, **kwargs):
            self.store = store
            self.embedder = embedder
            self.kwargs = kwargs

    class _ProcedureTagger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _ProfileFactExtractor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _PostResponseMemoryWorker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("agent.memory.MemoryStore", _MemoryStore)
    monkeypatch.setattr("agent.skills.SkillsLoader", _SkillsLoader)
    monkeypatch.setattr("agent.tools.memorize.MemorizeTool", _MemorizeTool)
    monkeypatch.setattr("agent.tools.filesystem.WriteFileTool", _WriteFileTool)
    monkeypatch.setattr("agent.tools.filesystem.EditFileTool", _EditFileTool)
    monkeypatch.setattr("core.memory.port.DefaultMemoryPort", _DefaultMemoryPort)
    monkeypatch.setattr("memory2.store.MemoryStore2", _Store2)
    monkeypatch.setattr("memory2.embedder.Embedder", _Embedder)
    monkeypatch.setattr("memory2.memorizer.Memorizer", _Memorizer)
    monkeypatch.setattr("memory2.retriever.Retriever", _Retriever)
    monkeypatch.setattr("memory2.procedure_tagger.ProcedureTagger", _ProcedureTagger)
    monkeypatch.setattr(
        "memory2.profile_extractor.ProfileFactExtractor",
        _ProfileFactExtractor,
    )
    monkeypatch.setattr(
        memory_module,
        "PostResponseMemoryWorker",
        _PostResponseMemoryWorker,
    )

    runtime = build_memory_runtime(
        config=Config(
            provider="test",
            model="gpt-test",
            api_key="k",
            system_prompt="hi",
            memory_v2=MemoryV2Config(enabled=True),
        ),
        workspace=tmp_path,
        tools=ToolRegistry(),
        provider=SimpleNamespace(),
        light_provider=None,
        http_resources=SimpleNamespace(external_default=SimpleNamespace()),
    )

    assert runtime.engine is not None
    assert runtime.engine.describe().name == "default"
    assert MemoryCapability.SEMANTICS_RICH_MEMORY in runtime.engine.describe().capabilities
