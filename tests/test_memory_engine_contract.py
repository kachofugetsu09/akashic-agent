from __future__ import annotations
from typing import Any, cast

import asyncio
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from agent.config_models import Config, MemoryConfig
from agent.tools.registry import ToolRegistry
from bootstrap.memory import build_memory_runtime
from plugins.default_memory.engine import DefaultMemoryEngine
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
from core.memory.events import ConsolidationCommitted, TurnIngested
from core.memory.markdown import (
    ConsolidateRequest,
    ConsolidateResult,
    _ConsolidationDraft,
    _ConsolidationWindow,
    MarkdownMemoryMaintenance,
    MarkdownMemoryStore,
    MemoryLifecycleBindRequest,
)
from core.memory.plugin import MemoryPluginRuntime


def _make_default_engine(
    *,
    config=None,
    provider=None,
    retriever=None,
    memorizer=None,
    tagger=None,
    post_response_worker=None,
    event_publisher=None,
):
    engine = DefaultMemoryEngine.__new__(DefaultMemoryEngine)
    engine._config = config or SimpleNamespace(model="lm")
    engine._workspace = Path(".")
    engine._provider = provider
    engine._light_provider = None
    engine._light_model = ""
    engine._v2_store = None
    engine._embedder = None
    engine._memorizer = memorizer
    engine._retriever = retriever
    engine._tagger = tagger
    engine._post_response_worker = post_response_worker
    engine._event_bus = event_publisher
    engine.closeables = []
    engine._wire_memory2_events()
    return engine


async def _drain_maintenance(maintenance: object) -> None:
    for _ in range(5):
        tasks = list(getattr(maintenance, "_maintenance_tasks").values())
        if not tasks:
            return
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)


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
    engine = _make_default_engine(retriever=cast(Any, retriever))

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
    engine = _make_default_engine(retriever=cast(Any, retriever))

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
    raw = cast(dict[str, object], result.raw)
    raw_items = cast(list[object], raw["items"])
    assert cast(dict[str, object], raw_items[0])["id"] == "e1"
    assert result.hits[0].id == "e1"
    assert result.hits[0].injected is True


async def test_default_memory_engine_retrieve_falls_back_to_session_scope():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=[]),
        build_injection_block=lambda items: ("", []),
    )
    engine = _make_default_engine(retriever=cast(Any, retriever))

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
    assert "keyword_only_enabled" not in kwargs


async def test_default_engine_keeps_history_injected_ids():
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
    engine = _make_default_engine(retriever=cast(Any, retriever))

    history_result = await engine.retrieve(
        MemoryEngineRetrieveRequest(
            query="Fitbit 型号",
            scope=MemoryScope(session_key="telegram:1", channel="telegram", chat_id="1"),
            mode="episodic",
            hints={"memory_types": ["event"], "require_scope_match": True},
            top_k=8,
        )
    )

    assert "用户昨天提过 FitBit" in history_result.text_block
    assert [hit.id for hit in history_result.hits if hit.injected] == ["e1"]


async def test_default_memory_engine_ingest_delegates_to_post_worker():
    worker = SimpleNamespace(run=AsyncMock())
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        post_response_worker=cast(Any, worker),
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


async def test_default_memory_engine_handles_turn_committed_via_event_bus():
    event_bus = EventBus()
    worker = SimpleNamespace(run=AsyncMock(), handle=AsyncMock())
    _ = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        post_response_worker=cast(Any, worker),
        event_publisher=event_bus,
    )

    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="以后用中文",
            persisted_user_message="以后用中文",
            assistant_response="好的",
            tools_used=[],
            tool_chain_raw=[{"text": "memo", "calls": []}],
        )
    )
    await event_bus.drain()

    worker.handle.assert_awaited_once()
    event = worker.handle.await_args.args[0]
    assert isinstance(event, TurnIngested)
    assert event.session_key == "cli:1"
    assert event.tool_chain == [{"text": "memo", "calls": []}]
    await event_bus.aclose()


async def test_default_memory_engine_respects_skip_post_memory_event_flag():
    event_bus = EventBus()
    worker = SimpleNamespace(run=AsyncMock(), handle=AsyncMock())
    _ = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        post_response_worker=cast(Any, worker),
        event_publisher=event_bus,
    )

    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="以后用中文",
            persisted_user_message="以后用中文",
            assistant_response="好的",
            tools_used=[],
            extra={"skip_post_memory": True},
        )
    )
    await event_bus.drain()

    worker.handle.assert_not_awaited()
    await event_bus.aclose()


async def test_default_memory_engine_refreshes_recent_context_from_lifecycle():
    event_bus = EventBus()
    session = SimpleNamespace(
        key="cli:1",
        messages=[{"role": "user", "content": "u"}],
        last_consolidated=0,
    )
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(Path(".")),
        provider=cast(Any, SimpleNamespace()),
        model="lm",
        keep_count=20,
        event_bus=event_bus,
    )
    maintenance.refresh_recent_turns = AsyncMock()
    save_session = AsyncMock()
    maintenance.bind_lifecycle(
        MemoryLifecycleBindRequest(
            get_session=lambda _key: session,
            save_session=save_session,
        )
    )

    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="hi",
            persisted_user_message="hi",
            assistant_response="ok",
            tools_used=[],
        )
    )
    await event_bus.drain()
    await _drain_maintenance(maintenance)

    maintenance.refresh_recent_turns.assert_awaited_once()
    save_session.assert_not_awaited()
    await event_bus.aclose()


async def test_default_memory_engine_consolidates_ready_session_from_lifecycle():
    event_bus = EventBus()
    session = SimpleNamespace(
        key="cli:1",
        messages=[{"role": "user", "content": "u"}] * 31,
        last_consolidated=0,
    )
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(Path(".")),
        provider=cast(Any, SimpleNamespace()),
        model="lm",
        keep_count=20,
        event_bus=event_bus,
    )
    maintenance._consolidate_unlocked = AsyncMock(
        return_value=ConsolidateResult(trace={"mode": "markdown"})
    )
    save_session = AsyncMock()
    maintenance.bind_lifecycle(
        MemoryLifecycleBindRequest(
            get_session=lambda _key: session,
            save_session=save_session,
        )
    )

    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="hi",
            persisted_user_message="hi",
            assistant_response="ok",
            tools_used=[],
        )
    )
    await event_bus.drain()
    await _drain_maintenance(maintenance)

    maintenance._consolidate_unlocked.assert_awaited_once()
    save_session.assert_awaited_once_with(session)
    await event_bus.aclose()


async def test_markdown_consolidation_keeps_window_when_consumer_fails(tmp_path: Path):
    event_bus = EventBus()

    async def _fail_consolidation(_event):
        raise RuntimeError("vector write failed")

    event_bus.on(ConsolidationCommitted, _fail_consolidation)
    session = SimpleNamespace(
        key="cli:1",
        messages=[{"role": "user", "content": "u"}] * 12,
        last_consolidated=0,
    )
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(tmp_path),
        provider=cast(Any, SimpleNamespace()),
        model="lm",
        keep_count=6,
        event_bus=event_bus,
    )
    draft = _ConsolidationDraft(
        window=_ConsolidationWindow(
            old_messages=list(session.messages[:6]),
            keep_count=6,
            consolidate_up_to=6,
        ),
        source_ref='["cli:1:0"]',
        history_entry_payloads=[("[2026-05-05 13:00] 用户测试记忆", 0)],
        pending_items="",
        conversation="USER: 测试记忆",
        recent_context_text="# Recent Context\n",
        scope_channel="cli",
        scope_chat_id="1",
    )
    maintenance._worker.prepare_consolidation = AsyncMock(return_value=draft)

    with pytest.raises(RuntimeError, match="vector write failed"):
        await maintenance.consolidate(ConsolidateRequest(session=session))

    assert session.last_consolidated == 0
    assert "用户测试记忆" in (tmp_path / "memory" / "HISTORY.md").read_text(
        encoding="utf-8"
    )
    await event_bus.aclose()


async def test_default_memory_engine_serializes_lifecycle_maintenance():
    event_bus = EventBus()
    session = SimpleNamespace(
        key="cli:1",
        messages=[{"role": "user", "content": "u"}],
        last_consolidated=0,
    )
    maintenance = MarkdownMemoryMaintenance(
        store=MarkdownMemoryStore(Path(".")),
        provider=cast(Any, SimpleNamespace()),
        model="lm",
        keep_count=20,
        event_bus=event_bus,
    )
    active = 0
    max_active = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def _refresh_recent_turns(_request) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if max_active == 1:
            first_started.set()
            await release_first.wait()
        active -= 1

    maintenance.refresh_recent_turns = AsyncMock(side_effect=_refresh_recent_turns)
    maintenance.bind_lifecycle(
        MemoryLifecycleBindRequest(
            get_session=lambda _key: session,
            save_session=AsyncMock(),
        )
    )

    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="a",
            persisted_user_message="a",
            assistant_response="ok",
            tools_used=[],
        )
    )
    await event_bus.drain()
    await first_started.wait()
    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="b",
            persisted_user_message="b",
            assistant_response="ok",
            tools_used=[],
        )
    )
    await event_bus.drain()
    release_first.set()
    await _drain_maintenance(maintenance)

    assert max_active == 1
    assert maintenance.refresh_recent_turns.await_count == 2
    await event_bus.aclose()


async def test_default_memory_engine_remember_uses_memorizer():
    memorizer = SimpleNamespace(
        save_item_with_supersede=AsyncMock(return_value="new:memu-1")
    )
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        memorizer=cast(Any, memorizer),
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
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        memorizer=cast(Any, memorizer),
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


async def test_default_memory_engine_consumes_markdown_consolidation_event():
    memorizer = SimpleNamespace(
        save_from_consolidation=AsyncMock(),
        save_item_with_supersede=AsyncMock(return_value="new:memu-1"),
    )
    provider = SimpleNamespace(
        chat=AsyncMock(
            return_value=SimpleNamespace(
                content='{"profile":[{"summary":"用户买了 Zigbee 网关","category":"purchase","emotional_weight":4}],"preference":[],"procedure":[]}'
            )
        )
    )
    engine = _make_default_engine(
        provider=cast(Any, provider),
        memorizer=cast(Any, memorizer),
    )

    await engine._on_consolidation_committed(
        ConsolidationCommitted(
            history_entry_payloads=[("[2026-03-15 10:00] 用户聊了 Zigbee", 6)],
            source_ref='["m1"]',
            scope_channel="cli",
            scope_chat_id="1",
            conversation="USER: 我买了 Zigbee 网关",
        )
    )

    memorizer.save_from_consolidation.assert_awaited_once()
    memorizer.save_item_with_supersede.assert_awaited_once()


async def test_default_memory_engine_reports_implicit_extraction_failure():
    memorizer = SimpleNamespace(
        save_from_consolidation=AsyncMock(),
        save_item_with_supersede=AsyncMock(return_value="new:memu-1"),
    )
    provider = SimpleNamespace(
        chat=AsyncMock(return_value=SimpleNamespace(content="not json"))
    )
    engine = _make_default_engine(
        provider=cast(Any, provider),
        memorizer=cast(Any, memorizer),
    )

    with pytest.raises(RuntimeError, match="long_term extraction failed"):
        await engine._on_consolidation_committed(
            ConsolidationCommitted(
                history_entry_payloads=[("[2026-03-15 10:00] 用户聊了 Zigbee", 6)],
                source_ref='["m1"]',
                scope_channel="cli",
                scope_chat_id="1",
                conversation="USER: 我买了 Zigbee 网关",
            )
        )

    memorizer.save_from_consolidation.assert_awaited_once()
    memorizer.save_item_with_supersede.assert_not_awaited()


async def test_default_memory_engine_ingest_accepts_conversation_batch_messages():
    worker = SimpleNamespace(run=AsyncMock())
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        post_response_worker=cast(Any, worker),
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
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        post_response_worker=cast(Any, worker),
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
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        post_response_worker=cast(Any, worker),
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
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
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


def test_build_memory_runtime_uses_memory_plugin(monkeypatch, tmp_path: Path):
    import bootstrap.memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "register_memory_meta_tools",
        lambda *args, **kwargs: None,
    )

    captured: dict[str, object] = {}

    class _CustomEngine:
        def describe(self):
            return SimpleNamespace(name="custom")

    class _CustomPlugin:
        plugin_id = "custom"

        def build(self, deps):
            captured["deps"] = deps
            return MemoryPluginRuntime(engine=cast(Any, _CustomEngine()))

    monkeypatch.setattr(
        "bootstrap.wiring.resolve_memory_plugin",
        lambda name: _CustomPlugin(),
    )

    runtime = build_memory_runtime(
        config=Config(
            provider="test",
            model="gpt-test",
            api_key="k",
            system_prompt="hi",
            memory=MemoryConfig(enabled=True, engine="custom"),
        ),
        workspace=tmp_path,
        tools=ToolRegistry(),
        provider=cast(Any, SimpleNamespace()),
        light_provider=None,
        http_resources=cast(Any, SimpleNamespace(external_default=SimpleNamespace())),
    )

    assert runtime.engine is not None
    assert runtime.engine.describe().name == "custom"
    deps = captured["deps"]
    assert deps.config.model == "gpt-test"
    assert deps.workspace == tmp_path
    assert deps.http_resources is not None


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

    class _PostResponseMemoryWorker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("agent.memory.MemoryStore", _MemoryStore)
    monkeypatch.setattr("agent.skills.SkillsLoader", _SkillsLoader)
    monkeypatch.setattr("agent.tools.memorize.MemorizeTool", _MemorizeTool)
    monkeypatch.setattr("agent.tools.filesystem.WriteFileTool", _WriteFileTool)
    monkeypatch.setattr("agent.tools.filesystem.EditFileTool", _EditFileTool)
    monkeypatch.setattr("memory2.store.MemoryStore2", _Store2)
    monkeypatch.setattr("memory2.embedder.Embedder", _Embedder)
    monkeypatch.setattr("memory2.memorizer.Memorizer", _Memorizer)
    monkeypatch.setattr("memory2.retriever.Retriever", _Retriever)
    monkeypatch.setattr("memory2.procedure_tagger.ProcedureTagger", _ProcedureTagger)

    runtime = build_memory_runtime(
        config=Config(
            provider="test",
            model="gpt-test",
            api_key="k",
            system_prompt="hi",
            memory=MemoryConfig(enabled=True),
        ),
        workspace=tmp_path,
        tools=ToolRegistry(),
        provider=cast(Any, SimpleNamespace()),
        light_provider=None,
        http_resources=cast(Any, SimpleNamespace(external_default=SimpleNamespace())),
    )

    assert runtime.engine is not None
    assert runtime.engine.describe().name == "default"
    assert MemoryCapability.SEMANTICS_RICH_MEMORY in runtime.engine.describe().capabilities
