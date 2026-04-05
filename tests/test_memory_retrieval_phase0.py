import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from bootstrap import app as bootstrap_app
from agent.config import Config
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps
from agent.looping.memory_gate import _update_session_runtime_metadata
from agent.memory import MemoryStore
from agent.provider import LLMResponse
from agent.retrieval.default_pipeline import _retrieve_episodic_items
from agent.tools.base import Tool
from agent.tools.memorize import MemorizeTool
from agent.tools.update_now import UpdateNowTool
from agent.tools.registry import ToolRegistry
from core.memory.port import DefaultMemoryPort
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from core.memory.engine import MemoryEngineRetrieveResult, MemoryHit
from memory2.retriever import Retriever
from session.manager import Session


class _NoopTool(Tool):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> str:
        return "ok"


class _FakeProvider:
    async def chat(self, **kwargs):
        return LLMResponse(content="ok", tool_calls=[])


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_memory_v2_top_k_history_compat_from_legacy_fields(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    _write_config(
        cfg_path,
        {
            "provider": "openai",
            "model": "x",
            "api_key": "k",
            "system_prompt": "s",
            "memory_v2": {
                "enabled": True,
                "retrieve_top_k": 9,
                "score_threshold": 0.5,
            },
        },
    )
    with pytest.warns(DeprecationWarning, match=r"memory_v2\.retrieve_top_k"):
        cfg = Config.load(cfg_path)
    assert cfg.memory_v2.top_k_history == 9
    assert cfg.memory_v2.retrieve_top_k == 9
    assert cfg.memory_v2.score_threshold_procedure == 0.60
    assert cfg.memory_v2.score_threshold_event == 0.68


def test_memory_v2_top_k_history_prefers_new_field(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    _write_config(
        cfg_path,
        {
            "provider": "openai",
            "model": "x",
            "api_key": "k",
            "system_prompt": "s",
            "memory_v2": {
                "enabled": True,
                "top_k_history": 12,
                "recall_top_k": 7,
                "retrieve_top_k": 5,
            },
        },
    )
    with pytest.warns(
        DeprecationWarning, match=r"memory_v2\.(recall_top_k|retrieve_top_k)"
    ):
        cfg = Config.load(cfg_path)
    assert cfg.memory_v2.top_k_history == 12
    assert cfg.memory_v2.retrieve_top_k == 12


def test_loop_updates_session_runtime_metadata(tmp_path: Path):
    tools = ToolRegistry()
    tools.register(_NoopTool())
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, _FakeProvider()),
            tools=tools,
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
        ),
        AgentLoopConfig(),
    )
    session = Session("telegram:1")

    _update_session_runtime_metadata(
        session,
        tools_used=["web_search", "task_note", "update_now"],
        tool_chain=[{"calls": [{"name": "a"}, {"name": "b"}]}],
    )

    assert session.metadata["last_turn_tool_calls_count"] == 2
    assert session.metadata["last_turn_had_task_tool"] is True
    assert "task_note" in session.metadata["recent_task_tools"]
    assert "update_now" in session.metadata["recent_task_tools"]
    assert isinstance(session.metadata.get("last_turn_ts"), str)

    _update_session_runtime_metadata(
        session,
        tools_used=["web_search"],
        tool_chain=[{"calls": [{"name": "c"}]}],
    )

    assert session.metadata["last_turn_tool_calls_count"] == 1
    assert isinstance(session.metadata.get("_task_tools_turns"), list)
    assert len(session.metadata["_task_tools_turns"]) <= 2


@pytest.mark.asyncio
async def test_update_now_tool_uses_memory_port():
    memory = MagicMock()
    tool = UpdateNowTool(cast(Any, memory))

    result = await tool.execute(add='["任务A"]', remove_keywords=["旧任务"])

    memory.update_now_ongoing.assert_called_once_with(
        add=["任务A"],
        remove_keywords=["旧任务"],
    )
    assert "NOW.md 已更新" in result


@pytest.mark.asyncio
async def test_memorize_tool_uses_memory_port():
    memory = MagicMock()
    memory.save_item_with_supersede = AsyncMock(return_value="mem-1")
    tool = MemorizeTool(cast(Any, memory))

    result = await tool.execute(
        summary="以后先查工具状态",
        memory_type="procedure",
        tool_requirement="task_note",
        steps=["先查", "再执行"],
    )

    memory.save_item_with_supersede.assert_awaited_once_with(
        summary="以后先查工具状态",
        memory_type="procedure",
        extra={
            "tool_requirement": "task_note",
            "steps": ["先查", "再执行"],
            "rule_schema": {
                "required_tools": ["task_note"],
                "forbidden_tools": [],
                "mentioned_tools": ["task_note"],
            },
        },
        source_ref="memorize_tool",
    )
    assert "已记住" in result


def test_agent_loop_accepts_memory_runtime(tmp_path: Path):
    tools = ToolRegistry()
    tools.register(_NoopTool())
    memory_port = cast(Any, MagicMock())
    post_mem_worker = MagicMock()
    runtime = MemoryRuntime(
        port=memory_port,
        post_response_worker=post_mem_worker,
    )

    loop = AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, _FakeProvider()),
            tools=tools,
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_runtime=runtime,
        ),
        AgentLoopConfig(),
    )

    assert loop._memory_port is memory_port
    assert loop._post_mem_worker is post_mem_worker
    assert loop.context.memory is memory_port


@pytest.mark.asyncio
async def test_build_memory_runtime_v2_enabled_returns_worker_and_port(tmp_path: Path):
    config = Config(
        provider="openai",
        model="test-model",
        api_key="test-key",
        system_prompt="test system prompt",
    )
    config.memory_v2.enabled = True

    tools = ToolRegistry()
    http_resources = SharedHttpResources()
    try:
        runtime = bootstrap_app.build_memory_runtime(
            config,
            tmp_path,
            tools,
            cast(Any, MagicMock()),
            None,
            http_resources,
        )

        assert runtime.port is not None
        assert runtime.post_response_worker is not None
        schema_names = {schema["function"]["name"] for schema in tools.get_schemas()}
        assert "memorize" in schema_names

        await runtime.aclose()
    finally:
        await http_resources.aclose()


@pytest.mark.asyncio
async def test_memory_runtime_aclose_closes_resources_in_reverse_order():
    calls: list[str] = []

    class _CloseOnly:
        def close(self) -> None:
            calls.append("first")

    class _AsyncCloseOnly:
        def __init__(self) -> None:
            self.aclose = AsyncMock(side_effect=self._aclose)

        async def _aclose(self) -> None:
            calls.append("second")

    runtime = MemoryRuntime(
        port=cast(Any, MagicMock()),
        closeables=[_CloseOnly(), _AsyncCloseOnly()],
    )
    await runtime.aclose()

    assert calls == ["second", "first"]


@pytest.mark.asyncio
async def test_memory_runtime_aclose_continues_after_failure():
    calls: list[str] = []

    class _FailingAsyncClose:
        async def aclose(self) -> None:
            calls.append("failing")
            raise RuntimeError("boom")

    class _CloseOnly:
        def close(self) -> None:
            calls.append("close")

    runtime = MemoryRuntime(
        port=cast(Any, MagicMock()),
        closeables=[_CloseOnly(), _FailingAsyncClose()],
    )

    with pytest.raises(RuntimeError, match="boom"):
        await runtime.aclose()

    assert calls == ["failing", "close"]


def test_retriever_select_for_injection_applies_type_threshold_and_relative_delta():
    retriever = Retriever(
        store=MagicMock(),
        embedder=MagicMock(),
        score_threshold=0.45,
        score_thresholds={
            "procedure": 0.60,
            "preference": 0.60,
            "event": 0.68,
            "profile": 0.68,
        },
        relative_delta=0.06,
    )
    items = [
        {"id": "a", "memory_type": "event", "score": 0.74, "summary": "A"},
        {
            "id": "b",
            "memory_type": "event",
            "score": 0.67,
            "summary": "B",
        },  # 低于 event 阈值
        {"id": "c", "memory_type": "procedure", "score": 0.63, "summary": "C"},
        {
            "id": "d",
            "memory_type": "procedure",
            "score": 0.57,
            "summary": "D",
        },  # 低于 proc 阈值
    ]

    selected = retriever.select_for_injection(items)
    ids = {i["id"] for i in selected}
    assert "a" in ids
    assert "c" in ids
    assert "b" not in ids
    assert "d" not in ids


def test_retriever_select_for_injection_keeps_protected_procedure():
    retriever = Retriever(
        store=MagicMock(),
        embedder=MagicMock(),
        score_threshold=0.7,
        score_thresholds={
            "procedure": 0.7,
            "preference": 0.7,
            "event": 0.7,
            "profile": 0.7,
        },
    )
    items = [
        {
            "id": "p1",
            "memory_type": "procedure",
            "score": 0.42,
            "summary": "必须先查工具状态",
            "extra_json": {"tool_requirement": "task_note"},
        },
        {"id": "e1", "memory_type": "event", "score": 0.75, "summary": "普通历史"},
    ]

    selected = retriever.select_for_injection(items)
    ids = {i["id"] for i in selected}
    assert "p1" in ids


def test_retriever_select_for_injection_can_drop_protected_when_guard_disabled():
    retriever = Retriever(
        store=MagicMock(),
        embedder=MagicMock(),
        score_threshold=0.7,
        score_thresholds={
            "procedure": 0.7,
            "preference": 0.7,
            "event": 0.7,
            "profile": 0.7,
        },
        procedure_guard_enabled=False,
    )
    items = [
        {
            "id": "p1",
            "memory_type": "procedure",
            "score": 0.42,
            "summary": "必须先查工具状态",
            "extra_json": {"tool_requirement": "task_note"},
        },
    ]

    selected = retriever.select_for_injection(items)
    ids = {i["id"] for i in selected}
    assert "p1" not in ids


def test_retriever_forced_limit_and_injected_ids_match_formatted_output():
    retriever = Retriever(
        store=MagicMock(),
        embedder=MagicMock(),
        inject_max_forced=1,
        procedure_guard_enabled=True,
    )
    items = [
        {
            "id": "p1",
            "memory_type": "procedure",
            "score": 0.95,
            "summary": "规则1",
            "extra_json": {"tool_requirement": "a"},
        },
        {
            "id": "p2",
            "memory_type": "procedure",
            "score": 0.94,
            "summary": "规则2",
            "extra_json": {"tool_requirement": "b"},
        },
    ]
    block, injected_ids = retriever.format_injection_with_ids(items)
    assert "规则1" in block
    assert "规则2" not in block
    assert injected_ids == ["p1"]


def test_retriever_format_injection_with_ids_empty_input_returns_tuple():
    retriever = Retriever(store=MagicMock(), embedder=MagicMock())
    block, injected_ids = retriever.format_injection_with_ids([])
    assert block == ""
    assert injected_ids == []


@pytest.mark.asyncio
async def test_retrieve_episodic_items_prefers_memory_engine_when_available():
    engine = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=MemoryEngineRetrieveResult(
                text_block="",
                hits=[
                    MemoryHit(
                        id="e1",
                        summary="用户昨天提过 FitBit",
                        content="用户昨天提过 FitBit",
                        score=0.81,
                        source_ref="telegram:7674283004@seed",
                        engine_kind="default",
                        metadata={"memory_type": "event", "origin": "engine"},
                    )
                ],
            )
        )
    )
    memory = SimpleNamespace(
        port=MagicMock(),
        engine=engine,
        hyde_enhancer=None,
    )

    items, scope_mode, hyde = await _retrieve_episodic_items(
        session_key="telegram:7674283004",
        channel="telegram",
        chat_id="7674283004",
        route_decision="RETRIEVE",
        rewritten_query="Fitbit 型号",
        history_memory_types=["event", "profile"],
        hyde_context="recent turns",
        memory=memory,
        config=AgentLoopConfig().memory,
    )

    assert scope_mode == "global"
    assert hyde is None
    assert items[0]["id"] == "e1"
    assert items[0]["memory_type"] == "event"
    assert items[0]["extra_json"] == {"origin": "engine"}
    assert items[0]["_retrieval_path"] == "history_raw"
    request = engine.retrieve.await_args.args[0]
    assert request.scope.session_key == "telegram:7674283004"
    assert request.hints["require_scope_match"] is True
    assert request.hints["memory_types"] == ["event", "profile"]


@pytest.mark.asyncio
async def test_retrieve_episodic_items_falls_back_to_legacy_port_path(monkeypatch):
    mocked = AsyncMock(return_value=([{"id": "h1"}], "local", "hyde"))
    monkeypatch.setattr(
        "agent.retrieval.default_pipeline.retrieve_episodic",
        mocked,
    )
    memory = SimpleNamespace(
        port=MagicMock(),
        engine=None,
        hyde_enhancer=None,
    )

    items, scope_mode, hyde = await _retrieve_episodic_items(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        route_decision="RETRIEVE",
        rewritten_query="历史查询",
        history_memory_types=["event"],
        hyde_context="recent turns",
        memory=memory,
        config=AgentLoopConfig().memory,
    )

    assert items == [{"id": "h1"}]
    assert scope_mode == "local"
    assert hyde == "hyde"
    mocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_episodic_items_keeps_legacy_hyde_path_when_enabled(monkeypatch):
    mocked = AsyncMock(
        return_value=([{"id": "h1", "_retrieval_path": "history_hyde"}], "global+hyde", "hypo")
    )
    monkeypatch.setattr(
        "agent.retrieval.default_pipeline.retrieve_episodic",
        mocked,
    )
    engine = SimpleNamespace(retrieve=AsyncMock())
    memory = SimpleNamespace(
        port=MagicMock(),
        engine=engine,
        hyde_enhancer=object(),
    )

    items, scope_mode, hyde = await _retrieve_episodic_items(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        route_decision="RETRIEVE",
        rewritten_query="历史查询",
        history_memory_types=["event"],
        hyde_context="recent turns",
        memory=memory,
        config=AgentLoopConfig().memory,
    )

    assert items == [{"id": "h1", "_retrieval_path": "history_hyde"}]
    assert scope_mode == "global+hyde"
    assert hyde == "hypo"
    mocked.assert_awaited_once()
    engine.retrieve.assert_not_called()


def test_retriever_norm_limit_uses_config_without_hardcoded_cap():
    retriever = Retriever(
        store=MagicMock(),
        embedder=MagicMock(),
        inject_max_procedure_preference=6,
        score_threshold=0.0,
    )
    items = [
        {
            "id": f"n{i}",
            "memory_type": "preference",
            "score": 0.9 - i * 0.01,
            "summary": f"偏好{i}",
        }
        for i in range(6)
    ]
    block, injected_ids = retriever.format_injection_with_ids(items)
    for i in range(6):
        assert f"偏好{i}" in block
    assert len(injected_ids) == 6


def test_retriever_forced_block_not_dropped_by_char_budget():
    retriever = Retriever(
        store=MagicMock(),
        embedder=MagicMock(),
        inject_max_chars=120,
        inject_max_forced=1,
    )
    long_summary = "A" * 500
    items = [
        {
            "id": "p1",
            "memory_type": "procedure",
            "score": 0.9,
            "summary": long_summary,
            "extra_json": {"tool_requirement": "web_search"},
        },
        {
            "id": "e1",
            "memory_type": "event",
            "score": 0.89,
            "summary": "普通事件",
        },
    ]
    block, injected_ids = retriever.format_injection_with_ids(items)
    assert "【强制约束】" in block
    assert "p1" in injected_ids


def test_retriever_build_injection_block_matches_legacy_format_api():
    retriever = Retriever(
        store=MagicMock(),
        embedder=MagicMock(),
        score_threshold=0.0,
    )
    items = [
        {"id": "p1", "memory_type": "procedure", "score": 0.91, "summary": "规则1"},
        {"id": "e1", "memory_type": "event", "score": 0.88, "summary": "事件1"},
    ]

    block, injected_ids = retriever.build_injection_block(items)
    legacy_block, legacy_ids = retriever.format_injection_with_ids(items)

    assert block == legacy_block
    assert injected_ids == legacy_ids


def test_retriever_build_injection_block_empty_input_returns_tuple():
    retriever = Retriever(store=MagicMock(), embedder=MagicMock())
    block, injected_ids = retriever.build_injection_block([])
    assert block == ""
    assert injected_ids == []
