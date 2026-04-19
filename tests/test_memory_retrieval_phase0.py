import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from bootstrap import app as bootstrap_app
from bootstrap.proactive import build_proactive_runtime
from agent.config import Config
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps
from agent.looping.memory_gate import _update_session_runtime_metadata
from agent.memory import MemoryStore
from agent.provider import LLMResponse
from agent.retrieval.default_pipeline import (
    _build_injection_payload,
    _retrieve_episodic_items,
)
from agent.tools.base import Tool
from agent.tools.memorize import MemorizeTool
from agent.tools.update_now import UpdateNowTool
from agent.tools.registry import ToolRegistry
from bootstrap.tools import _build_loop_deps
from core.memory.port import DefaultMemoryPort
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from core.memory.engine import MemoryEngineRetrieveResult, MemoryHit, RememberResult
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


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def _dump_toml(data: dict, prefix: tuple[str, ...] = ()) -> list[str]:
    lines: list[str] = []
    scalar_lines: list[str] = []

    for key, value in data.items():
        if isinstance(value, dict):
            continue
        scalar_lines.append(f"{key} = {_toml_value(value)}")

    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    lines.extend(scalar_lines)
    if scalar_lines:
        lines.append("")

    for key, value in data.items():
        if isinstance(value, dict):
            lines.extend(_dump_toml(value, prefix + (key,)))
    return lines


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text("\n".join(_dump_toml(payload)).strip() + "\n", encoding="utf-8")


def test_memory_v2_top_k_history_compat_from_legacy_fields(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
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
    cfg_path = tmp_path / "config.toml"
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


def test_memory_v2_reads_embed_endpoint_fields(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_config(
        cfg_path,
        {
            "provider": "openai",
            "model": "x",
            "api_key": "main-key",
            "system_prompt": "s",
            "memory_v2": {
                "enabled": True,
                "embed_model": "embed-x",
                "api_key": "embed-key",
                "base_url": "https://embed.example.com/v1",
            },
        },
    )
    cfg = Config.load(cfg_path)
    assert cfg.memory_v2.embed_model == "embed-x"
    assert cfg.memory_v2.api_key == "embed-key"
    assert cfg.memory_v2.base_url == "https://embed.example.com/v1"


def test_memory_reads_grouped_layout_fields(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_config(
        cfg_path,
        {
            "llm": {
                "provider": "openai",
                "main": {
                    "model": "x",
                    "api_key": "main-key",
                },
            },
            "agent": {
                "system_prompt": "s",
            },
            "memory": {
                "enabled": True,
                "embedding": {
                    "model": "embed-x",
                    "api_key": "embed-key",
                    "base_url": "https://embed.example.com/v1",
                },
                "retrieval": {
                    "top_k_history": 10,
                    "score_threshold": 0.5,
                    "thresholds": {
                        "procedure": 0.61,
                        "event": 0.71,
                    },
                    "inject": {
                        "max_chars": 2222,
                    },
                    "route_intention": True,
                },
                "gate": {
                    "llm_timeout_ms": 1234,
                    "max_tokens": 88,
                },
                "hyde": {
                    "enabled": True,
                    "timeout_ms": 3456,
                },
            },
        },
    )
    cfg = Config.load(cfg_path)
    assert cfg.memory_v2.enabled is True
    assert cfg.memory_v2.embed_model == "embed-x"
    assert cfg.memory_v2.api_key == "embed-key"
    assert cfg.memory_v2.base_url == "https://embed.example.com/v1"
    assert cfg.memory_v2.top_k_history == 10
    assert cfg.memory_v2.score_threshold == 0.5
    assert cfg.memory_v2.score_threshold_procedure == 0.61
    assert cfg.memory_v2.score_threshold_event == 0.71
    assert cfg.memory_v2.inject_max_chars == 2222
    assert cfg.memory_v2.route_intention_enabled is True
    assert cfg.memory_v2.gate_llm_timeout_ms == 1234
    assert cfg.memory_v2.gate_max_tokens == 88
    assert cfg.memory_v2.hyde_enabled is True
    assert cfg.memory_v2.hyde_timeout_ms == 3456


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
        tools_used=["web_search", "update_now"],
        tool_chain=[{"calls": [{"name": "a"}, {"name": "b"}]}],
    )

    assert session.metadata["last_turn_tool_calls_count"] == 2
    assert session.metadata["last_turn_had_task_tool"] is True
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
async def test_memorize_tool_uses_engine_remember_without_scope():
    engine = MagicMock()
    engine.remember = AsyncMock(
        return_value=RememberResult(item_id="mem-1", actual_type="procedure")
    )
    tool = MemorizeTool(cast(Any, engine))

    result = await tool.execute(
        summary="以后先查工具状态",
        memory_type="procedure",
        tool_requirement="update_now",
        steps=["先查", "再执行"],
    )

    engine.remember.assert_awaited_once()
    request = engine.remember.await_args.args[0]
    assert request.summary == "以后先查工具状态"
    assert request.memory_type == "procedure"
    assert request.raw_extra == {
        "tool_requirement": "update_now",
        "steps": ["先查", "再执行"],
    }
    assert "已记住" in result


@pytest.mark.asyncio
async def test_memorize_tool_uses_engine_remember():
    engine = MagicMock()
    engine.remember = AsyncMock(
        return_value=RememberResult(item_id="engine-1", actual_type="procedure")
    )
    tool = MemorizeTool(cast(Any, engine))

    result = await tool.execute(
        summary="以后先查工具状态",
        memory_type="procedure",
        tool_requirement="update_now",
        steps=["先查", "再执行"],
        channel="cli",
        chat_id="1",
    )

    engine.remember.assert_awaited_once()
    request = engine.remember.await_args.args[0]
    assert request.scope.session_key == "cli:1"
    assert request.scope.channel == "cli"
    assert request.scope.chat_id == "1"
    assert "engine-1" in result


@pytest.mark.asyncio
async def test_memorize_tool_uses_current_user_source_ref_by_default():
    engine = MagicMock()
    engine.remember = AsyncMock(
        return_value=RememberResult(item_id="engine-3", actual_type="preference")
    )
    tool = MemorizeTool(cast(Any, engine))

    await tool.execute(
        summary="用户并不反感星露谷物语",
        memory_type="preference",
        current_user_source_ref="telegram:test_user:9999",
    )

    request = engine.remember.await_args.args[0]
    assert request.source_ref == "telegram:test_user:9999"


@pytest.mark.asyncio
async def test_memorize_tool_without_any_source_ref_does_not_fall_back_to_memorize_tool():
    engine = MagicMock()
    engine.remember = AsyncMock(
        return_value=RememberResult(item_id="engine-4", actual_type="preference")
    )
    tool = MemorizeTool(cast(Any, engine))

    await tool.execute(
        summary="用户喜欢机制透明的游戏",
        memory_type="preference",
    )

    request = engine.remember.await_args.args[0]
    assert request.source_ref == ""


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


def test_phase0_loop_wiring_keeps_split_memory_entrypoints(tmp_path: Path):
    config = Config(
        provider="openai",
        model="test-model",
        api_key="test-key",
        system_prompt="test system prompt",
    )
    memory_port = MagicMock()
    profile_maint = MagicMock()
    memory_engine = MagicMock()
    post_mem_worker = MagicMock()
    runtime = MemoryRuntime(
        port=cast(Any, memory_port),
        engine=cast(Any, memory_engine),
        profile_maint=cast(Any, profile_maint),
        post_response_worker=post_mem_worker,
    )

    deps = _build_loop_deps(
        config=config,
        workspace=tmp_path,
        bus=MagicMock(),
        provider=cast(Any, _FakeProvider()),
        light_provider=None,
        tools=ToolRegistry(),
        session_manager=MagicMock(),
        presence=MagicMock(),
        processing_state=MagicMock(),
        memory_runtime=runtime,
        observe_writer=None,
    )

    assert deps.memory_services.engine is memory_engine
    assert deps.memory_services.facade is runtime.facade
    assert deps.post_turn_pipeline._engine is memory_engine
    assert deps.consolidation_service._memory_port is memory_port
    assert deps.consolidation_service._profile_maint is profile_maint


def test_phase0_proactive_runtime_keeps_memory_port_entrypoint(tmp_path: Path):
    config = Config(
        provider="openai",
        model="test-model",
        api_key="test-key",
        system_prompt="test system prompt",
    )
    config.proactive.enabled = True
    facade = MagicMock()
    session_manager = SimpleNamespace(workspace=tmp_path)
    agent_loop = SimpleNamespace(
        workspace=tmp_path,
        processing_state=SimpleNamespace(is_busy=MagicMock(return_value=False)),
    )

    tasks, proactive_loop = build_proactive_runtime(
        config,
        tmp_path,
        session_manager=session_manager,
        provider=cast(Any, _FakeProvider()),
        light_provider=None,
        push_tool=MagicMock(),
        memory_store=facade,
        presence=MagicMock(),
        agent_loop=agent_loop,
        observe_writer=None,
    )

    assert len(tasks) == 1
    assert proactive_loop is not None
    assert proactive_loop._memory is facade
    for task in tasks:
        close = getattr(task, "close", None)
        if callable(close):
            close()


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


def test_retriever_internal_select_for_injection_applies_type_threshold():
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

    selected = retriever._select_for_injection(items)
    ids = {i["id"] for i in selected}
    assert "a" in ids
    assert "c" in ids
    assert "b" not in ids
    assert "d" not in ids


def test_retriever_internal_select_for_injection_keeps_protected_procedure():
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
            "extra_json": {"tool_requirement": "update_now"},
        },
        {"id": "e1", "memory_type": "event", "score": 0.75, "summary": "普通历史"},
    ]

    selected = retriever._select_for_injection(items)
    ids = {i["id"] for i in selected}
    assert "p1" in ids


def test_retriever_internal_select_for_injection_can_drop_protected_when_guard_disabled():
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
            "extra_json": {"tool_requirement": "update_now"},
        },
    ]

    selected = retriever._select_for_injection(items)
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
    block, injected_ids = retriever.build_injection_block(items)
    assert "规则1" in block
    assert "规则2" not in block
    assert "有印象，不确定" not in block
    assert injected_ids == ["p1"]


def test_retriever_marks_borderline_items_as_uncertain():
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
        high_inject_delta=0.15,
    )
    items = [
        {
            "id": "e1",
            "memory_type": "event",
            "score": 0.74,
            "summary": "用户提过一次 Fitbit 相关情况",
        },
    ]

    block, injected_ids = retriever.build_injection_block(items)
    assert "有印象，不确定" in block
    assert injected_ids == ["e1"]


def test_retriever_keeps_full_summary_without_single_item_truncation():
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
        inject_max_chars=2000,
    )
    summary = "这是一段很长的历史摘要" + "A" * 220 + "结尾保留"
    items = [
        {
            "id": "e1",
            "memory_type": "event",
            "score": 0.9,
            "summary": summary,
        },
    ]

    block, injected_ids = retriever.build_injection_block(items)
    assert summary in block
    assert "…" not in block
    assert injected_ids == ["e1"]


def test_retriever_build_injection_block_empty_input_returns_tuple_duplicate_guard():
    retriever = Retriever(store=MagicMock(), embedder=MagicMock())
    block, injected_ids = retriever.build_injection_block([])
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
                        source_ref="telegram:test_user@seed",
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

    items, scope_mode, hyde, engine_result = await _retrieve_episodic_items(
        session_key="telegram:test_user",
        channel="telegram",
        chat_id="test_user",
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
    assert engine_result is not None
    request = engine.retrieve.await_args.args[0]
    assert request.scope.session_key == "telegram:test_user"
    assert request.hints["require_scope_match"] is True
    assert request.hints["memory_types"] == ["event", "profile"]


@pytest.mark.asyncio
async def test_retrieve_episodic_items_returns_disabled_without_engine():
    memory = SimpleNamespace(
        port=MagicMock(),
        engine=None,
        hyde_enhancer=None,
    )

    items, scope_mode, hyde, engine_result = await _retrieve_episodic_items(
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

    assert items == []
    assert scope_mode == "disabled"
    assert hyde is None
    assert engine_result is None


@pytest.mark.asyncio
async def test_retrieve_episodic_items_uses_engine_hyde_path_when_enabled():
    engine = SimpleNamespace(
        retrieve=AsyncMock(
            side_effect=[
                MemoryEngineRetrieveResult(
                    text_block="raw block",
                    hits=[
                        MemoryHit(
                            id="h1",
                            summary="原始命中",
                            content="原始命中",
                            score=0.8,
                            source_ref="seed",
                            engine_kind="default",
                            metadata={"memory_type": "event"},
                            injected=True,
                        )
                    ],
                    raw={"items": [{"id": "h1", "memory_type": "event", "summary": "原始命中"}]},
                ),
                MemoryEngineRetrieveResult(
                    text_block="hyde block",
                    hits=[
                        MemoryHit(
                            id="h2",
                            summary="HyDE 命中",
                            content="HyDE 命中",
                            score=0.79,
                            source_ref="seed",
                            engine_kind="default",
                            metadata={"memory_type": "event"},
                            injected=True,
                        )
                    ],
                    raw={"items": [{"id": "h2", "memory_type": "event", "summary": "HyDE 命中"}]},
                ),
            ]
        )
    )
    hyde = SimpleNamespace(generate_hypothesis=AsyncMock(return_value="hypo"))
    memory = SimpleNamespace(
        port=MagicMock(),
        engine=engine,
        hyde_enhancer=hyde,
    )

    items, scope_mode, hyde_hypothesis, engine_result = await _retrieve_episodic_items(
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

    assert [item["id"] for item in items] == ["h1", "h2"]
    assert [item["_retrieval_path"] for item in items] == ["history_raw", "history_hyde"]
    assert scope_mode == "global+hyde"
    assert hyde_hypothesis == "hypo"
    assert engine_result is not None
    assert engine_result.text_block == "raw block\n\nhyde block"
    assert engine.retrieve.await_count == 2


@pytest.mark.asyncio
async def test_retrieve_episodic_items_hyde_max_pools_same_id_to_higher_score():
    engine = SimpleNamespace(
        retrieve=AsyncMock(
            side_effect=[
                MemoryEngineRetrieveResult(
                    text_block="raw block",
                    hits=[
                        MemoryHit(
                            id="h1",
                            summary="原始摘要",
                            content="原始摘要",
                            score=0.6,
                            source_ref="seed",
                            engine_kind="default",
                            metadata={"memory_type": "event"},
                            injected=False,
                        )
                    ],
                    raw={
                        "items": [
                            {
                                "id": "h1",
                                "memory_type": "event",
                                "summary": "原始摘要",
                                "score": 0.6,
                            }
                        ]
                    },
                ),
                MemoryEngineRetrieveResult(
                    text_block="hyde block",
                    hits=[
                        MemoryHit(
                            id="h1",
                            summary="HyDE 更优摘要",
                            content="HyDE 更优摘要",
                            score=0.9,
                            source_ref="seed",
                            engine_kind="default",
                            metadata={"memory_type": "event"},
                            injected=True,
                        )
                    ],
                    raw={
                        "items": [
                            {
                                "id": "h1",
                                "memory_type": "event",
                                "summary": "HyDE 更优摘要",
                                "score": 0.9,
                            }
                        ]
                    },
                ),
            ]
        )
    )
    hyde = SimpleNamespace(generate_hypothesis=AsyncMock(return_value="hypo"))
    memory = SimpleNamespace(
        port=MagicMock(),
        engine=engine,
        hyde_enhancer=hyde,
    )

    items, _scope_mode, _hyde_hypothesis, engine_result = await _retrieve_episodic_items(
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

    assert len(items) == 1
    assert items[0]["summary"] == "HyDE 更优摘要"
    assert items[0]["_retrieval_path"] == "history_hyde"
    assert engine_result is not None
    assert len(engine_result.hits) == 1
    assert engine_result.hits[0].summary == "HyDE 更优摘要"
    assert engine_result.hits[0].injected is True


@pytest.mark.asyncio
async def test_retrieve_episodic_items_returns_engine_managed_injection():
    engine = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=MemoryEngineRetrieveResult(
                text_block="engine block",
                hits=[
                    MemoryHit(
                        id="e1",
                        summary="用户昨天提过 FitBit",
                        content="用户昨天提过 FitBit",
                        score=0.81,
                        source_ref="telegram:test_user@seed",
                        engine_kind="default",
                        metadata={"memory_type": "event", "origin": "engine"},
                        injected=True,
                    ),
                    MemoryHit(
                        id="e2",
                        summary="未注入候选",
                        content="未注入候选",
                        score=0.75,
                        source_ref="telegram:test_user@seed",
                        engine_kind="default",
                        metadata={"memory_type": "event"},
                        injected=False,
                    ),
                ],
                raw={
                    "items": [
                        {"id": "e1", "memory_type": "event", "summary": "用户昨天提过 FitBit"},
                        {"id": "e2", "memory_type": "event", "summary": "未注入候选"},
                    ]
                },
            )
        )
    )
    memory = SimpleNamespace(
        port=MagicMock(),
        engine=engine,
        hyde_enhancer=None,
    )

    items, scope_mode, hyde, engine_result = await _retrieve_episodic_items(
        session_key="telegram:test_user",
        channel="telegram",
        chat_id="test_user",
        route_decision="RETRIEVE",
        rewritten_query="Fitbit 型号",
        history_memory_types=["event", "profile"],
        hyde_context="recent turns",
        memory=memory,
        config=AgentLoopConfig().memory,
    )

    assert scope_mode == "global"
    assert hyde is None
    assert len(items) == 2
    assert engine_result is not None
    assert engine_result.text_block == "engine block"
    assert [item.id for item in engine_result.hits if item.injected] == ["e1"]


def test_build_injection_payload_uses_engine_result_text_block_and_injected_ids():
    procedure_result = MemoryEngineRetrieveResult(
        text_block="procedure block",
        hits=[
            MemoryHit(
                id="p1",
                summary="先查状态",
                content="先查状态",
                score=0.9,
                source_ref="seed",
                engine_kind="default",
                metadata={"memory_type": "procedure"},
                injected=True,
            )
        ],
    )
    history_result = MemoryEngineRetrieveResult(
        text_block="engine block",
        hits=[
            MemoryHit(
                id="e1",
                summary="用户昨天提过 FitBit",
                content="用户昨天提过 FitBit",
                score=0.81,
                source_ref="seed",
                engine_kind="compat",
                metadata={"memory_type": "event"},
                injected=True,
            ),
            MemoryHit(
                id="e2",
                summary="未注入候选",
                content="未注入候选",
                score=0.75,
                source_ref="seed",
                engine_kind="compat",
                metadata={"memory_type": "event"},
                injected=False,
            ),
        ],
    )

    selected_items, block, injected_ids = _build_injection_payload(
        procedure_items=[{"id": "p1", "memory_type": "procedure", "summary": "先查状态"}],
        procedure_result=procedure_result,
        history_items=[
            {"id": "e1", "memory_type": "event", "summary": "用户昨天提过 FitBit"},
            {"id": "e2", "memory_type": "event", "summary": "未注入候选"},
        ],
        history_result=history_result,
    )

    assert block == "procedure block\n\nengine block"
    assert injected_ids == ["p1", "e1"]
    assert [item["id"] for item in selected_items] == ["p1", "e1"]


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
    block, injected_ids = retriever.build_injection_block(items)
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
    block, injected_ids = retriever.build_injection_block(items)
    assert "【强制约束】" in block
    assert "p1" in injected_ids


def test_retriever_build_injection_block_empty_input_returns_tuple():
    retriever = Retriever(store=MagicMock(), embedder=MagicMock())
    block, injected_ids = retriever.build_injection_block([])
    assert block == ""
    assert injected_ids == []
