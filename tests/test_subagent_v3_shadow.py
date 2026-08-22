from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from agent.control.models import TurnRequest
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.control.turn_scope import get_current_turn_scope
from agent.plugin_composition import SCOPED_TURNS
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from session.store import SessionStore


@pytest.mark.asyncio
async def test_builtin_subagent_shadow_recurses_through_scoped_turn_service(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")
    observed: dict[str, object] = {}

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        scope = get_current_turn_scope()
        assert scope is not None
        observed.update(
            {
                "thread_id": request.thread_id,
                "input": request.input,
                "prompt_hints": scope.prompt_hints,
                "grant": scope.tool_grant.names,
                "memory_read": scope.memory_read,
                "memory_write": scope.memory_write,
                "stateless": scope.stateless,
                "tool_source": scope.tool_source,
            }
        )
        return ControlExecutionResult(response="child:done")

    runtime = ConversationRuntime(store, execute)
    manager = PluginManager(
        plugin_dirs=[Path(__file__).resolve().parents[1] / "plugins" / "subagent"],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(validate_semantic_schema=False),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        runtime,
        programmatic_session_creator=store.create_session,
    )
    manager.bind_continuation_publisher(lambda _item: _noop())
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None
    assert snapshot.tool_registry is not None

    lease = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    snapshot.tool_registry.set_context(turn_id="parent:turn")
    try:
        result = await snapshot.tool_registry.execute(
            "spawn",
            {"task": "inspect the fixture", "profile": "research"},
            raise_errors=True,
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    assert isinstance(result, str)
    assert "child:done" in result
    assert observed["input"] == "inspect the fixture"
    assert observed["memory_read"] is False
    assert observed["memory_write"] is False
    assert observed["stateless"] is True
    assert observed["tool_source"] == "subagent"
    assert observed["grant"] == frozenset(
        {"read_file", "list_dir", "web_fetch", "web_search"}
    )
    assert "调研型子 agent" in cast(tuple[str, ...], observed["prompt_hints"])[0]
    assert len(list((workspace / "subagent-runs").iterdir())) == 1
    trace = workspace / "memory" / "spawn_trace.jsonl"
    assert trace.read_text(encoding="utf-8").count("\n") == 2

    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_subagent_candidate_service_denies_child_turns(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")

    async def execute(_request: TurnRequest) -> str:
        raise AssertionError("candidate must not create a child Turn")

    runtime = ConversationRuntime(store, execute)
    manager = PluginManager(
        plugin_dirs=[Path(__file__).resolve().parents[1] / "plugins" / "subagent"],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        runtime,
        programmatic_session_creator=store.create_session,
    )
    manager.bind_continuation_publisher(lambda _item: _noop())
    await manager.load_all()
    candidate = await manager.prepare_candidate("subagent")
    assert candidate is not None
    assert candidate.runtime_snapshot is not None
    root = candidate.runtime_snapshot.composition_root
    assert root is not None
    service = root.context.require(SCOPED_TURNS)
    with pytest.raises(RuntimeError, match="candidate 验证期"):
        await service.create_session(metadata={"source": "subagent"})

    assert store.list_sessions() == []
    await manager.discard_prepared("subagent")
    await runtime.shutdown()
    store.close()


async def _noop() -> None:
    return None
