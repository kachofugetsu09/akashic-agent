from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from agent.control.models import TurnRequest
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.control.turn_scope import get_current_turn_scope
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bus.events import InboundMessage
from session.store import SessionStore


async def _wait_until(predicate: Any, *, attempts: int = 200) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


async def _loaded_runtime(
    tmp_path: Path,
    execute: Any,
) -> tuple[SessionStore, ConversationRuntime, PluginManager, list[InboundMessage]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")
    conversation = ConversationRuntime(store, execute)
    delivered: list[InboundMessage] = []

    async def publish(item: InboundMessage) -> None:
        delivered.append(item)

    manager = PluginManager(
        plugin_dirs=[Path(__file__).resolve().parents[1] / "plugins" / "subagent"],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(validate_semantic_schema=False),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
    )
    manager.bind_continuation_publisher(publish)
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None
    assert snapshot.tool_registry is not None
    assert {"spawn", "spawn_manage"}.issubset(
        snapshot.tool_registry.get_registered_names()
    )
    return store, conversation, manager, delivered


async def _execute_tool(
    manager: PluginManager,
    name: str,
    arguments: dict[str, object],
    *,
    turn_id: str,
    origin_channel: str = "",
    origin_chat_id: str = "",
) -> str:
    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.tool_registry is not None
    lease = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    snapshot.tool_registry.set_context(
        turn_id=turn_id,
        origin_channel=origin_channel,
        origin_chat_id=origin_chat_id,
    )
    try:
        result = await snapshot.tool_registry.execute(
            name, arguments, raise_errors=True
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
    assert isinstance(result, str)
    return result


@pytest.mark.asyncio
async def test_subagent_profiles_freeze_exact_tools_and_task_roots(
    tmp_path: Path,
) -> None:
    observed: list[object] = []

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        scope = get_current_turn_scope()
        assert scope is not None
        observed.append(scope)
        return ControlExecutionResult(response="done")

    store, conversation, manager, _ = await _loaded_runtime(tmp_path, execute)
    snapshot = manager.current_snapshot
    assert snapshot is not None
    for profile in ("research", "scripting", "general"):
        result = await _execute_tool(
            manager,
            "spawn",
            {"task": f"inspect {profile}", "profile": profile},
            turn_id=f"parent:{profile}",
        )
        assert "done" in result

    research, scripting, general = observed
    assert research.tool_overrides == {}
    assert set(scripting.tool_overrides) == {
        "write_file",
        "edit_file",
        "shell",
        "write_stdin",
        "task_stop",
    }
    assert scripting.tool_overrides["shell"]._allow_network is False
    assert general.tool_overrides["shell"]._allow_network is True
    scripting_root = scripting.tool_overrides["write_file"]._allowed_dir
    general_root = general.tool_overrides["write_file"]._allowed_dir
    assert scripting_root.parent == tmp_path / "workspace" / "subagent-runs"
    assert general_root.parent == tmp_path / "workspace" / "subagent-runs"
    assert scripting_root != general_root
    assert snapshot.lease_count == 0
    await manager.terminate_all()
    await conversation.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_background_completion_is_exactly_once_and_releases_lease(
    tmp_path: Path,
) -> None:
    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult(response="fixture-result")

    store, conversation, manager, delivered = await _loaded_runtime(tmp_path, execute)
    snapshot = manager.current_snapshot
    assert snapshot is not None
    receipt = await _execute_tool(
        manager,
        "spawn",
        {
            "task": "complete a bounded four-step fixture investigation",
            "label": "fixture",
            "profile": "research",
            "run_in_background": True,
        },
        turn_id="parent:success",
        origin_channel="web",
        origin_chat_id="chat-1",
    )
    assert "job_id=" in receipt
    await _wait_until(lambda: len(delivered) == 1 and snapshot.lease_count == 0)
    assert delivered[0].channel == "web"
    assert delivered[0].chat_id == "chat-1"
    assert "fixture-result" in delivered[0].content
    await asyncio.sleep(0.05)
    assert len(delivered) == 1
    assert snapshot.lease_count == 0
    await manager.terminate_all()
    await conversation.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_cancel_announces_before_interrupt_and_never_late_succeeds(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    store, conversation, manager, delivered = await _loaded_runtime(tmp_path, execute)
    snapshot = manager.current_snapshot
    assert snapshot is not None
    receipt = await _execute_tool(
        manager,
        "spawn",
        {
            "task": "run a bounded fixture until cancellation is requested",
            "label": "cancel",
            "run_in_background": True,
        },
        turn_id="parent:cancel",
        origin_channel="web",
        origin_chat_id="chat-2",
    )
    await started.wait()
    match = re.search(r"job_id=([0-9a-f]+)", receipt)
    assert match is not None
    result = await _execute_tool(
        manager,
        "spawn_manage",
        {"action": "cancel", "job_id": match.group(1)},
        turn_id="parent:cancel",
    )
    assert "cancel_requested" in result
    assert len(delivered) == 1
    assert "已取消" in delivered[0].content
    await _wait_until(lambda: snapshot.lease_count == 0)
    await asyncio.sleep(0.05)
    assert len(delivered) == 1
    assert "fixture-result" not in delivered[0].content
    assert snapshot.lease_count == 0
    await manager.terminate_all()
    await conversation.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_capacity_rejects_fourth_child_without_creating_turn(
    tmp_path: Path,
) -> None:
    blocker = asyncio.Event()

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        await blocker.wait()
        return ControlExecutionResult(response="released")

    store, conversation, manager, _ = await _loaded_runtime(tmp_path, execute)
    receipts: list[str] = []
    try:
        for index in range(3):
            receipts.append(
                await _execute_tool(
                    manager,
                    "spawn",
                    {
                        "task": f"complete bounded fixture investigation number {index}",
                        "run_in_background": True,
                    },
                    turn_id=f"parent:{index}",
                    origin_channel="web",
                    origin_chat_id="chat-capacity",
                )
            )
        rejected = await _execute_tool(
            manager,
            "spawn",
            {
                "task": "complete bounded fixture investigation number fourth",
                "run_in_background": True,
            },
            turn_id="parent:fourth",
            origin_channel="web",
            origin_chat_id="chat-capacity",
        )
        assert "上限 3" in rejected
        assert len(store.list_sessions()) == 3
    finally:
        blocker.set()
    await _wait_until(lambda: manager.current_snapshot.lease_count == 0)
    await manager.terminate_all()
    await conversation.shutdown()
    store.close()
