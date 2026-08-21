from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent.control.context import running_turn_id
from agent.core.response_parser import ResponseMetadata
from agent.lifecycle.composition import (
    AFTER_REASONING_PREPROCESS_EVENT,
    run_composition_lifecycle,
)
from agent.lifecycle.types import AfterReasoningCtx
from agent.plugin_composition import MemoryTurnRuntime
from agent.plugins.manager import PluginManager
from agent.plugins.mobile_ui import PluginMobileUiProvider
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from bus.event_bus import EventBus
from core.memory.plugin import ActiveRecallRecord, ActiveRecallView
from plugins.akasha.plugin import _persist_feedback


class _MemoryTurnEngine:
    def __init__(self, name: str = "akasha") -> None:
        self.name = name
        self.metadata_calls = 0
        self.recall_calls = 0

    def describe(self) -> object:
        return SimpleNamespace(name=self.name)

    def take_turn_user_metadata(self, turn_id: str) -> dict[str, object]:
        self.metadata_calls += 1
        return {"marker": {"turn_id": turn_id}}

    def wait_active_recall(
        self,
        session_key: str,
        turn_id: str,
    ) -> ActiveRecallView:
        self.recall_calls += 1
        return ActiveRecallView(
            query_id=f"{session_key}:{turn_id}",
            dense=(ActiveRecallRecord("user", "assistant", "now", 0.8),),
            completion=(),
        )


def test_memory_turn_runtime_copies_metadata_and_reads_bounded_recall() -> None:
    engine = _MemoryTurnEngine()
    runtime = MemoryTurnRuntime(engine)

    metadata = runtime.take_user_metadata("turn-1")
    recall = runtime.wait_active_recall("session-1", "turn-1")

    assert metadata == {"marker": {"turn_id": "turn-1"}}
    with pytest.raises(TypeError):
        metadata["marker"] = {}  # pyright: ignore[reportIndexIssue]
    assert recall is not None and recall.query_id == "session-1:turn-1"
    assert engine.metadata_calls == 1
    assert engine.recall_calls == 1


def test_candidate_memory_turn_runtime_rejects_formal_state_access() -> None:
    runtime = MemoryTurnRuntime.candidate_validation()

    assert not runtime.formal
    with pytest.raises(RuntimeError, match="candidate 验证期"):
        runtime.take_user_metadata("turn-1")
    with pytest.raises(RuntimeError, match="candidate 验证期"):
        runtime.wait_active_recall("session-1", "turn-1")


def test_akasha_feedback_moves_into_current_user_metadata_once() -> None:
    engine = _MemoryTurnEngine()
    event = AfterReasoningCtx(
        session_key="session-1",
        channel="test",
        chat_id="chat-1",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="reply"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="reply",
    )
    token = running_turn_id.set("turn-1")
    try:
        _persist_feedback(event, MemoryTurnRuntime(engine))
    finally:
        running_turn_id.reset(token)

    assert event.persist_user_metadata == {"marker": {"turn_id": "turn-1"}}
    assert engine.metadata_calls == 1


def _write_plugin(path: Path, *, version: str, probe_candidate: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    condition = (
        "'plugin-validation' in str(ctx.runtime.workspace)"
        if probe_candidate
        else "False"
    )
    (path / "plugin.py").write_text(
        "from agent.plugin_composition import MEMORY_TURN_RUNTIME\n"
        "api_version = 3\n"
        "name = 'memory_probe'\n"
        f"version = '{version}'\n"
        "inject = (MEMORY_TURN_RUNTIME,)\n"
        "bound = None\n"
        "async def apply(ctx, config):\n"
        "    global bound\n"
        "    bound = ctx.require(MEMORY_TURN_RUNTIME)\n"
        f"    if {condition}:\n"
        "        bound.take_user_metadata('candidate-turn')\n",
        encoding="utf-8",
    )


def _copy_akasha_plugin(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "plugins" / "akasha"
    target = tmp_path / "plugins" / "akasha"
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return target


@pytest.mark.asyncio
async def test_manager_provides_formal_port_and_candidate_cannot_consume_engine(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "memory_probe"
    _write_plugin(plugin_dir, version="1", probe_candidate=False)
    engine = _MemoryTurnEngine()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        memory_engine=engine,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    stable = manager.current_snapshot
    generation = manager.generation("memory_probe")
    assert stable is not None and generation is not None
    assert generation.instance.module.bound is not None

    _write_plugin(plugin_dir, version="2", probe_candidate=True)
    prepared = await manager.prepare_candidate("memory_probe")

    assert prepared is None
    assert manager.current_snapshot is stable
    assert engine.metadata_calls == 0
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_real_akasha_v3_rebuilds_formal_mobile_binding(
    tmp_path: Path,
) -> None:
    plugin_dir = _copy_akasha_plugin(tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    sentinel = workspace / "memory" / "stable.bin"
    sentinel.write_bytes(b"formal-memory")
    engine = _MemoryTurnEngine()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        memory_engine=engine,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    stable = manager.current_snapshot
    stable_generation = manager.generation("akasha")
    assert stable is not None and stable_generation is not None
    stable_registry = stable.mobile_ui_registry
    assert stable_registry is not None
    stable_binding = stable_registry["akasha"]

    feedback = AfterReasoningCtx(
        session_key="test:one",
        channel="test",
        chat_id="one",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="reply"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="reply",
    )
    lease = manager.snapshot_store.lease()
    binding_token = bind_runtime_snapshot(lease)
    turn_token = running_turn_id.set("turn-feedback")
    try:
        await run_composition_lifecycle(
            AFTER_REASONING_PREPROCESS_EVENT,
            feedback,
        )
    finally:
        running_turn_id.reset(turn_token)
        reset_runtime_snapshot(binding_token)
        await lease.release()
    assert feedback.persist_user_metadata == {"marker": {"turn_id": "turn-feedback"}}

    provider = PluginMobileUiProvider(manager)
    active = await provider.query(
        "akasha",
        stable_generation.source_revision,
        "recall.current",
        {"message_id": "assistant:turn-1"},
        session_id="test:one",
        turn_id="turn-1",
    )
    assert cast(str, active["query_id"]) == "test:one:turn-1"

    source = (plugin_dir / "plugin.py").read_text(encoding="utf-8")
    updated_source = source.replace('version = "3.0.0"', 'version = "3.0.1"')
    assert updated_source != source
    (plugin_dir / "plugin.py").write_text(
        updated_source,
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("akasha")
    assert candidate is not None and candidate.runtime_snapshot is not None
    assert manager.current_snapshot is stable
    assert sentinel.read_bytes() == b"formal-memory"

    candidate_feedback = AfterReasoningCtx(
        session_key="test:candidate",
        channel="programmatic",
        chat_id="candidate",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="candidate reply"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="candidate reply",
    )
    candidate_turn_token = running_turn_id.set("turn-candidate")
    try:
        result = await candidate.runtime_snapshot.composition_root.context.serial(
            AFTER_REASONING_PREPROCESS_EVENT,
            candidate_feedback,
        )
    finally:
        running_turn_id.reset(candidate_turn_token)
    assert result is None
    assert candidate_feedback.persist_user_metadata == {}
    assert engine.metadata_calls == 1

    candidate_registry = candidate.runtime_snapshot.mobile_ui_registry
    assert candidate_registry is not None
    with pytest.raises(RuntimeError, match="candidate 验证期"):
        candidate_registry["akasha"].query(
            "recall.current",
            {"message_id": "assistant:turn-2"},
            session_id="test:one",
            turn_id="turn-2",
        )
    assert engine.metadata_calls == 1
    assert engine.recall_calls == 1

    result = await manager.publish_prepared("akasha")
    assert result["publication_state"] == "committed"
    final = manager.current_snapshot
    final_generation = manager.generation("akasha")
    assert final is not None and final_generation is not None
    final_registry = final.mobile_ui_registry
    assert final_registry is not None
    final_binding = final_registry["akasha"]
    assert final_binding is not stable_binding
    await asyncio.sleep(0)
    assert not stable_binding.is_live()
    assert final_binding.is_live()
    assert sentinel.read_bytes() == b"formal-memory"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_real_akasha_v3_is_not_published_for_default_memory(
    tmp_path: Path,
) -> None:
    _copy_akasha_plugin(tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        memory_engine=_MemoryTurnEngine("default"),
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None
    assert snapshot.mobile_ui_registry is not None
    assert "akasha" not in snapshot.mobile_ui_registry
    assert not (workspace / "memory" / "akasha.db").exists()
    await manager.terminate_all()
