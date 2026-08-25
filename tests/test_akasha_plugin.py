from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
import struct
import threading
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.config_models import (
    Config,
    MemoryConfig as HostMemoryConfig,
    MemoryEmbeddingConfig,
)
from agent.control.context import running_turn_id
from agent.migrations.akasha_sidecar import rebuild_akasha_sidecars
from agent.plugin_composition import DashboardContext, TextEmbeddingSettings
from agent.plugins.composable import ComposablePlugin
from agent.plugins.mobile_ui import _normalize_rpc_result
from agent.plugins.manifest import (
    builtin_plugin_data_dir,
    ensure_workspace_plugin_data_dir,
)
from agent.tools.base import ToolExecutionContext, tool_execution_context_scope
from agent.tools.recall_memory import RecallMemoryTool
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.error_context import current_client_message_id, current_session_key
from core.memory.engine import MemoryQuery, MemoryQueryResult, MemoryScope
from plugins.akasha.application.cycle import MemoryCycle
from plugins.akasha.application.rebuild import rebuild_memory
from plugins.akasha.application.runtime import OnlineMemoryRuntime
from plugins.akasha.config import AkashaConfig, load_akasha_config, render_akasha_config
from plugins.akasha.dashboard import register as register_dashboard
from plugins.akasha.domain import features as features_module
from plugins.akasha.domain.features import BurstAwareFeaturePool
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.engine import (
    ActiveRecallSnapshot,
    AkashaMemoryEngine,
    PendingRetrieval,
    RetrievalRecords,
)
from plugins.akasha.inspector import AkashaInspectorReader, mobile_summary
from plugins.akasha.infrastructure.loader import load_turn_suffix, load_turns
from plugins.akasha.infrastructure.persistence import (
    logical_state_sha256,
)
from plugins.akasha.infrastructure.sparse_index import (
    BuildConfig,
    SparseIndexRebuildRequired,
    audit_source_embeddings,
    build_sparse_index,
    sparse_index_state_sha256,
)
from plugins.akasha.infrastructure.sparse_index.schema import (
    INDEX_VERSION,
    SCHEMA,
    TOOL_CHAIN_PROJECTION_VERSION,
)
from plugins.akasha import plugin as akasha_plugin
from plugins.akasha.plugin import (
    _AkashaMobileQuery,
    _empty_mobile_recall,
    _mobile_recall_lane,
)
from session.store import InteractionDeletion, SessionStore


class _Embedder:
    def __init__(self, **values: object) -> None:
        self.model = str(values["model"])
        self.output_dimensionality = int(cast(int, values["output_dimensionality"]))

    async def embed(self, text: str) -> list[float]:
        if self.output_dimensionality != 2:
            raise ValueError("test embedder requires two dimensions")
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    async def aclose(self) -> None:
        return None


def test_akasha_registers_v3_namespace() -> None:
    plugin = ComposablePlugin.from_module(akasha_plugin)

    assert plugin.api_version == 3
    assert plugin.name == "akasha"
    assert plugin.dashboard_module == "dashboard.py"
    assert plugin.workspace_roots == ("memory",)


def test_engine_and_inspector_resolve_sidecars_from_same_memory_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep engine writes and inspector reads on the same declared root."""

    # 1. Construct the real engine with the direct filename syntax.
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    config = AkashaConfig(
        db_path="akasha.db",
        index_path="akasha-v2-index.db",
    )
    engine = AkashaMemoryEngine(
        embedding=TextEmbeddingSettings(
            base_url="",
            api_key="chat-key",
            model="embedding-model",
            output_dimensionality=2,
        ),
        akasha_config=config,
        workspace=tmp_path,
        http_resources=cast(Any, SimpleNamespace(external_default=object())),
        event_publisher=None,
    )

    # 2. Resolve the same config through the dashboard/mobile inspector.
    inspector = AkashaInspectorReader(
        memory_root=tmp_path / "memory",
        config=config,
    )
    try:
        assert engine._runtime.memory_path == inspector.paths.memory  # noqa: SLF001
        assert engine._runtime.index_path == inspector.paths.index  # noqa: SLF001
        assert inspector.paths.memory == tmp_path / "memory" / "akasha.db"
        assert inspector.paths.index == tmp_path / "memory" / "akasha-v2-index.db"
    finally:
        engine._runtime.close()  # noqa: SLF001
        engine._embedding_store.close()  # noqa: SLF001


@pytest.mark.asyncio
async def test_feedback_tools_compose_correction_from_two_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose correction from forget plus remember without a third action."""

    # 1. Build one historical turn addressable by either Message ID.
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="old wrong answer",
        assistant="incorrect detail",
        started=started,
    )
    await engine._on_turn_committed(  # noqa: SLF001
        _event(
            sequence=0,
            user="old wrong answer",
            assistant="incorrect detail",
            started=started,
        )
    )
    await engine._wait_for_publication()  # noqa: SLF001

    # 2. Forget the old Message and remember the current user correction.
    specs = {spec.name: spec for spec in engine.tool_profile().tools}
    assert set(specs) == {"remember_memory", "forget_memory"}
    forget_spec = specs["forget_memory"]
    remember_spec = specs["remember_memory"]
    assert forget_spec.risk == "write"
    assert remember_spec.risk == "write"
    assert forget_spec.tool_class is not None
    assert remember_spec.tool_class is not None
    forget = forget_spec.tool_class(engine, forget_spec)
    remember = remember_spec.tool_class(engine, remember_spec)
    token = running_turn_id.set("turn:feedback")
    try:
        forgotten = json.loads(
            await forget.execute(
                message_ids=["message:1"],
                reason="old assistant answer is wrong",
            )
        )
        reinforced = json.loads(
            await remember.execute(
                message_ids=["current_user_message"],
                reason="the current user message supplies the correction",
            )
        )
        assert forgotten == {
            "status": "staged",
            "action": "forget",
            "target_message_ids": ["message:1"],
            "target_turn_ids": ["message:0::message:1"],
            "applies_after": "current_turn_commit",
        }
        assert reinforced == {
            "status": "staged",
            "action": "remember",
            "target_message_ids": ["current_user_message"],
            "target_turn_ids": ["current_turn"],
            "applies_after": "current_turn_commit",
        }

        # 3. Export both independent markers through the narrow Memory port.
        metadata = engine.take_turn_user_metadata("turn:feedback")
        forgotten_marker = cast(dict[str, object], metadata["akasha_forget"])
        remembered_marker = cast(dict[str, object], metadata["akasha_reinforce"])
        assert forgotten_marker["action"] == "forget"
        assert forgotten_marker["target_message_ids"] == ["message:1"]
        assert forgotten_marker["target_turn_ids"] == ["message:0::message:1"]
        assert remembered_marker["action"] == "remember"
        assert remembered_marker["target_message_ids"] == ["current_user_message"]
        assert remembered_marker["target_turn_ids"] == ["current_turn"]
        assert engine.take_staged_feedback("turn:feedback") == ()
    finally:
        running_turn_id.reset(token)
        _close_engine(engine)


@pytest.mark.asyncio
async def test_uncommitted_feedback_does_not_survive_engine_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    spec = next(
        item for item in engine.tool_profile().tools if item.name == "remember_memory"
    )
    assert spec.tool_class is not None
    tool = spec.tool_class(engine, spec)
    token = running_turn_id.set("turn:crashed")
    try:
        result = json.loads(
            await tool.execute(
                message_ids=["current_user_message"],
                reason="staged but not committed",
            )
        )
    finally:
        running_turn_id.reset(token)
    assert result["status"] == "staged"
    _close_engine(engine)

    restarted = _engine(tmp_path)
    try:
        assert restarted.take_turn_user_metadata("turn:crashed") == {}
    finally:
        _close_engine(restarted)


@pytest.mark.asyncio
async def test_feedback_tool_rejects_memory_item_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require Message identities even when the turn ID looks plausible."""

    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    try:
        spec = next(
            spec for spec in engine.tool_profile().tools if spec.name == "forget_memory"
        )
        assert spec.tool_class is not None
        tool = spec.tool_class(engine, spec)
        token = running_turn_id.set("turn:feedback")
        try:
            result = json.loads(
                await tool.execute(
                    message_ids=["message:0::message:1"],
                )
            )
        finally:
            running_turn_id.reset(token)
        assert result["status"] == "not_staged"
        assert result["error"] == "messages_not_in_akasha"

        token = running_turn_id.set("turn:feedback")
        try:
            current = json.loads(
                await tool.execute(
                    message_ids=["current_user_message"],
                )
            )
        finally:
            running_turn_id.reset(token)
        assert current["status"] == "not_staged"
        assert current["error"] == "cannot_forget_current_user_message"
    finally:
        _close_engine(engine)


@pytest.mark.asyncio
async def test_feedback_markers_change_future_recall_and_replay_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist tool markers, suppress future recall, and reproduce the graph."""

    # 1. Commit one wrong historical turn, then stage a current correction.
    sessions = tmp_path / "sessions.db"
    _create_sessions(sessions)
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        sessions,
        sequence=0,
        user="alpha old wrong claim",
        assistant="wrong answer",
        started=started,
    )
    await engine._on_turn_committed(  # noqa: SLF001
        _event(
            sequence=0,
            user="alpha old wrong claim",
            assistant="wrong answer",
            started=started,
        )
    )
    await engine._wait_for_publication()  # noqa: SLF001
    specs = {spec.name: spec for spec in engine.tool_profile().tools}
    forget_spec = specs["forget_memory"]
    remember_spec = specs["remember_memory"]
    assert forget_spec.tool_class is not None
    assert remember_spec.tool_class is not None
    forget = forget_spec.tool_class(engine, forget_spec)
    remember = remember_spec.tool_class(engine, remember_spec)
    token = running_turn_id.set("turn:correction")
    try:
        assert (
            json.loads(await forget.execute(message_ids=["message:1"]))["status"]
            == "staged"
        )
        assert (
            json.loads(await remember.execute(message_ids=["current_user_message"]))[
                "status"
            ]
            == "staged"
        )
        user_extra = engine.take_turn_user_metadata("turn:correction")
    finally:
        running_turn_id.reset(token)

    # 2. Persist those marker fields on the next canonical user Message.
    correction_time = started + timedelta(minutes=5)
    _append_turn(
        sessions,
        sequence=2,
        user="beta corrected claim",
        assistant="correction accepted",
        started=correction_time,
        user_extra=user_extra,
    )
    await engine._on_turn_committed(  # noqa: SLF001
        _event(
            sequence=2,
            user="beta corrected claim",
            assistant="correction accepted",
            started=correction_time,
        )
    )
    await engine._wait_for_publication()  # noqa: SLF001

    assert engine._runtime.cycle.inhibited_nodes == {0}  # noqa: SLF001
    correction = engine._runtime.cycle.turns[1]  # noqa: SLF001
    assert correction.feedback.forget_nodes == (0,)
    assert correction.feedback.remember_nodes == (1,)
    with closing(sqlite3.connect(tmp_path / "memory" / "akasha.db")) as connection:
        assert (
            connection.execute("""
            SELECT event_id, action, target_turn_node_id, boost
            FROM feedback_events
            ORDER BY event_id, action, target_turn_node_id
            """).fetchall()
            == [
                (1, "forget", 0, 1.0),
                (1, "remember", 1, 3.0),
            ]
        )

    # 3. Future direct-dense and graph completion lanes hide the old turn.
    recalled = await engine.query(
        _query(
            "alpha old wrong claim",
            correction_time + timedelta(minutes=5),
            intent="answer",
        )
    )
    assert all(record.id != "message:0::message:1" for record in recalled.records)

    # 4. A clean replay consumes the marker and hashes identically.
    replay = tmp_path / "memory" / "feedback-replay.db"
    rebuild_memory(
        tmp_path / "memory" / "akasha-v2-index.db",
        replay,
        target_sequences=(),
    )
    assert logical_state_sha256(
        tmp_path / "memory" / "akasha.db"
    ) == logical_state_sha256(replay)
    _close_engine(engine)


def test_mobile_recall_card_projection_preserves_bounded_lanes() -> None:
    lane = _mobile_recall_lane(
        [
            {
                "user_text": "🌙" * 1_000,
                "assistant_preview": "🌙" * 1_000,
                "assistant_text": "不应进入移动卡片",
                "ts": "2026-07-28T00:00:00Z",
                "score": 0.5,
            }
            for _ in range(40)
        ]
    )
    card = {
        "schema": "akasha.recall-card.v1",
        "query_id": "query",
        "recall_capture_available": True,
        "left": lane,
        "right": lane,
        "tool_left": lane,
        "tool_right": lane,
    }

    assert len(lane) == 40
    assert all(len(str(item["user_preview"])) == 103 for item in lane)
    assert all(len(str(item["assistant_preview"])) == 53 for item in lane)
    assert "assistant_text" not in json.dumps(card, ensure_ascii=False)
    assert (
        len(
            json.dumps(
                card,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        < 192 * 1024
    )


def test_mobile_inspector_detail_projects_large_assistant_text_to_bounded_rpc() -> None:
    large_answer = "答" * (256 * 1024)
    lane_item = {
        "query_id": "prior-query",
        "session_key": "test:one",
        "user_text": "问" * 1_000,
        "assistant_text": large_answer,
        "assistant_preview": large_answer,
        "ts": "2026-07-28T00:00:00Z",
        "score": 0.5,
        "sources": ["dense"],
    }
    detail = {
        "query_id": "query",
        "query_text": "当前问题",
        "query_preview": "当前问题",
        "ts": "2026-07-28T00:00:00Z",
        "seed_count": 1,
        "activation_capture_available": True,
        "recall_capture_available": True,
        "activation_count": 1,
        "left_count": 1,
        "right_count": 1,
        "pushes": 1,
        "residual_l1": 0.25,
        "tool_left_count": 1,
        "tool_right_count": 1,
        "left": [lane_item],
        "right": [lane_item],
        "tool_left": [lane_item],
        "tool_right": [lane_item],
    }

    projected = mobile_summary(detail)
    rpc = _normalize_rpc_result(
        projected,
        plugin_id="akasha",
        method="inspector.detail",
    )
    encoded = json.dumps(rpc, ensure_ascii=False, separators=(",", ":"))

    assert "assistant_text" not in encoded
    assert "user_text" not in encoded
    assert len(encoded.encode("utf-8")) < 192 * 1024
    for lane_name in ("left", "right", "tool_left", "tool_right"):
        lane = cast(list[dict[str, object]], rpc[lane_name])
        assert set(lane[0]) == {
            "user_preview",
            "assistant_preview",
            "ts",
            "score",
        }
        assert len(str(lane[0]["user_preview"])) == 103
        assert len(str(lane[0]["assistant_preview"])) == 53
    assert lane_item["assistant_text"] == large_answer


def test_active_mobile_recall_marks_temporary_absence_as_pending() -> None:
    assert _empty_mobile_recall()["pending"] is False
    assert _empty_mobile_recall(pending=True)["pending"] is True


def test_suffix_loader_and_appendable_features_match_full_replay(
    tmp_path: Path,
) -> None:
    """Keep the incremental online view identical to full replay features."""

    # 1. Build one causal source with dense, lexical, and temporal evidence.
    sessions = tmp_path / "sessions.db"
    index = tmp_path / "index.db"
    _create_sessions(sessions)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    for offset, (user, assistant) in enumerate(
        (
            ("alpha first", "first answer"),
            ("beta second", "alpha bridge"),
            ("alpha third", "final answer"),
        )
    ):
        _append_turn(
            sessions,
            sequence=offset * 2,
            user=user,
            assistant=assistant,
            started=started + timedelta(minutes=offset * 3),
            with_embeddings=True,
        )
    build_sparse_index(
        sessions,
        index,
        BuildConfig(
            embedding_model="embedding-model",
            embedding_dimension=2,
        ),
    )

    # 2. A suffix retains global node IDs, gaps, feedback, and feature bytes.
    full = load_turns(index)
    suffix = load_turn_suffix(index, 1)
    assert len(full) == 3
    assert len(suffix) == 2
    for expected, actual in zip(full[1:], suffix, strict=True):
        assert actual.node_id == expected.node_id
        assert actual.turn_id == expected.turn_id
        assert actual.inter_gap_seconds == expected.inter_gap_seconds
        assert actual.user_terms == expected.user_terms
        assert actual.assistant_terms == expected.assistant_terms
        assert actual.feedback == expected.feedback
        assert actual.user_dense is not None
        assert expected.user_dense is not None
        assert actual.assistant_dense is not None
        assert expected.assistant_dense is not None
        assert np.array_equal(actual.user_dense, expected.user_dense)
        assert np.array_equal(
            actual.assistant_dense,
            expected.assistant_dense,
        )

    # 3. The O(1) query view and incremental append preserve full-pool results.
    online = BurstAwareFeaturePool(full[:2], appendable=True)
    replay = BurstAwareFeaturePool(full)
    context = online.build_context(((1, 1.0),))
    view = online.query_view(full[2])
    online_decision = view.infer_burst_seed(2, context, (1,), True)
    replay_decision = replay.infer_burst_seed(2, context, (1,), True)
    assert view.turns is online.turns
    assert online_decision.evidence == replay_decision.evidence
    assert online_decision.base_continuation == (replay_decision.base_continuation)
    assert online_decision.context_dependence == (replay_decision.context_dependence)
    assert online_decision.context_mass == replay_decision.context_mass
    assert online_decision.continued == replay_decision.continued
    for name in online_decision.fields:
        assert np.array_equal(
            online_decision.fields[name],
            replay_decision.fields[name],
        )
    online.append_turn(full[2])
    assert np.array_equal(online.turn_dense[:3], replay.turn_dense)
    assert np.array_equal(online.lengths[:3], replay.lengths)
    assert np.array_equal(
        online.context_dependence[:3],
        replay.context_dependence,
    )

    # 4. A history without vectors can accept the first later dense turn.
    sparse_first = replace(
        full[0],
        user_dense=None,
        assistant_dense=None,
    )
    dense_online = BurstAwareFeaturePool(
        [sparse_first],
        appendable=True,
    )
    dense_online.append_turn(full[1])
    dense_replay = BurstAwareFeaturePool([sparse_first, full[1]])
    assert np.array_equal(dense_online.user_dense[:2], dense_replay.user_dense)
    assert np.array_equal(
        dense_online.assistant_dense[:2],
        dense_replay.assistant_dense,
    )
    assert np.array_equal(dense_online.turn_dense[:2], dense_replay.turn_dense)

    # 5. Diagnostic path capture cannot change committed online/replay state.
    online_cycle = MemoryCycle(
        MemoryConfig(),
        turn_capacity=len(full),
        feature_pool=BurstAwareFeaturePool(full),
    )
    replay_cycle = MemoryCycle(
        MemoryConfig(),
        turn_capacity=len(full),
        feature_pool=BurstAwareFeaturePool(full),
    )
    for turn in full:
        online_cycle.commit(
            turn,
            online_cycle.retrieve(turn, capture_paths=True),
        )
        replay_cycle.commit(
            turn,
            replay_cycle.retrieve(turn, capture_paths=False),
        )
    assert online_cycle.evidence == replay_cycle.evidence


@pytest.mark.asyncio
async def test_online_turn_recall_and_replay_share_one_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove one real host turn grows online and rebuilds identically."""

    # 1. Start the V2 host adapter on an isolated canonical source.
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    first_query = await engine.query(_query("alpha start", started, intent="context"))
    assert first_query.trace["effect"] == "stateful"
    assert first_query.records == []

    # 2. Persist the exact host messages, then commit their retrieval ticket.
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="A" * 80,
        started=started,
    )
    await engine._on_turn_committed(  # noqa: SLF001
        _event(
            sequence=0,
            user="alpha start",
            assistant="A" * 80,
            started=started,
        )
    )
    await engine._wait_for_publication()  # noqa: SLF001
    assert engine._runtime.cycle.state_version == 1  # noqa: SLF001

    # 3. Explicit recall is read-only and cannot replace context learning.
    next_time = started + timedelta(minutes=5)
    active_turn_id = "turn:alpha-follow"
    mobile_query = _AkashaMobileQuery(
        engine,
        memory_root=tmp_path / "memory",
        data_root=builtin_plugin_data_dir("akasha", tmp_path),
    )
    wait_started = threading.Event()
    wait_for_active_recall = engine.wait_for_active_recall

    def marked_wait_for_active_recall(
        session_key: str,
        turn_id: str,
        *,
        timeout: float = 15.0,
    ) -> ActiveRecallSnapshot | None:
        wait_started.set()
        return wait_for_active_recall(
            session_key,
            turn_id,
            timeout=timeout,
        )

    monkeypatch.setattr(
        engine,
        "wait_for_active_recall",
        marked_wait_for_active_recall,
    )
    active_query = asyncio.create_task(
        asyncio.to_thread(
            mobile_query,
            "recall.current",
            {"message_id": f"assistant:{active_turn_id}"},
            session_id="test:one",
            turn_id=active_turn_id,
        )
    )
    assert await asyncio.to_thread(wait_started.wait, 1.0)
    context = await engine.query(
        MemoryQuery(
            text="alpha follow",
            intent="context",
            scope=MemoryScope(
                session_key="test:one",
                channel="test",
                chat_id="one",
            ),
            context={"history": [], "turn_id": active_turn_id},
            timestamp=next_time,
        )
    )
    active_mobile = await active_query
    pending = engine._pending["test:one"]  # noqa: SLF001
    tool = RecallMemoryTool(
        engine,
        cast(Any, engine.tool_profile().recall),
    )
    before_recall = logical_state_sha256(tmp_path / "memory" / "akasha.db")
    with tool_execution_context_scope(
        ToolExecutionContext(
            origin_channel="test",
            origin_chat_id="one",
            origin_session_key="test:one",
            current_timestamp=next_time.isoformat(),
        )
    ):
        rendered = json.loads(
            await tool.execute(
                query="alpha details",
                limit=5,
            )
        )
    after_recall = logical_state_sha256(tmp_path / "memory" / "akasha.db")
    assert rendered["count"] == 1
    assert before_recall == after_recall
    assert engine._pending["test:one"] is pending  # noqa: SLF001
    assert context.text_block.startswith("# Akasha memory now=07-06")
    assert "## 左脑记忆：精确回忆" in context.text_block
    assert f'assistant="{"A" * 50}..."' in context.text_block
    assert (
        engine.wait_for_active_recall(
            "test:one",
            "turn:other",
            timeout=0,
        )
        is None
    )

    assert [
        item["user_preview"]
        for item in cast(
            list[dict[str, object]],
            active_mobile["left"],
        )
    ] == ["alpha start"]
    tool_payload = dict(rendered)
    tool_payload["items"] = [
        *cast(list[dict[str, object]], rendered["items"]),
        {
            "id": "tool:completion",
            "score": 0.25,
            "source_ref": "test:one",
            "signals": {
                "lane": "completion",
                "sources": ["basin_completion"],
                "started_at": started.isoformat(),
                "user_text": "associated memory",
                "assistant_preview": "associated answer",
            },
        },
    ]
    tool_chain = json.dumps(
        [
            {
                "calls": [
                    {
                        "name": "recall_memory",
                        "status": "success",
                        "result": json.dumps(tool_payload),
                    }
                ]
            }
        ]
    )

    # 4. Commit the second turn and compare online learned state with replay.
    _append_turn(
        tmp_path / "sessions.db",
        sequence=2,
        user="alpha follow",
        assistant="second answer",
        started=next_time,
        assistant_tool_chain=tool_chain,
    )
    await engine._on_turn_committed(  # noqa: SLF001
        _event(
            sequence=2,
            user="alpha follow",
            assistant="second answer",
            started=next_time,
        )
    )
    await engine._wait_for_publication()  # noqa: SLF001
    replay = tmp_path / "memory" / "replay.db"
    rebuild_memory(
        tmp_path / "memory" / "akasha-v2-index.db",
        replay,
        target_sequences=(),
    )
    assert logical_state_sha256(
        tmp_path / "memory" / "akasha.db"
    ) == logical_state_sha256(replay)
    with closing(sqlite3.connect(tmp_path / "memory" / "akasha.db")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM recall_runs").fetchone() == (2,)
        assert connection.execute(
            "SELECT COUNT(*) FROM activation_runs"
        ).fetchone() == (0,)

    # 5. Inspector reconstructs the exact prior-only lanes without writes.
    _write_inspector_config(tmp_path)
    before_memory = logical_state_sha256(tmp_path / "memory" / "akasha.db")
    reader = AkashaInspectorReader(
        memory_root=tmp_path / "memory",
        config=load_akasha_config(
            builtin_plugin_data_dir("akasha", tmp_path) / "config.local.toml"
        ),
    )
    overview = reader.get_overview()
    rows, total = reader.list_turns(q="alpha follow")
    detail = reader.get_turn(str(rows[0]["query_id"]))
    assert overview["total"] == 2
    assert total == 1
    assert detail is not None
    assert detail["query_text"] == "alpha follow"
    assert detail["assistant_text"] == "second answer"
    assert detail["recall_capture_available"] is True
    assert detail["left_count"] == 1
    assert detail["tool_left_count"] == 1
    assert detail["tool_right_count"] == 1
    assert (
        cast(list[dict[str, object]], detail["left"])[0]["user_text"] == "alpha start"
    )
    assert "## 左脑记忆：精确回忆" in str(detail["text_block_preview"])
    assert detail["activation_capture_available"] is False
    assert before_memory == logical_state_sha256(tmp_path / "memory" / "akasha.db")

    # 6. The desktop API exposes the same state through read-only routes.
    app = FastAPI()
    register_dashboard(
        app,
        DashboardContext(
            plugin_id="akasha",
            plugin_dir=Path("plugins/akasha"),
            data_root=builtin_plugin_data_dir("akasha", tmp_path),
            validation=False,
            _workspace_roots=(("memory", tmp_path / "memory"),),
        ),
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/dashboard/akasha-inspector/turns",
            params={"q": "alpha follow"},
        )
        assert response.status_code == 200
        assert response.json()["total"] == 1
        api_detail = client.get(
            f"/api/dashboard/akasha-inspector/turns/{rows[0]['query_id']}"
        )
        assert api_detail.status_code == 200
        assert api_detail.json()["left_count"] == 1

    # 7. Mobile projections resolve the same committed assistant message.
    mobile = mobile_query(
        "recall.current",
        {"message_id": "message:3"},
        session_id="test:one",
        turn_id=None,
    )
    recent = mobile_query(
        "inspector.recent",
        {},
        session_id=None,
        turn_id=None,
    )
    mobile_detail = mobile_query(
        "inspector.detail",
        {"query_id": str(rows[0]["query_id"])},
        session_id=None,
        turn_id=None,
    )
    assert len(cast(list[object], mobile["left"])) == 1
    assert len(cast(list[object], mobile["tool_left"])) == 1
    assert len(cast(list[object], mobile["tool_right"])) == 1
    assert mobile["schema"] == "akasha.recall-card.v1"
    mobile_left = cast(list[dict[str, object]], mobile["left"])
    assert mobile_left[0]["user_preview"] == "alpha start"
    assert active_mobile["left"] == mobile["left"]
    assert active_mobile["right"] == mobile["right"]
    assert "user_text" not in mobile_left[0]
    assert "assistant_text" not in json.dumps(mobile, ensure_ascii=False)
    assert (
        len(
            json.dumps(
                mobile,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        < 16 * 1024
    )
    assert recent["total"] == 2
    assert mobile_detail["query_text"] == "alpha follow"
    _close_engine(engine)


@pytest.mark.asyncio
async def test_turn_commit_returns_before_graph_publish_and_fences_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release the host turn after durable staging while fencing the next read."""

    # 1. Block only graph publication after the source and embedding are durable.
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    entered = threading.Event()
    release = threading.Event()
    publish = engine._runtime.publish_staged  # noqa: SLF001

    def blocked_publish(staged: object) -> object:
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test graph publication was not released")
        return publish(cast(Any, staged))

    monkeypatch.setattr(
        engine._runtime,  # noqa: SLF001
        "publish_staged",
        blocked_publish,
    )
    await engine._on_turn_committed(  # noqa: SLF001
        _event(
            sequence=0,
            user="alpha start",
            assistant="first answer",
            started=started,
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)
    assert engine._runtime.cycle.state_version == 0  # noqa: SLF001
    with closing(
        sqlite3.connect(tmp_path / "memory" / "akasha-v2-index.db")
    ) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sparse_turns").fetchone() == (
            1,
        )

    # 2. The next query waits until the staged graph becomes visible.
    query = asyncio.create_task(
        engine.query(
            _query(
                "alpha follow",
                started + timedelta(minutes=5),
                intent="context",
            )
        )
    )
    await asyncio.sleep(0)
    assert not query.done()
    release.set()
    result = await query

    assert engine._runtime.cycle.state_version == 1  # noqa: SLF001
    assert result.trace["state_version"] == 1
    _close_engine(engine)


def _milestone_events(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        cast(str, record.akashic_fields["event"])
        for record in caplog.records
        if getattr(record, "akashic_fields", None)
        and record.akashic_fields.get("flow") == "mobile_turn"
    ]


def _milestone_records(
    caplog: pytest.LogCaptureFixture,
    *events: str,
) -> list[Any]:
    return [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", None)
        and record.akashic_fields.get("flow") == "mobile_turn"
        and record.akashic_fields.get("event") in events
    ]


def _counts_map(counts: str) -> dict[str, str]:
    """Parse the akasha counts payload; operation/span_id are stable keys."""

    return dict(item.split("=", 1) for item in counts.split(",") if "=" in item)


def _milestone_triples(
    caplog: pytest.LogCaptureFixture,
) -> list[tuple[str, str, str]]:
    """Extract (span_id, operation, event) from every akasha milestone."""

    triples: list[tuple[str, str, str]] = []
    for record in caplog.records:
        fields = cast(
            Mapping[str, object] | None,
            getattr(record, "akashic_fields", None),
        )
        if not fields or fields.get("flow") != "mobile_turn":
            continue
        counts = _counts_map(str(fields.get("counts") or ""))
        triples.append(
            (
                counts.get("span_id", ""),
                counts.get("operation", ""),
                str(fields.get("event")),
            )
        )
    return triples


def _milestone_span(
    caplog: pytest.LogCaptureFixture,
    event: str,
) -> str:
    """Return the span_id carried by the first occurrence of an event."""

    records = _milestone_records(caplog, event)
    assert records, f"缺少里程碑: {event}"
    return _counts_map(str(records[0].akashic_fields["counts"]))["span_id"]


def _event_base(event: str) -> str:
    for suffix in (".start", ".done", ".error", ".cancelled", ".skip"):
        if event.endswith(suffix):
            return event[: -len(suffix)]
    return event


def _assert_span_closed(
    caplog: pytest.LogCaptureFixture,
    span_id: str,
    operation: str,
) -> None:
    """每个 (span, event base) 的 start 恰好一个终态；skip 也携同一 span。"""

    by_base: dict[str, list[str]] = {}
    for current_span, current_operation, event in _milestone_triples(caplog):
        if current_span != span_id:
            continue
        assert current_operation == operation
        base_events = by_base.setdefault(_event_base(event), [])
        base_events.append(event)
    assert by_base, f"span {span_id} 没有任何里程碑记录"
    for base, events in by_base.items():
        start = f"{base}.start"
        if start in events:
            assert events.count(start) == 1
            assert len([e for e in events if e != start]) == 1
        else:
            assert events == [f"{base}.skip"] or events == ["akasha.publish_scheduled"]


@pytest.mark.asyncio
async def test_turn_commit_blocked_embed_keeps_fanout_open_at_embed_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TurnCommitted fanout 在 embed 被阻塞时未完成，阶段停在 embed.start。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original_embed_batch = engine._embedder.embed_batch  # noqa: SLF001

    async def blocked_embed_batch(texts: list[str]) -> list[list[float]]:
        entered.set()
        await release.wait()
        return await original_embed_batch(texts)

    monkeypatch.setattr(
        engine._embedder,  # noqa: SLF001
        "embed_batch",
        blocked_embed_batch,
    )
    # contextvar 身份与事件不同，证明 commit 路径身份显式来自事件。
    session_token = current_session_key.set("ctx:session")
    client_token = current_client_message_id.set("ctx-client")
    turn_token = running_turn_id.set("ctx-turn")
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    commit_task: asyncio.Task[None] | None = None
    try:
        commit_task = asyncio.create_task(
            engine._on_turn_committed(event)  # noqa: SLF001
        )
        assert await asyncio.wait_for(entered.wait(), timeout=1)
        assert not commit_task.done()
        assert _milestone_events(caplog) == [
            "akasha.turn_commit.start",
            "akasha.source_gate.wait.start",
            "akasha.source_gate.wait.done",
            "akasha.embed.start",
        ]
        assert "akasha.embed.done" not in _milestone_events(caplog)
        assert "akasha.commit_gate.wait.done" not in _milestone_events(caplog)
        assert "akasha.stage.done" not in _milestone_events(caplog)
        assert "akasha.turn_commit.done" not in _milestone_events(caplog)
        embed_records = _milestone_records(caplog, "akasha.embed.start")
        embed_counts = _counts_map(str(embed_records[0].akashic_fields["counts"]))
        assert embed_counts["embed_mode"] == "batch"
        assert embed_counts["operation"] == "turn_commit"
        assert embed_counts["span_id"] == _milestone_span(
            caplog, "akasha.turn_commit.start"
        )
        for record in embed_records:
            assert record.akashic_fields["session_id"] == "test:one"
            assert record.akashic_fields["turn_id"] == "event-turn-1"
            assert record.akashic_fields["client_message_id"] == "event-client-1"
    finally:
        release.set()
        if commit_task is not None:
            await commit_task
        await engine._wait_for_publication()  # noqa: SLF001
        current_session_key.reset(session_token)
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        _close_engine(engine)


@pytest.mark.asyncio
async def test_turn_commit_release_orders_stage_before_turn_commit_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """解除 embed 阻塞后，阶段顺序完整，stage.done 先于 turn_commit.done。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "plugins.akasha.engine.perf_counter",
        lambda: clock["now"],
    )
    original_embed_batch = engine._embedder.embed_batch  # noqa: SLF001

    async def blocked_embed_batch(texts: list[str]) -> list[list[float]]:
        entered.set()
        await release.wait()
        clock["now"] += 0.25
        return await original_embed_batch(texts)

    monkeypatch.setattr(
        engine._embedder,  # noqa: SLF001
        "embed_batch",
        blocked_embed_batch,
    )
    session_token = current_session_key.set("ctx:session")
    client_token = current_client_message_id.set("ctx-client")
    turn_token = running_turn_id.set("ctx-turn")
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    commit_task: asyncio.Task[None] | None = None
    try:
        commit_task = asyncio.create_task(
            engine._on_turn_committed(event)  # noqa: SLF001
        )
        assert await asyncio.wait_for(entered.wait(), timeout=1)
        release.set()
        await commit_task
        await engine._wait_for_publication()  # noqa: SLF001
        assert _milestone_events(caplog) == [
            "akasha.turn_commit.start",
            "akasha.source_gate.wait.start",
            "akasha.source_gate.wait.done",
            "akasha.embed.start",
            "akasha.embed.done",
            "akasha.commit_gate.wait.start",
            "akasha.commit_gate.wait.done",
            "akasha.prior_publication.wait.start",
            "akasha.prior_publication.wait.done",
            "akasha.stage.start",
            "akasha.stage.done",
            "akasha.publish_scheduled",
            "akasha.turn_commit.done",
        ]
        assert commit_task.done()
        # 所有 commit 里程碑身份来自事件，而非 contextvar，且共享同一 span。
        span_id = _milestone_span(caplog, "akasha.turn_commit.start")
        for record in caplog.records:
            if not getattr(record, "akashic_fields", None):
                continue
            if record.akashic_fields.get("flow") != "mobile_turn":
                continue
            assert record.akashic_fields["session_id"] == "test:one"
            assert record.akashic_fields["turn_id"] == "event-turn-1"
            assert record.akashic_fields["client_message_id"] == "event-client-1"
            counts = _counts_map(str(cast(Any, record.akashic_fields)["counts"]))
            assert counts["span_id"] == span_id
            assert counts["operation"] == "turn_commit"
        _assert_span_closed(caplog, span_id, "turn_commit")
        # 可控单调时钟证明 stage.done 不把 embed 阻塞时间算入自身 span。
        embed_done = _milestone_records(caplog, "akasha.embed.done")[0]
        stage_done = _milestone_records(caplog, "akasha.stage.done")[0]
        assert cast(float, embed_done.akashic_fields["duration_ms"]) == 250.0
        assert cast(float, stage_done.akashic_fields["duration_ms"]) == 0.0
    finally:
        release.set()
        if commit_task is not None:
            await commit_task
        await engine._wait_for_publication()  # noqa: SLF001
        current_session_key.reset(session_token)
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        _close_engine(engine)


@pytest.mark.asyncio
async def test_query_blocked_publication_is_locatable_at_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """query 阻塞在 publication wait 时，可通过里程碑定位到等待阶段。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    entered = threading.Event()
    release = threading.Event()
    publish = engine._runtime.publish_staged  # noqa: SLF001

    def blocked_publish(staged: object) -> object:
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test graph publication was not released")
        return publish(cast(Any, staged))

    monkeypatch.setattr(
        engine._runtime,  # noqa: SLF001
        "publish_staged",
        blocked_publish,
    )
    session_token = current_session_key.set("test:one")
    client_token = current_client_message_id.set("client-message-1")
    turn_token = running_turn_id.set("turn-1")
    query_task: asyncio.Task[MemoryQueryResult] | None = None
    try:
        await engine._on_turn_committed(  # noqa: SLF001
            _event(
                sequence=0,
                user="alpha start",
                assistant="first answer",
                started=started,
                turn_id="event-turn-1",
                client_message_id="event-client-1",
            )
        )
        assert await asyncio.to_thread(entered.wait, 1)
        query_task = asyncio.create_task(
            engine.query(
                _query(
                    "alpha follow",
                    started + timedelta(minutes=5),
                    intent="context",
                )
            )
        )
        await asyncio.sleep(0)
        events = _milestone_events(caplog)
        assert "akasha.query.start" in events
        assert "akasha.embed.done" in events
        # commit 自身已产生一对 prior_publication.wait；query 的那对只到 start。
        assert events.count("akasha.prior_publication.wait.start") == 2
        assert events.count("akasha.prior_publication.wait.done") == 1
        assert "akasha.runtime.query.done" not in events
        assert "akasha.query.done" not in events
        assert not query_task.done()
        release.set()
        result = await query_task
        assert result.trace["state_version"] == 1
        assert _milestone_events(caplog)[-1] == "akasha.query.done"
        assert (
            _milestone_events(caplog).count("akasha.prior_publication.wait.done") == 2
        )
        done_records = _milestone_records(caplog, "akasha.query.done")
        assert "hits" in _counts_map(str(done_records[-1].akashic_fields["counts"]))
        # query 里程碑身份来自当前 turn contextvar。
        for record in _milestone_records(caplog, "akasha.query.start"):
            assert record.akashic_fields["session_id"] == "test:one"
            assert record.akashic_fields["turn_id"] == "turn-1"
            assert record.akashic_fields["client_message_id"] == "client-message-1"
        # 同 turn 的 query 与 turn_commit 各持独立 span，且都闭合。
        query_span = _milestone_span(caplog, "akasha.query.start")
        commit_span = _milestone_span(caplog, "akasha.turn_commit.start")
        assert query_span != commit_span
        _assert_span_closed(caplog, query_span, "query")
        _assert_span_closed(caplog, commit_span, "turn_commit")
        # embed 事件可按 operation 区分归属。
        assert {
            operation
            for _, operation, event in _milestone_triples(caplog)
            if event.startswith("akasha.embed.")
        } == {"query", "turn_commit"}
    finally:
        release.set()
        if query_task is not None and not query_task.done():
            await query_task
        current_session_key.reset(session_token)
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        _close_engine(engine)


@pytest.mark.asyncio
async def test_turn_commit_embed_failure_records_embed_error_and_turn_commit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """embed 异常产生 embed.error 与 turn_commit.error，均无 done。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )

    async def exploding_embed_batch(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed exploded")

    monkeypatch.setattr(
        engine._embedder,  # noqa: SLF001
        "embed_batch",
        exploding_embed_batch,
    )
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    with pytest.raises(RuntimeError, match="embed exploded"):
        await engine._on_turn_committed(event)  # noqa: SLF001
    assert _milestone_events(caplog) == [
        "akasha.turn_commit.start",
        "akasha.source_gate.wait.start",
        "akasha.source_gate.wait.done",
        "akasha.embed.start",
        "akasha.embed.error",
        "akasha.turn_commit.error",
    ]
    assert "akasha.embed.done" not in _milestone_events(caplog)
    assert "akasha.turn_commit.done" not in _milestone_events(caplog)
    for event_name in ("akasha.embed.error", "akasha.turn_commit.error"):
        for record in _milestone_records(caplog, event_name):
            assert record.levelno == logging.ERROR
            assert record.akashic_fields["session_id"] == "test:one"
            assert record.akashic_fields["turn_id"] == "event-turn-1"
            assert record.akashic_fields["client_message_id"] == "event-client-1"
    _assert_span_closed(
        caplog,
        _milestone_span(caplog, "akasha.turn_commit.start"),
        "turn_commit",
    )
    _close_engine(engine)


@pytest.mark.asyncio
async def test_turn_commit_source_gate_wait_records_blocked_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """source gate 被外部持有时，wait.start 后停在等待，释放后带 duration。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    gate_task = asyncio.create_task(engine._source_event_gate.acquire())  # noqa: SLF001
    await gate_task
    commit_task = asyncio.create_task(engine._on_turn_committed(event))  # noqa: SLF001
    try:
        await asyncio.sleep(0.1)
        assert not commit_task.done()
        events = _milestone_events(caplog)
        assert "akasha.turn_commit.start" in events
        assert "akasha.source_gate.wait.start" in events
        assert "akasha.source_gate.wait.done" not in events
        assert "akasha.embed.start" not in events
        assert "akasha.turn_commit.done" not in events
        engine._source_event_gate.release()  # noqa: SLF001
        await asyncio.wait_for(commit_task, timeout=5)
        await engine._wait_for_publication()  # noqa: SLF001
        wait_done = _milestone_records(caplog, "akasha.source_gate.wait.done")
        assert wait_done
        assert cast(float, wait_done[0].akashic_fields["duration_ms"]) >= 50.0
        assert _milestone_events(caplog)[-1] == "akasha.turn_commit.done"
    finally:
        if engine._source_event_gate.locked():  # noqa: SLF001
            engine._source_event_gate.release()  # noqa: SLF001
        if not commit_task.done():
            await commit_task
        _close_engine(engine)


@pytest.mark.asyncio
async def test_turn_commit_commit_gate_wait_records_blocked_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """commit gate 被外部持有时，embed 完成后停在 wait，释放后带 duration。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    gate_task = asyncio.create_task(engine._commit_gate.acquire())  # noqa: SLF001
    await gate_task
    commit_task = asyncio.create_task(engine._on_turn_committed(event))  # noqa: SLF001
    try:
        await asyncio.sleep(0.1)
        assert not commit_task.done()
        events = _milestone_events(caplog)
        assert "akasha.embed.done" in events
        assert "akasha.commit_gate.wait.start" in events
        assert "akasha.commit_gate.wait.done" not in events
        assert "akasha.stage.start" not in events
        assert "akasha.turn_commit.done" not in events
        engine._commit_gate.release()  # noqa: SLF001
        await asyncio.wait_for(commit_task, timeout=5)
        await engine._wait_for_publication()  # noqa: SLF001
        wait_done = _milestone_records(caplog, "akasha.commit_gate.wait.done")
        assert wait_done
        assert cast(float, wait_done[0].akashic_fields["duration_ms"]) >= 50.0
        assert _milestone_events(caplog)[-1] == "akasha.turn_commit.done"
    finally:
        if engine._commit_gate.locked():  # noqa: SLF001
            engine._commit_gate.release()  # noqa: SLF001
        if not commit_task.done():
            await commit_task
        _close_engine(engine)


@pytest.mark.asyncio
async def test_event_bus_fanout_swallows_akasha_error_but_keeps_error_milestones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """EventBus fanout 吞掉 Akasha handler 异常，但错误里程碑仍可定位。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    bus = EventBus()
    engine = _engine(tmp_path, event_publisher=bus)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )

    async def exploding_embed_batch(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed exploded")

    monkeypatch.setattr(
        engine._embedder,  # noqa: SLF001
        "embed_batch",
        exploding_embed_batch,
    )
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    await bus.fanout(event)
    events = _milestone_events(caplog)
    assert "akasha.turn_commit.start" in events
    assert "akasha.embed.error" in events
    assert "akasha.turn_commit.error" in events
    assert "akasha.embed.done" not in events
    assert "akasha.turn_commit.done" not in events
    for record in _milestone_records(caplog, "akasha.turn_commit.error"):
        assert record.levelno == logging.ERROR
        assert record.akashic_fields["session_id"] == "test:one"
        assert record.akashic_fields["turn_id"] == "event-turn-1"
        assert record.akashic_fields["client_message_id"] == "event-client-1"
    _assert_span_closed(
        caplog,
        _milestone_span(caplog, "akasha.turn_commit.start"),
        "turn_commit",
    )
    _close_engine(engine)


@pytest.mark.asyncio
async def test_query_embed_failure_closes_span_with_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """query embed 异常：embed.error 与 query.error 闭合，无 done。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)

    async def exploding_embed(text: str) -> list[float]:
        raise RuntimeError("query embed exploded")

    monkeypatch.setattr(engine._embedder, "embed", exploding_embed)  # noqa: SLF001
    session_token = current_session_key.set("test:one")
    client_token = current_client_message_id.set("client-message-1")
    turn_token = running_turn_id.set("turn-1")
    try:
        with pytest.raises(RuntimeError, match="query embed exploded"):
            await engine.query(
                _query(
                    "alpha follow",
                    started + timedelta(minutes=5),
                    intent="context",
                )
            )
        assert _milestone_events(caplog) == [
            "akasha.query.start",
            "akasha.embed.start",
            "akasha.embed.error",
            "akasha.query.error",
        ]
        assert "akasha.query.done" not in _milestone_events(caplog)
        for event_name in ("akasha.embed.error", "akasha.query.error"):
            for record in _milestone_records(caplog, event_name):
                assert record.levelno == logging.ERROR
                assert record.akashic_fields["session_id"] == "test:one"
                assert record.akashic_fields["turn_id"] == "turn-1"
                assert record.akashic_fields["client_message_id"] == "client-message-1"
        _assert_span_closed(
            caplog,
            _milestone_span(caplog, "akasha.query.start"),
            "query",
        )
    finally:
        current_session_key.reset(session_token)
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        _close_engine(engine)


@pytest.mark.asyncio
async def test_query_cancelled_while_blocked_on_commit_gate_closes_span(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """query 阻塞在 commit gate 时被取消：query.cancelled 闭合，无 done/error。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    session_token = current_session_key.set("test:one")
    client_token = current_client_message_id.set("client-message-1")
    turn_token = running_turn_id.set("turn-1")
    gate_task = asyncio.create_task(engine._commit_gate.acquire())  # noqa: SLF001
    await gate_task
    query_task = asyncio.create_task(
        engine.query(
            _query(
                "alpha follow",
                started + timedelta(minutes=5),
                intent="context",
            )
        )
    )
    try:
        await asyncio.sleep(0.05)
        assert not query_task.done()
        events = _milestone_events(caplog)
        assert "akasha.query.start" in events
        assert "akasha.embed.done" in events
        assert "akasha.commit_gate.wait.start" in events
        assert "akasha.commit_gate.wait.done" not in events
        assert "akasha.query.done" not in events
        query_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await query_task
        events = _milestone_events(caplog)
        assert "akasha.query.cancelled" in events
        assert "akasha.commit_gate.wait.cancelled" in events
        assert "akasha.query.done" not in events
        assert "akasha.query.error" not in events
        assert "akasha.commit_gate.wait.done" not in events
        for record in _milestone_records(
            caplog, "akasha.query.cancelled", "akasha.commit_gate.wait.cancelled"
        ):
            assert record.akashic_fields["outcome"] == "cancelled"
            assert record.akashic_fields["duration_ms"] is not None
            assert record.akashic_fields["session_id"] == "test:one"
            assert record.akashic_fields["turn_id"] == "turn-1"
            assert record.akashic_fields["client_message_id"] == "client-message-1"
        _assert_span_closed(
            caplog,
            _milestone_span(caplog, "akasha.query.start"),
            "query",
        )
    finally:
        engine._commit_gate.release()  # noqa: SLF001
        if not query_task.done():
            await query_task
        current_session_key.reset(session_token)
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        _close_engine(engine)


async def _wait_for_milestone(
    caplog: pytest.LogCaptureFixture,
    event: str,
    *,
    timeout_s: float = 5.0,
) -> None:
    """轮询等待某个里程碑出现；避免纯 sleep 的不稳定时序。"""

    for _ in range(int(timeout_s * 100)):
        if event in _milestone_events(caplog):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"milestone 未在 {timeout_s}s 内出现: {event}")


@pytest.mark.asyncio
async def test_turn_commit_source_skip_closes_total_span_with_skipped_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """source 代际失效的 skip 不得绕过 turn_commit.done，须以 skipped 收口。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    with engine._lock:  # noqa: SLF001
        engine._source_generation += 1  # noqa: SLF001
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    gate_task = asyncio.create_task(engine._source_event_gate.acquire())  # noqa: SLF001
    await gate_task
    commit_task = asyncio.create_task(engine._on_turn_committed(event))  # noqa: SLF001
    try:
        await _wait_for_milestone(caplog, "akasha.source_gate.wait.start")
        assert not commit_task.done()
        # 等 gate 期间 source 代际推进，handler 恢复后必须走 skip 而非静默 return。
        with engine._lock:  # noqa: SLF001
            engine._source_generation += 1  # noqa: SLF001
        engine._source_event_gate.release()  # noqa: SLF001
        await asyncio.wait_for(commit_task, timeout=5)
        assert _milestone_events(caplog) == [
            "akasha.turn_commit.start",
            "akasha.source_gate.wait.start",
            "akasha.source_gate.wait.done",
            "akasha.source_event.skip",
            "akasha.turn_commit.done",
        ]
        done = _milestone_records(caplog, "akasha.turn_commit.done")[0]
        assert done.akashic_fields["outcome"] == "skipped"
        assert done.akashic_fields["duration_ms"] is not None
        assert "akasha.embed.start" not in _milestone_events(caplog)
        assert "akasha.turn_commit.error" not in _milestone_events(caplog)
        assert "akasha.turn_commit.cancelled" not in _milestone_events(caplog)
        _assert_span_closed(
            caplog,
            _milestone_span(caplog, "akasha.turn_commit.start"),
            "turn_commit",
        )
    finally:
        if engine._source_event_gate.locked():  # noqa: SLF001
            engine._source_event_gate.release()  # noqa: SLF001
        if not commit_task.done():
            await commit_task
        _close_engine(engine)


@pytest.mark.asyncio
async def test_turn_commit_stage_failure_after_gate_records_stage_error_not_gate_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """commit_gate 取得后 stage 异常：记录 stage.error，不伪装成 gate.error。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )

    def exploding_stage(*args: object, **kwargs: object) -> object:
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(
        engine._runtime,  # noqa: SLF001
        "stage_from_source",
        exploding_stage,
    )
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    try:
        with pytest.raises(RuntimeError, match="stage exploded"):
            await engine._on_turn_committed(event)  # noqa: SLF001
        events = _milestone_events(caplog)
        assert "akasha.commit_gate.wait.done" in events
        assert "akasha.stage.start" in events
        assert "akasha.stage.error" in events
        assert "akasha.turn_commit.error" in events
        assert "akasha.commit_gate.wait.error" not in events
        assert "akasha.stage.done" not in events
        assert "akasha.turn_commit.done" not in events
        for event_name in ("akasha.stage.error", "akasha.turn_commit.error"):
            for record in _milestone_records(caplog, event_name):
                assert record.akashic_fields["outcome"] == "error"
                assert record.levelno == logging.ERROR
                assert record.akashic_fields["session_id"] == "test:one"
                assert record.akashic_fields["turn_id"] == "event-turn-1"
                assert record.akashic_fields["client_message_id"] == "event-client-1"
        _assert_span_closed(
            caplog,
            _milestone_span(caplog, "akasha.turn_commit.start"),
            "turn_commit",
        )
    finally:
        _close_engine(engine)


@pytest.mark.asyncio
async def test_turn_commit_cancelled_while_waiting_on_commit_gate_closes_wait_and_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """等待 commit gate 时取消：commit_gate.wait.cancelled + turn_commit.cancelled。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    event = _event(
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
        turn_id="event-turn-1",
        client_message_id="event-client-1",
    )
    gate_task = asyncio.create_task(engine._commit_gate.acquire())  # noqa: SLF001
    await gate_task
    commit_task = asyncio.create_task(engine._on_turn_committed(event))  # noqa: SLF001
    try:
        await _wait_for_milestone(caplog, "akasha.commit_gate.wait.start")
        assert not commit_task.done()
        assert "akasha.commit_gate.wait.done" not in _milestone_events(caplog)
        commit_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await commit_task
        events = _milestone_events(caplog)
        assert "akasha.commit_gate.wait.cancelled" in events
        assert "akasha.turn_commit.cancelled" in events
        assert "akasha.commit_gate.wait.done" not in events
        assert "akasha.stage.start" not in events
        assert "akasha.turn_commit.done" not in events
        for event_name in (
            "akasha.commit_gate.wait.cancelled",
            "akasha.turn_commit.cancelled",
        ):
            for record in _milestone_records(caplog, event_name):
                assert record.akashic_fields["outcome"] == "cancelled"
                assert record.akashic_fields["duration_ms"] is not None
                assert record.akashic_fields["session_id"] == "test:one"
                assert record.akashic_fields["turn_id"] == "event-turn-1"
                assert record.akashic_fields["client_message_id"] == "event-client-1"
        _assert_span_closed(
            caplog,
            _milestone_span(caplog, "akasha.turn_commit.start"),
            "turn_commit",
        )
    finally:
        if not commit_task.done():
            await commit_task
        engine._commit_gate.release()  # noqa: SLF001
        _close_engine(engine)


@pytest.mark.asyncio
async def test_query_runtime_failure_closes_runtime_query_with_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """runtime.query 异常：runtime.query.error + query.error，均不伪装 wait 失败。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)

    def exploding_query_turn(*args: object, **kwargs: object) -> object:
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(
        engine._runtime,  # noqa: SLF001
        "query_turn",
        exploding_query_turn,
    )
    session_token = current_session_key.set("test:one")
    client_token = current_client_message_id.set("client-message-1")
    turn_token = running_turn_id.set("turn-1")
    try:
        with pytest.raises(RuntimeError, match="graph exploded"):
            await engine.query(
                _query(
                    "alpha follow",
                    started + timedelta(minutes=5),
                    intent="context",
                )
            )
        assert _milestone_events(caplog) == [
            "akasha.query.start",
            "akasha.embed.start",
            "akasha.embed.done",
            "akasha.commit_gate.wait.start",
            "akasha.commit_gate.wait.done",
            "akasha.prior_publication.wait.start",
            "akasha.prior_publication.wait.done",
            "akasha.runtime.query.start",
            "akasha.runtime.query.error",
            "akasha.query.error",
        ]
        for event_name in ("akasha.runtime.query.error", "akasha.query.error"):
            for record in _milestone_records(caplog, event_name):
                assert record.akashic_fields["outcome"] == "error"
                assert record.levelno == logging.ERROR
                assert record.akashic_fields["session_id"] == "test:one"
                assert record.akashic_fields["turn_id"] == "turn-1"
                assert record.akashic_fields["client_message_id"] == "client-message-1"
        assert "akasha.runtime.query.done" not in _milestone_events(caplog)
        assert "akasha.query.done" not in _milestone_events(caplog)
        _assert_span_closed(
            caplog,
            _milestone_span(caplog, "akasha.query.start"),
            "query",
        )
    finally:
        current_session_key.reset(session_token)
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        _close_engine(engine)


def test_embedding_preflight_excludes_scheduler_but_reports_dialogue_gap(
    tmp_path: Path,
) -> None:
    """Keep background jobs out while failing on missing dialogue vectors."""

    # 1. Create one complete scheduler turn and one incomplete user turn.
    sessions = tmp_path / "sessions.db"
    _create_sessions(sessions)
    started = datetime(2026, 7, 6, tzinfo=timezone.utc)
    _append_turn(
        sessions,
        sequence=0,
        user="background",
        assistant="background answer",
        started=started,
        session_key="scheduler:job",
        with_embeddings=False,
        user_extra={"effects": {"post_commit": "suppress"}},
    )
    _append_turn(
        sessions,
        sequence=0,
        user="dialogue",
        assistant="dialogue answer",
        started=started,
        with_embeddings=False,
    )

    # 2. Only the real dialogue gap belongs in the strict migration report.
    audit = audit_source_embeddings(
        sessions,
        BuildConfig(
            embedding_model="embedding-model",
            embedding_dimension=2,
        ),
    )
    assert audit.eligible_turns == 1
    assert [issue.session_key for issue in audit.issues] == [
        "test:one",
        "test:one",
    ]


def test_embedding_preflight_excludes_legacy_interrupted_turn(
    tmp_path: Path,
) -> None:
    """Keep an interrupted placeholder outside the strict replay boundary."""

    # 1. Persist the historical marker without embeddings or structured flags.
    sessions = tmp_path / "sessions.db"
    _create_sessions(sessions)
    _append_turn(
        sessions,
        sequence=0,
        user="unfinished request",
        assistant="[interrupted]",
        started=datetime(2026, 7, 6, tzinfo=timezone.utc),
        with_embeddings=False,
    )

    # 2. Classify the turn as excluded instead of an embedding defect.
    audit = audit_source_embeddings(
        sessions,
        BuildConfig(
            embedding_model="embedding-model",
            embedding_dimension=2,
        ),
    )

    assert audit.complete
    assert audit.eligible_turns == 0
    assert audit.excluded_interrupted_turns == 1
    assert audit.issues == ()


def test_sparse_builder_ignores_twenty_proactive_messages_before_user_turn(
    tmp_path: Path,
) -> None:
    """忽略二十条无人回复的 proactive，只学习随后的完整 interaction。"""

    # 1. 构造 proactive 先提交、completed interaction 后提交的 canonical 顺序。
    sessions = tmp_path / "sessions.db"
    index = tmp_path / "index.db"
    _create_sessions(sessions)
    started = datetime(2026, 8, 6, tzinfo=timezone.utc)
    rows = [
        (
            "u1",
            20,
            "user",
            "alpha",
            {"control_turn_id": "t1", "turn_input_ordinal": 0},
        ),
        (
            "u2",
            21,
            "user",
            "beta",
            {"control_turn_id": "t1", "turn_input_ordinal": 1},
        ),
        (
            "u3",
            22,
            "user",
            "gamma",
            {"control_turn_id": "t1", "turn_input_ordinal": 2},
        ),
        (
            "a1",
            23,
            "assistant",
            "final",
            {
                "control_turn_id": "t1",
                "turn_terminal": True,
                "turn_input_count": 3,
            },
        ),
    ]
    vectors = {
        "u1": [1.0, 0.0],
        "u2": [0.0, 1.0],
        "u3": [1.0, 0.0],
        "a1": [0.0, 1.0],
    }
    with closing(sqlite3.connect(sessions)) as connection, connection:
        connection.execute(
            "INSERT INTO sessions VALUES ('test:one', ?, ?, 0, NULL)",
            (started.isoformat(), started.isoformat()),
        )
        for sequence in range(20):
            connection.execute(
                "INSERT INTO messages VALUES (?, 'test:one', ?, ?, ?, NULL, ?, ?)",
                (
                    f"p{sequence}",
                    sequence,
                    "assistant",
                    f"proactive {sequence}",
                    json.dumps(
                        {
                            "proactive": True,
                            "delivery_id": f"delivery-{sequence}",
                            "control_turn_id": f"proactive-turn-{sequence}",
                        }
                    ),
                    (started + timedelta(seconds=sequence)).isoformat(),
                ),
            )
        for message_id, seq, role, content, extra in rows:
            connection.execute(
                "INSERT INTO messages VALUES (?, 'test:one', ?, ?, ?, NULL, ?, ?)",
                (
                    message_id,
                    seq,
                    role,
                    content,
                    json.dumps(extra),
                    (started + timedelta(seconds=seq)).isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO message_embeddings VALUES (?, ?, 'embedding-model', ?, 2, ?, ?)",
                (
                    message_id,
                    hashlib.sha256(content.encode()).hexdigest(),
                    sqlite3.Binary(struct.pack("<2f", *vectors[message_id])),
                    started.isoformat(),
                    started.isoformat(),
                ),
            )

    # 2. 构建器只把显式 interaction 聚合成学习样本。
    result = build_sparse_index(
        sessions,
        index,
        BuildConfig(embedding_model="embedding-model", embedding_dimension=2),
    )
    with closing(sqlite3.connect(index)) as connection:
        turn = connection.execute(
            "SELECT user_message_id, assistant_message_id, user_text FROM sparse_turns"
        ).fetchone()
        dense = connection.execute(
            "SELECT source_id, embedding FROM turn_dense WHERE field = 'user'"
        ).fetchone()

    assert result.indexed_turns == 1
    assert turn == ("u1", "a1", "alpha\n\nbeta\n\ngamma")
    assert dense is not None and dense[0] == "u1"
    user_dense = struct.unpack("<2f", dense[1])
    assert user_dense == pytest.approx((2 / math.sqrt(5), 1 / math.sqrt(5)))


def test_sparse_builder_appends_resumed_turn_by_terminal_commit_time(
    tmp_path: Path,
) -> None:
    """跨天恢复的 Turn 按最终提交时间追加，不按首个输入时间倒插。"""

    # 1. 先建立已有高水位，再补入一个更早开始但更晚完成的 interaction。
    sessions = tmp_path / "sessions.db"
    index = tmp_path / "index.db"
    _create_sessions(sessions)
    existing_started = datetime(2026, 8, 15, tzinfo=timezone.utc)
    _append_turn(
        sessions,
        sequence=0,
        user="existing",
        assistant="existing answer",
        started=existing_started,
        session_key="test:existing",
        with_embeddings=True,
    )
    config = BuildConfig(
        embedding_model="embedding-model",
        embedding_dimension=2,
    )
    build_sparse_index(sessions, index, config)

    resumed_started = datetime(2026, 8, 12, tzinfo=timezone.utc)
    resumed_at = datetime(2026, 8, 16, tzinfo=timezone.utc)
    rows = (
        (
            "resume:u1",
            0,
            "user",
            "old input",
            {"control_turn_id": "turn:resume", "turn_input_ordinal": 0},
            resumed_started,
        ),
        (
            "resume:u2",
            1,
            "user",
            "current input",
            {"control_turn_id": "turn:resume", "turn_input_ordinal": 1},
            resumed_at,
        ),
        (
            "resume:a1",
            2,
            "assistant",
            "final answer",
            {
                "control_turn_id": "turn:resume",
                "turn_terminal": True,
                "turn_input_count": 2,
            },
            resumed_at + timedelta(seconds=10),
        ),
    )
    with closing(sqlite3.connect(sessions)) as connection, connection:
        connection.execute(
            "INSERT INTO sessions VALUES ('test:resume', ?, ?, 0, NULL)",
            (resumed_started.isoformat(), resumed_at.isoformat()),
        )
        for message_id, seq, role, content, extra, timestamp in rows:
            connection.execute(
                "INSERT INTO messages VALUES (?, 'test:resume', ?, ?, ?, NULL, ?, ?)",
                (
                    message_id,
                    seq,
                    role,
                    content,
                    json.dumps(extra),
                    timestamp.isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO message_embeddings VALUES (?, ?, 'embedding-model', ?, 2, ?, ?)",
                (
                    message_id,
                    hashlib.sha256(content.encode()).hexdigest(),
                    sqlite3.Binary(struct.pack("<2f", 1.0, 0.0)),
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                ),
            )

    # 2. 增量与 replay 都把恢复 Turn 放在最终提交位置。
    result = build_sparse_index(sessions, index, config)
    turns = load_turns(index)

    assert result.indexed_turns == 1
    assert [turn.turn_id for turn in turns] == [
        "test:existing:0::test:existing:1",
        "resume:u1::resume:a1",
    ]
    assert turns[1].started_at == resumed_started.isoformat()
    assert turns[1].committed_at == (resumed_at + timedelta(seconds=10)).isoformat()
    assert turns[1].inter_gap_seconds == pytest.approx(86400.0)


def test_sparse_builder_rejects_orphan_message_push_turn(tmp_path: Path) -> None:
    """不能把工具使用记录误当成合法的主动 Turn 身份。"""

    # 1. 构造只有 message_push 执行证据、没有 proactive 身份的孤儿 Turn。
    sessions = tmp_path / "sessions.db"
    index = tmp_path / "index.db"
    _create_sessions(sessions)
    started = datetime(2026, 8, 6, tzinfo=timezone.utc)
    with closing(sqlite3.connect(sessions)) as connection, connection:
        connection.execute(
            "INSERT INTO sessions VALUES ('test:one', ?, ?, 0, NULL)",
            (started.isoformat(), started.isoformat()),
        )
        connection.execute(
            "INSERT INTO messages VALUES (?, 'test:one', 0, ?, ?, NULL, ?, ?)",
            (
                "a1",
                "assistant",
                "orphan outbound",
                json.dumps(
                    {
                        "control_turn_id": "orphan-turn",
                        "tools_used": ["message_push"],
                    }
                ),
                started.isoformat(),
            ),
        )

    # 2. 普通孤儿 Turn 仍由 replay 边界明确拒绝。
    with pytest.raises(ValueError, match="同 turn transcript 结构无效: orphan-turn"):
        build_sparse_index(
            sessions,
            index,
            BuildConfig(
                embedding_model="embedding-model",
                embedding_dimension=2,
            ),
        )


def _seed_two_explicit_akasha_turns(workspace: Path) -> datetime:
    """Persist two canonical explicit turns with frozen embeddings."""

    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    started = datetime(2026, 8, 7, tzinfo=timezone.utc)
    rows = [
        (
            "u1",
            0,
            "user",
            "alpha",
            {"control_turn_id": "t1", "turn_input_ordinal": 0},
        ),
        (
            "u2",
            1,
            "user",
            "continue",
            {"control_turn_id": "t1", "turn_input_ordinal": 1},
        ),
        (
            "a1",
            2,
            "assistant",
            "first",
            {
                "control_turn_id": "t1",
                "turn_terminal": True,
                "turn_input_count": 2,
            },
        ),
        (
            "u3",
            3,
            "user",
            "beta",
            {"control_turn_id": "t2", "turn_input_ordinal": 0},
        ),
        (
            "a2",
            4,
            "assistant",
            "second",
            {
                "control_turn_id": "t2",
                "turn_terminal": True,
                "turn_input_count": 1,
            },
        ),
    ]
    with closing(sqlite3.connect(sessions)) as connection, connection:
        connection.execute(
            "INSERT INTO sessions VALUES ('test:one', ?, ?, 5, NULL)",
            (started.isoformat(), started.isoformat()),
        )
        for message_id, seq, role, content, extra in rows:
            connection.execute(
                "INSERT INTO messages VALUES (?, 'test:one', ?, ?, ?, NULL, ?, ?)",
                (
                    message_id,
                    seq,
                    role,
                    content,
                    json.dumps(extra),
                    (started + timedelta(seconds=seq)).isoformat(),
                ),
            )
            vector = (1.0, 0.0) if role == "user" else (0.0, 1.0)
            connection.execute(
                "INSERT INTO message_embeddings VALUES (?, ?, 'embedding-model', ?, 2, ?, ?)",
                (
                    message_id,
                    hashlib.sha256(content.encode()).hexdigest(),
                    sqlite3.Binary(struct.pack("<2f", *vector)),
                    started.isoformat(),
                    started.isoformat(),
                ),
            )
    return started


@pytest.mark.asyncio
async def test_interaction_deletion_rebuilds_akasha_and_clears_pending(
    tmp_path: Path,
) -> None:
    started = _seed_two_explicit_akasha_turns(tmp_path)
    sessions = tmp_path / "sessions.db"

    engine = _engine(tmp_path)
    first_turn = engine._runtime.cycle.turns[0]  # noqa: SLF001
    assert first_turn.user_dense is not None
    _, ticket = engine._runtime.query_turn(  # noqa: SLF001
        text=first_turn.user_text,
        dense=first_turn.user_dense,
        session_key="test:one",
        timestamp=started + timedelta(seconds=10),
    )
    engine._pending["test:one"] = PendingRetrieval(  # noqa: SLF001
        ticket=ticket,
        query_timestamp=started,
        query_text=first_turn.user_text,
        query_dense=first_turn.user_dense.copy(),
        turn_id="attempt-3",
        records=RetrievalRecords(dense=(), completion=()),
    )
    engine._pending["test:other"] = engine._pending["test:one"]  # noqa: SLF001
    store = SessionStore(sessions)
    deletion = await engine.delete_interaction_source(
        "t1",
        lambda: store.delete_interaction("t1"),
    )
    assert deletion is not None

    assert "test:one" not in engine._pending  # noqa: SLF001
    assert "test:other" not in engine._pending  # noqa: SLF001
    assert [turn.turn_id for turn in engine._runtime.cycle.turns] == [  # noqa: SLF001
        "u3::a2"
    ]
    with closing(
        sqlite3.connect(tmp_path / "memory" / "akasha-v2-index.db")
    ) as connection:
        assert connection.execute(
            "SELECT turn_id FROM sparse_turns ORDER BY turn_id"
        ).fetchall() == [("u3::a2",)]
    with closing(sqlite3.connect(tmp_path / "memory" / "akasha.db")) as connection:
        assert connection.execute(
            "SELECT turn_id FROM turn_nodes ORDER BY turn_id"
        ).fetchall() == [("u3::a2",)]
    store.close()
    engine._runtime.close()  # noqa: SLF001
    engine._embedding_store.close()  # noqa: SLF001


@pytest.mark.asyncio
async def test_interaction_deletion_waits_for_in_flight_source_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = _seed_two_explicit_akasha_turns(tmp_path)
    with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection, connection:
        connection.execute(
            "DELETE FROM message_embeddings WHERE message_id IN ('u3', 'a2')"
        )
        connection.execute("DELETE FROM messages WHERE id IN ('u3', 'a2')")
    engine = _engine(tmp_path)
    with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection, connection:
        connection.execute(
            "INSERT INTO messages VALUES ('u3', 'test:one', 3, 'user', 'beta', NULL, ?, ?)",
            (
                json.dumps({"control_turn_id": "t2", "turn_input_ordinal": 0}),
                (started + timedelta(seconds=3)).isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO messages VALUES ('a2', 'test:one', 4, 'assistant', 'second', NULL, ?, ?)",
            (
                json.dumps(
                    {
                        "control_turn_id": "t2",
                        "turn_terminal": True,
                        "turn_input_count": 1,
                    }
                ),
                (started + timedelta(seconds=4)).isoformat(),
            ),
        )
    embed_started = asyncio.Event()
    release_embed = asyncio.Event()

    async def blocked_embed_batch(texts: list[str]) -> list[list[float]]:
        embed_started.set()
        await release_embed.wait()
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(
        engine._embedder, "embed_batch", blocked_embed_batch
    )  # noqa: SLF001
    event = TurnCommitted(
        session_key="test:one",
        channel="test",
        chat_id="one",
        input_message="beta",
        persisted_user_message="beta",
        assistant_response="second",
        tools_used=[],
        persisted_user_message_id="u3",
        persisted_user_message_ids=("u3",),
        assistant_message_id="a2",
        timestamp=started,
    )
    commit_task = asyncio.create_task(engine._on_turn_committed(event))  # noqa: SLF001
    await embed_started.wait()

    store = SessionStore(tmp_path / "sessions.db")
    deletion_task = asyncio.create_task(
        engine.delete_interaction_source(
            "t1",
            lambda: store.delete_interaction("t1"),
        )
    )
    await asyncio.sleep(0)
    assert not deletion_task.done()
    release_embed.set()
    await commit_task
    deletion = await deletion_task
    assert deletion is not None

    with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection:
        assert (
            connection.execute(
                "SELECT message_id FROM message_embeddings WHERE message_id IN ('u1', 'u2', 'a1')"
            ).fetchall()
            == []
        )
        assert connection.execute(
            "SELECT message_id FROM message_embeddings WHERE message_id IN ('u3', 'a2') ORDER BY message_id"
        ).fetchall() == [("a2",), ("u3",)]
    assert [turn.turn_id for turn in engine._runtime.cycle.turns] == [  # noqa: SLF001
        "u3::a2"
    ]
    store.close()
    engine._runtime.close()  # noqa: SLF001
    engine._embedding_store.close()  # noqa: SLF001


def test_restart_repairs_sidecars_after_source_interaction_was_deleted(
    tmp_path: Path,
) -> None:
    _seed_two_explicit_akasha_turns(tmp_path)
    original = _engine(tmp_path)
    original._runtime.close()  # noqa: SLF001
    original._embedding_store.close()  # noqa: SLF001
    store = SessionStore(tmp_path / "sessions.db")
    deletion = store.delete_interaction("t1")
    assert deletion is not None
    store.close()

    restarted = _engine(tmp_path)

    assert [
        turn.turn_id for turn in restarted._runtime.cycle.turns
    ] == [  # noqa: SLF001
        "u3::a2"
    ]
    restarted._runtime.close()  # noqa: SLF001
    restarted._embedding_store.close()  # noqa: SLF001


@pytest.mark.asyncio
async def test_failed_interaction_rebuild_keeps_akasha_fail_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_sessions(tmp_path / "sessions.db")
    engine = _engine(tmp_path)

    def fail_rebuild() -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(
        engine._runtime,  # noqa: SLF001
        "rebuild_from_source",
        fail_rebuild,
    )
    deletion = InteractionDeletion(
        control_turn_id="t1",
        session_key="test:one",
        message_ids=("u1", "a1"),
        first_user_message_id="u1",
        old_last_consolidated=2,
        new_last_consolidated=0,
        backup_path=str(tmp_path / "backup.db"),
    )

    with pytest.raises(RuntimeError, match="failed to reconcile"):
        await engine.delete_interaction_source("t1", lambda: deletion)
    with pytest.raises(RuntimeError, match="derived state is stale"):
        engine.list_items_for_dashboard()
    engine._runtime.close()  # noqa: SLF001
    engine._embedding_store.close()  # noqa: SLF001


def test_sparse_builder_preserves_single_user_embedding_bytes(
    tmp_path: Path,
) -> None:
    """Keep legacy turn digests stable when multi-user projection is enabled."""

    # 1. Persist a legacy pair whose stored vector is deliberately not normalized.
    sessions = tmp_path / "sessions.db"
    index = tmp_path / "index.db"
    _create_sessions(sessions)
    started = datetime(2026, 8, 6, tzinfo=timezone.utc)
    with closing(sqlite3.connect(sessions)) as connection, connection:
        connection.execute(
            "INSERT INTO sessions VALUES ('test:one', ?, ?, 0, NULL)",
            (started.isoformat(), started.isoformat()),
        )
        for message_id, seq, role, content, vector in (
            ("u1", 0, "user", "legacy", (3.0, 4.0)),
            ("a1", 1, "assistant", "final", (0.0, 1.0)),
        ):
            connection.execute(
                "INSERT INTO messages VALUES (?, 'test:one', ?, ?, ?, NULL, NULL, ?)",
                (
                    message_id,
                    seq,
                    role,
                    content,
                    (started + timedelta(seconds=seq)).isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO message_embeddings VALUES (?, ?, 'embedding-model', ?, 2, ?, ?)",
                (
                    message_id,
                    hashlib.sha256(content.encode()).hexdigest(),
                    sqlite3.Binary(struct.pack("<2f", *vector)),
                    started.isoformat(),
                    started.isoformat(),
                ),
            )

    # 2. The incremental source digest must continue to see the original bytes.
    build_sparse_index(
        sessions,
        index,
        BuildConfig(embedding_model="embedding-model", embedding_dimension=2),
    )
    with closing(sqlite3.connect(index)) as connection:
        dense = connection.execute(
            "SELECT embedding FROM turn_dense WHERE field = 'user'"
        ).fetchone()

    assert dense is not None
    assert dense[0] == struct.pack("<2f", 3.0, 4.0)


def _engine(
    workspace: Path,
    *,
    event_publisher: EventBus | None = None,
) -> AkashaMemoryEngine:
    return AkashaMemoryEngine(
        embedding=TextEmbeddingSettings(
            base_url="",
            api_key="chat-key",
            model="embedding-model",
            output_dimensionality=2,
        ),
        akasha_config=AkashaConfig(),
        workspace=workspace,
        http_resources=cast(
            Any,
            SimpleNamespace(external_default=object()),
        ),
        event_publisher=event_publisher,
    )


def _query(
    text: str,
    timestamp: datetime,
    *,
    intent: str,
) -> MemoryQuery:
    return MemoryQuery(
        text=text,
        intent=cast(Any, intent),
        effect="stateful",
        scope=MemoryScope(
            session_key="test:one",
            channel="test",
            chat_id="one",
        ),
        limit=5,
        timestamp=timestamp,
    )


def _event(
    *,
    sequence: int,
    user: str,
    assistant: str,
    started: datetime,
    turn_id: str = "",
    client_message_id: str = "",
) -> TurnCommitted:
    return TurnCommitted(
        session_key="test:one",
        channel="test",
        chat_id="one",
        input_message=user,
        persisted_user_message=user,
        assistant_response=assistant,
        tools_used=[],
        turn_id=turn_id,
        client_message_id=client_message_id,
        persisted_user_message_id=f"message:{sequence}",
        assistant_message_id=f"message:{sequence + 1}",
        timestamp=started,
    )


def _create_sessions(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("""
            CREATE TABLE sessions (
                key               TEXT PRIMARY KEY,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                last_consolidated INTEGER NOT NULL DEFAULT 0,
                metadata          TEXT
            )
            """)
        connection.execute("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_chain TEXT,
                extra TEXT,
                ts TEXT NOT NULL,
                UNIQUE(session_key, seq)
            )
            """)
        connection.execute("""
            CREATE TABLE message_embeddings (
                message_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(message_id, model)
            )
            """)


def _append_turn(
    path: Path,
    *,
    sequence: int,
    user: str,
    assistant: str,
    started: datetime,
    session_key: str = "test:one",
    with_embeddings: bool = False,
    assistant_tool_chain: str | None = None,
    user_extra: dict[str, object] | None = None,
    session_metadata: dict[str, object] | None = None,
) -> None:
    assistant_time = started + timedelta(seconds=10)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
            VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(key) DO UPDATE SET metadata = excluded.metadata
            """,
            (
                session_key,
                started.isoformat(),
                assistant_time.isoformat(),
                (None if session_metadata is None else json.dumps(session_metadata)),
            ),
        )
        connection.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    (
                        f"{session_key}:{sequence}"
                        if session_key != "test:one"
                        else f"message:{sequence}"
                    ),
                    session_key,
                    sequence,
                    "user",
                    user,
                    None,
                    (None if user_extra is None else json.dumps(user_extra)),
                    started.isoformat(),
                ),
                (
                    (
                        f"{session_key}:{sequence + 1}"
                        if session_key != "test:one"
                        else f"message:{sequence + 1}"
                    ),
                    session_key,
                    sequence + 1,
                    "assistant",
                    assistant,
                    assistant_tool_chain,
                    None,
                    assistant_time.isoformat(),
                ),
            ],
        )
        if with_embeddings:
            for offset, text in enumerate((user, assistant)):
                message_id = (
                    f"{session_key}:{sequence + offset}"
                    if session_key != "test:one"
                    else f"message:{sequence + offset}"
                )
                vector = (
                    b"\x00\x00\x80?\x00\x00\x00\x00"
                    if "alpha" in text
                    else b"\x00\x00\x00\x00\x00\x00\x80?"
                )
                connection.execute(
                    """
                    INSERT INTO message_embeddings
                    VALUES (?, ?, 'embedding-model', ?, 2, ?, ?)
                    """,
                    (
                        message_id,
                        hashlib.sha256(text.encode()).hexdigest(),
                        vector,
                        started.isoformat(),
                        started.isoformat(),
                    ),
                )


def _close_engine(engine: AkashaMemoryEngine) -> None:
    engine._runtime.close()  # noqa: SLF001
    engine._embedding_store.close()  # noqa: SLF001


def test_build_sparse_index_excludes_marked_and_scheduler_sessions(
    tmp_path: Path,
) -> None:
    sessions_path = tmp_path / "sessions.db"
    _create_sessions(sessions_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        sessions_path,
        sequence=0,
        user="normal user",
        assistant="normal reply",
        started=started,
        session_key="telegram:1",
        with_embeddings=True,
    )
    _append_turn(
        sessions_path,
        sequence=2,
        user="pr noise",
        assistant="pr reply",
        started=started + timedelta(minutes=1),
        session_key="github:owner/repo:pr:1",
        with_embeddings=True,
        user_extra={"effects": {"post_commit": "suppress"}},
    )
    _append_turn(
        sessions_path,
        sequence=4,
        user="scheduler prompt",
        assistant="scheduler reply",
        started=started + timedelta(minutes=2),
        session_key="scheduler:job-1",
        with_embeddings=True,
        user_extra={"effects": {"post_commit": "suppress"}},
    )

    result = build_sparse_index(sessions_path, tmp_path / "index.db")

    # 1. 只有普通 session 成为学习样本，排除计数可见。
    assert result.discovered_turns == 1
    assert result.excluded_memory_turns == 2
    with closing(sqlite3.connect(tmp_path / "index.db")) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='turns_excluded_memory'"
        ).fetchone()
        assert row[0] == "2"


def test_audit_source_embeddings_counts_excluded_memory_turns(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.db"
    _create_sessions(sessions_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        sessions_path,
        sequence=0,
        user="pr noise",
        assistant="pr reply",
        started=started,
        session_key="github:owner/repo:pr:1",
        with_embeddings=True,
        user_extra={"effects": {"post_commit": "suppress"}},
    )

    audit = audit_source_embeddings(sessions_path, BuildConfig())

    assert audit.eligible_turns == 0
    assert audit.excluded_memory_turns == 1


def test_build_sparse_index_fails_loud_on_orphan_messages(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.db"
    _create_sessions(sessions_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        sessions_path,
        sequence=0,
        user="user",
        assistant="reply",
        started=started,
    )
    # 1. 制造孤儿消息：删除 session 行后重建必须 fail-loud，不得静默消失。
    with closing(sqlite3.connect(sessions_path)) as connection, connection:
        connection.execute("DELETE FROM sessions")

    with pytest.raises(ValueError, match="孤儿"):
        build_sparse_index(sessions_path, tmp_path / "index.db")


def test_build_sparse_index_models_arrival_during_previous_reply_as_overlap(
    tmp_path: Path,
) -> None:
    sessions_path = tmp_path / "sessions.db"
    index_path = tmp_path / "index.db"
    _create_sessions(sessions_path)
    started = datetime(2026, 8, 5, 2, 23, tzinfo=timezone.utc)
    _append_turn(
        sessions_path,
        sequence=0,
        user="first",
        assistant="slow reply",
        started=started,
    )
    _append_turn(
        sessions_path,
        sequence=2,
        user="arrived while busy",
        assistant="second reply",
        started=started + timedelta(seconds=5),
    )

    build_sparse_index(sessions_path, index_path)

    with closing(sqlite3.connect(index_path)) as connection:
        timing = connection.execute("""
            SELECT response_delta_seconds, idle_gap_seconds, log_idle_gap,
                   overlap_seconds, log_overlap
            FROM time_observations WHERE turn_id = 'message:2::message:3'
            """).fetchone()
        feature = connection.execute("""
            SELECT value FROM sparse_features
            WHERE turn_id = 'message:2::message:3'
              AND family = 'time_overlap' AND feature_id = 'test'
            """).fetchone()
        stats = connection.execute("""
            SELECT idle_gap_count, mean_log_idle_gap
            FROM time_stats WHERE channel = 'test'
            """).fetchone()

    assert timing == pytest.approx((-5.0, 0.0, 0.0, 5.0, math.log1p(5.0)))
    assert feature == pytest.approx((math.log1p(5.0),))
    assert stats == pytest.approx((1, 0.0))


def test_rebuild_akasha_sidecars_backs_up_and_publishes_verified_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True)
    sessions_path = workspace / "sessions.db"
    index_path = memory_dir / "akasha-v2-index.db"
    graph_path = memory_dir / "akasha.db"
    backup_dir = workspace / "backups" / "v9-test"
    _create_sessions(sessions_path)
    _append_turn(
        sessions_path,
        sequence=0,
        user="alpha request",
        assistant="beta reply",
        started=datetime(2026, 8, 5, tzinfo=timezone.utc),
        with_embeddings=True,
    )
    with closing(sqlite3.connect(index_path)) as connection, connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO metadata VALUES ('index_version', '8')")
    source_sha = hashlib.sha256(sessions_path.read_bytes()).hexdigest()
    host = Config(
        provider="openai",
        model="chat-model",
        api_key="chat-key",
        system_prompt="system",
        memory=HostMemoryConfig(
            enabled=True,
            embedding=MemoryEmbeddingConfig(
                model="embedding-model",
                output_dimensionality=2,
            ),
        ),
    )
    monkeypatch.setattr(Config, "load", lambda *_args, **_kwargs: host)

    rebuilt = rebuild_akasha_sidecars(
        config_path=tmp_path / "config.toml",
        workspace=workspace,
        backup_dir=backup_dir,
        accepted_versions={"8"},
    )

    with closing(sqlite3.connect(index_path)) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'index_version'"
        ).fetchone() == ("10",)
        assert connection.execute("SELECT COUNT(*) FROM sparse_turns").fetchone() == (
            1,
        )
    with closing(sqlite3.connect(backup_dir / "index-before.db")) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'index_version'"
        ).fetchone() == ("8",)
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    assert rebuilt
    assert graph_path.is_file()
    assert manifest["indexVersion"] == "10"
    assert manifest["candidateMemorySha256"]
    assert hashlib.sha256(sessions_path.read_bytes()).hexdigest() == source_sha


def test_online_runtime_rebuilds_previous_sparse_index_version(
    tmp_path: Path,
) -> None:
    """Boot rebuilds an obsolete derived index without changing sessions.db."""

    # 1. Freeze one canonical source turn and an obsolete v9 sidecar.
    sessions_path = tmp_path / "sessions.db"
    index_path = tmp_path / "memory" / "akasha-v2-index.db"
    memory_path = tmp_path / "memory" / "akasha.db"
    _create_sessions(sessions_path)
    _append_turn(
        sessions_path,
        sequence=0,
        user="alpha request",
        assistant="beta reply",
        started=datetime(2026, 8, 5, tzinfo=timezone.utc),
        with_embeddings=True,
    )
    index_path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(index_path)) as connection, connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO metadata VALUES ('index_version', '9')")
    source_sha = hashlib.sha256(sessions_path.read_bytes()).hexdigest()

    # 2. Boot through the production runtime and verify the derived pair only.
    runtime = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    runtime.close()

    with closing(sqlite3.connect(index_path)) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'index_version'"
        ).fetchone() == (INDEX_VERSION,)
        assert connection.execute("SELECT COUNT(*) FROM sparse_turns").fetchone() == (
            1,
        )
    assert memory_path.is_file()
    assert hashlib.sha256(sessions_path.read_bytes()).hexdigest() == source_sha


def test_online_runtime_keeps_previous_sidecar_when_version_rebuild_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed version rebuild leaves both canonical input and old sidecar intact."""

    # 1. Freeze the source and obsolete sidecar before the injected rebuild failure.
    sessions_path = tmp_path / "sessions.db"
    index_path = tmp_path / "memory" / "akasha-v2-index.db"
    memory_path = tmp_path / "memory" / "akasha.db"
    _create_sessions(sessions_path)
    index_path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(index_path)) as connection, connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO metadata VALUES ('index_version', '9')")
    source_sha = hashlib.sha256(sessions_path.read_bytes()).hexdigest()
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()

    def fail_rebuild(
        _source_path: Path,
        output_path: Path,
        _config: BuildConfig,
    ) -> object:
        if output_path == index_path:
            raise SparseIndexRebuildRequired("obsolete test index")
        raise RuntimeError("injected candidate rebuild failure")

    monkeypatch.setattr(
        "plugins.akasha.application.runtime.build_sparse_index",
        fail_rebuild,
    )

    # 2. Startup fails loudly without replacing either protected input.
    with pytest.raises(RuntimeError, match="candidate rebuild failure"):
        OnlineMemoryRuntime(
            sessions_path=sessions_path,
            index_path=index_path,
            memory_path=memory_path,
            embedding_model="embedding-model",
            embedding_dimension=2,
            config=MemoryConfig(),
        )

    assert hashlib.sha256(sessions_path.read_bytes()).hexdigest() == source_sha
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == index_sha
    assert not memory_path.exists()


def test_online_runtime_rebuilds_pair_after_crash_between_sidecar_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boot rejects a new index paired with the previous memory snapshot."""

    # 1. Publish one valid pair, then change its canonical source in place.
    sessions_path = tmp_path / "sessions.db"
    index_path = tmp_path / "memory" / "akasha-v2-index.db"
    memory_path = tmp_path / "memory" / "akasha.db"
    _create_sessions(sessions_path)
    _append_turn(
        sessions_path,
        sequence=0,
        user="alpha request",
        assistant="beta reply",
        started=datetime(2026, 8, 5, tzinfo=timezone.utc),
        with_embeddings=True,
    )
    runtime = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    with closing(sqlite3.connect(sessions_path)) as connection, connection:
        connection.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("replacement request", "message:0"),
        )

    # 2. Crash after publishing the rebuilt index but before its memory pair.
    real_replace = os.replace

    def fail_memory_publication(source: Path, destination: Path) -> None:
        if Path(destination) == memory_path:
            raise OSError("injected crash between sidecar replaces")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_memory_publication)
    with pytest.raises(OSError, match="between sidecar replaces"):
        runtime.rebuild_from_source()
    runtime.close()
    monkeypatch.setattr(os, "replace", real_replace)
    mixed_index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    mixed_index_state_sha = sparse_index_state_sha256(index_path)
    with closing(sqlite3.connect(memory_path)) as connection:
        stale_memory_sha = connection.execute(
            "SELECT value FROM metadata WHERE key = 'source_index_sha256'"
        ).fetchone()
        stale_memory_state_sha = connection.execute(
            "SELECT value FROM metadata " "WHERE key = 'source_index_state_sha256'"
        ).fetchone()
    assert stale_memory_sha != (mixed_index_sha,)
    assert stale_memory_state_sha != (mixed_index_state_sha,)

    # 3. The next boot detects the mixed pair and deterministically republishes it.
    recovered = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    recovered.close()
    final_index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    with closing(sqlite3.connect(memory_path)) as connection:
        recovered_memory_sha = connection.execute(
            "SELECT value FROM metadata WHERE key = 'source_index_sha256'"
        ).fetchone()
        recovered_memory_state_sha = connection.execute(
            "SELECT value FROM metadata " "WHERE key = 'source_index_state_sha256'"
        ).fetchone()
    assert recovered_memory_sha == (final_index_sha,)
    assert recovered_memory_state_sha == (sparse_index_state_sha256(index_path),)


def test_online_runtime_reopens_unchanged_sidecars_without_rebuilding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正常进程重启必须复用未变化的派生库组合。"""

    # 1. 发布一组完整的稀疏索引与记忆快照。
    sessions_path = tmp_path / "sessions.db"
    index_path = tmp_path / "memory" / "akasha-v2-index.db"
    memory_path = tmp_path / "memory" / "akasha.db"
    _create_sessions(sessions_path)
    _append_turn(
        sessions_path,
        sequence=0,
        user="alpha request",
        assistant="beta reply",
        started=datetime(2026, 8, 5, tzinfo=timezone.utc),
        with_embeddings=True,
    )
    first = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    first.close()
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    memory_sha = hashlib.sha256(memory_path.read_bytes()).hexdigest()

    # 2. 再次打开同一组合，不得进入全量重建路径。
    def reject_rebuild(_runtime: OnlineMemoryRuntime) -> MemoryCycle:
        raise AssertionError("unchanged sidecars must not rebuild")

    monkeypatch.setattr(
        OnlineMemoryRuntime,
        "_fresh_rebuild_from_source",
        reject_rebuild,
    )
    reopened = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    reopened.close()

    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == index_sha
    assert hashlib.sha256(memory_path.read_bytes()).hexdigest() == memory_sha


def test_online_runtime_ignores_excluded_turn_diagnostics_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """排除计数增长不得触发既有记忆图的全量重放。"""

    # 1. 发布一个旧格式快照，随后只新增明确排除的 session。
    sessions_path = tmp_path / "sessions.db"
    index_path = tmp_path / "memory" / "akasha-v2-index.db"
    memory_path = tmp_path / "memory" / "akasha.db"
    _create_sessions(sessions_path)
    started = datetime(2026, 8, 5, tzinfo=timezone.utc)
    _append_turn(
        sessions_path,
        sequence=0,
        user="alpha request",
        assistant="beta reply",
        started=started,
        with_embeddings=True,
    )
    first = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    first.close()
    with closing(sqlite3.connect(memory_path)) as connection, connection:
        connection.execute("DELETE FROM metadata WHERE key='source_index_state_sha256'")
    _append_turn(
        sessions_path,
        sequence=0,
        user="programmatic request",
        assistant="programmatic reply",
        started=started + timedelta(minutes=1),
        session_key="github:owner/repo:pr:1",
        with_embeddings=True,
        user_extra={"effects": {"post_commit": "suppress"}},
    )

    # 2. 重启只更新诊断计数，并把旧快照升级为逻辑索引身份。
    def reject_rebuild(_runtime: OnlineMemoryRuntime) -> MemoryCycle:
        raise AssertionError("excluded turns must not rebuild learned memory")

    monkeypatch.setattr(
        OnlineMemoryRuntime,
        "_fresh_rebuild_from_source",
        reject_rebuild,
    )
    reopened = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    try:
        assert reopened.cycle.state_version == 1
    finally:
        reopened.close()

    with closing(sqlite3.connect(index_path)) as connection:
        excluded = connection.execute(
            "SELECT value FROM metadata WHERE key='turns_excluded_memory'"
        ).fetchone()
    with closing(sqlite3.connect(memory_path)) as connection:
        state_hash = connection.execute(
            "SELECT value FROM metadata WHERE key='source_index_state_sha256'"
        ).fetchone()
    assert excluded == ("1",)
    assert state_hash is not None

    # 3. 已迁移快照再次遇到排除计数增长，仍只更新诊断元数据。
    _append_turn(
        sessions_path,
        sequence=0,
        user="scheduler request",
        assistant="scheduler reply",
        started=started + timedelta(minutes=2),
        session_key="scheduler:job-1",
        with_embeddings=True,
        user_extra={"effects": {"post_commit": "suppress"}},
    )
    modern = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    try:
        assert modern.cycle.state_version == 1
    finally:
        modern.close()
    with closing(sqlite3.connect(index_path)) as connection:
        modern_excluded = connection.execute(
            "SELECT value FROM metadata WHERE key='turns_excluded_memory'"
        ).fetchone()
    assert modern_excluded == ("2",)


def test_online_runtime_replays_an_appended_suffix_without_rebuilding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """进程离线期间新增的 Turn 必须增量追平而不是重建完整图。"""

    # 1. 先持久化一个与稀疏索引完全对齐的记忆快照。
    sessions_path = tmp_path / "sessions.db"
    index_path = tmp_path / "memory" / "akasha-v2-index.db"
    memory_path = tmp_path / "memory" / "akasha.db"
    _create_sessions(sessions_path)
    _append_turn(
        sessions_path,
        sequence=0,
        user="alpha request",
        assistant="beta reply",
        started=datetime(2026, 8, 5, tzinfo=timezone.utc),
        with_embeddings=True,
    )
    first = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    first.close()

    # 2. 模拟 Core 停机窗口新增一个合法 append-only Turn。
    _append_turn(
        sessions_path,
        sequence=2,
        user="gamma request",
        assistant="delta reply",
        started=datetime(2026, 8, 5, 0, 1, tzinfo=timezone.utc),
        with_embeddings=True,
    )

    def reject_rebuild(_runtime: OnlineMemoryRuntime) -> MemoryCycle:
        raise AssertionError("append-only suffix must not rebuild")

    monkeypatch.setattr(
        OnlineMemoryRuntime,
        "_fresh_rebuild_from_source",
        reject_rebuild,
    )

    # 3. 重启只回放新增后缀，并发布新的完整配对快照。
    reopened = OnlineMemoryRuntime(
        sessions_path=sessions_path,
        index_path=index_path,
        memory_path=memory_path,
        embedding_model="embedding-model",
        embedding_dimension=2,
        config=MemoryConfig(),
    )
    try:
        assert reopened.cycle.state_version == 2
        assert len(reopened.cycle.turns) == 2
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("context_mass", "query", "context", "expected_node"),
    (
        (1.0, np.asarray([1.0, 0.0]), np.zeros(2), 0),
        (0.0, np.zeros(2), np.asarray([0.0, 1.0]), 1),
    ),
)
def test_burst_seed_keeps_the_only_available_evidence_source(
    context_mass: float,
    query: np.ndarray,
    context: np.ndarray,
    expected_node: int,
) -> None:
    """An unavailable source cannot erase the other side of the mixture."""

    seed = features_module._mix_sources(  # pyright: ignore[reportPrivateUsage]
        {
            "query_dense": query,
            "query_bm25": np.zeros(2),
            "context_dense": context,
            "context_bm25": np.zeros(2),
        },
        context_mass,
    )

    assert seed == ((expected_node, 1.0),)


def _write_inspector_config(workspace: Path) -> None:
    plugin_dir = builtin_plugin_data_dir("akasha", workspace)
    ensure_workspace_plugin_data_dir(plugin_dir, workspace)
    (plugin_dir / "config.local.toml").write_text(
        render_akasha_config(),
        encoding="utf-8",
    )


def test_inspector_overview_is_empty_before_first_akasha_commit(tmp_path: Path) -> None:
    _write_inspector_config(tmp_path)
    reader = AkashaInspectorReader(
        memory_root=tmp_path / "memory",
        config=load_akasha_config(
            builtin_plugin_data_dir("akasha", tmp_path) / "config.local.toml"
        ),
    )
    sidecars = (reader.paths.memory, reader.paths.index)

    assert reader.get_overview() == {
        "available": True,
        "total": 0,
        "latest_at": None,
        "earliest_at": None,
    }
    assert reader.list_turns() == ([], 0)
    assert reader.latest_for_session("fresh:empty") is None
    assert reader.get_turn("missing") is None
    assert reader.for_assistant_message("fresh:empty", "missing") is None
    assert all(not path.exists() for path in sidecars)


def test_inspector_accepts_valid_empty_sparse_projection_without_memory(
    tmp_path: Path,
) -> None:
    _write_inspector_config(tmp_path)
    reader = AkashaInspectorReader(
        memory_root=tmp_path / "memory",
        config=load_akasha_config(
            builtin_plugin_data_dir("akasha", tmp_path) / "config.local.toml"
        ),
    )
    reader.paths.index.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(reader.paths.index)) as connection:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("index_version", INDEX_VERSION),
                ("tool_chain_projection_version", TOOL_CHAIN_PROJECTION_VERSION),
            ),
        )
        connection.commit()

    assert reader.get_overview() == {
        "available": True,
        "total": 0,
        "latest_at": None,
        "earliest_at": None,
    }
    assert reader.list_turns() == ([], 0)
    assert reader.latest_for_session("fresh:empty") is None
    assert reader.get_turn("missing") is None
    assert reader.for_assistant_message("fresh:empty", "missing") is None
    assert not reader.paths.memory.exists()


@pytest.mark.parametrize("present", ("memory", "index"))
def test_inspector_partial_sidecar_fails_loud(
    tmp_path: Path,
    present: str,
) -> None:
    _write_inspector_config(tmp_path)
    reader = AkashaInspectorReader(
        memory_root=tmp_path / "memory",
        config=load_akasha_config(
            builtin_plugin_data_dir("akasha", tmp_path) / "config.local.toml"
        ),
    )
    present_path = getattr(reader.paths, present)
    present_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(present_path).close()

    with pytest.raises(sqlite3.OperationalError):
        reader.list_turns()


@pytest.mark.asyncio
async def test_concurrent_queries_in_one_turn_pair_distinct_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同一 turn 并发两次 query：span 独立，各自 total/子阶段同 span 且闭合。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    session_token = current_session_key.set("test:one")
    client_token = current_client_message_id.set("client-message-1")
    turn_token = running_turn_id.set("turn-1")
    try:
        results = await asyncio.gather(
            engine.query(_query("alpha one", started, intent="context")),
            engine.query(_query("beta two", started, intent="context")),
        )
        assert len(results) == 2
        triples = _milestone_triples(caplog)
        span_ids = sorted({span for span, _, _ in triples})
        assert len(span_ids) == 2
        for span_id in span_ids:
            _assert_span_closed(caplog, span_id, "query")
        for span_id, operation, event in triples:
            assert operation == "query"
        assert sum(1 for _, _, event in triples if event == "akasha.query.start") == 2
        assert sum(1 for _, _, event in triples if event == "akasha.query.done") == 2
    finally:
        current_session_key.reset(session_token)
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        _close_engine(engine)


@pytest.mark.asyncio
async def test_query_and_same_turn_commit_use_distinct_spans_and_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同 turn 的 query 与 turn_commit：span 不同，embed 可按 operation 区分。"""

    caplog.set_level(logging.INFO, logger="plugins.akasha.engine")
    _create_sessions(tmp_path / "sessions.db")
    monkeypatch.setattr("plugins.akasha.engine.Embedder", _Embedder)
    engine = _engine(tmp_path)
    started = datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    _append_turn(
        tmp_path / "sessions.db",
        sequence=0,
        user="alpha start",
        assistant="first answer",
        started=started,
    )
    session_token = current_session_key.set("test:one")
    client_token = current_client_message_id.set("client-message-1")
    turn_token = running_turn_id.set("turn-1")
    try:
        await engine._on_turn_committed(  # noqa: SLF001
            _event(
                sequence=0,
                user="alpha start",
                assistant="first answer",
                started=started,
                turn_id="event-turn-1",
                client_message_id="event-client-1",
            )
        )
        await engine._wait_for_publication()  # noqa: SLF001
        await engine.query(
            _query(
                "alpha follow",
                started + timedelta(minutes=5),
                intent="context",
            )
        )
        triples = _milestone_triples(caplog)
        spans = {(span, operation) for span, operation, _ in triples}
        assert len(spans) == 2
        assert sorted(operation for _, operation in spans) == [
            "query",
            "turn_commit",
        ]
        query_span = next(span for span, operation in spans if operation == "query")
        commit_span = next(
            span for span, operation in spans if operation == "turn_commit"
        )
        assert query_span != commit_span
        _assert_span_closed(caplog, query_span, "query")
        _assert_span_closed(caplog, commit_span, "turn_commit")
        # 同一事件名 akasha.embed.* 可按 operation 区分归属。
        assert {
            (span, operation)
            for span, operation, event in triples
            if event == "akasha.embed.start"
        } == {(query_span, "query"), (commit_span, "turn_commit")}
        assert {
            operation
            for _, operation, event in triples
            if event.startswith("akasha.embed.")
        } == {"query", "turn_commit"}
    finally:
        current_session_key.reset(session_token)
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        _close_engine(engine)
