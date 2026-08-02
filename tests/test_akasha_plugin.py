from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.config_models import (
    Config,
    MemoryConfig as HostMemoryConfig,
    MemoryEmbeddingConfig,
)
from agent.control.context import current_turn_id
from agent.looping.ports import MemoryServices
from agent.plugins import Plugin
from agent.plugins.context import PluginContext, PluginKVStore
from agent.plugins.manifest import (
    builtin_plugin_data_dir,
    ensure_workspace_plugin_data_dir,
)
from agent.tools.base import ToolExecutionContext, tool_execution_context_scope
from agent.tools.recall_memory import RecallMemoryTool
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import RetrievalRequest
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryQuery, MemoryScope
from core.memory.plugin import MemoryPlugin as MemoryPluginProtocol
from plugins.akasha.application.rebuild import rebuild_memory
from plugins.akasha.config import AkashaConfig, render_akasha_config
from plugins.akasha.dashboard import register as register_dashboard
from plugins.akasha.engine import (
    ActiveRecallSnapshot,
    AkashaFeedbackPersistModule,
    AkashaMemoryEngine,
)
from plugins.akasha.inspector import AkashaInspectorReader
from plugins.akasha.infrastructure.persistence import (
    logical_state_sha256,
)
from plugins.akasha.infrastructure.sparse_index import (
    BuildConfig,
    audit_source_embeddings,
    build_sparse_index,
)
from plugins.akasha.memory_plugin import MemoryPlugin
from plugins.akasha.plugin import (
    AkashaPlugin,
    _empty_mobile_recall,
    _mobile_recall_lane,
)


class _Embedder:
    def __init__(self, **values: object) -> None:
        self.model = str(values["model"])
        self.output_dimensionality = int(
            cast(int, values["output_dimensionality"])
        )

    async def embed(self, text: str) -> list[float]:
        if self.output_dimensionality != 2:
            raise ValueError("test embedder requires two dimensions")
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    async def aclose(self) -> None:
        return None


def test_akasha_v2_registers_both_host_protocols() -> None:
    plugin = AkashaPlugin()
    assert isinstance(plugin, Plugin)
    assert isinstance(MemoryPlugin(), MemoryPluginProtocol)
    assert MemoryPlugin.plugin_id == "akasha"
    assert len(plugin.after_reasoning_modules()) == 1
    assert plugin.dashboard_module() == "dashboard.py"
    mobile = plugin.mobile_ui()
    assert mobile.module == "mobile_ui.js"
    assert mobile.stylesheet == "mobile_ui.css"
    assert mobile.slots == ("turn.before_reasoning",)
    assert mobile.navigation is not None
    assert mobile.navigation.label == "Akasha Inspector"


def test_feedback_marker_is_exported_before_user_message_persistence() -> None:
    # Import after passive-turn modules settle their lifecycle import cycle.
    from agent.lifecycle.phases import default_after_reasoning_modules

    plugin = AkashaPlugin()
    modules = default_after_reasoning_modules(
        EventBus(),
        cast(Any, SimpleNamespace()),
        cast(Any, plugin.after_reasoning_modules()),
    )
    slots = [module.slot for module in modules]

    assert slots.index("akasha.feedback.persist") < slots.index(
        "after_reasoning.persist_user"
    )


@pytest.mark.asyncio
async def test_feedback_persistence_is_inert_when_akasha_is_not_active() -> None:
    frame = SimpleNamespace(slots={"existing": "value"})
    owner = SimpleNamespace(
        context=SimpleNamespace(memory_engine=None)
    )

    result = await AkashaFeedbackPersistModule(owner).run(frame)

    assert result is frame
    assert frame.slots == {"existing": "value"}


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
    token = current_turn_id.set("turn:feedback")
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

        # 3. Export both independent markers onto the current user Message.
        frame = SimpleNamespace(slots={})
        owner = SimpleNamespace(
            context=SimpleNamespace(memory_engine=engine)
        )
        await AkashaFeedbackPersistModule(owner).run(frame)
        forgotten_marker = frame.slots["persist:user:akasha_forget"]
        remembered_marker = frame.slots["persist:user:akasha_reinforce"]
        assert forgotten_marker["action"] == "forget"
        assert forgotten_marker["target_message_ids"] == ["message:1"]
        assert forgotten_marker["target_turn_ids"] == [
            "message:0::message:1"
        ]
        assert remembered_marker["action"] == "remember"
        assert remembered_marker["target_message_ids"] == [
            "current_user_message"
        ]
        assert remembered_marker["target_turn_ids"] == ["current_turn"]
        assert engine.take_staged_feedback("turn:feedback") == ()
    finally:
        current_turn_id.reset(token)
        _close_engine(engine)


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
            spec
            for spec in engine.tool_profile().tools
            if spec.name == "forget_memory"
        )
        assert spec.tool_class is not None
        tool = spec.tool_class(engine, spec)
        token = current_turn_id.set("turn:feedback")
        try:
            result = json.loads(
                await tool.execute(
                    message_ids=["message:0::message:1"],
                )
            )
        finally:
            current_turn_id.reset(token)
        assert result["status"] == "not_staged"
        assert result["error"] == "messages_not_in_akasha"

        token = current_turn_id.set("turn:feedback")
        try:
            current = json.loads(
                await tool.execute(
                    message_ids=["current_user_message"],
                )
            )
        finally:
            current_turn_id.reset(token)
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
    token = current_turn_id.set("turn:correction")
    try:
        assert json.loads(
            await forget.execute(message_ids=["message:1"])
        )["status"] == "staged"
        assert json.loads(
            await remember.execute(
                message_ids=["current_user_message"]
            )
        )["status"] == "staged"
        frame = SimpleNamespace(slots={})
        owner = SimpleNamespace(
            context=SimpleNamespace(memory_engine=engine)
        )
        await AkashaFeedbackPersistModule(owner).run(frame)
    finally:
        current_turn_id.reset(token)

    # 2. Persist those marker fields on the next canonical user Message.
    correction_time = started + timedelta(minutes=5)
    user_extra = {
        key.removeprefix("persist:user:"): value
        for key, value in frame.slots.items()
    }
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
    with closing(
        sqlite3.connect(tmp_path / "memory" / "akasha.db")
    ) as connection:
        assert connection.execute(
            """
            SELECT event_id, action, target_turn_node_id, boost
            FROM feedback_events
            ORDER BY event_id, action, target_turn_node_id
            """
        ).fetchall() == [
            (1, "forget", 0, 1.0),
            (1, "remember", 1, 3.0),
        ]

    # 3. Future direct-dense and graph completion lanes hide the old turn.
    recalled = await engine.query(
        _query(
            "alpha old wrong claim",
            correction_time + timedelta(minutes=5),
            intent="answer",
        )
    )
    assert all(
        record.id != "message:0::message:1"
        for record in recalled.records
    )

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


def test_active_mobile_recall_marks_temporary_absence_as_pending() -> None:
    assert _empty_mobile_recall()["pending"] is False
    assert _empty_mobile_recall(pending=True)["pending"] is True


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
    first_query = await engine.query(
        _query("alpha start", started, intent="context")
    )
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
    plugin = AkashaPlugin()
    plugin.context = PluginContext(
        event_bus=cast(Any, None),
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=Path("plugins/akasha"),
        data_dir=builtin_plugin_data_dir("akasha", tmp_path),
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        memory_engine=engine,
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
            plugin.mobile_ui_query,
            "recall.current",
            {"message_id": f"assistant:{active_turn_id}"},
            session_id="test:one",
            turn_id=active_turn_id,
        )
    )
    assert await asyncio.to_thread(wait_started.wait, 1.0)
    context = await DefaultMemoryRetrievalPipeline(
        MemoryServices(engine=engine)
    ).retrieve(
        RetrievalRequest(
            message="alpha follow",
            session_key="test:one",
            channel="test",
            chat_id="one",
            history=[],
            session_metadata={},
            turn_id=active_turn_id,
            timestamp=next_time,
        )
    )
    active_mobile = await active_query
    pending = engine._pending["test:one"]  # noqa: SLF001
    tool = RecallMemoryTool(
        engine,
        cast(Any, engine.tool_profile().recall),
    )
    before_recall = logical_state_sha256(
        tmp_path / "memory" / "akasha.db"
    )
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
    after_recall = logical_state_sha256(
        tmp_path / "memory" / "akasha.db"
    )
    assert rendered["count"] == 1
    assert before_recall == after_recall
    assert engine._pending["test:one"] is pending  # noqa: SLF001
    assert context.block.startswith("# Akasha memory now=07-06")
    assert "## 左脑记忆：精确回忆" in context.block
    assert f'assistant="{"A" * 50}..."' in context.block
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
    with closing(
        sqlite3.connect(tmp_path / "memory" / "akasha.db")
    ) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM recall_runs"
        ).fetchone() == (2,)
        assert connection.execute(
            "SELECT COUNT(*) FROM activation_runs"
        ).fetchone() == (0,)

    # 5. Inspector reconstructs the exact prior-only lanes without writes.
    _write_inspector_config(tmp_path)
    before_memory = logical_state_sha256(
        tmp_path / "memory" / "akasha.db"
    )
    reader = AkashaInspectorReader(tmp_path)
    overview = reader.get_overview()
    rows, total = reader.list_turns(q="alpha follow")
    detail = reader.get_turn(str(rows[0]["query_id"]))
    assert overview["total"] == 2
    assert total == 1
    assert detail is not None
    assert detail["query_text"] == "alpha follow"
    assert detail["recall_capture_available"] is True
    assert detail["left_count"] == 1
    assert detail["tool_left_count"] == 1
    assert detail["tool_right_count"] == 1
    assert cast(list[dict[str, object]], detail["left"])[0][
        "user_text"
    ] == "alpha start"
    assert "## 左脑记忆：精确回忆" in str(
        detail["text_block_preview"]
    )
    assert detail["activation_capture_available"] is False
    assert before_memory == logical_state_sha256(
        tmp_path / "memory" / "akasha.db"
    )

    # 6. The desktop API exposes the same state through read-only routes.
    app = FastAPI()
    app.state.memory_admin = engine
    register_dashboard(app, Path("plugins/akasha"), tmp_path)
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
    mobile = plugin.mobile_ui_query(
        "recall.current",
        {"message_id": "message:3"},
        session_id="test:one",
        turn_id=None,
    )
    recent = plugin.mobile_ui_query(
        "inspector.recent",
        {},
        session_id=None,
        turn_id=None,
    )
    mobile_detail = plugin.mobile_ui_query(
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
        assert connection.execute(
            "SELECT COUNT(*) FROM sparse_turns"
        ).fetchone() == (1,)

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


def _engine(workspace: Path) -> AkashaMemoryEngine:
    return AkashaMemoryEngine(
        config=Config(
            provider="openai",
            model="chat-model",
            api_key="chat-key",
            system_prompt="system",
            memory=HostMemoryConfig(
                embedding=MemoryEmbeddingConfig(
                    model="embedding-model",
                    output_dimensionality=2,
                )
            ),
        ),
        akasha_config=AkashaConfig(),
        workspace=workspace,
        http_resources=cast(
            Any,
            SimpleNamespace(external_default=object()),
        ),
        event_publisher=None,
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
) -> TurnCommitted:
    return TurnCommitted(
        session_key="test:one",
        channel="test",
        chat_id="one",
        input_message=user,
        persisted_user_message=user,
        assistant_response=assistant,
        tools_used=[],
        persisted_user_message_id=f"message:{sequence}",
        assistant_message_id=f"message:{sequence + 1}",
        timestamp=started,
    )


def _create_sessions(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE sessions (
                key               TEXT PRIMARY KEY,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                last_consolidated INTEGER NOT NULL DEFAULT 0,
                metadata          TEXT
            )
            """
        )
        connection.execute(
            """
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
            """
        )
        connection.execute(
            """
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
            """
        )


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
                (
                    None
                    if session_metadata is None
                    else json.dumps(session_metadata)
                ),
            ),
        )
        connection.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    f"{session_key}:{sequence}"
                    if session_key != "test:one"
                    else f"message:{sequence}",
                    session_key,
                    sequence,
                    "user",
                    user,
                    None,
                    (
                        None
                        if user_extra is None
                        else json.dumps(user_extra)
                    ),
                    started.isoformat(),
                ),
                (
                    f"{session_key}:{sequence + 1}"
                    if session_key != "test:one"
                    else f"message:{sequence + 1}",
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
        session_metadata={"skip_post_memory": True},
    )
    _append_turn(
        sessions_path,
        sequence=4,
        user="scheduler prompt",
        assistant="scheduler reply",
        started=started + timedelta(minutes=2),
        session_key="scheduler:job-1",
        with_embeddings=True,
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
        session_metadata={"skip_post_memory": True},
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


def _write_inspector_config(workspace: Path) -> None:
    plugin_dir = builtin_plugin_data_dir("akasha", workspace)
    ensure_workspace_plugin_data_dir(plugin_dir, workspace)
    (plugin_dir / "config.local.toml").write_text(
        render_akasha_config(),
        encoding="utf-8",
    )
