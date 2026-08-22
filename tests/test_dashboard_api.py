from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
import sqlite3
import shutil
import threading
import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient as _RawTestClient

import bootstrap.dashboard_api as dashboard_api
from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.dashboard_host import (
    PluginDashboardHost,
    SnapshotDashboardMiddleware,
)
from agent.plugins.generation_activity_host import ActivityHost
from agent.plugins.generation_private_proactive_host import PrivateProactiveHost
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from bootstrap.dashboard_api import (
    _dashboard_plugin_dirs,
    create_dashboard_app as _create_dashboard_app,
)
from plugins.akasha.engine import AkashaMemoryEngine
from plugins.default_memory.engine import DefaultMemoryEngine
from memory2.store import MemoryStore2
from proactive_v2.state import ProactiveStateStore
from session.embedding_store import MessageEmbeddingStore
from session.store import SessionStore
from agent.model_runtime.context_compaction import source_plan_digest


class _TrackedTestClient(_RawTestClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._is_closed = False

    def close(self) -> None:
        if self._is_closed:
            return
        self._is_closed = True
        super().close()

    def __del__(self) -> None:
        if not self._is_closed:
            try:
                self.close()
            except Exception:
                pass


TestClient = _TrackedTestClient


@pytest.fixture(autouse=True)
def _isolate_dashboard_plugin_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """让 Dashboard 测试只观察各自声明的 HOME/manifest。"""

    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))


def _use_writable_dashboard_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plugin_ids: set[str],
) -> None:
    """复制指定插件源码，并为只读 Gate 合成最小测试面板产物。"""

    discovered = _dashboard_plugin_dirs(Path.cwd())
    missing = plugin_ids - discovered.keys()
    assert not missing
    writable: dict[str, Path] = {}
    for plugin_id in sorted(plugin_ids):
        target = tmp_path / "dashboard-plugins" / plugin_id
        shutil.copytree(discovered[plugin_id], target)
        for source in target.glob("dashboard_panel*.ts*"):
            source.with_suffix(".js").write_text(
                "export default {};\n",
                encoding="utf-8",
            )
        writable[plugin_id] = target
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _project_root: dict(writable),
    )


class _DashboardMemoryAdmin:
    def __init__(self, workspace) -> None:
        self._store = MemoryStore2(workspace / "memory" / "memory2.db")

    def describe(self):
        return DefaultMemoryEngine.DESCRIPTOR

    def keyword_match_procedures(self, action_tokens: list[str]):
        return self._store.keyword_match_procedures(action_tokens)

    def list_events_by_time_range(self, time_start, time_end, *, limit: int = 200):
        return self._store.list_events_by_time_range(time_start, time_end, limit=limit)

    def list_items_for_dashboard(self, **kwargs):
        return self._store.list_items_for_dashboard(**kwargs)

    def get_item_for_dashboard(self, item_id: str, *, include_embedding: bool = False):
        return self._store.get_item_for_dashboard(
            item_id, include_embedding=include_embedding
        )

    def update_item_for_dashboard(self, item_id: str, **kwargs):
        return self._store.update_item_for_dashboard(item_id, **kwargs)

    def delete_item(self, item_id: str) -> bool:
        return self._store.delete_item(item_id)

    def delete_items_batch(self, ids: list[str]) -> int:
        return self._store.delete_items_batch(ids)

    def find_similar_items_for_dashboard(self, item_id: str, **kwargs):
        return self._store.find_similar_items_for_dashboard(item_id, **kwargs)

    def close(self) -> None:
        self._store.close()


class _AkashaDashboardMemoryAdmin:
    def describe(self):
        return AkashaMemoryEngine.DESCRIPTOR

    def close(self) -> None:
        return None


def create_dashboard_app(tmp_path, **kwargs):
    kwargs.setdefault("memory_admin", _DashboardMemoryAdmin(tmp_path))
    return _create_dashboard_app(tmp_path, **kwargs)


def _seed_explicit_interaction(
    workspace: Path,
    *,
    last_consolidated: int,
) -> tuple[str, list[str]]:
    store = SessionStore(workspace / "sessions.db")
    timestamp = "2026-08-07T10:00:00+08:00"
    rows = store.persist_session(
        "mobile:review",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={},
        messages=[
            {
                "role": "user",
                "content": "legacy",
                "timestamp": timestamp,
                "extra": {},
            },
            {
                "role": "assistant",
                "content": "old",
                "timestamp": timestamp,
                "extra": {},
            },
            {
                "role": "user",
                "content": "u1",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn-review",
                    "turn_input_ordinal": 0,
                },
            },
            {
                "role": "user",
                "content": "u2",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn-review",
                    "turn_input_ordinal": 1,
                },
            },
            {
                "role": "user",
                "content": "u3",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn-review",
                    "turn_input_ordinal": 2,
                },
            },
            {
                "role": "assistant",
                "content": "final",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn-review",
                    "turn_terminal": True,
                    "turn_input_count": 3,
                },
            },
            {
                "role": "user",
                "content": "later",
                "timestamp": timestamp,
                "extra": {},
            },
            {
                "role": "assistant",
                "content": "later-a",
                "timestamp": timestamp,
                "extra": {},
            },
        ],
    )
    if last_consolidated:
        legacy = rows[0]
        store.persist_compaction(
            session_key="mobile:review",
            trigger="test",
            summary="legacy checkpoint",
            source_ref="test:legacy",
            source_plan_digest=source_plan_digest(
                (
                    {
                        "id": str(legacy["id"]),
                        "seq": int(legacy["seq"]),
                        "unit_ref": "test:legacy",
                        "message": dict(legacy),
                    },
                )
            ),
            source_from_seq=int(legacy["seq"]),
            consolidated_through_seq=int(legacy["seq"]),
            source_message_ids=[str(legacy["id"])],
            retained_tail=[],
            model_runtime_id="test",
            model="test",
            context_window=100,
            threshold_tokens=80,
            hard_input_tokens=90,
            keep_recent_tokens=10,
            tokens_before=10,
            tokens_after=5,
            summary_usage={},
            generation=2,
            parent_generation=0,
        )
    if last_consolidated > 2:
        interaction = rows[2]
        store.persist_compaction(
            session_key="mobile:review",
            trigger="test",
            summary="interaction checkpoint",
            source_ref="test:interaction",
            source_plan_digest=source_plan_digest(
                (
                    {
                        "id": str(interaction["id"]),
                        "seq": int(interaction["seq"]),
                        "unit_ref": "test:interaction",
                        "message": dict(interaction),
                    },
                )
            ),
            source_from_seq=int(interaction["seq"]),
            consolidated_through_seq=int(interaction["seq"]),
            source_message_ids=[str(interaction["id"])],
            retained_tail=[],
            model_runtime_id="test",
            model="test",
            context_window=100,
            threshold_tokens=80,
            hard_input_tokens=90,
            keep_recent_tokens=10,
            tokens_before=10,
            tokens_after=5,
            summary_usage={},
            generation=last_consolidated,
            parent_generation=2,
        )
    store.close()
    return "turn-review", [str(row["id"]) for row in rows[2:6]]


@pytest.mark.asyncio
async def test_dashboard_lifespan_swallows_its_own_compile_cancellation(
    tmp_path, monkeypatch
):
    async def _pending_compile(
        _queue: dashboard_api._PluginPanelBuildQueue,
    ) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        dashboard_api,
        "_compile_pending_plugins_async",
        _pending_compile,
    )
    app = create_dashboard_app(tmp_path)

    async with app.router.lifespan_context(app):
        pass


@pytest.mark.asyncio
async def test_dashboard_waits_for_async_closeable() -> None:
    closed = False

    class Closeable:
        async def close(self) -> None:
            nonlocal closed
            closed = True

    await dashboard_api._close_dashboard_value(Closeable())

    assert closed


@pytest.mark.asyncio
async def test_dashboard_lifespan_exposes_unexpected_compile_failure(
    tmp_path, monkeypatch
):
    async def _failed_compile(
        _queue: dashboard_api._PluginPanelBuildQueue,
    ) -> None:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(
        dashboard_api,
        "_compile_pending_plugins_async",
        _failed_compile,
    )
    app = create_dashboard_app(tmp_path)

    with pytest.raises(RuntimeError, match="compile failed"):
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_pending_panel_probe_is_terminated_and_drained_on_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = dashboard_api._PluginPanelBuildQueue()
    plugin_dir = tmp_path / "plugin"
    output_dir = tmp_path / "output"
    plugin_dir.mkdir()
    queue.add(tmp_path, plugin_dir, output_dir)
    communicate_started = asyncio.Event()
    wait_started = asyncio.Event()
    release_wait = asyncio.Event()

    class Process:
        returncode: int | None = None
        terminated = False
        waited = False

        async def communicate(self) -> tuple[bytes, bytes]:
            communicate_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            raise AssertionError("graceful termination should finish")

        async def wait(self) -> int:
            self.waited = True
            wait_started.set()
            await release_wait.wait()
            self.returncode = -15
            return self.returncode

    process = Process()

    async def create_process(*_args: object, **_kwargs: object) -> Process:
        return process

    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: ["npx"])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    task = asyncio.create_task(dashboard_api._compile_pending_plugins_async(queue))
    await communicate_started.wait()
    task.cancel()
    await wait_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    release_wait.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated is True
    assert process.waited is True


class _ManualMemoryOptimizer:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        block: bool = False,
    ) -> None:
        self.error = error
        self.block = block
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._running = False
        self.raise_busy = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def optimize(self) -> None:
        if self.raise_busy:
            from proactive_v2.memory_optimizer import MemoryOptimizerBusy

            raise MemoryOptimizerBusy("busy")
        self._running = True
        self.calls += 1
        self.started.set()
        try:
            if self.block:
                await asyncio.to_thread(self.release.wait, 1.0)
            if self.error is not None:
                raise self.error
        finally:
            self._running = False


def _seed_workspace(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(
        key="telegram:100",
        metadata={"title": "alpha room"},
        last_user_at="2026-04-19T10:00:00+08:00",
    )
    store.create_session(
        key="cli:local",
        metadata={"title": "beta room"},
        last_proactive_at="2026-04-19T09:00:00+08:00",
    )
    store.insert_message(
        "telegram:100",
        role="user",
        content="你好，今晚睡觉了吗",
        ts="2026-04-19T10:01:00+08:00",
        seq=0,
        extra={"pinned": True},
    )
    store.insert_message(
        "telegram:100",
        role="assistant",
        content="还没睡呢",
        ts="2026-04-19T10:02:00+08:00",
        seq=1,
        tool_chain=[{"text": "reply", "calls": []}],
        extra={"source": "test"},
    )
    store.insert_message(
        "cli:local",
        role="user",
        content="hello from cli",
        ts="2026-04-19T09:01:00+08:00",
        seq=0,
    )
    store.close()

    memory_store = MemoryStore2(tmp_path / "memory" / "memory2.db", vec_dim=2)
    memory_store.upsert_item(
        memory_type="preference",
        summary="喜欢奶茶，少糖去冰",
        embedding=[1.0, 0.0],
        source_ref="telegram:100:pref",
        extra={"scope_channel": "telegram", "scope_chat_id": "100"},
        happened_at="2026-04-19T10:03:00+08:00",
        emotional_weight=6,
    )
    memory_store.upsert_item(
        memory_type="event",
        summary="昨晚和朋友去散步",
        embedding=[0.9, 0.1],
        source_ref="telegram:100:event",
        extra={"scope_channel": "telegram", "scope_chat_id": "100"},
        happened_at="2026-04-18T20:00:00+08:00",
    )
    memory_store.upsert_item(
        memory_type="profile",
        summary="常驻上海",
        embedding=None,
        source_ref="cli:local:profile",
        extra={"scope_channel": "cli", "scope_chat_id": "local"},
    )
    memory_store.close()

    proactive_store = ProactiveStateStore(tmp_path / "proactive.db")
    proactive_store.mark_delivery(
        "telegram:100",
        "delivery-a",
        now=datetime.fromisoformat("2026-04-19T02:05:00+00:00"),
    )
    proactive_store.mark_delivery(
        "cli:local",
        "delivery-b",
        now=datetime.fromisoformat("2026-04-19T02:06:00+00:00"),
    )
    proactive_store.mark_context_only_send(
        "telegram:100",
        now=datetime.fromisoformat("2026-04-19T02:31:00+00:00"),
    )
    proactive_store.mark_drift_run(
        "telegram:100",
        now=datetime.fromisoformat("2026-04-19T02:32:00+00:00"),
    )
    proactive_store.close()

    conn = sqlite3.connect(tmp_path / "proactive.db")
    conn.execute(
        """
        INSERT INTO tick_log(
            tick_id, session_key, started_at, finished_at, gate_exit,
            terminal_action, skip_reason, steps_taken, alert_count,
            content_count, context_count, interesting_ids, discarded_ids,
            cited_ids, drift_entered, final_message
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "tick-1",
            "telegram:100",
            "2026-04-19T02:40:00+00:00",
            "2026-04-19T02:40:05+00:00",
            None,
            "reply",
            None,
            3,
            1,
            2,
            1,
            '["mcp:feed:feed-1"]',
            '["rss:news:rss-9"]',
            '["mcp:feed:feed-1"]',
            0,
            "记得早点休息",
        ),
    )
    conn.execute(
        """
        INSERT INTO tick_log(
            tick_id, session_key, started_at, finished_at, gate_exit,
            terminal_action, skip_reason, steps_taken, alert_count,
            content_count, context_count, interesting_ids, discarded_ids,
            cited_ids, drift_entered, final_message
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "tick-2",
            "cli:local",
            "2026-04-19T03:00:00+00:00",
            "2026-04-19T03:00:01+00:00",
            "busy",
            "skip",
            "busy",
            0,
            0,
            0,
            0,
            "[]",
            "[]",
            "[]",
            1,
            None,
        ),
    )
    conn.execute(
        """
        INSERT INTO tick_step_log(
            tick_id, step_index, phase, tool_name, tool_call_id, tool_args_json,
            tool_result_text, terminal_action_after, skip_reason_after,
            interesting_ids_after, discarded_ids_after, cited_ids_after,
            final_message_after
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "tick-1",
            1,
            "loop",
            "message_push",
            "call-1",
            '{"message":"记得早点休息","evidence":["mcp:feed:feed-1"]}',
            '{"ok": true}',
            None,
            "",
            '["mcp:feed:feed-1"]',
            "[]",
            "[]",
            "",
        ),
    )
    conn.execute(
        """
        INSERT INTO tick_step_log(
            tick_id, step_index, phase, tool_name, tool_call_id, tool_args_json,
            tool_result_text, terminal_action_after, skip_reason_after,
            interesting_ids_after, discarded_ids_after, cited_ids_after,
            final_message_after
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "tick-1",
            2,
            "loop",
            "finish_turn",
            "call-2",
            '{"decision":"reply"}',
            '{"ok": true}',
            "reply",
            "",
            '["mcp:feed:feed-1"]',
            "[]",
            '["mcp:feed:feed-1"]',
            "记得早点休息",
        ),
    )
    conn.commit()
    conn.close()


def _seed_pending_compaction_prepare(
    workspace: Path,
    session_key: str,
    message_ids: list[str],
) -> str:
    store = SessionStore(workspace / "sessions.db")
    meta = store.get_session_meta(session_key)
    assert meta is not None
    rows = {str(row["id"]): row for row in store.fetch_session_messages(session_key)}
    selected = [rows[message_id] for message_id in message_ids]
    assert selected
    source_ref = f"test:pending:{session_key}"
    store.prepare_compaction(
        session_key=session_key,
        session_created_at=str(meta["created_at"]),
        generation=1,
        parent_generation=0,
        source_ref=source_ref,
        source_from_seq=min(int(row["seq"]) for row in selected),
        consolidated_through_seq=max(int(row["seq"]) for row in selected),
        source_message_ids=tuple(message_ids),
        retained_tail=(),
    )
    store.close()
    return source_ref


def test_list_sessions_with_filters(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        resp = client.get(
            "/api/dashboard/sessions",
            params={"q": "alpha", "channel": "telegram"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total"] == 1
        assert payload["items"][0]["key"] == "telegram:100"
        assert payload["items"][0]["message_count"] == 2

        messages_resp = client.get(
            "/api/dashboard/messages",
            params={"sort_by": "seq", "sort_order": "asc"},
        )
        assert messages_resp.status_code == 200
        assert messages_resp.json()["items"][0]["seq"] == 0


def test_update_and_delete_session(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        patch_resp = client.patch(
            "/api/dashboard/sessions/telegram:100",
            json={"metadata": {"title": "patched"}},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["metadata"]["title"] == "patched"

        retired_patch = client.patch(
            "/api/dashboard/sessions/telegram:100",
            json={"last_consolidated": 9},
        )
        assert retired_patch.status_code == 422
        assert (
            client.get("/api/dashboard/sessions/telegram:100").json()[
                "last_consolidated"
            ]
            == 0
        )

        delete_resp = client.delete("/api/dashboard/sessions/telegram:100")
        assert delete_resp.status_code == 200
        delete_payload = delete_resp.json()
        assert delete_payload["action_source"] == "dashboard.session_delete"
        assert delete_payload["result"] == "committed"
        assert delete_payload["audit_id"]
        assert Path(delete_payload["backup_path"]).is_file()

        get_resp = client.get("/api/dashboard/sessions/telegram:100")
        assert get_resp.status_code == 404


def test_dashboard_update_message_returns_409_when_session_is_active(tmp_path) -> None:
    _seed_workspace(tmp_path)
    runtime_store = SessionStore(tmp_path / "sessions.db")
    message = runtime_store.get_message("telegram:100:1")
    assert message is not None
    assert runtime_store.acquire_session_admission(
        "telegram:100",
        "admission:dashboard-edit",
    )
    runtime_store.close()

    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.patch(
            f"/api/dashboard/messages/{message['id']}",
            json={"content": "blocked"},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "session_busy",
        "session_key": "telegram:100",
    }
    inspector = SessionStore(tmp_path / "sessions.db")
    assert inspector.get_message(str(message["id"]))["content"] == "还没睡呢"
    inspector.release_session_admission("admission:dashboard-edit")
    inspector.close()


@pytest.mark.parametrize("operation", ("edit", "delete", "batch", "interaction"))
def test_dashboard_returns_distinct_409_for_pending_compaction_prepare(
    tmp_path,
    operation: str,
) -> None:
    if operation == "interaction":
        turn_id, message_ids = _seed_explicit_interaction(
            tmp_path,
            last_consolidated=0,
        )
        session_key = "mobile:review"
        target_id = message_ids[1]
    else:
        _seed_workspace(tmp_path)
        session_key = "telegram:100"
        target_id = "telegram:100:1"
        message_ids = [target_id]
        turn_id = ""
    source_ref = _seed_pending_compaction_prepare(
        tmp_path,
        session_key,
        message_ids,
    )

    with TestClient(create_dashboard_app(tmp_path)) as client:
        if operation == "edit":
            response = client.patch(
                f"/api/dashboard/messages/{target_id}",
                json={"content": "blocked"},
            )
        elif operation == "delete":
            response = client.delete(f"/api/dashboard/messages/{target_id}")
        elif operation == "batch":
            response = client.post(
                "/api/dashboard/messages/batch-delete",
                json={"ids": message_ids},
            )
        else:
            response = client.delete(f"/api/dashboard/interactions/{turn_id}")

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "session_compaction_pending",
        "session_key": session_key,
        "source_ref": source_ref,
    }


@pytest.mark.parametrize("operation", ("single", "batch"))
def test_dashboard_rejects_session_delete_with_pending_prepare(
    tmp_path,
    operation: str,
) -> None:
    _seed_workspace(tmp_path)
    session_key = "telegram:100"
    message_id = "telegram:100:1"
    source_ref = _seed_pending_compaction_prepare(
        tmp_path,
        session_key,
        [message_id],
    )

    with TestClient(create_dashboard_app(tmp_path)) as client:
        if operation == "single":
            response = client.delete(f"/api/dashboard/sessions/{session_key}")
        else:
            response = client.post(
                "/api/dashboard/sessions/batch-delete",
                json={"keys": [session_key], "cascade": True},
            )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "session_compaction_pending"
    assert detail["session_key"] == session_key
    assert detail["source_ref"] == source_ref
    assert detail["audit_id"]

    inspector = SessionStore(tmp_path / "sessions.db")
    assert inspector.session_exists(session_key)
    assert inspector.get_message(message_id) is not None
    prepare = inspector.get_compaction_prepare(session_key, source_ref=source_ref)
    assert prepare is not None
    audit = inspector.get_session_delete_audit(detail["audit_id"])
    assert audit is not None
    assert audit.result == "rejected"
    assert audit.backup_path is None
    inspector.close()
    assert not list((tmp_path / "backups" / "session-deletions").glob("sessions-*.db"))


def test_manual_memory_optimizer_uses_runtime_entrypoint(tmp_path) -> None:
    optimizer = _ManualMemoryOptimizer()
    with TestClient(
        create_dashboard_app(tmp_path, manual_memory_optimizer=optimizer)
    ) as client:
        resp = client.post("/api/dashboard/memory/optimize")

    assert resp.status_code == 202
    assert resp.json()["status"] == "started"
    assert optimizer.started.wait(1.0)
    assert optimizer.calls == 1


def test_manual_memory_optimizer_reports_unavailable_runtime(tmp_path) -> None:
    with TestClient(create_dashboard_app(tmp_path)) as client:
        status_resp = client.get("/api/dashboard/memory/optimizer")
        resp = client.post("/api/dashboard/memory/optimize")

        assert status_resp.status_code == 200
        assert status_resp.json()["enabled"] is False
        assert resp.status_code == 503


def test_manual_memory_optimizer_reports_busy_runtime(tmp_path) -> None:
    optimizer = _ManualMemoryOptimizer(block=True)
    with TestClient(
        create_dashboard_app(tmp_path, manual_memory_optimizer=optimizer)
    ) as client:
        first_resp = client.post("/api/dashboard/memory/optimize")
        assert first_resp.status_code == 202
        assert optimizer.started.wait(1.0)
        status_resp = client.get("/api/dashboard/memory/optimizer")

        busy_resp = client.post("/api/dashboard/memory/optimize")
        optimizer.release.set()

    assert status_resp.status_code == 200
    assert status_resp.json()["enabled"] is True
    assert status_resp.json()["running"] is True
    assert status_resp.json()["last_status"] == "running"
    assert busy_resp.status_code == 409
    assert optimizer.calls == 1


def test_manual_memory_optimizer_skips_when_backend_reports_busy(tmp_path) -> None:
    optimizer = _ManualMemoryOptimizer()
    optimizer.raise_busy = True
    with TestClient(
        create_dashboard_app(tmp_path, manual_memory_optimizer=optimizer)
    ) as client:
        start_resp = client.post("/api/dashboard/memory/optimize")
        status_resp = client.get("/api/dashboard/memory/optimizer")

    assert start_resp.status_code == 202
    assert status_resp.status_code == 200
    assert status_resp.json()["running"] is False
    assert status_resp.json()["last_status"] == "skipped"


def test_list_update_and_batch_delete_messages(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        list_resp = client.get(
            "/api/dashboard/sessions/telegram:100/messages",
            params={"q": "睡", "role": "assistant"},
        )
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert payload["total"] == 1
        message_id = payload["items"][0]["id"]
        embedding_store = MessageEmbeddingStore(tmp_path / "sessions.db")
        embedding_store.upsert(
            message_id=message_id,
            content="还没睡呢",
            model="m",
            embedding=[1.0, 0.0],
        )
        embedding_store.close()

        patch_resp = client.patch(
            f"/api/dashboard/messages/{message_id}",
            json={"content": "已经睡了", "extra": {"edited": True}},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["content"] == "已经睡了"
        assert patch_resp.json()["edited"] is True
        embedding_store = MessageEmbeddingStore(tmp_path / "sessions.db")
        assert (
            embedding_store.get(
                message_id=message_id,
                content="还没睡呢",
                model="m",
            )
            is None
        )
        embedding_store.upsert(
            message_id=message_id,
            content="已经睡了",
            model="m",
            embedding=[1.0, 0.0],
        )
        embedding_store.upsert(
            message_id="cli:local:0",
            content="hello from cli",
            model="m",
            embedding=[0.0, 1.0],
        )
        embedding_store.close()

        batch_resp = client.post(
            "/api/dashboard/messages/batch-delete",
            json={"ids": [message_id, "cli:local:0"]},
        )
        assert batch_resp.status_code == 200
        assert batch_resp.json()["deleted_count"] == 2
        with closing(sqlite3.connect(tmp_path / "sessions.db")) as db:
            embedding_count = db.execute(
                """
                SELECT COUNT(*)
                FROM message_embeddings
                WHERE message_id IN (?, ?)
                """,
                (message_id, "cli:local:0"),
            ).fetchone()[0]
        assert embedding_count == 0

        remain_resp = client.get(
            "/api/dashboard/messages", params={"session_key": "telegram:100"}
        )
        assert remain_resp.status_code == 200
        assert remain_resp.json()["total"] == 1


def test_explicit_interaction_rejects_generic_message_deletes(tmp_path) -> None:
    turn_id, message_ids = _seed_explicit_interaction(
        tmp_path,
        last_consolidated=6,
    )
    with TestClient(create_dashboard_app(tmp_path)) as client:
        single = client.delete(f"/api/dashboard/messages/{message_ids[1]}")
        batch = client.post(
            "/api/dashboard/messages/batch-delete",
            json={"ids": message_ids},
        )

    assert single.status_code == 409
    assert single.json()["detail"] == {
        "code": "interaction_delete_required",
        "message_id": message_ids[1],
        "control_turn_id": turn_id,
    }
    assert batch.status_code == 409
    assert batch.json()["detail"]["control_turn_id"] == turn_id
    store = SessionStore(tmp_path / "sessions.db")
    assert all(store.get_message(message_id) is not None for message_id in message_ids)
    store.close()


def test_delete_interaction_returns_409_when_session_has_active_admission(
    tmp_path,
) -> None:
    turn_id, message_ids = _seed_explicit_interaction(
        tmp_path,
        last_consolidated=0,
    )
    runtime_store = SessionStore(tmp_path / "sessions.db")
    assert runtime_store.acquire_session_admission(
        "mobile:review",
        "admission:dashboard-conflict",
    )

    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.delete(f"/api/dashboard/interactions/{turn_id}")

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "session_busy",
        "session_key": "mobile:review",
    }
    inspector = SessionStore(tmp_path / "sessions.db")
    assert all(
        inspector.get_message(message_id) is not None for message_id in message_ids
    )
    assert inspector.get_session_meta("mobile:review")["last_consolidated"] == 0
    inspector.close()
    runtime_store.release_session_admission("admission:dashboard-conflict")
    runtime_store.close()


@pytest.mark.parametrize(
    ("old_cursor", "expected_cursor"),
    ((0, 0), (4, 2), (8, 2)),
)
def test_delete_interaction_is_atomic_and_repairs_cursor(
    tmp_path,
    old_cursor: int,
    expected_cursor: int,
) -> None:
    turn_id, message_ids = _seed_explicit_interaction(
        tmp_path,
        last_consolidated=old_cursor,
    )
    embedding_store = MessageEmbeddingStore(tmp_path / "sessions.db")
    message_store = SessionStore(tmp_path / "sessions.db")
    for message_id in message_ids:
        message = message_store.get_message(message_id)
        assert message is not None
        embedding_store.upsert(
            message_id=message_id,
            content=str(message["content"]),
            model="m",
            embedding=[1.0, 0.0],
        )
    message_store.close()
    embedding_store.close()

    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.delete(f"/api/dashboard/interactions/{turn_id}")

    assert response.status_code == 200
    assert response.json()["message_ids"] == message_ids
    assert response.json()["old_last_consolidated"] == old_cursor
    assert response.json()["new_last_consolidated"] == expected_cursor
    backup_path = Path(response.json()["backup_path"])
    assert backup_path.is_file()
    assert backup_path.stat().st_mode & 0o777 == 0o600
    assert list(backup_path.parent.glob(".sessions-*.db.tmp")) == []
    store = SessionStore(tmp_path / "sessions.db")
    assert [
        item["content"] for item in store.fetch_session_messages("mobile:review")
    ] == [
        "legacy",
        "old",
        "later",
        "later-a",
    ]
    meta = store.get_session_meta("mobile:review")
    assert meta is not None
    assert meta["last_consolidated"] == expected_cursor
    with closing(sqlite3.connect(tmp_path / "sessions.db")) as database:
        assert (
            database.execute(
                "SELECT COUNT(*) FROM message_embeddings WHERE message_id IN (?, ?, ?, ?)",
                tuple(message_ids),
            ).fetchone()[0]
            == 0
        )
    store.close()

    restored_path = tmp_path / "restored" / "sessions.db"
    restored_path.parent.mkdir()
    shutil.copy2(backup_path, restored_path)
    restored = SessionStore(restored_path)
    assert [
        restored.get_message(message_id) is not None for message_id in message_ids
    ] == [True, True, True, True]
    restored_meta = restored.get_session_meta("mobile:review")
    assert restored_meta is not None
    assert restored_meta["last_consolidated"] == old_cursor
    restored.close()


def test_list_memory_items_with_filters(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        resp = client.get(
            "/api/dashboard/memories",
            params={
                "q": "奶茶",
                "memory_type": "preference",
                "scope_channel": "telegram",
                "has_embedding": "true",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total"] == 1
        assert payload["items"][0]["memory_type"] == "preference"
        assert payload["items"][0]["scope_chat_id"] == "100"
        assert payload["items"][0]["has_embedding"] is True

        status_resp = client.get(
            "/api/dashboard/memories",
            params={
                "memory_type": "profile",
                "status": "active",
                "page_size": 1,
            },
        )
        assert status_resp.status_code == 200
        assert status_resp.json()["total"] == 1
        assert status_resp.json()["items"][0]["memory_type"] == "profile"


def test_list_memory_items_sorts_by_created_at_desc(tmp_path) -> None:
    _seed_workspace(tmp_path)
    conn = sqlite3.connect(tmp_path / "memory" / "memory2.db")
    try:
        conn.execute(
            "UPDATE memory_items SET created_at=? WHERE source_ref=?",
            ("2026-04-19T10:00:00+08:00", "telegram:100:pref"),
        )
        conn.execute(
            "UPDATE memory_items SET created_at=? WHERE source_ref=?",
            ("2026-04-19T11:00:00+08:00", "telegram:100:event"),
        )
        conn.execute(
            "UPDATE memory_items SET created_at=? WHERE source_ref=?",
            ("2026-04-19T12:00:00+08:00", "cli:local:profile"),
        )
        conn.commit()
    finally:
        conn.close()
    with TestClient(create_dashboard_app(tmp_path)) as client:
        resp = client.get(
            "/api/dashboard/memories",
            params={"sort_by": "created_at", "sort_order": "desc"},
        )

        assert resp.status_code == 200
        assert [item["source_ref"] for item in resp.json()["items"]] == [
            "cli:local:profile",
            "telegram:100:event",
            "telegram:100:pref",
        ]


def test_list_memory_items_default_sort_is_created_at_desc(tmp_path) -> None:
    _seed_workspace(tmp_path)
    conn = sqlite3.connect(tmp_path / "memory" / "memory2.db")
    try:
        conn.execute(
            "UPDATE memory_items SET created_at=?, updated_at=? WHERE source_ref=?",
            (
                "2026-04-19T10:00:00+08:00",
                "2026-04-19T13:00:00+08:00",
                "telegram:100:pref",
            ),
        )
        conn.execute(
            "UPDATE memory_items SET created_at=?, updated_at=? WHERE source_ref=?",
            (
                "2026-04-19T11:00:00+08:00",
                "2026-04-19T12:00:00+08:00",
                "telegram:100:event",
            ),
        )
        conn.execute(
            "UPDATE memory_items SET created_at=?, updated_at=? WHERE source_ref=?",
            (
                "2026-04-19T12:00:00+08:00",
                "2026-04-19T11:00:00+08:00",
                "cli:local:profile",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    with TestClient(create_dashboard_app(tmp_path)) as client:
        resp = client.get("/api/dashboard/memories")

        assert resp.status_code == 200
        assert resp.json()["items"][0]["source_ref"] == "cli:local:profile"


def test_get_update_and_delete_memory(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        list_resp = client.get("/api/dashboard/memories", params={"q": "奶茶"})
        memory_id = list_resp.json()["items"][0]["id"]

        get_resp = client.get(
            f"/api/dashboard/memories/{memory_id}",
            params={"include_embedding": "true"},
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["embedding_dim"] == 2

        patch_resp = client.patch(
            f"/api/dashboard/memories/{memory_id}",
            json={
                "status": "superseded",
                "source_ref": "telegram:100:pref:patched",
                "emotional_weight": 9,
                "extra_json": {"scope_channel": "telegram", "scope_chat_id": "100"},
            },
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["status"] == "superseded"
        assert patch_resp.json()["emotional_weight"] == 9
        assert patch_resp.json()["source_ref"] == "telegram:100:pref:patched"

        delete_resp = client.delete(f"/api/dashboard/memories/{memory_id}")
        assert delete_resp.status_code == 200

        missing_resp = client.get(f"/api/dashboard/memories/{memory_id}")
        assert missing_resp.status_code == 404


def test_memory_similar_and_batch_delete(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        list_resp = client.get(
            "/api/dashboard/memories", params={"scope_channel": "telegram"}
        )
        items = list_resp.json()["items"]
        pref = next(item for item in items if item["memory_type"] == "preference")
        event = next(item for item in items if item["memory_type"] == "event")

        similar_resp = client.get(f"/api/dashboard/memories/{pref['id']}/similar")
        assert similar_resp.status_code == 200
        assert similar_resp.json()["total"] >= 1
        assert similar_resp.json()["items"][0]["id"] == event["id"]

        batch_resp = client.post(
            "/api/dashboard/memories/batch-delete",
            json={"ids": [pref["id"], event["id"]]},
        )
        assert batch_resp.status_code == 200
        assert batch_resp.json()["deleted_count"] == 2


def test_memory_dashboard_filters_survive_parallel_requests(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:

        def _fetch(memory_type: str) -> tuple[int, dict]:
            resp = client.get(
                "/api/dashboard/memories",
                params={
                    "status": "active",
                    "memory_type": memory_type,
                    "page_size": 1,
                    "sort_by": "updated_at",
                    "sort_order": "desc",
                },
            )
            return resp.status_code, resp.json()

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(_fetch, ["procedure", "preference", "profile", "event"])
            )

        for status_code, payload in results:
            assert status_code == 200
            assert "total" in payload


def test_standalone_dashboard_does_not_execute_proactive_backend(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "bootstrap.dashboard_api.load_package_manifest",
        lambda: {"default-proactive": True, "wake-proactive": False},
    )
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        assert client.get("/api/dashboard/proactive/overview").status_code == 404
        assert client.get("/api/dashboard/proactive/deliveries").status_code == 404
        assert client.get("/api/dashboard/proactive/tick_logs").status_code == 404


def test_proactive_reader_rejects_corrupt_json() -> None:
    with pytest.raises(ValueError, match="JSON 列表损坏"):
        dashboard_api.ProactiveDashboardReader._decode_json_list("{")

    with pytest.raises(ValueError, match="列表元素类型错误"):
        dashboard_api.ProactiveDashboardReader._decode_json_list('["ok", 1]')

    with pytest.raises(ValueError, match="存储类型错误"):
        dashboard_api.ProactiveDashboardReader._decode_json_object(b"{}")


def test_wake_package_owns_dashboard_visibility(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "bootstrap.dashboard_api.load_package_manifest",
        lambda: {"default-proactive": False, "wake-proactive": True},
    )
    _use_writable_dashboard_plugins(
        tmp_path,
        monkeypatch,
        {"wake-proactive"},
    )
    with TestClient(create_dashboard_app(tmp_path)) as client:
        plugin_ids = {
            item["id"] for item in client.get("/api/dashboard/plugins").json()
        }
        assert "wake-proactive" in plugin_ids
        assert "default-proactive" not in plugin_ids
        assert client.get("/api/dashboard/proactive/overview").status_code == 404
        assert client.get("/api/dashboard/wake-proactive/runs").status_code == 404
        assert client.get("/api/dashboard/wake-proactive/meter").status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("family", "members", "endpoint"),
    (
        (
            "default",
            ("default_proactive", "proactive_flow", "drift_flow"),
            "/api/dashboard/proactive/overview",
        ),
        (
            "wake",
            ("wake_proactive", "wake_proactive_flow", "wake_drift_flow"),
            "/api/dashboard/wake-proactive/runs?page=1&page_size=1",
        ),
    ),
)
async def test_real_private_proactive_snapshot_serves_dashboard_api(
    tmp_path: Path,
    family: Literal["default", "wake"],
    members: tuple[str, ...],
    endpoint: str,
) -> None:
    """真实 committed 私有 catalog 通过 Dashboard HTTP 边界读取领域数据库。"""

    plugins_root = Path(__file__).parents[1] / "plugins"
    manager = PluginManager(
        plugin_dirs=[plugins_root / member for member in members],
        event_bus=EventBus(),
        workspace=tmp_path,
        installed_cache_root=tmp_path / "plugin-home" / "cache",
    )
    manager.bind_activity_host(ActivityHost((PrivateProactiveHost(family),)))
    await manager.load_all()
    try:
        app = create_dashboard_app(tmp_path, plugin_manager=manager)
        with TestClient(app) as client:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert isinstance(response.json(), dict)
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_private_proactive_dashboard_follows_exact_snapshot_switch(
    tmp_path: Path,
) -> None:
    """Default/Wake 路由必须随请求 lease 的 exact snapshot 切换。"""

    # 1. 分别构造两个真实 committed 私有 catalog。
    plugins_root = Path(__file__).parents[1] / "plugins"

    async def load_family(
        family: Literal["default", "wake"],
        members: tuple[str, ...],
    ) -> PluginManager:
        manager = PluginManager(
            plugin_dirs=[plugins_root / member for member in members],
            event_bus=EventBus(),
            workspace=tmp_path / family,
            installed_cache_root=tmp_path / f"plugin-home-{family}" / "cache",
        )
        manager.bind_activity_host(ActivityHost((PrivateProactiveHost(family),)))
        await manager.load_all()
        return manager

    default_manager = await load_family(
        "default",
        ("default_proactive", "proactive_flow", "drift_flow"),
    )
    wake_manager = await load_family(
        "wake",
        ("wake_proactive", "wake_proactive_flow", "wake_drift_flow"),
    )
    default_snapshot = default_manager.current_snapshot
    wake_snapshot = wake_manager.current_snapshot
    assert default_snapshot is not None and wake_snapshot is not None

    # 2. 同一 Dashboard host 为两个 snapshot 建立各自 binding。
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    host = PluginDashboardHost(core_routes=tuple(app.routes))
    host.prepare_initial_snapshot(default_snapshot)
    host.prepare_snapshot(wake_snapshot)

    class Lease:
        def __init__(self, snapshot: object) -> None:
            self.snapshot = snapshot
            self.active = True

        async def __aenter__(self) -> object:
            return self.snapshot

        async def __aexit__(self, *_exc_info: object) -> None:
            self.active = False

    class SwitchingStore:
        def __init__(self) -> None:
            self.current = default_snapshot

        async def acquire(self) -> object:
            return Lease(self.current)

    store = SwitchingStore()
    app.add_middleware(
        SnapshotDashboardMiddleware,
        snapshot_store=cast(Any, store),
    )

    try:
        with TestClient(app) as client:
            assert client.get("/api/dashboard/proactive/overview").status_code == 200
            assert client.get("/api/dashboard/wake-proactive/runs").status_code == 404
            store.current = wake_snapshot
            assert client.get("/api/dashboard/proactive/overview").status_code == 404
            assert client.get("/api/dashboard/wake-proactive/runs").status_code == 200
    finally:
        await default_manager.terminate_all()
        await wake_manager.terminate_all()


def test_dashboard_lists_installed_plugin_panels(tmp_path, monkeypatch) -> None:
    _seed_workspace(tmp_path)
    home = tmp_path / "home"
    plugin_base = (
        home / ".akashic-plugin" / "cache" / "github" / "status_commands"
    )
    plugin_dir = plugin_base / ".artifacts" / "1.0.0-test"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "dashboard.py").write_text(
        "from fastapi import FastAPI\n"
        "def register(app: FastAPI, context):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard_panel.js").write_text(
        "export default {};\n", encoding="utf-8"
    )
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'status_commands'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'status_commands'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '[plugins."status_commands@github"]\nenabled = true\n', encoding="utf-8"
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))

    with TestClient(create_dashboard_app(tmp_path)) as client:
        plugins = client.get("/api/dashboard/plugins").json()
    installed = next(item for item in plugins if item["id"] == "status_commands@github")
    assert installed == {
        "id": "status_commands@github",
        "panels": [
            {
                "name": "dashboard_panel",
                "js_version": str(
                    (plugin_dir / "dashboard_panel.js").stat().st_mtime_ns
                ),
                "has_css": False,
            }
        ],
    }


def test_installed_typescript_panel_uses_runtime_cache_without_source_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace(tmp_path)
    home = tmp_path / "home"
    plugin_base = home / ".akashic-plugin/cache/github/read_only_panel"
    plugin_dir = plugin_base / ".artifacts/1.0.0-test"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'read_only_panel'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'read_only_panel'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    (plugin_dir / "db.py").write_text(
        "def ping(): return 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, context):\n"
        "    from .db import ping\n"
        "    assert ping() == 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard_panel.tsx").write_text(
        "export default function Panel() { return null }\n",
        encoding="utf-8",
    )
    manifest = home / ".akashic-plugin/manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '[plugins."read_only_panel@github"]\nenabled = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))
    compiled_outputs: list[Path] = []
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"read_only_panel@github": plugin_dir},
    )

    def compile_to_cache(
        _cmd: list[str],
        _source: Path,
        output: Path,
        _name: str,
    ) -> None:
        compiled_outputs.append(output)
        output.write_text("export default {};\n", encoding="utf-8")

    monkeypatch.setattr(dashboard_api, "_run_esbuild", compile_to_cache)
    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: ["esbuild"])
    source_before = _test_tree_digest(plugin_dir)
    cache_root = tmp_path / "runtime/dashboard-panels"

    with TestClient(create_dashboard_app(tmp_path)) as client:
        plugins = client.get("/api/dashboard/plugins").json()
        installed = next(
            item for item in plugins if item["id"] == "read_only_panel@github"
        )
        assert len(installed["panels"]) == 1
        assert installed["panels"][0]["name"] == "dashboard_panel"
        assert installed["panels"][0]["has_css"] is False
        assert str(installed["panels"][0]["js_version"]).isdigit()
        response = client.get("/plugins/read_only_panel@github/dashboard_panel.js")
        assert response.status_code == 200
        assert response.text == "export default {};\n"
        assert compiled_outputs
        assert all(output.is_relative_to(cache_root) for output in compiled_outputs)
        assert not (plugin_dir / "dashboard_panel.js").exists()
        assert _test_tree_digest(plugin_dir) == source_before

    assert not cache_root.exists()
    assert _test_tree_digest(plugin_dir) == source_before


def test_installed_fresh_javascript_panel_stays_in_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace(tmp_path)
    home = tmp_path / "home"
    plugin_base = home / ".akashic-plugin/cache/github/published_panel"
    plugin_dir = plugin_base / ".artifacts/1.0.0-test"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'published_panel'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'published_panel'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, context): return None\n",
        encoding="utf-8",
    )
    source = plugin_dir / "dashboard_panel.ts"
    source.write_text("export default 1;\n", encoding="utf-8")
    published = plugin_dir / "dashboard_panel.js"
    published.write_text("export default 1;\n", encoding="utf-8")
    published.touch()
    manifest = home / ".akashic-plugin/manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '[plugins."published_panel@github"]\nenabled = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"published_panel@github": plugin_dir},
    )

    def unexpected_compile(*_args: object) -> None:
        raise AssertionError("fresh published JavaScript must not compile")

    monkeypatch.setattr(dashboard_api, "_run_esbuild", unexpected_compile)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.get("/plugins/published_panel@github/dashboard_panel.js")
        assert response.status_code == 200
        assert response.text == "export default 1;\n"
    assert not (tmp_path / "runtime/dashboard-panels").exists()


def test_panel_cache_identity_tracks_transitive_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace(tmp_path)
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace): return None\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard_panel.ts").write_text(
        "import value from './fragment'; export default value;\n",
        encoding="utf-8",
    )
    fragment = plugin_dir / "fragment.ts"
    fragment.write_text("export default 'first';\n", encoding="utf-8")
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"panel": plugin_dir},
    )
    outputs: list[Path] = []

    def compile_fragment(
        _cmd: list[str],
        _source: Path,
        output: Path,
        _name: str,
    ) -> None:
        outputs.append(output)
        output.write_text(fragment.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(dashboard_api, "_run_esbuild", compile_fragment)
    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: ["esbuild"])

    with TestClient(create_dashboard_app(tmp_path)) as client:
        first = client.get("/plugins/panel/dashboard_panel.js")
        runtime_metadata = plugin_dir / ".venv/metadata.json"
        runtime_metadata.parent.mkdir()
        runtime_metadata.write_text('{"runtime": 1}\n', encoding="utf-8")
        unrelated = client.get("/plugins/panel/dashboard_panel.js")
        fragment.write_text("export default 'second';\n", encoding="utf-8")
        second = client.get("/plugins/panel/dashboard_panel.js")

    assert first.text == "export default 'first';\n"
    assert unrelated.text == "export default 'first';\n"
    assert second.text == "export default 'second';\n"
    assert len(outputs) == 2
    assert outputs[0].parent != outputs[1].parent


def test_stale_published_panel_is_unavailable_when_compile_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace(tmp_path)
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace): return None\n",
        encoding="utf-8",
    )
    published = plugin_dir / "dashboard_panel.js"
    published.write_text("export default 'stale';\n", encoding="utf-8")
    source = plugin_dir / "dashboard_panel.ts"
    source.write_text("export default 'new';\n", encoding="utf-8")
    published.touch()
    source.touch()
    source_mtime = published.stat().st_mtime + 1
    os.utime(source, (source_mtime, source_mtime))
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"panel": plugin_dir},
    )
    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: ["esbuild"])
    monkeypatch.setattr(dashboard_api, "_run_esbuild", lambda *_args: None)

    with TestClient(create_dashboard_app(tmp_path)) as client:
        plugins = client.get("/api/dashboard/plugins").json()
        response = client.get("/plugins/panel/dashboard_panel.js")

    assert plugins == []
    assert response.status_code == 404
    assert not list((tmp_path / "runtime/dashboard-panels").rglob("*.js"))


def test_dashboard_rejects_symlink_panel_cache(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "runtime/dashboard-panels"
    cache.parent.mkdir(parents=True)
    cache.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="不能穿过符号链接"):
        create_dashboard_app(tmp_path)


def test_dashboard_rejects_symlink_runtime_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime = tmp_path / "runtime"
    runtime.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="不能穿过符号链接"):
        create_dashboard_app(tmp_path)


@pytest.mark.asyncio
async def test_dashboard_pending_panel_builds_are_app_lifecycle_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace): return None\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard_panel.ts").write_text(
        "export default 1;\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"panel": plugin_dir},
    )
    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: None)

    started = [asyncio.Event(), asyncio.Event()]
    release_first = asyncio.Event()
    first_compiled = asyncio.Event()
    call_count = 0

    async def compile_owned_queue(
        queue: dashboard_api._PluginPanelBuildQueue,
    ) -> None:
        nonlocal call_count
        call_index = call_count
        call_count += 1
        started[call_index].set()
        if call_index == 1:
            await asyncio.Event().wait()
            return
        await release_first.wait()
        for _root, _plugin, output_dir in queue.take_all():
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dashboard_panel.js").write_text(
                "export default 1;\n",
                encoding="utf-8",
            )
        first_compiled.set()

    monkeypatch.setattr(
        dashboard_api,
        "_compile_pending_plugins_async",
        compile_owned_queue,
    )
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    app_a = create_dashboard_app(workspace_a)
    app_b = create_dashboard_app(workspace_b)

    async with app_a.router.lifespan_context(app_a):
        await started[0].wait()
        async with app_b.router.lifespan_context(app_b):
            await started[1].wait()
        release_first.set()
        await first_compiled.wait()
        assert list((workspace_a / "runtime/dashboard-panels").rglob("*.js"))
        assert not (workspace_b / "runtime/dashboard-panels").exists()


def _test_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_standalone_dashboard_honors_builtin_plugin_manifest(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "home"
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "[plugins.akasha]\nenabled = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))

    plugins = _dashboard_plugin_dirs(Path.cwd())

    assert "akasha" not in plugins
    assert "default_memory" in plugins


def test_standalone_dashboard_rejects_invalid_manifest(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("invalid = [\n", encoding="utf-8")
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))

    with pytest.raises(tomllib.TOMLDecodeError):
        _dashboard_plugin_dirs(Path.cwd())


def test_standalone_dashboard_does_not_import_plugin_backend(
    tmp_path, monkeypatch
) -> None:
    _seed_workspace(tmp_path)
    home = tmp_path / "home"
    plugin_base = home / ".akashic-plugin" / "cache" / "github" / "observe"
    plugin_dir = plugin_base / ".artifacts" / "1.0.0-test"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "db.py").write_text(
        "def ping():\n" "    return 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('backend-imported').write_text('yes')\n"
        "from .db import ping\n"
        "def register(app, context):\n"
        "    @app.get('/api/dashboard/test-relative-import')\n"
        "    def route():\n"
        "        return {'value': ping()}\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'observe'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'observe'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '[plugins."observe@github"]\nenabled = true\n', encoding="utf-8"
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))
    source_before = _test_tree_digest(plugin_dir)

    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.get("/api/dashboard/test-relative-import")
    assert response.status_code == 404
    assert not (plugin_dir / "backend-imported").exists()
    assert _test_tree_digest(plugin_dir) == source_before


def test_two_standalone_dashboard_apps_do_not_import_plugin_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "db.py").write_text(
        "def ping(): return 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('backend-imported').write_text('yes')\n"
        "def register(app, context):\n"
        "    @app.get('/api/dashboard/deferred-relative-import')\n"
        "    def route():\n"
        "        from .db import ping\n"
        "        return {'value': ping()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"panel": plugin_dir},
    )
    source_before = _test_tree_digest(plugin_dir)
    app_a = create_dashboard_app(tmp_path / "workspace-a")
    app_b = create_dashboard_app(tmp_path / "workspace-b")

    with TestClient(app_a) as client_a:
        with TestClient(app_b) as client_b:
            assert client_b.get("/api/dashboard/deferred-relative-import").status_code == 404
        assert client_a.get("/api/dashboard/deferred-relative-import").status_code == 404

    assert not (plugin_dir / "backend-imported").exists()
    assert _test_tree_digest(plugin_dir) == source_before


def test_plugin_asset_paths_reject_cross_platform_traversal(tmp_path) -> None:
    with TestClient(create_dashboard_app(tmp_path)) as client:
        for path in (
            "/plugins/..%5Csecret/dashboard_panel.js",
            "/plugins/C:%5Csecret/dashboard_panel.js",
            "/plugins/%5C%5Cserver%5Cshare/dashboard_panel.css",
        ):
            response = client.get(path)
            assert response.status_code == 400

        assert (
            client.get(
                "/plugins/default_memory/dashboard_panel..%5Csecret.js"
            ).status_code
            == 400
        )

        assert client.get("/plugins/missing/dashboard_panel.js").status_code == 404


def test_memory_engine_plugins_only_expose_active_engine_panels(
    tmp_path,
    monkeypatch,
) -> None:
    _use_writable_dashboard_plugins(
        tmp_path,
        monkeypatch,
        {"default_memory"},
    )
    with TestClient(create_dashboard_app(tmp_path)) as client:
        plugins = client.get("/api/dashboard/plugins").json()
        memory_plugins = {
            item["id"]: [panel["name"] for panel in item["panels"]]
            for item in plugins
            if item["id"] in {"default_memory", "cross_memory"}
        }
        assert memory_plugins == {
            "default_memory": ["dashboard_panel", "dashboard_panel_inspector"]
        }
        assert (
            client.get(
                "/plugins/default_memory/dashboard_panel_inspector.js"
            ).status_code
            == 200
        )
        assert (
            client.get("/plugins/cross_memory/dashboard_panel_inspector.js").status_code
            == 404
        )


def test_akasha_only_exposes_read_only_inspector_panel(
    tmp_path,
    monkeypatch,
) -> None:
    _use_writable_dashboard_plugins(
        tmp_path,
        monkeypatch,
        {"akasha"},
    )
    with TestClient(
        create_dashboard_app(
            tmp_path,
            memory_admin=_AkashaDashboardMemoryAdmin(),
        )
    ) as client:
        plugins = client.get("/api/dashboard/plugins").json()
        akasha = next(item for item in plugins if item["id"] == "akasha")
        assert [panel["name"] for panel in akasha["panels"]] == [
            "dashboard_panel_inspector"
        ]
        assert (
            client.get("/plugins/akasha/dashboard_panel_inspector.js").status_code
            == 200
        )
        assert (
            client.get("/plugins/akasha/dashboard_panel_inspector.css").status_code
            == 200
        )
        assert client.get("/api/dashboard/akasha/graph").status_code == 404
