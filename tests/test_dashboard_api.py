from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
from datetime import datetime

from fastapi.testclient import TestClient

from bootstrap.dashboard_api import create_dashboard_app as _create_dashboard_app
from plugins.default_memory.engine import DefaultMemoryEngine
from memory2.store import MemoryStore2
from proactive_v2.state import ProactiveStateStore
from session.store import SessionStore


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
        return self._store.get_item_for_dashboard(item_id, include_embedding=include_embedding)

    def update_item_for_dashboard(self, item_id: str, **kwargs):
        return self._store.update_item_for_dashboard(item_id, **kwargs)

    def delete_item(self, item_id: str) -> bool:
        return self._store.delete_item(item_id)

    def delete_items_batch(self, ids: list[str]) -> int:
        return self._store.delete_items_batch(ids)

    def find_similar_items_for_dashboard(self, item_id: str, **kwargs):
        return self._store.find_similar_items_for_dashboard(item_id, **kwargs)


def create_dashboard_app(tmp_path, **kwargs):
    kwargs.setdefault("memory_admin", _DashboardMemoryAdmin(tmp_path))
    return _create_dashboard_app(tmp_path, **kwargs)


class _ManualConsolidator:
    def __init__(self, *, result: bool = True, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, bool, bool]] = []

    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
        force: bool = False,
    ) -> bool:
        self.calls.append((session_key, archive_all, force))
        if self.error is not None:
            raise self.error
        return self.result


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
        last_consolidated=2,
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
    proactive_store.mark_items_seen(
        [
            ("mcp:feed:event-1", "feed-1"),
            ("mcp:feed:event-2", "feed-2"),
            ("rss:news", "rss-1"),
        ],
        now=datetime.fromisoformat("2026-04-19T02:00:00+00:00"),
    )
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
    proactive_store.mark_rejection_cooldown(
        [("mcp:feed:event-3", "feed-3")],
        hours=24,
        now=datetime.fromisoformat("2026-04-19T02:10:00+00:00"),
    )
    proactive_store.mark_semantic_items(
        [
            {
                "source_key": "rss:news",
                "item_id": "rss-1",
                "text": "今天有新游戏资讯",
            },
            {
                "source_key": "mcp:feed",
                "item_id": "feed-2",
                "text": "用户昨天提到过奶茶",
            },
        ],
        now=datetime.fromisoformat("2026-04-19T02:20:00+00:00"),
    )
    proactive_store.mark_bg_context_main_send(
        now=datetime.fromisoformat("2026-04-19T02:30:00+00:00")
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


def test_list_sessions_with_filters(tmp_path) -> None:
    _seed_workspace(tmp_path)
    client = TestClient(create_dashboard_app(tmp_path))

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
    client = TestClient(create_dashboard_app(tmp_path))

    patch_resp = client.patch(
        "/api/dashboard/sessions/telegram:100",
        json={"metadata": {"title": "patched"}, "last_consolidated": 9},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["metadata"]["title"] == "patched"
    assert patch_resp.json()["last_consolidated"] == 9

    delete_resp = client.delete("/api/dashboard/sessions/telegram:100")
    assert delete_resp.status_code == 200

    get_resp = client.get("/api/dashboard/sessions/telegram:100")
    assert get_resp.status_code == 404


def test_manual_consolidate_session_uses_runtime_entrypoint(tmp_path) -> None:
    _seed_workspace(tmp_path)
    consolidator = _ManualConsolidator(result=True)
    client = TestClient(
        create_dashboard_app(tmp_path, manual_consolidator=consolidator)
    )

    resp = client.post(
        "/api/dashboard/sessions/telegram:100/consolidate",
        json={"archive_all": True},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["triggered"] is True
    assert payload["archive_all"] is True
    assert payload["force"] is True
    assert payload["session"]["key"] == "telegram:100"
    assert consolidator.calls == [("telegram:100", True, True)]


def test_manual_consolidate_session_requires_existing_session(tmp_path) -> None:
    _seed_workspace(tmp_path)
    consolidator = _ManualConsolidator(result=True)
    client = TestClient(
        create_dashboard_app(tmp_path, manual_consolidator=consolidator)
    )

    resp = client.post("/api/dashboard/sessions/missing/consolidate", json={})

    assert resp.status_code == 404
    assert consolidator.calls == []


def test_manual_consolidate_session_reports_unavailable_runtime(tmp_path) -> None:
    _seed_workspace(tmp_path)
    client = TestClient(create_dashboard_app(tmp_path))

    resp = client.post("/api/dashboard/sessions/telegram:100/consolidate", json={})

    assert resp.status_code == 503


def test_manual_consolidate_session_reports_concurrency_timeout(tmp_path) -> None:
    _seed_workspace(tmp_path)
    client = TestClient(
        create_dashboard_app(
            tmp_path,
            manual_consolidator=_ManualConsolidator(error=TimeoutError("busy")),
        )
    )

    resp = client.post("/api/dashboard/sessions/telegram:100/consolidate", json={})

    assert resp.status_code == 409
    assert resp.json()["detail"] == "busy"


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
    client = TestClient(create_dashboard_app(tmp_path))

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
    client = TestClient(create_dashboard_app(tmp_path))

    list_resp = client.get(
        "/api/dashboard/sessions/telegram:100/messages",
        params={"q": "睡", "role": "assistant"},
    )
    assert list_resp.status_code == 200
    payload = list_resp.json()
    assert payload["total"] == 1
    message_id = payload["items"][0]["id"]

    patch_resp = client.patch(
        f"/api/dashboard/messages/{message_id}",
        json={"content": "已经睡了", "extra": {"edited": True}},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["content"] == "已经睡了"
    assert patch_resp.json()["edited"] is True

    batch_resp = client.post(
        "/api/dashboard/messages/batch-delete",
        json={"ids": [message_id, "cli:local:0"]},
    )
    assert batch_resp.status_code == 200
    assert batch_resp.json()["deleted_count"] == 2

    remain_resp = client.get("/api/dashboard/messages", params={"session_key": "telegram:100"})
    assert remain_resp.status_code == 200
    assert remain_resp.json()["total"] == 1


def test_list_memory_items_with_filters(tmp_path) -> None:
    _seed_workspace(tmp_path)
    client = TestClient(create_dashboard_app(tmp_path))

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
    client = TestClient(create_dashboard_app(tmp_path))

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
    client = TestClient(create_dashboard_app(tmp_path))

    resp = client.get("/api/dashboard/memories")

    assert resp.status_code == 200
    assert resp.json()["items"][0]["source_ref"] == "cli:local:profile"


def test_get_update_and_delete_memory(tmp_path) -> None:
    _seed_workspace(tmp_path)
    client = TestClient(create_dashboard_app(tmp_path))

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
    client = TestClient(create_dashboard_app(tmp_path))

    list_resp = client.get("/api/dashboard/memories", params={"scope_channel": "telegram"})
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
    client = TestClient(create_dashboard_app(tmp_path))

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


def test_proactive_dashboard_endpoints(tmp_path) -> None:
    _seed_workspace(tmp_path)
    client = TestClient(create_dashboard_app(tmp_path))

    overview_resp = client.get("/api/dashboard/proactive/overview")
    assert overview_resp.status_code == 200
    overview = overview_resp.json()
    assert overview["counts"]["seen_items"] == 3
    assert overview["counts"]["deliveries"] == 2
    assert overview["counts"]["tick_logs"] == 2
    assert overview["flow_counts"]["drift"] == 1
    assert overview["flow_counts"]["proactive"] == 1
    assert overview["last_tick_at"] == "2026-04-19T03:00:00+00:00"
    assert overview["last_send_at"] == "2026-04-19T02:06:00+00:00"
    assert overview["last_skip_reason"] == "busy"

    deliveries_resp = client.get(
        "/api/dashboard/proactive/deliveries",
        params={"session_key": "telegram:100"},
    )
    assert deliveries_resp.status_code == 200
    assert deliveries_resp.json()["total"] == 1
    assert deliveries_resp.json()["items"][0]["delivery_key"] == "delivery-a"

    seen_resp = client.get(
        "/api/dashboard/proactive/seen_items",
        params={"source_key": "mcp:feed"},
    )
    assert seen_resp.status_code == 200
    assert seen_resp.json()["total"] == 2

    semantic_resp = client.get(
        "/api/dashboard/proactive/semantic_items",
        params={"window_hours": 100000},
    )
    assert semantic_resp.status_code == 200
    assert semantic_resp.json()["total"] == 2

    tick_logs_resp = client.get(
        "/api/dashboard/proactive/tick_logs",
        params={"terminal_action": "skip"},
    )
    assert tick_logs_resp.status_code == 200
    assert tick_logs_resp.json()["total"] == 1
    assert tick_logs_resp.json()["items"][0]["tick_id"] == "tick-2"

    drift_logs_resp = client.get(
        "/api/dashboard/proactive/tick_logs",
        params={"flow": "drift"},
    )
    assert drift_logs_resp.status_code == 200
    assert drift_logs_resp.json()["total"] == 1
    assert drift_logs_resp.json()["items"][0]["tick_id"] == "tick-2"

    proactive_sorted_resp = client.get(
        "/api/dashboard/proactive/tick_logs",
        params={"sort_by": "started_at", "sort_order": "asc"},
    )
    assert proactive_sorted_resp.status_code == 200
    assert proactive_sorted_resp.json()["items"][0]["tick_id"] == "tick-1"

    tick_detail_resp = client.get("/api/dashboard/proactive/tick_logs/tick-1")
    assert tick_detail_resp.status_code == 200
    assert tick_detail_resp.json()["interesting_ids"] == ["mcp:feed:feed-1"]
    assert tick_detail_resp.json()["final_message"] == "记得早点休息"

    tick_steps_resp = client.get("/api/dashboard/proactive/tick_logs/tick-1/steps")
    assert tick_steps_resp.status_code == 200
    assert tick_steps_resp.json()["total"] == 2
    assert tick_steps_resp.json()["items"][0]["tool_name"] == "message_push"
    assert tick_steps_resp.json()["items"][0]["tool_args"]["message"] == "记得早点休息"
    assert tick_steps_resp.json()["items"][1]["terminal_action_after"] == "reply"


def test_status_commands_kvcache_dashboard_uses_workspace_observe(tmp_path) -> None:
    _seed_workspace(tmp_path)
    observe_dir = tmp_path / "observe"
    observe_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(observe_dir / "observe.db")
    try:
        conn.execute(
            """
            CREATE TABLE turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT NOT NULL,
                session_key TEXT NOT NULL,
                user_msg TEXT,
                llm_output TEXT NOT NULL DEFAULT '',
                react_cache_prompt_tokens INTEGER,
                react_cache_hit_tokens INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO turns(
                ts, source, session_key, user_msg, llm_output,
                react_cache_prompt_tokens, react_cache_hit_tokens
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-04-19T03:20:00+00:00",
                "agent",
                "telegram:100",
                "again",
                "ok",
                300,
                260,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    client = TestClient(create_dashboard_app(tmp_path))

    overview = client.get("/api/dashboard/status-commands/kvcache/overview")
    turns = client.get("/api/dashboard/status-commands/kvcache/turns")

    assert overview.status_code == 200
    assert overview.json()["tracked_turn_count"] == 1
    assert overview.json()["hit_rate"] == 260 / 300
    assert turns.status_code == 200
    payload = turns.json()
    assert payload["total"] == 1
    assert payload["items"][0]["session_key"] == "telegram:100"
    assert payload["items"][0]["user_preview"] == "again"


def test_proactive_dashboard_batch_delete(tmp_path) -> None:
    _seed_workspace(tmp_path)
    client = TestClient(create_dashboard_app(tmp_path))

    seen_delete_resp = client.request(
        "DELETE",
        "/api/dashboard/proactive/seen_items/batch",
        json={"source_key": "mcp:feed", "item_ids": ["feed-1"]},
    )
    assert seen_delete_resp.status_code == 200
    assert seen_delete_resp.json()["deleted_count"] == 1

    seen_resp = client.get(
        "/api/dashboard/proactive/seen_items",
        params={"source_key": "mcp:feed"},
    )
    assert seen_resp.json()["total"] == 1

    cooldown_delete_resp = client.request(
        "DELETE",
        "/api/dashboard/proactive/rejection_cooldown/batch",
        json={"source_key": "mcp:feed", "item_ids": ["feed-3"]},
    )
    assert cooldown_delete_resp.status_code == 200
    assert cooldown_delete_resp.json()["deleted_count"] == 1

    cooldown_resp = client.get(
        "/api/dashboard/proactive/rejection_cooldown",
        params={"source_key": "mcp:feed"},
    )
    assert cooldown_resp.status_code == 200
    assert cooldown_resp.json()["total"] == 0


def test_proactive_dashboard_batch_delete_rejects_empty_payload(tmp_path) -> None:
    _seed_workspace(tmp_path)
    client = TestClient(create_dashboard_app(tmp_path))

    seen_delete_resp = client.request(
        "DELETE",
        "/api/dashboard/proactive/seen_items/batch",
        json={},
    )
    assert seen_delete_resp.status_code == 400
    assert seen_delete_resp.json()["detail"] == "至少提供 source_key 或 item_ids"

    cooldown_delete_resp = client.request(
        "DELETE",
        "/api/dashboard/proactive/rejection_cooldown/batch",
        json={},
    )
    assert cooldown_delete_resp.status_code == 400
    assert cooldown_delete_resp.json()["detail"] == "至少提供 source_key 或 item_ids"


def test_plugin_asset_paths_reject_cross_platform_traversal(tmp_path) -> None:
    client = TestClient(create_dashboard_app(tmp_path))

    for path in (
        "/plugins/..%5Csecret/panel.js",
        "/plugins/C:%5Csecret/panel.js",
        "/plugins/%5C%5Cserver%5Cshare/panel.css",
    ):
        response = client.get(path)
        assert response.status_code == 400

    assert client.get("/plugins/missing/panel.js").status_code == 404
