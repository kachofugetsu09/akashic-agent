from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from bootstrap.dashboard_api import create_dashboard_app
from memory2.store import MemoryStore2
from session.store import SessionStore


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
