from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agent.memory import MemoryStore
from bootstrap.dashboard_api import create_dashboard_app
from memory2.store import MemoryStore2


def test_memory_rollup_generates_and_commits_pending(tmp_path: Path):
    memory2 = MemoryStore2(tmp_path / "memory" / "memory2.db")
    memory2.upsert_item(
        memory_type="preference",
        summary="用户希望回复温暖直接，先给结论，不喜欢成功学式鼓励",
        embedding=None,
        emotional_weight=7,
    )
    memory2.upsert_item(
        memory_type="preference",
        summary="用户希望回复不要模板化，像朋友自然提醒",
        embedding=None,
        emotional_weight=6,
    )
    memory2.upsert_item(
        memory_type="profile",
        summary="用户常驻上海",
        embedding=None,
    )
    memory2.close()

    client = TestClient(create_dashboard_app(tmp_path))

    generated = client.post("/api/dashboard/memory-rollup/generate").json()
    assert generated["total"] >= 1
    candidate = generated["items"][0]

    committed = client.post(
        "/api/dashboard/memory-rollup/commit",
        json={"items": [{"id": candidate["id"]}]},
    ).json()

    assert committed["appended_count"] == 1
    after_commit = client.get("/api/dashboard/memory-rollup/candidates").json()
    assert candidate["id"] not in {
        item["id"] for item in after_commit["items"]
    }
    pending = MemoryStore(tmp_path).read_pending()
    assert "- [preference]" in pending
    assert "用户希望回复温暖直接，先给结论，不喜欢成功学式鼓励" in pending

    marked_store = MemoryStore2(tmp_path / "memory" / "memory2.db")
    try:
        for source_id in candidate["source_ids"]:
            item = marked_store.get_item_for_dashboard(source_id)
            assert item is not None
            assert item["extra_json"]["_rollup"]["candidate_id"] == candidate["id"]
    finally:
        marked_store.close()

    regenerated = client.post("/api/dashboard/memory-rollup/generate").json()
    regenerated_source_ids = {
        source_id
        for item in regenerated["items"]
        for source_id in item["source_ids"]
    }
    assert not set(candidate["source_ids"]) & regenerated_source_ids


def test_memory_rollup_profile_candidate_uses_profile_tag(tmp_path: Path):
    memory2 = MemoryStore2(tmp_path / "memory" / "memory2.db")
    memory2.upsert_item(
        memory_type="profile",
        summary="用户希望回复温暖、直接、先给结论",
        embedding=None,
        emotional_weight=6,
    )
    memory2.close()

    client = TestClient(create_dashboard_app(tmp_path))
    generated = client.post("/api/dashboard/memory-rollup/generate").json()

    assert generated["items"][0]["title"] == "身份信息"
    assert generated["items"][0]["tag"] == "identity"


def test_memory_rollup_uses_real_memory_content(tmp_path: Path):
    memory2 = MemoryStore2(tmp_path / "memory" / "memory2.db")
    memory2.upsert_item(
        memory_type="preference",
        summary="用户希望回复简洁、冷淡",
        embedding=None,
        emotional_weight=6,
    )
    memory2.close()

    client = TestClient(create_dashboard_app(tmp_path))
    generated = client.post("/api/dashboard/memory-rollup/generate").json()

    assert generated["items"][0]["content"] == "用户希望回复简洁、冷淡"


def test_memory_rollup_does_not_group_by_keywords(tmp_path: Path):
    memory2 = MemoryStore2(tmp_path / "memory" / "memory2.db")
    memory2.upsert_item(
        memory_type="preference",
        summary="用户希望回复简洁",
        embedding=None,
    )
    memory2.upsert_item(
        memory_type="preference",
        summary="用户希望语气自然",
        embedding=None,
    )
    memory2.close()

    client = TestClient(create_dashboard_app(tmp_path))
    generated = client.post("/api/dashboard/memory-rollup/generate").json()

    contents = {item["content"] for item in generated["items"]}
    assert contents == {"用户希望回复简洁", "用户希望语气自然"}


def test_memory_rollup_scores_with_memory_table_metrics(tmp_path: Path):
    memory2 = MemoryStore2(tmp_path / "memory" / "memory2.db")
    memory2.upsert_item(
        memory_type="preference",
        summary="用户偏好高频出现的记忆",
        embedding=None,
    )
    memory2.upsert_item(
        memory_type="preference",
        summary="用户偏好高频出现的记忆",
        embedding=None,
        emotional_weight=8,
    )
    memory2.upsert_item(
        memory_type="preference",
        summary="用户偏好低频出现的记忆",
        embedding=None,
    )
    memory2.close()

    client = TestClient(create_dashboard_app(tmp_path))
    generated = client.post("/api/dashboard/memory-rollup/generate").json()

    assert generated["items"][0]["content"] == "用户偏好高频出现的记忆"
    assert generated["items"][0]["reinforcement"] == 2
    assert generated["items"][0]["emotional_weight"] == 8
    assert generated["items"][0]["score"] > generated["items"][1]["score"]


def test_memory_rollup_shows_sensitive_profile_content(tmp_path: Path):
    memory2 = MemoryStore2(tmp_path / "memory" / "memory2.db")
    memory2.upsert_item(
        memory_type="profile",
        summary="用户的邮箱是 hanayue@example.com",
        embedding=None,
    )
    memory2.close()

    client = TestClient(create_dashboard_app(tmp_path))
    generated = client.post("/api/dashboard/memory-rollup/generate").json()

    contents = {item["content"] for item in generated["items"]}
    assert "用户的邮箱是 hanayue@example.com" in contents


def test_memory_rollup_ignore_marks_sources_without_pending(tmp_path: Path):
    memory2 = MemoryStore2(tmp_path / "memory" / "memory2.db")
    memory2.upsert_item(
        memory_type="preference",
        summary="用户喜欢川菜和重口味",
        embedding=None,
    )
    memory2.close()

    client = TestClient(create_dashboard_app(tmp_path))
    generated = client.post("/api/dashboard/memory-rollup/generate").json()
    candidate = generated["items"][0]

    ignored = client.post(
        "/api/dashboard/memory-rollup/ignore",
        json={"id": candidate["id"]},
    ).json()

    assert ignored["ignored"] is True
    assert MemoryStore(tmp_path).read_pending() == ""
    after_ignore = client.get("/api/dashboard/memory-rollup/candidates").json()
    assert candidate["id"] not in {item["id"] for item in after_ignore["items"]}

    marked_store = MemoryStore2(tmp_path / "memory" / "memory2.db")
    try:
        item = marked_store.get_item_for_dashboard(candidate["source_ids"][0])
        assert item is not None
        assert item["extra_json"]["_rollup"]["action"] == "ignored"
    finally:
        marked_store.close()

    regenerated = client.post("/api/dashboard/memory-rollup/generate").json()
    regenerated_source_ids = {
        source_id
        for item in regenerated["items"]
        for source_id in item["source_ids"]
    }
    assert candidate["source_ids"][0] not in regenerated_source_ids


def test_memory_rollup_delete_sources_removes_memory_without_pending(tmp_path: Path):
    memory2 = MemoryStore2(tmp_path / "memory" / "memory2.db")
    memory2.upsert_item(
        memory_type="preference",
        summary="用户喜欢粤菜和清淡口味",
        embedding=None,
    )
    memory2.close()

    client = TestClient(create_dashboard_app(tmp_path))
    generated = client.post("/api/dashboard/memory-rollup/generate").json()
    candidate = generated["items"][0]
    source_id = candidate["source_ids"][0]

    deleted = client.post(
        "/api/dashboard/memory-rollup/delete-sources",
        json={"id": candidate["id"]},
    ).json()

    assert deleted["deleted_count"] == 1
    assert MemoryStore(tmp_path).read_pending() == ""
    after_delete = client.get("/api/dashboard/memory-rollup/candidates").json()
    assert candidate["id"] not in {item["id"] for item in after_delete["items"]}

    check_store = MemoryStore2(tmp_path / "memory" / "memory2.db")
    try:
        assert check_store.get_item_for_dashboard(source_id) is None
    finally:
        check_store.close()
