"""Tests for current MemoryStore behavior."""

from agent.memory import MemoryStore


def test_pending_file_created_on_init(tmp_path):
    store = MemoryStore(tmp_path)
    assert store.pending_file.exists()


def test_snapshot_and_commit_clear_snapshot_file(tmp_path):
    store = MemoryStore(tmp_path)
    store.append_pending("- fact A")

    snap = store.snapshot_pending()
    assert "fact A" in snap
    assert store._snapshot_path.exists()

    store.commit_pending_snapshot()
    assert not store._snapshot_path.exists()
    assert store.pending_file.exists()


def test_snapshot_and_rollback_merges_new_pending(tmp_path):
    store = MemoryStore(tmp_path)
    store.append_pending("- old")

    _ = store.snapshot_pending()
    store.append_pending("- new")
    store.rollback_pending_snapshot()

    pending = store.read_pending()
    assert "- old" in pending
    assert "- new" in pending


def test_get_memory_context_empty_and_nonempty(tmp_path):
    store = MemoryStore(tmp_path)
    assert store.get_memory_context() == ""

    store.write_long_term("- user profile")
    assert store.get_memory_context().startswith("## Long-term Memory")


def test_append_pending_once_is_idempotent_and_hidden_from_read(tmp_path):
    store = MemoryStore(tmp_path)

    assert store.append_pending_once(
        "- pref A",
        source_ref="session@1-10",
        kind="user_facts",
    )
    assert not store.append_pending_once(
        "- pref A duplicated",
        source_ref="session@1-10",
        kind="user_facts",
    )

    pending = store.read_pending()
    raw = store.pending_file.read_text(encoding="utf-8")

    assert "- pref A" in pending
    assert "duplicated" not in pending
    assert "<!-- consolidation:session@1-10:user_facts -->" in raw
    assert raw.count("<!-- consolidation:session@1-10:user_facts -->") == 1


def test_append_pending_once_repairs_file_when_db_ahead(tmp_path):
    store = MemoryStore(tmp_path)
    assert store.append_pending_once(
        "- pref A",
        source_ref="session@1-10",
        kind="user_facts",
    )

    # 模拟文件被回滚/覆盖但 sidecar 仍保留写入记录
    store.pending_file.write_text("", encoding="utf-8")

    # 同一 source_ref 再次写入时应被判重，但会自动把缺失内容补回文件
    assert not store.append_pending_once(
        "- pref A should be ignored",
        source_ref="session@1-10",
        kind="user_facts",
    )
    pending = store.read_pending()
    raw = store.pending_file.read_text(encoding="utf-8")

    assert "- pref A" in pending
    assert "ignored" not in pending
    assert "<!-- consolidation:session@1-10:user_facts -->" in raw
