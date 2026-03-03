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


def test_update_now_ongoing_add_and_remove_by_keyword(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_now("## 近期进行中\n\n- 任务A\n\n## 待确认事项\n\n- 问题1\n")

    store.update_now_ongoing(add=["任务B"], remove_keywords=["任务A"])

    now_text = store.read_now()
    assert "任务A" not in now_text
    assert "- 任务B" in now_text


def test_read_now_ongoing_extracts_section_body(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_now("## 近期进行中\n\n- A\n- B\n\n## 待确认事项\n\n- C\n")

    ongoing = store.read_now_ongoing()

    assert "- A" in ongoing
    assert "- B" in ongoing
    assert "- C" not in ongoing


def test_get_memory_context_empty_and_nonempty(tmp_path):
    store = MemoryStore(tmp_path)
    assert store.get_memory_context() == ""

    store.write_long_term("- user profile")
    assert store.get_memory_context().startswith("## Long-term Memory")
