"""Tests for current MemoryStore behavior."""

import builtins
import sqlite3
from pathlib import Path

import pytest

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


def test_snapshot_rollback_replace_failure_keeps_both_recovery_files(
    tmp_path, monkeypatch
):
    store = MemoryStore(tmp_path)
    store.append_pending("- old")
    _ = store.snapshot_pending()
    store.append_pending("- new")
    original_replace = Path.replace

    def fail_pending_replace(path: Path, target: Path) -> Path:
        if target == store.pending_file:
            raise OSError("replace failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_pending_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.rollback_pending_snapshot()

    assert "- old" in store._snapshot_path.read_text(encoding="utf-8")
    assert "- new" in store.pending_file.read_text(encoding="utf-8")
    assert not list(store.memory_dir.glob("PENDING.md.*.tmp"))


def test_profile_replace_failure_preserves_existing_file(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    store.write_long_term("old profile")
    original_replace = Path.replace

    def fail_memory_replace(path: Path, target: Path) -> Path:
        if target == store.memory_file:
            raise OSError("replace failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_memory_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.write_long_term("new profile")

    assert store.read_long_term() == "old profile"
    assert not list(store.memory_dir.glob("MEMORY.md.*.tmp"))


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


@pytest.mark.parametrize(
    "reader_name",
    ["_tail_contains_marker", "_file_contains_marker"],
)
def test_marker_reader_exposes_file_read_failure(tmp_path, monkeypatch, reader_name):
    marker_file = tmp_path / "PENDING.md"
    marker_file.write_text("existing", encoding="utf-8")

    def deny_read(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(builtins, "open", deny_read)

    reader = getattr(MemoryStore, reader_name)
    with pytest.raises(PermissionError, match="denied"):
        reader(marker_file, "marker")


@pytest.mark.parametrize("reader_name", ["_tail_contains_marker"])
def test_marker_reader_exposes_invalid_utf8(tmp_path, reader_name):
    marker_file = tmp_path / "PENDING.md"
    marker_file.write_bytes(b"\xff")

    reader = getattr(MemoryStore, reader_name)
    with pytest.raises(UnicodeDecodeError):
        reader(marker_file, "marker")


@pytest.mark.parametrize(
    ("payload", "trailing", "error"),
    [
        (None, 0, "payload is missing"),
        ("content", 2, "must be 0 or 1"),
    ],
)
def test_consolidation_index_rejects_unrecoverable_rows(
    tmp_path, payload, trailing, error
):
    store = MemoryStore(tmp_path)
    source_ref = "source"
    kind = "pending"
    conn = sqlite3.connect(store._consolidation_db)
    try:
        conn.execute(
            """INSERT INTO consolidation_writes(
                source_ref, kind, payload, trailing_blank_line, done_at
            ) VALUES (?, ?, ?, ?, datetime('now'))""",
            (source_ref, kind, payload, trailing),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match=error):
        store.append_pending_once("new content", source_ref=source_ref, kind=kind)
