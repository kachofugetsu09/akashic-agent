from pathlib import Path

import pytest

from infra.channels.base import AttachmentStore, MessageDeduper, SessionIdentityIndex
from session.manager import SessionManager


def test_attachment_store_writes_under_configured_root(tmp_path: Path):
    store = AttachmentStore(tmp_path / "uploads")

    path = store.write_bytes(b"hello", prefix="img_", suffix=".png")

    assert path.parent == tmp_path / "uploads"
    assert path.suffix == ".png"
    assert path.read_bytes() == b"hello"


def test_attachment_store_fails_when_root_is_not_a_directory(tmp_path: Path):
    root = tmp_path / "not-a-directory"
    root.write_text("occupied", encoding="utf-8")
    store = AttachmentStore(root)

    with pytest.raises(FileExistsError):
        store.write_bytes(b"hello", prefix="img_", suffix=".png")


def test_attachment_store_rejects_symlink_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "uploads"
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="符号链接"):
        AttachmentStore(root).write_bytes(b"hello", prefix="img_", suffix=".png")

    assert list(outside.iterdir()) == []


def test_message_deduper_evicts_oldest_keys():
    deduper = MessageDeduper(max_size=2)

    assert deduper.seen("a") is False
    assert deduper.seen("b") is False
    assert deduper.seen("a") is True
    assert deduper.seen("c") is False
    assert deduper.seen("a") is False


@pytest.mark.asyncio
async def test_session_identity_index_rebuilds_and_persists_metadata(tmp_path: Path):
    manager = SessionManager(tmp_path)
    existing = manager.get_or_create("telegram:123")
    existing.metadata["username"] = "alice"
    manager.save(existing)

    index = SessionIdentityIndex(
        manager,
        channel="telegram",
        metadata_key="username",
        normalizer=lambda value: value.lower(),
    )

    rebuilt = index.rebuild()
    assert rebuilt == {"alice": "123"}
    assert index.resolve("ALICE") == "123"

    await index.remember("Bob", "456")

    assert index.mapping["bob"] == "456"
    saved = manager.get_or_create("telegram:456")
    assert saved.metadata["username"] == "bob"


@pytest.mark.asyncio
async def test_session_identity_index_rolls_back_failed_metadata_save(tmp_path: Path):
    manager = SessionManager(tmp_path)

    async def fail_save(_session):
        raise OSError("metadata store unavailable")

    manager.save_async = fail_save  # type: ignore[method-assign]
    index = SessionIdentityIndex(manager, channel="telegram", metadata_key="username")
    session = manager.get_or_create("telegram:123")

    with pytest.raises(OSError, match="metadata store unavailable"):
        await index.remember("alice", "123")

    assert index.mapping == {}
    assert session.metadata == {}
