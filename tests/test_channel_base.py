import asyncio
from pathlib import Path

import pytest

from agent.control.context import running_turn_id
from infra.channels.base import AttachmentStore, MessageDeduper, SessionIdentityIndex
from session.manager import SessionManager
from session.store import SessionStore


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
    token = running_turn_id.set("turn:identity-cache")
    try:
        grant = saved.issue_projection_grant("turn:identity-cache")
        saved.revoke_projection_grant(grant)
    finally:
        running_turn_id.reset(token)


@pytest.mark.asyncio
async def test_session_identity_index_rolls_back_failed_metadata_save(tmp_path: Path):
    manager = SessionManager(tmp_path)

    def fail_persist(**_kwargs: object) -> None:
        raise OSError("metadata store unavailable")

    manager.control_store.persist_channel_identity = fail_persist  # type: ignore[method-assign]
    index = SessionIdentityIndex(manager, channel="telegram", metadata_key="username")

    with pytest.raises(OSError, match="metadata store unavailable"):
        await index.remember("alice", "123")

    assert index.mapping == {}
    assert manager.get_channel_identities("telegram") == {}
    assert manager.control_store.get_session_meta("telegram:123") is None
    assert "telegram:123" not in manager._cache


@pytest.mark.asyncio
async def test_session_identity_index_preserves_existing_session_on_failure(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("telegram:123")
    session.metadata["marker"] = "before"
    manager.save(session)

    def fail_persist(**_kwargs: object) -> None:
        raise OSError("metadata store unavailable")

    manager.control_store.persist_channel_identity = fail_persist  # type: ignore[method-assign]
    index = SessionIdentityIndex(manager, channel="telegram", metadata_key="username")

    with pytest.raises(OSError, match="metadata store unavailable"):
        await index.remember("alice", "123")

    assert index.mapping == {}
    assert session.metadata == {"marker": "before"}
    assert manager.control_store.get_session_meta("telegram:123")["metadata"] == {
        "marker": "before"
    }


@pytest.mark.asyncio
async def test_session_identity_index_keeps_latest_owner_across_restart(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    manager.get_or_create("feishu:new")
    index = SessionIdentityIndex(
        manager,
        channel="feishu",
        metadata_key="feishu_open_id",
    )

    await index.remember("open-id", "old")
    await index.remember("open-id", "new")

    assert manager.get_channel_identities("feishu") == {"open-id": "new"}
    manager.close()

    reopened = SessionManager(tmp_path)
    rebuilt = SessionIdentityIndex(
        reopened,
        channel="feishu",
        metadata_key="feishu_open_id",
    )

    assert rebuilt.rebuild() == {"open-id": "new"}
    assert rebuilt.resolve("open-id") == "new"
    assert reopened.get_channel_identities("feishu") == {"open-id": "new"}
    reopened.close()


@pytest.mark.asyncio
async def test_session_identity_index_concurrent_move_has_one_durable_owner(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    index = SessionIdentityIndex(
        manager,
        channel="feishu",
        metadata_key="feishu_open_id",
    )

    await asyncio.gather(
        index.remember("open-id", "first"),
        index.remember("open-id", "second"),
    )

    durable = manager.get_channel_identities("feishu")
    assert durable in ({"open-id": "first"}, {"open-id": "second"})
    assert index.mapping == durable
    manager.close()


@pytest.mark.asyncio
async def test_session_identity_index_rolls_back_only_new_acceptance_state(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    index = SessionIdentityIndex(
        manager,
        channel="web",
        metadata_key="web_identity",
    )
    assert index.rebuild() == {}
    assert manager.channel_identity_migration_completed("web") is True

    receipt = await index.remember("abc", "abc")
    assert receipt is not None
    assert await index.rollback(receipt) is True

    assert index.mapping == {}
    assert manager.get_channel_identities("web") == {}
    assert manager.control_store.get_session_meta("web:abc") is None
    assert manager.channel_identity_migration_completed("web") is True
    assert "web:abc" not in manager._cache
    manager.close()


@pytest.mark.asyncio
async def test_session_identity_index_rollback_restores_existing_session(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("web:abc")
    session.metadata["marker"] = "before"
    manager.save(session)
    before = manager.control_store.get_session_meta("web:abc")
    index = SessionIdentityIndex(
        manager,
        channel="web",
        metadata_key="web_identity",
    )

    receipt = await index.remember("abc", "abc")
    assert receipt is not None
    assert await index.rollback(receipt) is True

    assert manager.control_store.get_session_meta("web:abc") == before
    assert manager.get_channel_identities("web") == {}
    assert index.mapping == {}
    manager.close()


@pytest.mark.asyncio
async def test_session_identity_index_rollback_refuses_superseded_write(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    index = SessionIdentityIndex(
        manager,
        channel="web",
        metadata_key="web_identity",
    )

    first = await index.remember("abc", "abc")
    second = await index.remember("abc", "abc")
    assert first is not None and second is not None
    assert first.committed_updated_at != second.committed_updated_at
    assert await index.rollback(first) is False

    assert manager.get_channel_identities("web") == {"abc": "abc"}
    assert manager.control_store.get_session_meta("web:abc") is not None
    assert index.mapping == {"abc": "abc"}
    manager.close()


@pytest.mark.asyncio
async def test_session_delete_removes_identity_owner_and_backup_can_restore_it(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    index = SessionIdentityIndex(
        manager,
        channel="feishu",
        metadata_key="feishu_open_id",
    )
    await index.remember("open-id", "old")
    await index.remember("open-id", "new")

    audit = manager.delete_session_with_audit("feishu:new")

    assert audit.result == "committed"
    assert manager.get_channel_identities("feishu") == {}
    assert index.resolve("open-id") is None
    assert audit.backup_path is not None
    backup = SessionStore(audit.backup_path)
    assert backup.get_channel_identities("feishu") == {"open-id": "new"}
    backup.close()
    manager.close()

    reopened = SessionManager(tmp_path)
    rebuilt = SessionIdentityIndex(
        reopened,
        channel="feishu",
        metadata_key="feishu_open_id",
    )
    assert rebuilt.rebuild() == {}
    assert reopened.channel_identity_migration_completed("feishu") is True
    assert rebuilt.resolve("open-id") is None
    reopened.close()
