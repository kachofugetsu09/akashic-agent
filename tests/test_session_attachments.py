from __future__ import annotations

from datetime import UTC, datetime
import sqlite3

import pytest

from session.store import SessionStore


NOW = datetime(2026, 8, 17, 6, 0, tzinfo=UTC).isoformat()
ARTIFACT_ID = "a" * 32


def _register(store: SessionStore, artifact_id: str = ARTIFACT_ID) -> None:
    store.begin_attachment_import(
        artifact_id=artifact_id,
        storage_key=f"uploads/artifacts/{artifact_id}.bin",
        expected_size_bytes=4,
        expected_sha256="b" * 64,
        created_at=NOW,
    )
    store.mark_attachment_import_file_published(artifact_id, updated_at=NOW)
    store.register_ready_attachment(
        artifact_id=artifact_id,
        storage_key=f"uploads/artifacts/{artifact_id}.bin",
        kind="file",
        filename="report.bin",
        media_type="application/octet-stream",
        size_bytes=4,
        sha256="b" * 64,
        created_at=NOW,
    )


def test_message_and_attachment_binding_commit_atomically(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    _register(store)

    rows = store.persist_session(
        "telegram:one",
        created_at=NOW,
        updated_at=NOW,
        metadata={},
        messages=[
            {
                "role": "user",
                "content": "file",
                "timestamp": NOW,
                "tool_chain": None,
                "extra": {"attachment_ids": [ARTIFACT_ID]},
            }
        ],
    )

    assert [row["id"] for row in rows] == ["telegram:one:0"]
    assert store.message_attachment_ids("telegram:one:0") == (ARTIFACT_ID,)
    assert store.get_attachment(ARTIFACT_ID) is not None
    report = store.validate_attachment_metadata_integrity()
    assert report.artifact_count == 1
    assert report.binding_count == 1
    assert report.bound_message_count == 1
    assert report.incomplete_import_ids == ()
    store.close()


def test_missing_attachment_rolls_back_session_and_message(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    with pytest.raises(ValueError, match="未发布"):
        store.persist_session(
            "telegram:missing",
            created_at=NOW,
            updated_at=NOW,
            metadata={"before": True},
            messages=[
                {
                    "role": "user",
                    "content": "missing",
                    "timestamp": NOW,
                    "tool_chain": None,
                    "extra": {"attachment_ids": ["missing"]},
                }
            ],
        )

    assert not store.session_exists("telegram:missing")
    assert store.count_messages("telegram:missing") == 0
    assert store.message_attachment_ids("telegram:missing:0") == ()
    store.close()


def test_explicit_message_delete_removes_binding_but_retains_artifact(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="telegram:delete")
    _register(store)
    message = store.insert_message(
        "telegram:delete",
        role="user",
        content="delete binding only",
        ts=NOW,
        seq=0,
        extra={"attachment_ids": [ARTIFACT_ID]},
    )

    assert store.delete_message(
        str(message["id"]),
        action_source="test.attachment_binding_delete",
    )
    assert store.message_attachment_ids(str(message["id"])) == ()
    assert store.get_attachment(ARTIFACT_ID) is not None
    store.close()


def test_attachment_identity_and_message_order_are_fail_loud(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    _register(store)

    with pytest.raises(ValueError, match="不得重复"):
        store.persist_session(
            "telegram:duplicate",
            created_at=NOW,
            updated_at=NOW,
            metadata={},
            messages=[
                {
                    "role": "user",
                    "content": "duplicate",
                    "timestamp": NOW,
                    "tool_chain": None,
                    "extra": {"attachment_ids": [ARTIFACT_ID, ARTIFACT_ID]},
                }
            ],
        )

    assert not store.session_exists("telegram:duplicate")
    store.close()


def test_binding_foreign_keys_and_message_edit_cannot_drift(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="telegram:edit")
    _register(store)
    message = store.insert_message(
        "telegram:edit",
        role="user",
        content="original",
        ts=NOW,
        seq=0,
        extra={"attachment_ids": [ARTIFACT_ID]},
    )
    message_id = str(message["id"])

    with pytest.raises(ValueError, match="不允许由 message_edit 改写"):
        store.update_message(
            message_id,
            extra={"attachment_ids": []},
            action_source="test.attachment_drift",
        )
    assert store.message_attachment_ids(message_id) == (ARTIFACT_ID,)

    with store._lock, pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """
            INSERT INTO message_attachments (
                message_id, ordinal, artifact_id, direction
            ) VALUES ('missing-message', 0, ?, 'inbound')
            """,
            (ARTIFACT_ID,),
        )
    with store._lock:
        store._conn.rollback()
    store.close()


def test_session_cascade_removes_bindings_without_deleting_artifact(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="telegram:cascade")
    _register(store)
    message = store.insert_message(
        "telegram:cascade",
        role="user",
        content="session delete",
        ts=NOW,
        seq=0,
        extra={"attachment_ids": [ARTIFACT_ID]},
    )

    assert store.delete_session(
        "telegram:cascade",
        cascade=True,
        action_source="test.attachment_session_delete",
    )
    assert store.message_attachment_ids(str(message["id"])) == ()
    assert store.get_attachment(ARTIFACT_ID) is not None
    store.close()


def test_integrity_gate_rejects_projection_drift(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="telegram:corrupt")
    _register(store)
    message = store.insert_message(
        "telegram:corrupt",
        role="user",
        content="corrupt only through raw SQL",
        ts=NOW,
        seq=0,
        extra={"attachment_ids": [ARTIFACT_ID]},
    )
    with store._lock:
        store._conn.execute(
            "UPDATE messages SET extra = '{}' WHERE id = ?",
            (str(message["id"]),),
        )
        store._conn.commit()

    with pytest.raises(ValueError, match="projection 已漂移"):
        store.validate_attachment_metadata_integrity()
    store.close()
