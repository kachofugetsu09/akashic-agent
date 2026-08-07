from datetime import UTC, datetime, timedelta
import asyncio
import logging
import sqlite3
import threading
from pathlib import Path

import pytest

from agent.control.errors import TurnNotFoundError, TurnStateTransitionError
from agent.control.models import (
    TurnError,
    TurnItem,
    TurnItemKind,
    TurnRecord,
    TurnStatus,
    TurnUsage,
)
from session.manager import Session, SessionManager
from session.store import (
    SessionAdmissionConflictError,
    SessionStore,
    _decode_message_extra,
)
from bus.events import InboundMessage
from bus.queue import MessageBus

NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


def _seed_compaction_message(store: SessionStore, session_key: str) -> dict:
    """Create the canonical row referenced by compaction fixtures."""

    return store.insert_message(
        session_key,
        role="user",
        content="tail",
        ts=NOW.isoformat(),
        seq=1,
    )


def _compaction_kwargs(
    session_key: str,
    message: dict[str, object],
    *,
    generation: int | None = None,
) -> dict:
    message_id = str(message["id"])
    message_seq = int(message["seq"])
    return {
        "session_key": session_key,
        "trigger": "soft_limit",
        "summary": "## Goal\nsummary",
        "source_ref": f"source:{generation or 1}",
        "source_from_seq": message_seq,
        "consolidated_through_seq": message_seq,
        "source_message_ids": [message_id],
        "retained_tail": [
            {
                "id": message_id,
                "seq": message_seq,
                "unit_ref": f"turn:{message_seq}",
                "message": {"role": "user", "content": "tail"},
            }
        ],
        "model_runtime_id": "main",
        "model": "test-model",
        "context_window": 100_000,
        "threshold_tokens": 74_000,
        "hard_input_tokens": 90_000,
        "keep_recent_tokens": 20_000,
        "tokens_before": 80_000,
        "tokens_after": 30_000,
        "summary_usage": {"input_tokens": 10, "output_tokens": 5},
        **({"generation": generation} if generation is not None else {}),
    }


def _seed_interaction_with_compactions(
    store: SessionStore,
    session_key: str = "mobile:cache",
) -> tuple[list[dict[str, object]], str]:
    """Seed an explicit interaction plus an ancestor and descendant checkpoint."""

    timestamp = NOW.isoformat()
    rows = store.persist_session(
        session_key,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={},
        messages=[
            {
                "role": "user",
                "content": "ancestor",
                "timestamp": timestamp,
                "extra": {},
            },
            {
                "role": "user",
                "content": "u1",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn:cache",
                    "turn_input_ordinal": 0,
                },
            },
            {
                "role": "assistant",
                "content": "final",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn:cache",
                    "turn_terminal": True,
                    "turn_input_count": 1,
                },
            },
        ],
    )
    for generation, source in ((1, rows[0]), (2, rows[1]), (3, rows[2])):
        store.persist_compaction(
            session_key=session_key,
            trigger="test",
            summary=f"checkpoint-{generation}",
            source_ref=f"test:cache:{generation}",
            source_from_seq=int(source["seq"]),
            consolidated_through_seq=int(source["seq"]),
            source_message_ids=[str(source["id"])],
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
            generation=generation,
            parent_generation=generation - 1,
        )
    return rows, "turn:cache"


def _queued(turn_id: str = "turn:1", thread_id: str = "programmatic:1") -> TurnRecord:
    return TurnRecord(
        id=turn_id,
        thread_id=thread_id,
        status=TurnStatus.QUEUED,
        input="你好",
        metadata={"request": 1},
        items=[],
        usage=None,
        error=None,
        created_at=NOW,
    )


def test_turn_reopens_with_terminal_usage_and_items(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    store.create_turn(_queued())
    store.transition_turn(
        "turn:1",
        expected_status=TurnStatus.QUEUED,
        status=TurnStatus.IN_PROGRESS,
        now=NOW + timedelta(seconds=1),
    )
    item = TurnItem(
        id="item:1",
        kind=TurnItemKind.ASSISTANT_MESSAGE,
        data={"content": "完成"},
    )
    store.transition_turn(
        "turn:1",
        expected_status=TurnStatus.IN_PROGRESS,
        status=TurnStatus.COMPLETED,
        items=[item],
        usage=TurnUsage(input_tokens=12, output_tokens=3, coverage="exact"),
        final_response="完成",
        now=NOW + timedelta(seconds=2),
    )
    store.close()

    reopened = SessionStore(db_path)
    record = reopened.read_turn("turn:1")

    assert record is not None
    assert record.status is TurnStatus.COMPLETED
    assert record.items == [item]
    assert record.usage == TurnUsage(input_tokens=12, output_tokens=3, coverage="exact")
    assert record.final_response == "完成"
    reopened.close()


@pytest.mark.asyncio
async def test_mobile_inbound_handoff_survives_queue_restart_and_deduplicates(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    message = InboundMessage(
        channel="mobile",
        sender="device:1",
        chat_id="mobile:session",
        content="你好",
        metadata={"client_message_id": "client:1"},
    )

    await bus.publish_inbound(message)
    assert len(store.list_inbound_handoffs()) == 1

    restarted = MessageBus()
    recovered_store = SessionStore(db_path)
    restarted.bind_durable_inbound_store(recovered_store)
    await restarted.recover_durable_inbounds()
    recovered = await restarted.consume_inbound()
    assert recovered.content == "你好"
    assert recovered.handoff_id == message.handoff_id

    duplicate = InboundMessage(
        channel="mobile",
        sender="device:1",
        chat_id="mobile:session",
        content="你好",
        metadata={"client_message_id": "client:1"},
    )
    await bus.publish_inbound(duplicate)
    assert bus.inbound_size == 1

    await restarted.complete_inbound(recovered)
    assert store.list_inbound_handoffs() == []
    recovered_store.close()
    store.close()


@pytest.mark.asyncio
async def test_mobile_handoff_recovery_pages_durable_rows_and_completes_them(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    seed = MessageBus()
    seed.bind_durable_inbound_store(store)
    for index in range(3):
        await seed.publish_inbound(
            InboundMessage(
                channel="mobile",
                sender="device:1",
                chat_id=f"mobile:session-{index}",
                content=f"message-{index}",
                metadata={"client_message_id": f"client:{index}"},
            )
        )

    recovered_store = SessionStore(db_path)
    restarted = MessageBus()
    restarted.bind_durable_inbound_store(recovered_store)
    await restarted.recover_durable_inbounds()
    assert restarted.inbound_size == 3
    assert len(recovered_store.list_inbound_handoffs()) == 3

    for index in range(3):
        item = await restarted.consume_inbound()
        assert item.content == f"message-{index}"
        await restarted.complete_inbound(item)
        assert restarted.inbound_size == 2 - index
    assert recovered_store.list_inbound_handoffs() == []
    recovered_store.close()
    store.close()


def test_mobile_handoff_recovery_rejects_corrupt_json(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store._conn.execute(
        """
        INSERT INTO inbound_handoffs(
            handoff_id, dedupe_key, channel, sender, chat_id, session_key,
            content, timestamp, media_json, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "handoff-corrupt",
            "mobile:session:client-corrupt",
            "mobile",
            "device:1",
            "mobile:session",
            "mobile:session",
            "bad",
            NOW.isoformat(),
            "{}",
            "[]",
            NOW.isoformat(),
        ),
    )
    store._conn.commit()
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    with pytest.raises(ValueError, match="inbound handoff media invalid"):
        asyncio.run(bus.recover_durable_inbounds())
    store.close()


def test_mobile_handoff_conflicting_reuse_fails_loud(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    base = {
        "handoff_id": "handoff-1",
        "dedupe_key": "mobile:session:client-1",
        "channel": "mobile",
        "sender": "device:1",
        "chat_id": "mobile:session",
        "session_key": "mobile:session",
        "content": "hello",
        "timestamp": NOW.isoformat(),
        "media_json": "[]",
        "metadata_json": '{"client_message_id":"client-1"}',
        "created_at": NOW.isoformat(),
    }
    assert store.reserve_inbound_handoff(**base) == ("handoff-1", True)
    assert store.reserve_inbound_handoff(**base | {"handoff_id": "handoff-2"}) == (
        "handoff-1",
        False,
    )
    with pytest.raises(RuntimeError, match="identity conflict"):
        store.reserve_inbound_handoff(**base | {"content": "tampered"})
    with pytest.raises(RuntimeError, match="identity conflict"):
        store.reserve_inbound_handoff(
            **base
            | {
                "handoff_id": "handoff-2",
                "content": "tampered",
            }
        )
    store.close()


@pytest.mark.asyncio
async def test_mobile_handoff_delete_failure_retains_owner_until_retry(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    message = InboundMessage(
        channel="mobile",
        sender="device:1",
        chat_id="mobile:session",
        content="hello",
        metadata={"client_message_id": "client:delete-retry"},
    )
    await bus.publish_inbound(message)
    consumed = await bus.consume_inbound()
    original_complete = store.complete_inbound_handoff
    failed = True

    def fail_once(handoff_id: str) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise OSError("delete unavailable")
        original_complete(handoff_id)

    store.complete_inbound_handoff = fail_once  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError, match="delete unavailable"):
            await bus.complete_inbound(consumed)
    assert len(bus._inbound_accepted) == 1
    assert len(store.list_inbound_handoffs()) == 1
    assert "cleanup_degraded" in caplog.text

    async def retry_finished() -> None:
        while bus._inbound_accepted:
            await asyncio.sleep(0.02)

    await asyncio.wait_for(retry_finished(), timeout=1)
    assert bus._inbound_accepted == {}
    assert store.list_inbound_handoffs() == []
    store.close()


def test_queued_turn_can_be_cancelled(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_turn(_queued())

    cancelled = store.transition_turn(
        "turn:1",
        expected_status=TurnStatus.QUEUED,
        status=TurnStatus.CANCELLED,
        now=NOW + timedelta(seconds=1),
    )

    assert cancelled.status is TurnStatus.CANCELLED
    assert cancelled.completed_at == NOW + timedelta(seconds=1)
    assert cancelled.started_at is None


def test_illegal_transition_fails_before_writing(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_turn(_queued())

    with pytest.raises(TurnStateTransitionError, match="非法"):
        store.transition_turn(
            "turn:1",
            expected_status=TurnStatus.QUEUED,
            status=TurnStatus.COMPLETED,
        )

    assert store.read_turn("turn:1").status is TurnStatus.QUEUED  # type: ignore[union-attr]


def test_stale_compare_and_set_fails_loud(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_turn(_queued())
    store.transition_turn(
        "turn:1",
        expected_status=TurnStatus.QUEUED,
        status=TurnStatus.IN_PROGRESS,
    )

    with pytest.raises(TurnStateTransitionError, match="CAS 失败"):
        store.transition_turn(
            "turn:1",
            expected_status=TurnStatus.QUEUED,
            status=TurnStatus.CANCELLED,
        )


def test_transition_rejects_wrong_thread_identity(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_turn(_queued())

    with pytest.raises(TurnNotFoundError, match="不属于 thread"):
        store.transition_turn(
            "turn:1",
            thread_id="programmatic:other",
            expected_status=TurnStatus.QUEUED,
            status=TurnStatus.IN_PROGRESS,
        )


def test_failed_turn_requires_structured_error(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_turn(_queued())
    store.transition_turn(
        "turn:1",
        expected_status=TurnStatus.QUEUED,
        status=TurnStatus.IN_PROGRESS,
    )

    with pytest.raises(TurnStateTransitionError, match="必须包含 error"):
        store.transition_turn(
            "turn:1",
            expected_status=TurnStatus.IN_PROGRESS,
            status=TurnStatus.FAILED,
        )

    failed = store.transition_turn(
        "turn:1",
        expected_status=TurnStatus.IN_PROGRESS,
        status=TurnStatus.FAILED,
        error=TurnError(
            type="provider_error", message="provider failed", retryable=True
        ),
    )
    assert failed.error is not None
    assert failed.error.type == "provider_error"


def test_read_missing_turn_returns_none_and_transition_raises(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    assert store.read_turn("turn:missing") is None

    with pytest.raises(TurnNotFoundError, match="不存在"):
        store.transition_turn(
            "turn:missing",
            expected_status=TurnStatus.QUEUED,
            status=TurnStatus.IN_PROGRESS,
        )


def test_delivery_id_resolves_only_unique_proactive_assistant(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_key = "mobile:test"
    expected = store.insert_message(
        session_key,
        role="assistant",
        content="主动消息",
        ts=NOW.isoformat(),
        seq=0,
        extra={"proactive": True, "delivery_id": "delivery-1"},
    )
    _ = store.insert_message(
        session_key,
        role="user",
        content="不能占用投递身份",
        ts=NOW.isoformat(),
        seq=1,
        extra={"delivery_id": "delivery-user"},
    )
    _ = store.insert_message(
        session_key,
        role="assistant",
        content="非主动消息",
        ts=NOW.isoformat(),
        seq=2,
        extra={"delivery_id": "delivery-passive"},
    )

    assert store.get_message_by_delivery_id(session_key, "delivery-1") == expected
    assert store.get_message_by_delivery_id(session_key, "delivery-user") is None
    assert store.get_message_by_delivery_id(session_key, "delivery-passive") is None
    assert store.get_message_by_delivery_id("mobile:other", "delivery-1") is None


def test_duplicate_proactive_delivery_id_fails_loud(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    for seq in range(2):
        _ = store.insert_message(
            "mobile:test",
            role="assistant",
            content=f"主动消息 {seq}",
            ts=NOW.isoformat(),
            seq=seq,
            extra={"proactive": True, "delivery_id": "delivery-1"},
        )

    with pytest.raises(RuntimeError, match="重复 delivery_id"):
        store.get_message_by_delivery_id("mobile:test", "delivery-1")


def test_list_turns_is_thread_scoped_and_stable(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_turn(_queued("turn:1"))
    store.create_turn(
        TurnRecord(
            **{
                **_queued("turn:2").__dict__,
                "created_at": NOW + timedelta(seconds=1),
            }
        )
    )
    store.create_turn(_queued("turn:other", "programmatic:other"))

    page = store.list_turns("programmatic:1", limit=1)
    next_page = store.list_turns(
        "programmatic:1",
        limit=10,
        before=(page[-1].created_at.isoformat(), page[-1].id),
    )

    assert [turn.id for turn in page] == ["turn:2"]
    assert [turn.id for turn in next_page] == ["turn:1"]


def test_delete_thread_turns_does_not_touch_other_threads(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_turn(_queued("turn:1"))
    store.create_turn(_queued("turn:2"))
    store.create_turn(_queued("turn:other", "programmatic:other"))

    assert store.delete_thread_turns("programmatic:1") == 2
    assert store.list_turns("programmatic:1") == []
    assert store.read_turn("turn:other") is not None


def test_session_cascade_deletes_turns_in_same_store_transaction(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="programmatic:1")
    store.create_turn(_queued())

    with pytest.raises(ValueError, match="messages 或 turns"):
        store.delete_session("programmatic:1")

    assert store.delete_session("programmatic:1", cascade=True)
    assert store.read_turn("turn:1") is None
    store.close()


def test_session_delete_backup_and_audit_preserve_ledger_lineage(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    rows, _ = _seed_interaction_with_compactions(store, "mobile:delete")

    audit = store.delete_session_with_audit(
        "mobile:delete",
        cascade=True,
        action_source="test.session_delete",
    )

    assert audit.result == "committed"
    assert audit.targets == ("mobile:delete",)
    assert audit.message_ids == tuple(str(row["id"]) for row in rows)
    assert [item["generation"] for item in audit.compactions] == [1, 2, 3]
    assert [item["source_ref"] for item in audit.compactions] == [
        "test:cache:1",
        "test:cache:2",
        "test:cache:3",
    ]
    assert audit.backup_path is not None
    backup_path = Path(audit.backup_path)
    assert backup_path.is_file()
    assert backup_path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(backup_path) as database:
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    with sqlite3.connect(tmp_path / "sessions.db") as database:
        columns = {
            row[1] for row in database.execute("PRAGMA table_info(session_delete_audits)")
        }
    assert {
        "audit_id",
        "targets_json",
        "message_ids_json",
        "compactions_json",
        "action_source",
        "backup_path",
        "result",
    } <= columns
    restored = SessionStore(backup_path)
    assert [item["id"] for item in restored.fetch_session_messages("mobile:delete")] == [
        str(row["id"]) for row in rows
    ]
    assert [item.generation for item in restored.list_compactions("mobile:delete")] == [
        1,
        2,
        3,
    ]
    restored.close()
    assert store.get_session_meta("mobile:delete") is None
    assert store.get_session_delete_audit(audit.audit_id) == audit
    store.close()


def test_session_batch_delete_returns_audit_for_each_target(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="batch:one")
    store.create_session(key="batch:two")

    audit = store.delete_sessions_batch_with_audit(
        ["batch:one", "batch:two"],
        cascade=True,
        action_source="test.session_batch_delete",
    )

    assert audit.result == "committed"
    assert audit.targets == ("batch:one", "batch:two")
    assert audit.deleted_count == 2
    assert audit.backup_path is not None
    assert store.list_session_delete_audits(limit=1)[0].audit_id == audit.audit_id
    store.close()


def test_session_admission_blocks_delete_from_another_connection(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    runtime_store = SessionStore(db_path)
    dashboard_store = SessionStore(db_path)
    runtime_store.create_session(key="mobile:one")

    assert runtime_store.acquire_session_admission("mobile:one", "admission:one")
    with pytest.raises(SessionAdmissionConflictError, match="正在处理消息") as exc_info:
        dashboard_store.delete_session("mobile:one", cascade=True)

    audit = dashboard_store.get_session_delete_audit(exc_info.value.audit_id)
    assert audit is not None
    assert audit.result == "rejected"
    assert audit.backup_path is None

    runtime_store.release_session_admission("admission:one")
    assert dashboard_store.delete_session("mobile:one", cascade=True)
    runtime_store.close()
    dashboard_store.close()


def test_update_message_rejects_active_admission_atomically(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    runtime_store = SessionStore(db_path)
    dashboard_store = SessionStore(db_path)
    runtime_store.create_session(key="mobile:edit")
    message = runtime_store.insert_message(
        "mobile:edit",
        role="user",
        content="before",
        ts=NOW.isoformat(),
        seq=0,
    )
    assert runtime_store.acquire_session_admission("mobile:edit", "admission:edit")

    with pytest.raises(SessionAdmissionConflictError, match="正在处理消息"):
        dashboard_store.update_message(str(message["id"]), content="after")
    assert dashboard_store.get_message(str(message["id"]))["content"] == "before"

    runtime_store.release_session_admission("admission:edit")
    assert dashboard_store.update_message(str(message["id"]), content="after")["content"] == "after"
    runtime_store.close()
    dashboard_store.close()


def test_source_mutation_audits_authorize_interaction_delete_and_legacy_edit(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    _rows, control_turn_id = _seed_interaction_with_compactions(store)
    deletion = store.delete_interaction(
        control_turn_id,
        action_source="test.interaction_delete",
    )
    assert deletion is not None
    interaction_audits = store.find_authorized_source_mutations(
        session_key="mobile:cache",
        source_ids=list(deletion.message_ids),
        prepared_at="2000-01-01T00:00:00+00:00",
    )
    assert len(interaction_audits) == 1
    assert interaction_audits[0].operation == "interaction_delete"
    assert interaction_audits[0].action_source == "test.interaction_delete"
    assert interaction_audits[0].backup_path == deletion.backup_path

    store.create_session(key="legacy:edit")
    message = store.insert_message(
        "legacy:edit",
        role="user",
        content="before",
        ts=NOW.isoformat(),
        seq=0,
    )
    assert store.update_message(
        str(message["id"]),
        content="after",
        action_source="test.message_edit",
    ) is not None
    edit_audits = store.find_authorized_source_mutations(
        session_key="legacy:edit",
        source_ids=[str(message["id"])],
        prepared_at="2000-01-01T00:00:00+00:00",
    )
    assert len(edit_audits) == 1
    assert edit_audits[0].operation == "message_edit"
    assert edit_audits[0].backup_path is None

    store.create_session(key="legacy:delete")
    deleted = store.insert_message(
        "legacy:delete",
        role="user",
        content="delete",
        ts=NOW.isoformat(),
        seq=0,
    )
    assert store.delete_message(
        str(deleted["id"]),
        action_source="test.message_delete",
    )
    delete_audits = store.find_authorized_source_mutations(
        session_key="legacy:delete",
        source_ids=[str(deleted["id"])],
        prepared_at="2000-01-01T00:00:00+00:00",
    )
    assert len(delete_audits) == 1
    assert delete_audits[0].operation == "message_delete"
    assert delete_audits[0].backup_path is not None
    store.close()


def test_source_mutation_audit_rejects_failed_and_direct_sql_changes(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="legacy:guard")
    message = store.insert_message(
        "legacy:guard",
        role="user",
        content="before",
        ts=NOW.isoformat(),
        seq=0,
    )
    message_id = str(message["id"])
    assert store.acquire_session_admission("legacy:guard", "admission:guard")
    with pytest.raises(SessionAdmissionConflictError):
        store.update_message(
            message_id,
            content="blocked",
            action_source="test.blocked_edit",
        )
    store.release_session_admission("admission:guard")
    assert store.find_authorized_source_mutations(
        session_key="legacy:guard",
        source_ids=[message_id],
        prepared_at="2000-01-01T00:00:00+00:00",
    ) == []

    with store._lock:
        store._conn.execute(
            "UPDATE messages SET content = 'direct' WHERE id = ?",
            (message_id,),
        )
        store._conn.commit()
    assert store.find_authorized_source_mutations(
        session_key="legacy:guard",
        source_ids=[message_id],
        prepared_at="2000-01-01T00:00:00+00:00",
    ) == []
    store.close()


def test_session_manager_reloads_cached_session_after_dashboard_interaction_delete(
    tmp_path,
) -> None:
    runtime = SessionManager(tmp_path)
    dashboard_store = SessionStore(tmp_path / "sessions.db")
    rows, control_turn_id = _seed_interaction_with_compactions(dashboard_store)

    cached = runtime.get_existing("mobile:cache")
    assert [str(message["content"]) for message in cached.messages] == [
        "ancestor",
        "u1",
        "final",
    ]
    assert cached.last_consolidated == 3

    deletion = dashboard_store.delete_interaction(control_turn_id)
    assert deletion is not None
    refreshed = runtime.get_existing("mobile:cache")
    assert refreshed is not cached
    assert [str(message["content"]) for message in refreshed.messages] == ["ancestor"]
    assert refreshed.last_consolidated == 1
    assert runtime.get_or_create("mobile:cache") is refreshed
    assert runtime._store.get_session_meta("mobile:cache")["last_consolidated"] == 1

    runtime.close()
    dashboard_store.close()


def test_interaction_delete_conflict_is_atomic_until_admission_release(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    runtime_store = SessionStore(db_path)
    dashboard_store = SessionStore(db_path)
    rows, control_turn_id = _seed_interaction_with_compactions(dashboard_store)
    assert runtime_store.acquire_session_admission("mobile:cache", "admission:cache")

    with pytest.raises(SessionAdmissionConflictError, match="正在处理消息"):
        dashboard_store.delete_interaction(control_turn_id)

    assert [
        message["id"]
        for message in dashboard_store.fetch_session_messages("mobile:cache")
    ] == [str(row["id"]) for row in rows]
    assert dashboard_store.get_session_meta("mobile:cache")["last_consolidated"] == 3
    blocked_compactions = dashboard_store.list_compactions("mobile:cache")
    assert [item.invalidated_at for item in blocked_compactions] == [None, None, None]

    runtime_store.release_session_admission("admission:cache")
    deletion = dashboard_store.delete_interaction(control_turn_id)
    assert deletion is not None
    assert deletion.old_last_consolidated == 3
    assert deletion.new_last_consolidated == 1
    compactions = dashboard_store.list_compactions("mobile:cache")
    assert compactions[0].invalidated_at is None
    assert compactions[1].invalidated_at is not None
    assert compactions[1].invalidated_reason == "interaction_deleted:turn:cache"
    assert compactions[2].invalidated_at is not None
    assert compactions[2].invalidated_reason == "interaction_deleted:turn:cache"
    assert dashboard_store.get_session_meta("mobile:cache")["last_consolidated"] == 1

    runtime_store.close()
    dashboard_store.close()


def test_session_manager_only_clears_admissions_when_runtime_owns_workspace(tmp_path) -> None:
    runtime = SessionManager(tmp_path)
    runtime._store.create_session(key="mobile:one")
    assert runtime._store.acquire_session_admission("mobile:one", "admission:one")

    inspector = SessionManager(tmp_path)
    with pytest.raises(ValueError, match="正在处理消息"):
        inspector._store.delete_session("mobile:one", cascade=True)

    inspector.clear_stale_admissions()
    assert inspector._store.delete_session("mobile:one", cascade=True)
    runtime._store.close()
    inspector._store.close()


@pytest.mark.parametrize("batch", [False, True])
def test_session_delete_serializes_admission_check_with_delete(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    batch: bool,
) -> None:
    db_path = tmp_path / "sessions.db"
    runtime_store = SessionStore(db_path)
    dashboard_store = SessionStore(db_path)
    runtime_store.create_session(key="mobile:one")
    checked = threading.Event()
    continue_delete = threading.Event()
    original_check = dashboard_store._require_sessions_not_admitted_locked

    def pause_after_check(keys: list[str]) -> None:
        original_check(keys)
        checked.set()
        assert continue_delete.wait(timeout=2)

    monkeypatch.setattr(
        dashboard_store,
        "_require_sessions_not_admitted_locked",
        pause_after_check,
    )
    result: dict[str, bool] = {}

    def delete_session() -> None:
        result["deleted"] = (
            dashboard_store.delete_sessions_batch(["mobile:one"], cascade=True) == 1
            if batch
            else dashboard_store.delete_session("mobile:one", cascade=True)
        )

    def admit_session() -> None:
        result["admitted"] = runtime_store.acquire_session_admission(
            "mobile:one",
            "admission:one",
        )

    delete_thread = threading.Thread(target=delete_session)
    admit_thread = threading.Thread(target=admit_session)
    delete_thread.start()
    assert checked.wait(timeout=2)
    admit_thread.start()
    assert admit_thread.is_alive()
    continue_delete.set()
    delete_thread.join(timeout=2)
    admit_thread.join(timeout=2)

    assert result == {"deleted": True, "admitted": False}
    runtime_store.close()
    dashboard_store.close()


def test_session_admission_requires_existing_session_and_known_release(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    assert not store.acquire_session_admission("mobile:missing", "admission:missing")
    with pytest.raises(RuntimeError, match="admission 不存在"):
        store.release_session_admission("admission:missing")
    store.close()


@pytest.mark.parametrize(
    "column,bad_value,match",
    [
        ("input_json", "[]", "input 必须是 JSON object"),
        ("items_json", "{}", "items 必须是 JSON array"),
        ("usage_json", "[]", "usage 必须是 JSON object"),
        ("error_json", "{broken", "error JSON 损坏"),
    ],
)
def test_turn_corrupted_json_fails_loud(
    tmp_path, column: str, bad_value: str, match: str
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    record = _queued()
    store.create_turn(record)
    store._conn.execute(
        f"UPDATE turns SET {column} = ? WHERE id = ?", (bad_value, record.id)
    )
    store._conn.commit()

    with pytest.raises(ValueError, match=match):
        store.read_turn(record.id)


def test_compaction_head_is_store_owned_and_monotonic(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="cli:head")
    message = _seed_compaction_message(store, "cli:head")

    initial = store.get_compaction_head("cli:head")
    assert (initial.parent_generation, initial.next_generation) == (0, 1)

    first = store.persist_compaction(
        **_compaction_kwargs("cli:head", message, generation=1),
        parent_generation=initial.parent_generation,
    )
    assert first.generation == 1
    second = store.persist_compaction(
        **(
            _compaction_kwargs("cli:head", message, generation=2)
            | {"source_ref": "source:2"}
        ),
        parent_generation=first.generation,
    )
    assert second.generation == 2

    head = store.get_compaction_head("cli:head")
    assert (head.parent_generation, head.next_generation) == (2, 3)


def test_compaction_retained_unit_ref_survives_store_reopen(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    store.create_session(key="cli:reopen")
    message = _seed_compaction_message(store, "cli:reopen")
    persisted = store.persist_compaction(
        **_compaction_kwargs("cli:reopen", message, generation=1),
        parent_generation=0,
    )
    store.close()

    reopened = SessionStore(db_path)
    loaded = reopened.get_compaction("cli:reopen", persisted.generation)

    assert loaded is not None
    assert loaded.retained_tail[0]["id"] == message["id"]
    assert loaded.retained_tail[0]["unit_ref"] == "turn:1"
    reopened.close()


def test_compaction_head_rejects_cursor_without_active_generation(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="cli:invalid-head")
    store._conn.execute(
        "UPDATE sessions SET last_consolidated = 4 WHERE key = ?",
        ("cli:invalid-head",),
    )
    store._conn.commit()

    with pytest.raises(ValueError, match="超出 ledger head"):
        store.get_compaction_head("cli:invalid-head")


def test_compaction_source_ref_is_idempotent_after_cursor_advances(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="cli:idempotent")
    message = _seed_compaction_message(store, "cli:idempotent")
    first = store.persist_compaction(
        **_compaction_kwargs("cli:idempotent", message, generation=1),
        parent_generation=0,
    )
    _ = store.persist_compaction(
        **(
            _compaction_kwargs("cli:idempotent", message, generation=2)
            | {"source_ref": "source:2"}
        ),
        parent_generation=1,
    )

    replay = store.persist_compaction(
        **_compaction_kwargs("cli:idempotent", message, generation=None),
    )
    assert replay.generation == first.generation
    assert store.get_compaction_head("cli:idempotent").parent_generation == 2

    with pytest.raises(ValueError, match="source_ref 内容冲突"):
        store.persist_compaction(
            **(
                _compaction_kwargs("cli:idempotent", message, generation=None)
                | {"summary": "different"}
            ),
        )


def test_legacy_react_compaction_extra_is_preserved_without_runtime_read(tmp_path) -> None:
    payload = '{"react_compaction":{"compacted_tool_groups":999,"summary":"old"}}'

    decoded = _decode_message_extra(payload, "cli:legacy:1")

    assert decoded["react_compaction"] == {
        "compacted_tool_groups": 999,
        "summary": "old",
    }


def test_new_insert_rejects_retired_react_compaction_extra(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="cli:retired-insert")

    with pytest.raises(ValueError, match="assistant extra 字段已退役"):
        store.insert_message(
            "cli:retired-insert",
            role="assistant",
            content="reply",
            ts=NOW.isoformat(),
            seq=0,
            extra={"react_compaction": {"summary": "new"}},
        )


def test_new_persist_rejects_retired_react_compaction_extra(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    with pytest.raises(ValueError, match="assistant extra 字段已退役"):
        store.persist_session(
            "cli:retired-persist",
            created_at=NOW.isoformat(),
            updated_at=NOW.isoformat(),
            metadata={},
            messages=[
                {
                    "role": "assistant",
                    "content": "reply",
                    "timestamp": NOW.isoformat(),
                    "extra": {"react_compaction": {"summary": "new"}},
                }
            ],
        )


def test_new_update_rejects_retired_react_compaction_extra_without_role(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="cli:retired-update")
    message = store.insert_message(
        "cli:retired-update",
        role="assistant",
        content="reply",
        ts=NOW.isoformat(),
        seq=0,
    )

    with pytest.raises(ValueError, match="assistant extra 字段已退役"):
        store.update_message(
            str(message["id"]),
            extra={"react_compaction": {"summary": "new"}},
        )


def test_new_update_accepts_non_retired_assistant_extra_without_role(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="cli:normal-update")
    message = store.insert_message(
        "cli:normal-update",
        role="assistant",
        content="reply",
        ts=NOW.isoformat(),
        seq=0,
    )

    updated = store.update_message(
        str(message["id"]),
        extra={"trace_id": "trace-1"},
    )

    assert updated is not None
    assert updated["trace_id"] == "trace-1"


def test_session_save_cannot_regress_ledger_cursor(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(key="cli:stale-save")
    message = _seed_compaction_message(store, "cli:stale-save")
    _ = store.persist_compaction(
        **_compaction_kwargs("cli:stale-save", message, generation=1),
        parent_generation=0,
    )

    store.persist_session(
        "cli:stale-save",
        created_at=NOW.isoformat(),
        updated_at=NOW.isoformat(),
        metadata={},
        messages=[],
    )

    assert store.get_session_meta("cli:stale-save")["last_consolidated"] == 1


def test_session_manager_save_rejects_nonzero_cursor_for_new_session(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session = Session("cli:new-stale", last_consolidated=1)

    with pytest.raises(ValueError, match="必须由 ledger 建立"):
        manager.save(session)

    assert not manager.control_store.session_exists("cli:new-stale")
    manager.close()
