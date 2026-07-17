from datetime import UTC, datetime, timedelta
import threading

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
from session.manager import SessionManager
from session.store import SessionStore

NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


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


def test_session_admission_blocks_delete_from_another_connection(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    runtime_store = SessionStore(db_path)
    dashboard_store = SessionStore(db_path)
    runtime_store.create_session(key="mobile:one")

    assert runtime_store.acquire_session_admission("mobile:one", "admission:one")
    with pytest.raises(ValueError, match="正在处理消息"):
        dashboard_store.delete_session("mobile:one", cascade=True)

    runtime_store.release_session_admission("admission:one")
    assert dashboard_store.delete_session("mobile:one", cascade=True)
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
