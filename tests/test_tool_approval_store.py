from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from agent.policies.tool_approval import canonical_args_hash
from agent.policies.tool_approval_store import ToolApprovalStore


UTC = timezone.utc


def _store(tmp_path: Path) -> ToolApprovalStore:
    return ToolApprovalStore(tmp_path / "approvals.db")


def _now() -> datetime:
    return datetime(2026, 7, 24, 9, 0, tzinfo=UTC)


def _create_pending(
    store: ToolApprovalStore,
    *,
    request_id: str = "req-1",
    session_key: str = "cli:session-1",
    tool_name: str = "write_file",
    approval_scope: str = "tool_call",
    arguments: Mapping[str, object] | None = None,
    now: datetime | None = None,
    ttl: timedelta = timedelta(minutes=5),
):
    return store.create_or_get_pending_request(
        request_id=request_id,
        session_key=session_key,
        channel="cli",
        chat_id="chat-1",
        source="passive",
        tool_name=tool_name,
        risk="write",
        approval_scope=approval_scope,
        policy_reason="risk_strategy_write_requires_approval",
        arguments=arguments
        if arguments is not None
        else {"path": "notes.md", "content": "secret-content"},
        now=now or _now(),
        ttl=ttl,
    )


def _raw_args_summary_json(db_path: Path) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT args_summary_json FROM tool_approval_requests"
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_create_or_get_pending_request_is_idempotent_and_sanitized(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    arguments = {"path": "notes.md", "content": "raw-secret-content"}

    first = _create_pending(store, arguments=arguments)
    second = _create_pending(store, arguments=arguments)

    assert second.approval_request_id == first.approval_request_id
    assert second.args_hash == canonical_args_hash(arguments)
    assert second.args_summary["path"] == "notes.md"
    content_summary = cast(dict[str, Any], second.args_summary["content"])
    assert content_summary["kind"] == "text"
    persisted = _raw_args_summary_json(tmp_path / "approvals.db")
    assert "raw-secret-content" not in persisted
    assert "content" in persisted


def test_create_or_get_pending_request_does_not_reuse_terminal_request(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = _create_pending(store)
    store.deny_request(
        approval_request_id=first.approval_request_id,
        request_id=first.request_id,
        session_key=first.session_key,
        tool_name=first.tool_name,
        approval_scope=first.approval_scope,
        args_hash=first.args_hash,
        actor="user",
        reason="user_denied",
        now=_now(),
    )

    second = _create_pending(store)

    assert second.status == "pending"
    assert second.approval_request_id != first.approval_request_id
    old = store.get_request(first.approval_request_id)
    assert old is not None
    assert old.status == "denied"


def test_approve_requires_request_session_tool_scope_and_args_hash(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    record = _create_pending(store)

    mismatch = store.approve_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope="different_scope",
        args_hash=record.args_hash,
        actor="user",
        now=_now(),
    )
    approved = store.approve_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        now=_now(),
    )

    assert mismatch.action == "mismatch"
    assert approved.action == "approved"
    assert store.get_request(record.approval_request_id).status == "approved"


def test_deny_requires_request_session_tool_scope_and_args_hash(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    record = _create_pending(store)

    mismatch = store.deny_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash="wrong-hash",
        actor="user",
        reason="not now",
        now=_now(),
    )
    denied = store.deny_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        reason="not now",
        now=_now(),
    )

    assert mismatch.action == "mismatch"
    assert denied.action == "denied"
    assert store.get_request(record.approval_request_id).status == "denied"


def test_list_pending_expires_stale_requests(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created_at = _now()
    stale = _create_pending(
        store,
        request_id="req-stale",
        now=created_at,
        ttl=timedelta(seconds=1),
    )
    fresh = _create_pending(store, request_id="req-fresh", now=created_at)

    pending = store.list_pending_requests(
        session_key="cli:session-1",
        now=created_at + timedelta(seconds=2),
    )

    assert [record.approval_request_id for record in pending] == [
        fresh.approval_request_id
    ]
    assert store.get_request(stale.approval_request_id).status == "expired"


def test_expire_pending_requests_returns_expired_decisions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created_at = _now()
    stale = _create_pending(store, now=created_at, ttl=timedelta(seconds=1))

    decisions = store.expire_pending_requests(
        now=created_at + timedelta(seconds=2)
    )

    assert [decision.action for decision in decisions] == ["expired"]
    assert decisions[0].approval_request_id == stale.approval_request_id
    assert decisions[0].args_hash == stale.args_hash


def test_consume_approved_request_is_single_use_and_atomic(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _create_pending(store)
    store.approve_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        now=_now(),
    )

    first = store.consume_approved_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="runtime",
        now=_now(),
    )
    second = store.consume_approved_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="runtime",
        now=_now(),
    )

    assert first.action == "consumed"
    assert first.allows_invoker is True
    assert second.action == "mismatch"
    assert second.allows_invoker is False


def test_approved_request_cannot_be_consumed_after_expiry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created_at = _now()
    record = _create_pending(store, now=created_at, ttl=timedelta(seconds=1))
    store.approve_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        now=created_at,
    )

    decision = store.consume_approved_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="runtime",
        now=created_at + timedelta(seconds=2),
    )

    assert decision.action == "expired"
    assert store.get_request(record.approval_request_id).status == "expired"


def test_denied_request_cannot_be_consumed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _create_pending(store)
    store.deny_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        reason="deny",
        now=_now(),
    )

    decision = store.consume_approved_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="runtime",
        now=_now(),
    )

    assert decision.action == "denied"
    persisted = _raw_args_summary_json(tmp_path / "approvals.db")
    assert "secret-content" not in persisted


def test_finalize_requires_consumed_state_and_full_binding_tuple(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    record = _create_pending(store)
    pending_finalize = store.finalize_consumed_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        execution_status="executed",
        now=_now(),
    )
    store.approve_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        now=_now(),
    )
    store.consume_approved_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="runtime",
        now=_now(),
    )
    mismatch = store.finalize_consumed_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash="wrong-hash",
        execution_status="executed",
        now=_now(),
    )
    executed = store.finalize_consumed_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        execution_status="executed",
        now=_now(),
    )

    assert pending_finalize.action == "pending"
    assert mismatch.action == "mismatch"
    assert executed.action == "executed"
    assert store.get_request(record.approval_request_id).status == "executed"
    assert json.dumps(executed.metadata) == "{}"
