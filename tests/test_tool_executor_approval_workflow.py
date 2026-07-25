from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent.policies.tool_approval_context import (
    TrustedApprovalContext,
    trusted_approval_from_runtime,
)
from agent.policies.tool_approval_runtime import ToolApprovalRuntime
from agent.policies.tool_approval_store import ToolApprovalStore
from agent.tool_hooks import ToolExecutionRequest, ToolExecutor


UTC = timezone.utc


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 24, 11, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class RecordingInvoker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, tool_name: str, arguments: dict[str, Any]) -> object:
        self.calls.append((tool_name, dict(arguments)))
        return {"tool": tool_name, "arguments": dict(arguments)}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _runtime(
    tmp_path: Path,
    *,
    clock: MutableClock | None = None,
    ttl: timedelta = timedelta(minutes=5),
) -> ToolApprovalRuntime:
    return ToolApprovalRuntime(
        ToolApprovalStore(tmp_path / "approvals.db"),
        now_factory=clock or MutableClock(),
        approval_ttl=ttl,
    )


def _write_request(
    *,
    call_id: str = "call-approval",
    arguments: Mapping[str, object] | None = None,
    trusted_approval_context: TrustedApprovalContext | None = None,
    resource_roots: tuple[str, ...] = (),
) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        call_id=call_id,
        tool_name="write_file",
        arguments=dict(arguments)
        if arguments is not None
        else {"path": "notes.md", "content": "hello"},
        source="passive",
        session_key="cli:session-1",
        channel="cli",
        chat_id="chat-1",
        registered=True,
        registry_risk="write",
        trusted_approval_context=trusted_approval_context,
        resource_roots=resource_roots,
    )


def _defer_write(
    runtime: ToolApprovalRuntime,
    invoker: RecordingInvoker,
    *,
    call_id: str = "call-approval",
):
    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(call_id=call_id), invoker
        )
    )
    payload = json.loads(result.output)
    approval_request_id = payload["approval_request"]["approval_request_id"]
    record = runtime.store.get_request(approval_request_id)
    assert record is not None
    return result, record


def _approve(runtime: ToolApprovalRuntime, record) -> None:
    runtime.store.approve_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        now=datetime(2026, 7, 24, 11, 1, tzinfo=UTC),
    )


def test_deferred_write_persists_pending_approval_id(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    invoker = RecordingInvoker()

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(_write_request(), invoker)
    )

    payload = json.loads(result.output)
    approval_request_id = payload["approval_request"]["approval_request_id"]
    record = runtime.store.get_request(approval_request_id)
    assert result.status == "deferred"
    assert result.invoker_reached is False
    assert invoker.calls == []
    assert record is not None
    assert record.status == "pending"
    assert payload["approval_request"]["expires_at"] == record.expires_at
    assert "hello" not in str(payload["approval_request"]["args_summary"])


def test_executor_defer_trace_includes_approval_requested_event(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    invoker = RecordingInvoker()

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(_write_request(), invoker)
    )

    payload = json.loads(result.output)
    event = result.approval_lifecycle[0]
    assert event["event_type"] == "tool_approval_lifecycle"
    assert event["status"] == "requested"
    assert (
        event["approval_request_id"]
        == payload["approval_request"]["approval_request_id"]
    )
    assert event["request_id"] == "call-approval"
    assert event["tool_name"] == "write_file"
    assert event["risk"] == "write"
    assert event["args_hash"]
    assert "args_summary" not in event
    assert "raw-secret-content" not in str(event)


def test_trusted_approved_write_executes_once_and_marks_executed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    invoker = RecordingInvoker()
    _, record = _defer_write(runtime, invoker)
    _approve(runtime, record)
    trusted = trusted_approval_from_runtime(
        approval_request_id=record.approval_request_id,
        actor="user",
        source="status_command",
    )

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(trusted_approval_context=trusted), invoker
        )
    )

    assert result.status == "success"
    assert result.invoker_reached is True
    assert result.invoker_succeeded is True
    assert invoker.calls == [("write_file", {"path": "notes.md", "content": "hello"})]
    assert runtime.store.get_request(record.approval_request_id).status == "executed"


def test_executor_approved_execution_trace_includes_consumed_and_executed_events(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    invoker = RecordingInvoker()
    _, record = _defer_write(runtime, invoker)
    _approve(runtime, record)
    trusted = trusted_approval_from_runtime(
        approval_request_id=record.approval_request_id,
        actor="user",
        source="status_command",
    )

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(trusted_approval_context=trusted), invoker
        )
    )

    statuses = [event["status"] for event in result.approval_lifecycle]
    assert statuses == ["consumed", "executed"]
    for event in result.approval_lifecycle:
        assert event["actor"] == "user"
        assert event["approval_request_id"] == record.approval_request_id
        assert event["request_id"] == record.request_id
        assert event["session_key"] == record.session_key
        assert event["tool_name"] == "write_file"
        assert event["args_hash"] == record.args_hash
        assert "content" not in event
        assert "args_summary" not in event


def test_reusing_consumed_approval_does_not_execute(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    invoker = RecordingInvoker()
    _, record = _defer_write(runtime, invoker)
    _approve(runtime, record)
    trusted = trusted_approval_from_runtime(
        approval_request_id=record.approval_request_id,
        actor="user",
        source="status_command",
    )
    _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(trusted_approval_context=trusted), invoker
        )
    )

    second = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(trusted_approval_context=trusted), invoker
        )
    )

    assert second.status == "deferred"
    assert second.invoker_reached is False
    assert len(invoker.calls) == 1


def test_approved_write_with_changed_arguments_does_not_execute(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    invoker = RecordingInvoker()
    _, record = _defer_write(runtime, invoker)
    _approve(runtime, record)
    trusted = trusted_approval_from_runtime(
        approval_request_id=record.approval_request_id,
        actor="user",
        source="status_command",
    )

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(
                arguments={"path": "notes.md", "content": "changed"},
                trusted_approval_context=trusted,
            ),
            invoker,
        )
    )

    assert result.status == "deferred"
    assert result.invoker_reached is False
    assert invoker.calls == []
    assert runtime.store.get_request(record.approval_request_id).status == "approved"


def test_untrusted_approval_id_in_arguments_does_not_execute(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    invoker = RecordingInvoker()
    _, record = _defer_write(runtime, invoker)
    _approve(runtime, record)

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(
                arguments={
                    "path": "notes.md",
                    "content": "hello",
                    "approval_request_id": record.approval_request_id,
                },
            ),
            invoker,
        )
    )

    assert result.status == "deferred"
    assert result.invoker_reached is False
    assert invoker.calls == []
    assert runtime.store.get_request(record.approval_request_id).status == "approved"


def test_approved_write_still_cannot_escape_resource_roots(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    invoker = RecordingInvoker()
    root = tmp_path / "allowed"
    root.mkdir()
    arguments = {"path": "../outside.md", "content": "hello"}
    record = runtime.record_defer_request(
        request_id="call-approval",
        session_key="cli:session-1",
        channel="cli",
        chat_id="chat-1",
        source="passive",
        tool_name="write_file",
        risk="write",
        approval_scope="tool_call",
        policy_reason="risk_strategy_write_requires_approval",
        arguments=arguments,
    )
    _approve(runtime, record)
    trusted = trusted_approval_from_runtime(
        approval_request_id=record.approval_request_id,
        actor="user",
        source="status_command",
    )

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(
                arguments=arguments,
                trusted_approval_context=trusted,
                resource_roots=(str(root),),
            ),
            invoker,
        )
    )

    assert result.status == "denied"
    assert result.invoker_reached is False
    assert result.policy_trace["reason"] == "resource_policy_file_path_outside_roots"
    assert invoker.calls == []
    assert runtime.store.get_request(record.approval_request_id).status == "approved"


def test_denied_or_expired_approval_does_not_execute(tmp_path: Path) -> None:
    clock = MutableClock()
    runtime = _runtime(tmp_path, clock=clock, ttl=timedelta(seconds=1))
    invoker = RecordingInvoker()
    _, denied_record = _defer_write(runtime, invoker)
    runtime.store.deny_request(
        approval_request_id=denied_record.approval_request_id,
        request_id=denied_record.request_id,
        session_key=denied_record.session_key,
        tool_name=denied_record.tool_name,
        approval_scope=denied_record.approval_scope,
        args_hash=denied_record.args_hash,
        actor="user",
        reason="deny",
        now=clock(),
    )
    denied_trusted = trusted_approval_from_runtime(
        approval_request_id=denied_record.approval_request_id,
        actor="user",
        source="status_command",
    )

    denied_result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(trusted_approval_context=denied_trusted), invoker
        )
    )

    clock.now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
    expired_result, expired_record = _defer_write(
        runtime, invoker, call_id="call-expired"
    )
    assert expired_result.status == "deferred"
    runtime.store.approve_request(
        approval_request_id=expired_record.approval_request_id,
        request_id=expired_record.request_id,
        session_key=expired_record.session_key,
        tool_name=expired_record.tool_name,
        approval_scope=expired_record.approval_scope,
        args_hash=expired_record.args_hash,
        actor="user",
        now=clock(),
    )
    clock.now = clock.now + timedelta(seconds=2)
    expired_trusted = trusted_approval_from_runtime(
        approval_request_id=expired_record.approval_request_id,
        actor="user",
        source="status_command",
    )
    expired_consume_result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(
                call_id="call-expired",
                trusted_approval_context=expired_trusted,
            ),
            invoker,
        )
    )

    assert denied_result.status == "deferred"
    assert denied_result.invoker_reached is False
    assert expired_consume_result.status == "deferred"
    assert expired_consume_result.invoker_reached is False
    assert invoker.calls == []
    assert runtime.store.get_request(expired_record.approval_request_id).status == (
        "expired"
    )
