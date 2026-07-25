from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from agent.policies.tool_approval_context import (
    TrustedApprovalContext,
    trusted_approval_from_runtime,
)
from agent.policies.tool_approval_decision import ToolApprovalDecision
from agent.policies.tool_approval_runtime import ToolApprovalRuntime
from agent.policies.tool_approval_store import ToolApprovalStore
from agent.policies.tool_audit import build_tool_approval_audit_event
from agent.tool_hooks import ToolExecutionRequest, ToolExecutor
from tests.test_task_execution_reasoner import (
    ReasonerExecutionFixture,
    final_reply,
    tool_call,
)

UTC = timezone.utc


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class _RecordingInvoker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, tool_name: str, arguments: dict[str, Any]) -> object:
        self.calls.append((tool_name, dict(arguments)))
        return {"ok": True, "tool_name": tool_name}


async def _raising_invoker(tool_name: str, arguments: dict[str, Any]) -> object:
    raise AssertionError(f"invoker reached for {tool_name}: {arguments}")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _runtime(
    tmp_path: Path,
    *,
    clock: _Clock | None = None,
    ttl: timedelta = timedelta(minutes=5),
) -> ToolApprovalRuntime:
    return ToolApprovalRuntime(
        ToolApprovalStore(tmp_path / "approvals.db"),
        now_factory=clock or _Clock(),
        approval_ttl=ttl,
    )


def _write_request(
    *,
    call_id: str = "p3-call",
    arguments: Mapping[str, object] | None = None,
    trusted_approval_context: TrustedApprovalContext | None = None,
    resource_roots: tuple[str, ...] = (),
) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        call_id=call_id,
        tool_name="write_file",
        arguments=dict(arguments)
        if arguments is not None
        else {"path": "notes.md", "content": "private"},
        source="passive",
        session_key="cli:p3",
        channel="cli",
        chat_id="p3",
        registered=True,
        registry_risk="write",
        trusted_approval_context=trusted_approval_context,
        resource_roots=resource_roots,
    )


def _approve(runtime: ToolApprovalRuntime, approval_request_id: str) -> object:
    record = runtime.store.get_request(approval_request_id)
    assert record is not None
    runtime.store.approve_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        now=datetime(2026, 7, 24, 12, 1, tzinfo=UTC),
    )
    return record


def _defer_write(runtime: ToolApprovalRuntime, invoker: _RecordingInvoker) -> str:
    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(_write_request(), invoker)
    )
    payload = json.loads(result.output)
    assert result.status == "deferred"
    return str(payload["approval_request"]["approval_request_id"])


def test_p3_real_defer_approve_resume_executes_once(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    invoker = _RecordingInvoker()
    approval_request_id = _defer_write(runtime, invoker)
    record = _approve(runtime, approval_request_id)
    trusted = trusted_approval_from_runtime(
        approval_request_id=record.approval_request_id,
        actor="user",
        source="status_command",
    )

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(trusted_approval_context=trusted),
            invoker,
        )
    )
    reused = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(trusted_approval_context=trusted),
            invoker,
        )
    )

    assert result.status == "success"
    assert result.invoker_reached is True
    assert invoker.calls == [("write_file", {"path": "notes.md", "content": "private"})]
    assert runtime.store.get_request(record.approval_request_id).status == "executed"
    assert reused.status == "deferred"
    assert reused.invoker_reached is False
    assert len(invoker.calls) == 1


def test_p3_changed_args_denied_expired_or_reused_approval_do_not_execute(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    runtime = _runtime(tmp_path, clock=clock, ttl=timedelta(seconds=1))
    invoker = _RecordingInvoker()

    changed_id = _defer_write(runtime, invoker)
    changed_record = _approve(runtime, changed_id)
    changed_trusted = trusted_approval_from_runtime(
        approval_request_id=changed_record.approval_request_id,
        actor="user",
        source="status_command",
    )
    changed = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(
                arguments={"path": "notes.md", "content": "changed"},
                trusted_approval_context=changed_trusted,
            ),
            invoker,
        )
    )

    denied_id = _defer_write(runtime, invoker)
    denied_record = runtime.store.get_request(denied_id)
    assert denied_record is not None
    runtime.store.deny_request(
        approval_request_id=denied_record.approval_request_id,
        request_id=denied_record.request_id,
        session_key=denied_record.session_key,
        tool_name=denied_record.tool_name,
        approval_scope=denied_record.approval_scope,
        args_hash=denied_record.args_hash,
        actor="user",
        reason="user_denied",
        now=clock(),
    )
    denied = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(
                trusted_approval_context=trusted_approval_from_runtime(
                    approval_request_id=denied_record.approval_request_id,
                    actor="user",
                    source="status_command",
                )
            ),
            invoker,
        )
    )

    expired_id = _defer_write(runtime, invoker)
    expired_record = _approve(runtime, expired_id)
    clock.now = clock.now + timedelta(seconds=2)
    expired = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(
                trusted_approval_context=trusted_approval_from_runtime(
                    approval_request_id=expired_record.approval_request_id,
                    actor="user",
                    source="status_command",
                )
            ),
            invoker,
        )
    )

    assert changed.status == "deferred"
    assert denied.status == "deferred"
    assert expired.status == "deferred"
    assert changed.invoker_reached is False
    assert denied.invoker_reached is False
    assert expired.invoker_reached is False
    assert invoker.calls == []


def test_p3_p1_resource_deny_still_wins_after_approval(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    arguments = {"path": "../outside.md", "content": "private"}
    record = runtime.record_defer_request(
        request_id="p3-call",
        session_key="cli:p3",
        channel="cli",
        chat_id="p3",
        source="passive",
        tool_name="write_file",
        risk="write",
        approval_scope="tool_call",
        policy_reason="risk_strategy_write_requires_approval",
        arguments=arguments,
    )
    _approve(runtime, record.approval_request_id)
    root = tmp_path / "workspace"
    root.mkdir()

    result = _run(
        ToolExecutor(approval_runtime=runtime).execute(
            _write_request(
                arguments=arguments,
                trusted_approval_context=trusted_approval_from_runtime(
                    approval_request_id=record.approval_request_id,
                    actor="user",
                    source="status_command",
                ),
                resource_roots=(str(root),),
            ),
            _raising_invoker,
        )
    )

    assert result.status == "denied"
    assert result.invoker_reached is False
    assert result.policy_trace["reason"] == "resource_policy_file_path_outside_roots"
    assert runtime.store.get_request(record.approval_request_id).status == "approved"


@pytest.mark.asyncio
async def test_p3_task_execution_metadata_bridge_does_not_resume_side_effects(
    tmp_path: Path,
) -> None:
    fixture = ReasonerExecutionFixture(tmp_path)
    fixture.reasoner._context.workspace = tmp_path
    fixture.llm.responses = [
        tool_call("begin_task_step_execution", {}),
        tool_call("write_file", {"path": "x.txt", "content": "raw-secret-content"}),
        final_reply("Waiting for authorization"),
    ]

    await fixture.run_turn("继续执行下一步")
    attempt = fixture.latest_attempt()
    store = ToolApprovalStore(
        ToolApprovalRuntime.approval_db_path_from_workspace(tmp_path)
    )
    record = store.get_request(str(attempt.requested_arguments["approval_request_id"]))
    assert record is not None
    store.approve_request(
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
        actor="user",
        now=fixture.clock(),
    )

    assert attempt.status == "waiting_authorization"
    assert attempt.requested_arguments["approval_scope"] == "task_execution_step"
    assert "raw-secret-content" not in json.dumps(
        attempt.requested_arguments,
        ensure_ascii=False,
    )
    assert fixture.write_executor_calls == []


def test_p3_lifecycle_audit_is_bounded() -> None:
    trace = build_tool_approval_audit_event(
        ToolApprovalDecision(
            action="executed",
            reason="approval_executed",
            approval_request_id="approval-1",
            request_id="call-1",
            session_key="cli:p3",
            tool_name="write_file",
            approval_scope="tool_call",
            args_hash="abc123",
            metadata={
                "actor": "user",
                "source": "passive",
                "risk": "write",
                "policy_reason": "risk_strategy_write_requires_approval",
                "created_at": "2026-07-24T12:00:00+00:00",
                "executed_at": "2026-07-24T12:01:00+00:00",
                "args_summary": {"content": {"sha256": "secret"}},
                "command": "rm file.txt",
                "content": "raw-secret-content",
            },
        )
    ).to_trace_metadata()

    assert trace["event_type"] == "tool_approval_lifecycle"
    assert trace["status"] == "executed"
    assert trace["approval_request_id"] == "approval-1"
    assert trace["args_hash"] == "abc123"
    assert "args_summary" not in trace
    assert "command" not in trace
    assert "content" not in trace
