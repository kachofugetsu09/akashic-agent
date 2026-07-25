from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.policies.tool_approval import canonical_args_hash
from agent.policies.tool_approval_context import TrustedApprovalContext
from agent.policies.tool_approval_decision import ToolApprovalDecision
from agent.policies.tool_audit import build_tool_approval_audit_event
from agent.policies.tool_approval_store import (
    ToolApprovalRequestRecord,
    ToolApprovalStore,
)


class ToolApprovalRuntime:
    def __init__(
        self,
        store: ToolApprovalStore,
        *,
        now_factory: Callable[[], datetime] | None = None,
        approval_ttl: timedelta = timedelta(minutes=15),
    ) -> None:
        self._store = store
        self._now_factory = now_factory or _utcnow
        self._approval_ttl = approval_ttl

    @staticmethod
    def approval_db_path_from_workspace(workspace: str | Path) -> Path:
        path = (
            Path(workspace).expanduser().resolve()
            / "tool_approvals"
            / "approvals.db"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def store(self) -> ToolApprovalStore:
        return self._store

    def record_defer_request(
        self,
        *,
        request_id: str,
        session_key: str,
        channel: str,
        chat_id: str,
        source: str,
        tool_name: str,
        risk: str,
        approval_scope: str,
        policy_reason: str,
        arguments: Mapping[str, object],
    ) -> ToolApprovalRequestRecord:
        return self._store.create_or_get_pending_request(
            request_id=request_id,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            source=source,
            tool_name=tool_name,
            risk=risk,
            approval_scope=approval_scope,
            policy_reason=policy_reason,
            arguments=arguments,
            now=self._now(),
            ttl=self._approval_ttl,
        )

    def consume_for_execution(
        self,
        *,
        trusted_context: TrustedApprovalContext | None,
        request_id: str,
        session_key: str,
        tool_name: str,
        approval_scope: str,
        arguments: Mapping[str, object],
    ) -> ToolApprovalDecision:
        if trusted_context is None:
            return ToolApprovalDecision(
                action="not_applicable",
                reason="trusted_approval_context_missing",
                request_id=request_id,
                session_key=session_key,
                tool_name=tool_name,
                approval_scope=approval_scope or "tool_call",
                args_hash=canonical_args_hash(arguments),
            )
        return self._store.consume_approved_request(
            approval_request_id=trusted_context.approval_request_id,
            request_id=request_id,
            session_key=session_key,
            tool_name=tool_name,
            approval_scope=approval_scope,
            args_hash=canonical_args_hash(arguments),
            actor=trusted_context.actor,
            now=self._now(),
        )

    def finalize_execution(
        self,
        *,
        approval_request_id: str,
        request_id: str,
        session_key: str,
        tool_name: str,
        approval_scope: str,
        arguments: Mapping[str, object],
        execution_status: str,
    ) -> ToolApprovalDecision:
        return self._store.finalize_consumed_request(
            approval_request_id=approval_request_id,
            request_id=request_id,
            session_key=session_key,
            tool_name=tool_name,
            approval_scope=approval_scope,
            args_hash=canonical_args_hash(arguments),
            execution_status=execution_status,
            now=self._now(),
        )

    @staticmethod
    def lifecycle_event_from_record(
        record: ToolApprovalRequestRecord,
        *,
        status: str,
        actor: str = "model",
    ) -> dict[str, object]:
        decision = ToolApprovalDecision(
            action="pending",
            reason=f"approval_{status}",
            approval_request_id=record.approval_request_id,
            request_id=record.request_id,
            session_key=record.session_key,
            tool_name=record.tool_name,
            approval_scope=record.approval_scope,
            args_hash=record.args_hash,
            metadata={
                "actor": actor,
                "source": record.source,
                "risk": record.risk,
                "policy_reason": record.policy_reason,
                "created_at": record.created_at,
                "decided_at": record.decided_at,
                "consumed_at": record.consumed_at,
                "executed_at": record.executed_at,
            },
        )
        return build_tool_approval_audit_event(
            decision,
            status=status,
            actor=actor,
        ).to_trace_metadata()

    def lifecycle_event_from_decision(
        self,
        decision: ToolApprovalDecision,
        *,
        actor: str = "",
    ) -> dict[str, object]:
        if decision.approval_request_id:
            record = self._store.get_request(decision.approval_request_id)
            if record is not None:
                return self.lifecycle_event_from_record(
                    record,
                    status=decision.action,
                    actor=actor or str(decision.metadata.get("actor") or ""),
                )
        return build_tool_approval_audit_event(
            decision,
            actor=actor,
        ).to_trace_metadata()

    def _now(self) -> datetime:
        value = self._now_factory()
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def _utcnow() -> datetime:
    return datetime.now(UTC)
