from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.policies.tool_approval import canonical_args_hash, summarize_arguments
from agent.policies.tool_approval_decision import ToolApprovalDecision


_APPROVAL_LIFECYCLE_EVENT_TYPE = "tool_approval_lifecycle"


@dataclass(frozen=True)
class ToolAuditEvent:
    event_type: str
    request_id: str
    session_key: str
    channel: str
    chat_id: str
    tool_name: str
    source: str
    risk: str
    policy_action: str
    policy_reason: str
    args_hash: str
    args_summary: dict[str, Any] = field(default_factory=dict)
    invoker_reached: bool = False
    invoker_succeeded: bool = False

    def to_trace_metadata(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "request_id": self.request_id,
            "session_key": self.session_key,
            "channel": self.channel,
            "chat_id": self.chat_id,
            "tool_name": self.tool_name,
            "source": self.source,
            "risk": self.risk,
            "policy_action": self.policy_action,
            "policy_reason": self.policy_reason,
            "args_hash": self.args_hash,
            "args_summary": dict(self.args_summary),
            "invoker_reached": self.invoker_reached,
            "invoker_succeeded": self.invoker_succeeded,
        }


def build_tool_audit_event(
    *,
    request_id: str,
    session_key: str,
    channel: str,
    chat_id: str,
    tool_name: str,
    source: str,
    risk: str,
    policy_action: str,
    policy_reason: str,
    arguments: dict[str, Any],
    invoker_reached: bool,
    invoker_succeeded: bool,
) -> ToolAuditEvent:
    return ToolAuditEvent(
        event_type="tool_invocation_policy_decision",
        request_id=request_id,
        session_key=session_key,
        channel=channel,
        chat_id=chat_id,
        tool_name=tool_name,
        source=source,
        risk=risk,
        policy_action=policy_action,
        policy_reason=policy_reason,
        args_hash=canonical_args_hash(arguments),
        args_summary=summarize_arguments(arguments),
        invoker_reached=invoker_reached,
        invoker_succeeded=invoker_succeeded,
    )


@dataclass(frozen=True)
class ToolApprovalAuditEvent:
    event_type: str
    approval_request_id: str
    request_id: str
    session_key: str
    actor: str
    source: str
    tool_name: str
    risk: str
    approval_scope: str
    policy_reason: str
    status: str
    args_hash: str
    created_at: str = ""
    decided_at: str = ""
    consumed_at: str = ""
    executed_at: str = ""

    def to_trace_metadata(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "approval_request_id": self.approval_request_id,
            "request_id": self.request_id,
            "session_key": self.session_key,
            "actor": self.actor,
            "source": self.source,
            "tool_name": self.tool_name,
            "risk": self.risk,
            "approval_scope": self.approval_scope,
            "policy_reason": self.policy_reason,
            "status": self.status,
            "args_hash": self.args_hash,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "consumed_at": self.consumed_at,
            "executed_at": self.executed_at,
        }


def build_tool_approval_audit_event(
    decision: ToolApprovalDecision,
    *,
    status: str | None = None,
    actor: str = "",
    source: str = "",
) -> ToolApprovalAuditEvent:
    metadata = decision.metadata
    return ToolApprovalAuditEvent(
        event_type=_APPROVAL_LIFECYCLE_EVENT_TYPE,
        approval_request_id=decision.approval_request_id,
        request_id=decision.request_id,
        session_key=decision.session_key,
        actor=actor or _metadata_str(metadata, "actor"),
        source=source or _metadata_str(metadata, "source"),
        tool_name=decision.tool_name,
        risk=_metadata_str(metadata, "risk"),
        approval_scope=decision.approval_scope or "tool_call",
        policy_reason=_metadata_str(metadata, "policy_reason"),
        status=status or decision.action,
        args_hash=decision.args_hash,
        created_at=_metadata_str(metadata, "created_at"),
        decided_at=_metadata_str(metadata, "decided_at"),
        consumed_at=_metadata_str(metadata, "consumed_at"),
        executed_at=_metadata_str(metadata, "executed_at"),
    )


def _metadata_str(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    return value if isinstance(value, str) else ""
