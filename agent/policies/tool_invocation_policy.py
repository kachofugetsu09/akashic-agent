from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from agent.policies.resource_policy import (
    ResourcePolicyContext,
    ResourcePolicyDecision,
    ResourcePolicyEngine,
)
from agent.policies.tool_risk_strategy import (
    DefaultToolRiskStrategy,
    RiskStrategyContext,
)

ToolInvocationAction = Literal["allow", "deny", "defer"]
ToolInvocationSource = Literal["passive", "proactive", "subagent", "task_execution"]
ToolInvocationTaskExecutionPhase = Literal[
    "",
    "inactive",
    "claim",
    "work",
    "waiting_authorization",
    "finish",
    "terminal",
]

_ACTIONS = frozenset({"allow", "deny", "defer"})
_SOURCES = frozenset({"passive", "proactive", "subagent", "task_execution"})
_TASK_EXECUTION_PHASES = frozenset(
    {
        "",
        "inactive",
        "claim",
        "work",
        "waiting_authorization",
        "finish",
        "terminal",
    }
)
_TASK_EXECUTION_CONTROL_CAPABILITIES = frozenset(
    {
        "task_execution.begin",
        "task_execution.finish",
        "task_execution.defer",
        "task_execution.inspect",
        "task_execution.abort",
    }
)


class ResourcePolicy(Protocol):
    def evaluate(self, context: ResourcePolicyContext) -> ResourcePolicyDecision: ...


@dataclass(frozen=True)
class ToolInvocationContext:
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    registered: bool = True
    registry_risk: str = "unknown"
    capabilities: frozenset[str] = field(default_factory=frozenset)
    source: ToolInvocationSource = "passive"
    session_key: str = ""
    request_id: str = ""
    turn_id: int | None = None
    user_text: str = ""
    task_execution_active: bool = False
    task_execution_phase: ToolInvocationTaskExecutionPhase = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError("unsupported tool invocation source")
        if self.task_execution_phase not in _TASK_EXECUTION_PHASES:
            raise ValueError("unsupported task execution phase")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ToolInvocationDecision:
    action: ToolInvocationAction
    reason: str
    risk: str
    policy_name: str = "ToolInvocationPolicyEngine"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ValueError("unsupported tool invocation action")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    def to_trace_metadata(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "risk": self.risk,
            "policy_name": self.policy_name,
            "metadata": dict(self.metadata),
        }


class ToolInvocationPolicyEngine:
    policy_name = "ToolInvocationPolicyEngine"

    def __init__(
        self,
        resource_policy: ResourcePolicy | None = None,
        risk_strategy: DefaultToolRiskStrategy | None = None,
    ) -> None:
        self._resource_policy = resource_policy or ResourcePolicyEngine()
        self._risk_strategy = risk_strategy or DefaultToolRiskStrategy()

    def evaluate(self, context: ToolInvocationContext) -> ToolInvocationDecision:
        risk = _normalize_risk(context.registry_risk)
        metadata = _base_metadata(context, risk)
        if not context.registered:
            return ToolInvocationDecision(
                action="deny",
                reason="tool_invocation_unregistered_tool",
                risk=risk,
                policy_name=self.policy_name,
                metadata=metadata,
            )
        if risk == "destructive":
            return ToolInvocationDecision(
                action="deny",
                reason="tool_invocation_destructive_denied",
                risk=risk,
                policy_name=self.policy_name,
                metadata=metadata,
            )
        resource_decision = self._resource_policy.evaluate(
            ResourcePolicyContext(
                tool_name=context.tool_name,
                arguments=context.arguments,
                resource_roots=_resource_roots(context.metadata.get("resource_roots")),
                source=context.source,
                registry_risk=risk,
            )
        )
        if resource_decision.action in {"deny", "defer"}:
            return ToolInvocationDecision(
                action=cast(ToolInvocationAction, resource_decision.action),
                reason=resource_decision.reason,
                risk=risk,
                policy_name=self.policy_name,
                metadata={
                    **metadata,
                    "resource_policy": resource_decision.to_trace_metadata(),
                },
            )
        if resource_decision.action == "allow":
            metadata = {
                **metadata,
                "resource_policy": resource_decision.to_trace_metadata(),
            }
        if context.task_execution_active and context.task_execution_phase == "work":
            return _task_execution_work_decision(
                context=context,
                risk=risk,
                policy_name=self.policy_name,
                metadata=metadata,
            )
        strategy_decision = self._risk_strategy.evaluate(
            RiskStrategyContext(
                tool_name=context.tool_name,
                registry_risk=risk,
                capabilities=context.capabilities,
                source=context.source,
                task_execution_active=context.task_execution_active,
                task_execution_phase=context.task_execution_phase,
            )
        )
        if strategy_decision.effective:
            return ToolInvocationDecision(
                action=cast(ToolInvocationAction, strategy_decision.action),
                reason=strategy_decision.reason,
                risk=risk,
                policy_name=self.policy_name,
                metadata={
                    **metadata,
                    "risk_strategy": strategy_decision.to_trace_metadata(),
                    "approval_scope": strategy_decision.approval_scope,
                    "approval_user_prompt": strategy_decision.user_prompt,
                },
            )
        return ToolInvocationDecision(
            action="allow",
            reason="tool_invocation_default_allow",
            risk=risk,
            policy_name=self.policy_name,
            metadata=metadata,
        )


def _normalize_risk(value: str) -> str:
    return value if value else "unknown"


def _base_metadata(
    context: ToolInvocationContext,
    risk: str,
) -> dict[str, object]:
    return {
        "tool_name": context.tool_name,
        "risk": risk,
        "source": context.source,
        "registered": context.registered,
        "capabilities": sorted(context.capabilities),
        "task_execution_active": context.task_execution_active,
        "task_execution_phase": context.task_execution_phase,
    }


def _resource_roots(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _task_execution_work_decision(
    *,
    context: ToolInvocationContext,
    risk: str,
    policy_name: str,
    metadata: dict[str, object],
) -> ToolInvocationDecision:
    if context.capabilities & _TASK_EXECUTION_CONTROL_CAPABILITIES:
        return ToolInvocationDecision(
            action="allow",
            reason="tool_invocation_task_execution_control_allowed",
            risk=risk,
            policy_name=policy_name,
            metadata=metadata,
        )
    if context.tool_name != "shell" and risk == "read-only":
        return ToolInvocationDecision(
            action="allow",
            reason="tool_invocation_task_execution_read_only_allowed",
            risk=risk,
            policy_name=policy_name,
            metadata=metadata,
        )
    return ToolInvocationDecision(
        action="defer",
        reason="tool_invocation_task_execution_authorization_required",
        risk=risk,
        policy_name=policy_name,
        metadata={
            **metadata,
            "durable_transition": "waiting_authorization",
        },
    )
