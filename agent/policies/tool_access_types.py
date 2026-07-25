from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent.policies.task_execution_contract import TaskExecutionTurnContract
    from agent.policies.task_plan_contract import TaskPlanTurnContract


@dataclass(frozen=True)
class ToolAccessContext:
    session_key: str
    user_text: str
    always_on_tools: frozenset[str]
    lru_preloaded_tools: frozenset[str]
    disabled_tools: frozenset[str]
    turn_metadata: Mapping[str, Any] = field(default_factory=dict)
    registered_tools: frozenset[str] = frozenset()
    tool_capabilities: Mapping[str, frozenset[str]] = field(default_factory=dict)
    tool_risks: Mapping[str, str] = field(default_factory=dict)
    tool_discovery_enabled: bool = True


@dataclass(frozen=True)
class ToolAccessPlan:
    visible_add: frozenset[str] = frozenset()
    visible_suppress: frozenset[str] = frozenset()
    tool_search_block: frozenset[str] = frozenset()
    execution_block: frozenset[str] = frozenset()
    reason: str = "no_tool_access_policy"
    matched_terms: tuple[str, ...] = ()
    policies: tuple[str, ...] = ()
    filter_error: bool = False
    local_source_allowed: bool = False
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)
    task_plan_contract: TaskPlanTurnContract | None = None
    task_execution_contract: TaskExecutionTurnContract | None = None
    strict_capability_scope: bool = False
    context_retrieval_tools: frozenset[str] = frozenset()
    context_retrieval_consumed: bool = False
    execution_dynamic_tools: frozenset[str] = frozenset()
    final_only: bool = False
    model_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolExecutionGateResult:
    allowed: bool
    error_code: str = ""
    message: str = ""
    recommended_tools: tuple[str, ...] = ()
    reason: str = ""


class ToolAccessPolicy(Protocol):
    name: str

    def build_plan(self, context: ToolAccessContext) -> ToolAccessPlan:
        ...

    def observe_tool_result(
        self,
        plan: ToolAccessPlan,
        tool_name: str,
        result_text: str,
        *,
        execution_status: str = "success",
    ) -> ToolAccessPlan:
        ...
