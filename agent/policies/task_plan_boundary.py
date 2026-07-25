from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from agent.policies.task_plan_contract import (
    BackgroundPassthroughMode,
    TaskPlanTurnContract,
    infer_task_plan_turn_decision,
)
from agent.policies.task_control_arbiter import resolve_task_control_context
from agent.policies.tool_access_types import ToolAccessContext, ToolAccessPlan

TASK_PLAN_TOOL_NAMES = frozenset(
    {"create_task_plan", "inspect_task_plan", "update_task_step"}
)
SPAWN_TOOL_NAMES = frozenset({"spawn", "spawn_manage", "task_output"})
TASK_PLAN_LOCAL_TOOL_NAMES = frozenset({"shell", "read_file", "list_dir"})

TaskPlanIntentKind = Literal[
    "none",
    "plan_create",
    "plan_inspect",
    "plan_update",
    "background_job",
]

_BACKGROUND_TOOLS: dict[BackgroundPassthroughMode, frozenset[str]] = {
    "none": frozenset(),
    "start": frozenset({"spawn"}),
    "observe": frozenset({"spawn_manage"}),
    "output": frozenset({"spawn_manage", "task_output"}),
    "cancel": frozenset({"spawn_manage"}),
}
_CONTEXT_CAPABILITIES = frozenset({"memory.recall", "history.search"})


@dataclass(frozen=True)
class TaskPlanIntent:
    kind: TaskPlanIntentKind
    matched_terms: tuple[str, ...] = ()


def infer_task_plan_intent(
    user_text: str,
    has_active_task: bool,
) -> TaskPlanIntent:
    """Compatibility view for existing callers while contracts carry policy state."""
    decision = infer_task_plan_turn_decision(
        user_text,
        has_active_task=has_active_task,
    )
    if decision.background_mode != "none":
        return TaskPlanIntent(
            "background_job",
            decision.contract.matched_terms,
        )
    return TaskPlanIntent(
        decision.contract.action,
        decision.contract.matched_terms,
    )


class TaskPlanAccessPolicy:
    name = "TaskPlanAccessPolicy"

    def build_plan(self, context: ToolAccessContext) -> ToolAccessPlan:
        context = resolve_task_control_context(context)
        background_mode = context.turn_metadata.get("_task_control_background_mode")
        contract = context.turn_metadata.get("task_plan_contract")
        if background_mode != "none" and background_mode in _BACKGROUND_TOOLS:
            return ToolAccessPlan(
                visible_add=_BACKGROUND_TOOLS[background_mode],
                reason=f"background_{background_mode}_passthrough",
                matched_terms=(
                    contract.matched_terms
                    if isinstance(contract, TaskPlanTurnContract)
                    else ()
                ),
                policies=(self.name,),
                policy_metadata={
                    "task_plan": {
                        **(
                            contract.to_trace_metadata()
                            if isinstance(contract, TaskPlanTurnContract)
                            else TaskPlanTurnContract.inactive().to_trace_metadata()
                        ),
                        "background_mode": background_mode,
                    }
                },
            )

        if not isinstance(contract, TaskPlanTurnContract) or not contract.active:
            return ToolAccessPlan()

        universe = _registered_universe(context)
        providers = _providers_by_capability(context, universe)
        missing_required = sorted(
            capability
            for capability in contract.required_capabilities
            if not providers.get(capability)
        )
        optional_capabilities = (
            contract.allowed_capabilities - contract.required_capabilities
        )
        missing_optional = sorted(
            capability
            for capability in optional_capabilities
            if not providers.get(capability)
        )
        resolved_capabilities = {
            capability: sorted(providers.get(capability, frozenset()))
            for capability in sorted(contract.allowed_capabilities)
        }
        trace_metadata = {
            **contract.to_trace_metadata(),
            "resolved_capabilities": resolved_capabilities,
            "missing_required_capabilities": missing_required,
            "optional_capability_unavailable": missing_optional,
        }

        if missing_required:
            missing_text = ", ".join(missing_required)
            return ToolAccessPlan(
                visible_suppress=universe,
                tool_search_block=universe,
                execution_block=universe,
                reason="task_plan_required_capability_missing",
                matched_terms=contract.matched_terms,
                policies=(self.name,),
                filter_error=True,
                policy_metadata={"task_plan": trace_metadata},
                task_plan_contract=contract,
                strict_capability_scope=True,
                model_hints=(
                    f"TaskPlan action {contract.action} is unavailable because "
                    f"required capability {missing_text} has no enabled registered "
                    "provider. Do not use unrelated tools as a fallback.",
                ),
            )

        allowed_tools = frozenset(
            tool_name
            for capability in contract.allowed_capabilities
            for tool_name in providers.get(capability, frozenset())
        )
        retrieval_tools = frozenset(
            tool_name
            for capability in contract.allowed_capabilities & _CONTEXT_CAPABILITIES
            for tool_name in providers.get(capability, frozenset())
        )
        blocked_tools = universe - allowed_tools
        model_hints: tuple[str, ...] = ()
        if optional_capabilities & _CONTEXT_CAPABILITIES and (
            set(missing_optional) & _CONTEXT_CAPABILITIES
        ):
            model_hints = (
                "Optional planning context retrieval is unavailable. Use the "
                "current prompt/context or ask one necessary clarification; do "
                "not discover unrelated retrieval tools.",
            )
        return ToolAccessPlan(
            visible_add=allowed_tools,
            visible_suppress=blocked_tools,
            tool_search_block=blocked_tools,
            execution_block=blocked_tools,
            reason=contract.reason,
            matched_terms=contract.matched_terms,
            policies=(self.name,),
            policy_metadata={"task_plan": trace_metadata},
            task_plan_contract=contract,
            strict_capability_scope=True,
            context_retrieval_tools=retrieval_tools,
            model_hints=model_hints,
        )

    def observe_tool_result(
        self,
        plan: ToolAccessPlan,
        tool_name: str,
        result_text: str,
        *,
        execution_status: str = "success",
    ) -> ToolAccessPlan:
        if (
            plan.task_plan_contract is None
            or tool_name not in plan.context_retrieval_tools
        ):
            return plan

        retired = plan.context_retrieval_tools
        contract = plan.task_plan_contract
        if contract.context_requirement == "session_history":
            hint = (
                "Session-history search is complete. Treat its preview as planning "
                "context; fetch_messages is unavailable in this turn. Create the "
                "task plan now or ask one necessary clarification."
            )
        else:
            hint = (
                "Planning context lookup is complete. Create the task plan from the "
                "available context or ask one necessary clarification; do not call "
                "more retrieval tools."
            )
        metadata = dict(plan.policy_metadata)
        raw_task_plan_metadata = metadata.get("task_plan")
        task_plan_metadata = (
            dict(raw_task_plan_metadata)
            if isinstance(raw_task_plan_metadata, Mapping)
            else {}
        )
        task_plan_metadata.update(
            {
                "context_retrieval_consumed": True,
                "context_retrieval_tool": tool_name,
                "context_retrieval_execution_status": execution_status,
            }
        )
        metadata["task_plan"] = task_plan_metadata
        return replace(
            plan,
            visible_suppress=plan.visible_suppress | retired,
            tool_search_block=plan.tool_search_block | retired,
            context_retrieval_consumed=True,
            policy_metadata=metadata,
            model_hints=_dedupe_tuple((*plan.model_hints, hint)),
        )


def _registered_universe(context: ToolAccessContext) -> frozenset[str]:
    if context.registered_tools:
        return context.registered_tools
    return frozenset(
        set(context.tool_capabilities)
        | set(context.always_on_tools)
        | set(context.lru_preloaded_tools)
    )


def _providers_by_capability(
    context: ToolAccessContext,
    universe: frozenset[str],
) -> dict[str, frozenset[str]]:
    providers: dict[str, set[str]] = {}
    for tool_name, capabilities in context.tool_capabilities.items():
        if tool_name not in universe or tool_name in context.disabled_tools:
            continue
        for capability in capabilities:
            providers.setdefault(capability, set()).add(tool_name)
    return {
        capability: frozenset(tool_names)
        for capability, tool_names in providers.items()
    }


def _dedupe_tuple(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))
