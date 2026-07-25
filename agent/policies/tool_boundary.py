from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from agent.policies.evidence_completion import EvidenceCompletionPolicy
from agent.policies.task_plan_context_budget import TaskPlanContextBudgetPolicy
from agent.policies.task_plan_contract import TaskPlanTurnContract
from agent.policies.task_execution_boundary import TaskExecutionRiskPolicy
from agent.policies.task_execution_budget import (
    TaskExecutionBudgetPolicy,
    TaskExecutionEventClassifier,
)
from agent.policies.tool_access import ToolAccessGateway
from agent.policies.tool_access_types import (
    ToolAccessContext,
    ToolAccessPlan,
)
from agent.policies.tool_budget import (
    TaskIntent,
    ToolBoundaryDecision,
    ToolBudgetPolicy,
    infer_task_intent,
)
from agent.policies.tool_ledger import (
    ToolCallLedger,
    ToolCallRecord,
    classify_tool_name,
    extract_tool_result_facts,
    stable_args_hash,
    summarize_args,
)
from agent.tools.base import ToolResult, normalize_tool_result
from agent.policies.task_execution_contract import TaskExecutionTurnContract

logger = logging.getLogger(__name__)


@dataclass
class ToolBoundaryContext:
    access_context: ToolAccessContext
    access_plan: ToolAccessPlan
    intent: TaskIntent
    task_plan_contract: TaskPlanTurnContract | None = None
    task_execution_contract: TaskExecutionTurnContract | None = None
    ledger: ToolCallLedger = field(default_factory=ToolCallLedger)
    pending_hints: list[str] = field(default_factory=list)
    decisions: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class BoundaryExecutionDecision:
    action: str
    reason: str
    execute: bool
    result_payload: str | None = None
    model_hint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TurnToolBoundaryManager:
    def __init__(
        self,
        *,
        access_gateway: ToolAccessGateway | None = None,
        budget_policy: ToolBudgetPolicy | None = None,
        evidence_policy: EvidenceCompletionPolicy | None = None,
        task_plan_context_budget_policy: TaskPlanContextBudgetPolicy | None = None,
        task_execution_risk_policy: TaskExecutionRiskPolicy | None = None,
        task_execution_budget_policy: TaskExecutionBudgetPolicy | None = None,
    ) -> None:
        self._access = access_gateway or ToolAccessGateway()
        self._budget = budget_policy or ToolBudgetPolicy()
        self._evidence = evidence_policy or EvidenceCompletionPolicy()
        self._task_plan_context_budget = (
            task_plan_context_budget_policy or TaskPlanContextBudgetPolicy()
        )
        self._task_execution_risk = (
            task_execution_risk_policy or TaskExecutionRiskPolicy()
        )
        self._task_execution_budget = (
            task_execution_budget_policy or TaskExecutionBudgetPolicy()
        )
        self._task_execution_event_classifier = TaskExecutionEventClassifier()

    def build_context(self, access_context: ToolAccessContext) -> ToolBoundaryContext:
        access_plan = self._access.build_plan(access_context)
        contract = access_plan.task_plan_contract
        return ToolBoundaryContext(
            access_context=access_context,
            access_plan=access_plan,
            intent=(
                "task_plan_state"
                if contract is not None and contract.active
                else infer_task_intent(access_context.user_text)
            ),
            task_plan_contract=contract,
            task_execution_contract=access_plan.task_execution_contract,
            pending_hints=list(access_plan.model_hints),
        )

    def refresh_task_execution_contract(
        self,
        context: ToolBoundaryContext,
        contract: TaskExecutionTurnContract,
    ) -> ToolBoundaryContext:
        metadata = dict(context.access_context.turn_metadata)
        metadata["task_execution_contract"] = contract
        access_context = replace(context.access_context, turn_metadata=metadata)
        refreshed = self.build_context(access_context)
        if (
            contract.active
            and contract.phase == "work"
            and refreshed.access_plan.execution_dynamic_tools
        ):
            refreshed.access_plan = replace(
                refreshed.access_plan,
                visible_add=(
                    refreshed.access_plan.visible_add
                    | refreshed.access_plan.execution_dynamic_tools
                ),
            )
        refreshed.ledger = context.ledger
        refreshed.decisions = context.decisions
        refreshed.pending_hints = [
            *context.pending_hints,
            *(
                hint
                for hint in refreshed.pending_hints
                if hint not in context.pending_hints
            ),
        ]
        return refreshed

    def compute_visible_names(self, context: ToolBoundaryContext) -> set[str]:
        return self._access.compute_visible_names(
            context.access_context,
            context.access_plan,
        )

    def filter_tool_search_matches(
        self,
        context: ToolBoundaryContext,
        tool_search_payload: str,
    ) -> tuple[str, tuple[str, ...]]:
        return self._access.filter_tool_search_matches(
            context.access_plan,
            tool_search_payload,
        )

    def merge_tool_search_unlocks(
        self,
        *,
        context: ToolBoundaryContext,
        current_visible: set[str],
        unlocked: set[str],
    ) -> set[str]:
        return self._access.merge_tool_search_unlocks(
            current_visible=current_visible,
            unlocked=unlocked,
            context=context.access_context,
            plan=context.access_plan,
        )

    def recent_decisions(
        self,
        context: ToolBoundaryContext,
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(context.decisions)

    def observe_access_tool_result(
        self,
        context: ToolBoundaryContext,
        tool_name: str,
        result_text: str | ToolResult,
        *,
        execution_status: str = "success",
    ) -> None:
        previous_plan = context.access_plan
        updated_plan = self._access.observe_tool_result(
            previous_plan,
            tool_name,
            normalize_tool_result(result_text).preview(),
            execution_status=execution_status,
        )
        context.access_plan = updated_plan
        previous_hints = set(previous_plan.model_hints)
        context.pending_hints.extend(
            hint for hint in updated_plan.model_hints if hint not in previous_hints
        )

    def evaluate_tool_call(
        self,
        context: ToolBoundaryContext,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        visible_names: set[str] | None,
    ) -> BoundaryExecutionDecision:
        execution_contract = context.access_plan.task_execution_contract
        registered_tools = context.access_context.registered_tools
        registered = (
            tool_name in registered_tools
            if registered_tools
            else tool_name in context.access_context.tool_risks
        )
        is_execution_control = bool(
            context.access_context.tool_capabilities.get(tool_name, frozenset())
            & {
                "task_execution.begin",
                "task_execution.inspect",
                "task_execution.finish",
                "task_execution.defer",
                "task_execution.abort",
            }
        )
        if (
            execution_contract is not None
            and execution_contract.active
            and execution_contract.phase == "work"
            and (
                context.access_plan.filter_error
                or tool_name in context.access_context.disabled_tools
            )
        ):
            gate = self._access.check_tool_call(
                context.access_plan,
                tool_name,
                dict(arguments),
            )
            return self._access_block(
                context,
                tool_name,
                arguments,
                error_code=gate.error_code,
                message=gate.message,
                recommended_tools=gate.recommended_tools,
                reason=gate.error_code or gate.reason or "tool_access_block",
            )
        if not is_execution_control:
            risk_decision = self._task_execution_risk.evaluate(
                contract=execution_contract,
                tool_name=tool_name,
                registered=registered,
                registry_risk=context.access_context.tool_risks.get(tool_name, "unknown"),
                registry_capabilities=context.access_context.tool_capabilities.get(
                    tool_name,
                    frozenset(),
                ),
            )
            if risk_decision is not None and risk_decision.action == "unknown_tool":
                return self._boundary_block(
                    context,
                    tool_name,
                    arguments,
                    risk_decision.reason,
                    risk_decision.metadata,
                )
            if risk_decision is not None and risk_decision.action == "deny":
                return self._boundary_block(
                    context,
                    tool_name,
                    arguments,
                    risk_decision.reason,
                    risk_decision.metadata,
                )
            if risk_decision is not None and risk_decision.action == "defer":
                return self._authorization_defer(
                    context,
                    tool_name,
                    arguments,
                    metadata=risk_decision.metadata,
                    shell_compatibility=(tool_name == "shell"),
                )

        gate = self._access.check_tool_call(
            context.access_plan,
            tool_name,
            dict(arguments),
        )
        if not gate.allowed:
            return self._access_block(
                context,
                tool_name,
                arguments,
                error_code=gate.error_code,
                message=gate.message,
                recommended_tools=gate.recommended_tools,
                reason=gate.error_code or gate.reason or "tool_access_block",
            )

        task_execution_budget_decision = self._task_execution_budget.evaluate(
            contract=execution_contract,
            ledger=context.ledger,
            tool_name=tool_name,
            arguments=dict(arguments),
            tool_risk=context.access_context.tool_risks.get(tool_name, "unknown"),
            tool_capabilities=context.access_context.tool_capabilities,
        )
        if task_execution_budget_decision is not None:
            if task_execution_budget_decision.action == "soft_stop":
                return self._soft_stop(
                    context,
                    tool_name,
                    arguments,
                    task_execution_budget_decision,
                )
            if task_execution_budget_decision.action != "allow":
                decision = BoundaryExecutionDecision(
                    action=task_execution_budget_decision.action,
                    reason=task_execution_budget_decision.reason,
                    execute=True,
                    model_hint=task_execution_budget_decision.model_hint,
                    metadata=dict(task_execution_budget_decision.metadata),
                )
                self._record_decision(context, tool_name, decision, arguments)
                return decision

        task_plan_budget_decision = self._task_plan_context_budget.evaluate_call(
            contract=context.task_plan_contract,
            ledger=context.ledger,
            tool_name=tool_name,
            tool_capabilities=context.access_context.tool_capabilities,
        )
        if task_plan_budget_decision is not None:
            if task_plan_budget_decision.action == "soft_stop":
                return self._soft_stop(
                    context,
                    tool_name,
                    arguments,
                    task_plan_budget_decision,
                )
            decision = BoundaryExecutionDecision(
                action=task_plan_budget_decision.action,
                reason=task_plan_budget_decision.reason,
                execute=True,
                model_hint=task_plan_budget_decision.model_hint,
                metadata=dict(task_plan_budget_decision.metadata),
            )
            self._record_decision(context, tool_name, decision, arguments)
            return decision

        evidence_decision = self._evidence.evaluate_call(
            intent=context.intent,
            ledger=context.ledger,
            tool_name=tool_name,
            arguments=arguments,
        )
        budget_decision = self._budget.evaluate_call(
            intent=context.intent,
            ledger=context.ledger,
            tool_name=tool_name,
            arguments=arguments,
            visible_names=visible_names,
        )
        final = _more_restrictive(evidence_decision, budget_decision)
        if final.action == "soft_stop":
            return self._soft_stop(context, tool_name, arguments, final)

        decision = BoundaryExecutionDecision(
            action=final.action,
            reason=final.reason,
            execute=True,
            model_hint=final.model_hint,
            metadata=dict(final.metadata),
        )
        self._record_decision(context, tool_name, decision, arguments)
        return decision

    def record_tool_result(
        self,
        context: ToolBoundaryContext,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        result_text: str | ToolResult,
        visible_before_call: bool,
        decision_action: str,
        decision_reason: str,
        execution_status: str = "",
        tool_call_id: str = "",
        invoker_reached: bool | None = None,
        invoker_succeeded: bool | None = None,
        requested_unlocks: tuple[str, ...] = (),
        unlocked_tools: tuple[str, ...] = (),
        blocked_tools: tuple[str, ...] = (),
    ) -> None:
        normalized = normalize_tool_result(result_text)
        rendered_text = normalized.preview()
        facts = extract_tool_result_facts(tool_name, result_text)
        execution_contract = context.access_plan.task_execution_contract
        is_execution_work = bool(
            execution_contract is not None
            and execution_contract.active
            and execution_contract.phase == "work"
        )
        event = self._task_execution_event_classifier.classify(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            registry_risk=context.access_context.tool_risks.get(tool_name, "unknown"),
            invoker_reached=(
                execution_status == "success"
                if invoker_reached is None
                else invoker_reached
            ),
            invoker_succeeded=(
                execution_status == "success"
                if invoker_succeeded is None
                else invoker_succeeded
            ),
            execution_status=execution_status,
            result_ok=facts.result_ok,
        )
        context.ledger.add_record(
            ToolCallRecord(
                tool_name=tool_name,
                tool_call_id=event.tool_call_id if is_execution_work else "",
                tool_class=classify_tool_name(tool_name),
                args_hash=stable_args_hash(arguments),
                args_summary=summarize_args(arguments),
                call_index=context.ledger.next_call_index(),
                visible_before_call=visible_before_call,
                decision_action=decision_action,
                decision_reason=decision_reason,
                requested_unlocks=requested_unlocks,
                unlocked_tools=unlocked_tools,
                blocked_tools=blocked_tools,
                execution_status=execution_status,
                result_ok=facts.result_ok,
                hit_count=facts.hit_count,
                citation_refs=facts.citation_refs,
                chunk_keys=facts.chunk_keys,
                terminal_scope=facts.terminal_scope,
                result_summary=rendered_text[:240],
                result_text=rendered_text,
                result_has_evidence=facts.result_has_evidence,
                result_has_citation=facts.result_has_citation,
                result_error_code=facts.result_error_code,
                tool_risk=event.tool_risk if is_execution_work else "",
                tool_capabilities=tuple(
                    sorted(context.access_context.tool_capabilities.get(tool_name, frozenset()))
                )
                if is_execution_work
                else (),
                counts_as_work=event.counts_as_work if is_execution_work else False,
                invoker_reached=event.invoker_reached if is_execution_work else False,
                invoker_succeeded=(
                    event.invoker_succeeded if is_execution_work else False
                ),
            )
        )

    def consume_pending_hint(self, context: ToolBoundaryContext) -> str | None:
        if not context.pending_hints:
            return None
        return context.pending_hints.pop(0)

    def trace(self, context: ToolBoundaryContext) -> dict[str, Any]:
        contract = context.task_plan_contract
        context_consumed_count = 0
        if contract is not None and contract.context_requirement != "none":
            capability = (
                "memory.recall"
                if contract.context_requirement == "long_term_memory"
                else "history.search"
            )
            context_consumed_count = sum(
                1
                for record in context.ledger.records
                if capability
                in context.access_context.tool_capabilities.get(
                    record.tool_name, frozenset()
                )
            )
        budget_decision_reason = next(
            (
                str(decision.get("reason") or "")
                for decision in reversed(context.decisions)
                if str(decision.get("reason") or "").startswith(
                    "task_plan_context_"
                )
            ),
            "",
        )
        task_plan_metadata = context.access_plan.policy_metadata.get(
            "task_plan", {}
        )
        last_execution_status = (
            str(task_plan_metadata.get("context_retrieval_execution_status") or "")
            if isinstance(task_plan_metadata, Mapping)
            else ""
        )
        completion_capability = contract.completion_capability if contract else None
        completion_providers = sorted(
            tool_name
            for tool_name, capabilities in context.access_context.tool_capabilities.items()
            if completion_capability is not None
            and completion_capability in capabilities
            and tool_name in context.access_context.registered_tools
            and tool_name not in context.access_context.disabled_tools
        )
        return {
            "intent": context.intent,
            "task_plan_contract": (
                contract.to_trace_metadata() if contract is not None else None
            ),
            "task_plan_context_budget": {
                "retrieval_budget": contract.retrieval_budget if contract else 0,
                "consumed_count": context_consumed_count,
                "consumed": context.access_plan.context_retrieval_consumed,
                "decision_reason": budget_decision_reason,
                "last_execution_status": last_execution_status,
            },
            "task_plan_completion": {
                "completion_capability": completion_capability,
                "resolved_provider_tools": completion_providers,
            },
            "tool_access": {
                "reason": context.access_plan.reason,
                "policies": list(context.access_plan.policies),
                "visible_add": sorted(context.access_plan.visible_add),
                "visible_suppress": sorted(context.access_plan.visible_suppress),
                "tool_search_block": sorted(context.access_plan.tool_search_block),
                "execution_block": sorted(context.access_plan.execution_block),
                "matched_terms": list(context.access_plan.matched_terms),
                "filter_error": context.access_plan.filter_error,
                "policy_metadata": dict(context.access_plan.policy_metadata),
            },
            "decisions": list(context.decisions),
            "ledger_summary": context.ledger.summary(),
        }

    def _soft_stop(
        self,
        context: ToolBoundaryContext,
        tool_name: str,
        arguments: Mapping[str, Any],
        policy_decision: ToolBoundaryDecision,
    ) -> BoundaryExecutionDecision:
        if policy_decision.model_hint:
            context.pending_hints.append(policy_decision.model_hint)
        logger.info(
            "[tool_boundary] soft_stop tool=%s reason=%s",
            tool_name,
            policy_decision.reason,
        )
        decision = BoundaryExecutionDecision(
            action="soft_stop",
            reason=policy_decision.reason,
            execute=False,
            result_payload=_soft_stop_payload(policy_decision),
            model_hint=policy_decision.model_hint,
            metadata=dict(policy_decision.metadata),
        )
        self._record_decision(context, tool_name, decision, arguments)
        return decision

    def _authorization_defer(
        self,
        context: ToolBoundaryContext,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        metadata: Mapping[str, object],
        shell_compatibility: bool,
    ) -> BoundaryExecutionDecision:
        decision = BoundaryExecutionDecision(
            action="soft_stop" if shell_compatibility else "defer",
            reason="task_execution_authorization_required",
            execute=False,
            result_payload=json.dumps(
                {
                    "ok": False,
                    "error_code": "task_execution_authorization_required",
                    "fallback_allowed": False,
                },
                ensure_ascii=False,
            ),
            metadata=dict(metadata),
        )
        self._record_decision(context, tool_name, decision, arguments)
        return decision

    def _access_block(
        self,
        context: ToolBoundaryContext,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        error_code: str = "",
        message: str = "",
        recommended_tools: tuple[str, ...] = (),
        reason: str | None = None,
    ) -> BoundaryExecutionDecision:
        return self._boundary_block(
            context,
            tool_name,
            arguments,
            reason or error_code or "tool_access_block",
            {"recommended_tools": list(recommended_tools)},
            error_code=error_code,
            message=message,
            recommended_tools=recommended_tools,
        )

    def _boundary_block(
        self,
        context: ToolBoundaryContext,
        tool_name: str,
        arguments: Mapping[str, Any],
        reason: str,
        metadata: Mapping[str, object],
        *,
        error_code: str | None = None,
        message: str = "",
        recommended_tools: tuple[str, ...] = (),
    ) -> BoundaryExecutionDecision:
        decision = BoundaryExecutionDecision(
            action="block",
            reason=reason,
            execute=False,
            result_payload=json.dumps(
                {
                    "ok": False,
                    "error_code": error_code or reason,
                    "message": message,
                    "recommended_tools": list(recommended_tools),
                    "fallback_allowed": False,
                },
                ensure_ascii=False,
            ),
            metadata=dict(metadata),
        )
        self._record_decision(context, tool_name, decision, arguments)
        return decision

    def _record_decision(
        self,
        context: ToolBoundaryContext,
        tool_name: str,
        decision: BoundaryExecutionDecision,
        arguments: Mapping[str, Any],
    ) -> None:
        context.decisions.append(
            {
                "tool": tool_name,
                "action": decision.action,
                "reason": decision.reason,
                "execute": decision.execute,
                "arguments": _decision_arguments(tool_name, arguments),
                "args_summary": summarize_args(arguments),
                "metadata": dict(decision.metadata),
            }
        )


def _soft_stop_payload(decision: ToolBoundaryDecision) -> str:
    return json.dumps(
        {
            "ok": False,
            "error_code": "tool_boundary_soft_stop",
            "terminal_scope": "current_turn_tool_budget",
            "fallback_allowed": False,
            "recommended_action": "answer_from_existing_evidence",
            "message": decision.model_hint
            or "Current turn already has enough evidence; answer from existing evidence.",
            "reason": decision.reason,
        },
        ensure_ascii=False,
    )


def _decision_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, object]:
    if tool_name == "fetch_doc_chunk":
        chunk_id = arguments.get("chunk_id")
        return {"chunk_id": chunk_id} if isinstance(chunk_id, str) else {}
    if tool_name == "search_docs":
        query = arguments.get("query")
        return {"query": query} if isinstance(query, str) else {}
    if tool_name == "tool_search":
        query = arguments.get("query")
        return {"query": query} if isinstance(query, str) else {}
    return {}


_ACTION_RANK = {
    "allow": 0,
    "warn": 1,
    "require_reason": 2,
    "soft_stop": 3,
    "block": 4,
}


def _more_restrictive(
    left: ToolBoundaryDecision,
    right: ToolBoundaryDecision,
) -> ToolBoundaryDecision:
    if _ACTION_RANK[right.action] > _ACTION_RANK[left.action]:
        return right
    return left
