from __future__ import annotations

from collections.abc import Mapping

from agent.policies.task_execution_boundary import TaskExecutionRiskPolicy
from agent.policies.task_execution_budget import (
    TaskExecutionBudgetPolicy,
    TaskExecutionEventClassifier,
)
from agent.policies.task_execution_contract import TaskExecutionTurnContract
from agent.policies.tool_ledger import ToolCallLedger, ToolCallRecord, stable_args_hash


def _work_contract(*, work_call_budget: int = 3) -> TaskExecutionTurnContract:
    return TaskExecutionTurnContract(
        active=True,
        action="continue",
        phase="work",
        attempt_id="attempt-1",
        target_step_id="step-1",
        required_capabilities=frozenset({"task_execution.finish"}),
        allowed_capabilities=frozenset(
            {
                "task_execution.finish",
                "task_execution.defer",
                "task_execution.abort",
            }
        ),
        allowed_risks=frozenset({"read-only"}),
        work_call_budget=work_call_budget,
        tool_search_budget=1,
        completion_capability="task_execution.finish",
        reason="attempt_running",
        matched_terms=(),
    )


def _work_record(name: str, arguments: Mapping[str, object], index: int) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=name,
        tool_class="local_file",
        args_hash=stable_args_hash(arguments),
        args_summary="{}",
        call_index=index,
        visible_before_call=True,
        execution_status="success",
        result_ok=True,
        tool_risk="read-only",
        invoker_reached=True,
        invoker_succeeded=True,
        counts_as_work=True,
    )


def test_risk_policy_defers_shell_even_when_metadata_says_read_only() -> None:
    decision = TaskExecutionRiskPolicy().evaluate(
        contract=_work_contract(),
        tool_name="shell",
        registered=True,
        registry_risk="read-only",
    )

    assert decision is not None
    assert decision.action == "defer"
    assert decision.reason == "task_execution_authorization_required"
    assert decision.metadata["durable_transition"] == "waiting_authorization"


def test_risk_policy_denies_destructive_before_visibility() -> None:
    decision = TaskExecutionRiskPolicy().evaluate(
        contract=_work_contract(),
        tool_name="delete_workspace",
        registered=True,
        registry_risk="destructive",
    )

    assert decision is not None
    assert decision.action == "deny"
    assert decision.reason == "task_execution_destructive_denied"


def test_budget_skips_work_beyond_remaining_budget_with_failed_transition() -> None:
    ledger = ToolCallLedger(
        records=[
            _work_record("read_file", {"path": "a"}, 1),
            _work_record("read_file", {"path": "b"}, 2),
            _work_record("list_dir", {"path": "."}, 3),
        ]
    )

    decision = TaskExecutionBudgetPolicy().evaluate(
        contract=_work_contract(),
        ledger=ledger,
        tool_name="read_file",
        arguments={"path": "c"},
        tool_risk="read-only",
        tool_capabilities={},
    )

    assert decision is not None
    assert decision.action == "soft_stop"
    assert decision.reason == "task_execution_batch_budget_skip"
    assert decision.metadata["terminal_transition"] == "failed"


def test_repeated_read_only_work_stops_with_failed_transition() -> None:
    arguments = {"path": "README.md"}
    ledger = ToolCallLedger(records=[_work_record("read_file", arguments, 1)])

    decision = TaskExecutionBudgetPolicy().evaluate(
        contract=_work_contract(),
        ledger=ledger,
        tool_name="read_file",
        arguments=arguments,
        tool_risk="read-only",
        tool_capabilities={},
    )

    assert decision is not None
    assert decision.reason == "task_execution_repeated_work_call"
    assert decision.metadata["terminal_transition"] == "failed"


def test_tool_search_attempt_is_budgeted_but_never_counts_as_work() -> None:
    classifier = TaskExecutionEventClassifier()
    event = classifier.classify(
        tool_name="tool_search",
        tool_call_id="call-search",
        registry_risk="read-only",
        invoker_reached=True,
        invoker_succeeded=True,
        execution_status="success",
        result_ok=True,
    )
    ledger = ToolCallLedger(
        records=[
            ToolCallRecord(
                tool_name="tool_search",
                tool_class="discovery",
                args_hash="search",
                args_summary="{}",
                call_index=1,
                visible_before_call=True,
            )
        ]
    )

    decision = TaskExecutionBudgetPolicy().evaluate(
        contract=_work_contract(),
        ledger=ledger,
        tool_name="tool_search",
        arguments={"query": "read files"},
        tool_risk="read-only",
        tool_capabilities={},
    )

    assert event.counts_as_work is False
    assert decision is not None
    assert decision.reason == "task_execution_tool_search_budget_exhausted"
    assert decision.metadata["terminal_transition"] == "failed"
