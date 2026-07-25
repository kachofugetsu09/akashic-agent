from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from agent.config_models import TaskExecutionConfig
from agent.task_plan.execution_models import RuntimeToolEvent
from agent.task_plan.execution_service import (
    TaskExecutionConflictError,
    TaskExecutionService,
)
from agent.task_plan.orchestrator import TaskExecutionOrchestrator
from agent.task_plan.service import TaskPlanConflictError, TaskPlanService
from agent.task_plan.store import TaskPlanStore


@pytest.fixture
def execution_service(tmp_path: Path) -> TaskExecutionService:
    store = TaskPlanStore(tmp_path / "task_execution.db")
    return TaskExecutionService(
        store=store,
        plan_service=TaskPlanService(store),
        runtime_instance_id="runtime-1",
        config=TaskExecutionConfig(lease_seconds=60),
        clock=lambda: datetime(2030, 7, 15, tzinfo=UTC),
    )


def _record_successful_work(
    service: TaskExecutionService, *, attempt_id: str
) -> None:
    service.record_tool_event(
        session_key="cli:s1",
        attempt_id=attempt_id,
        event=RuntimeToolEvent(
            event_type="tool_finished",
            tool_name="read_file",
            tool_call_id="call-1",
            source_turn_id=10,
            tool_risk="read-only",
            tool_capabilities=("filesystem:read",),
            counts_as_work=True,
            invoker_reached=True,
            invoker_succeeded=True,
            execution_status="success",
            result_ok=True,
            error_code="",
            arguments_hash="hash-1",
            result_preview="README",
        ),
    )


def _fail_first_step(service: TaskExecutionService) -> str:
    claimed = service.begin_next_step(session_key="cli:s1", request_id="req-fail")
    service.start_attempt(session_key="cli:s1", attempt_id=claimed.attempt.attempt_id)
    service.finish_attempt(
        session_key="cli:s1",
        attempt_id=claimed.attempt.attempt_id,
        success=False,
        result_summary="read failed",
        error_code="read_error",
    )
    return claimed.attempt.attempt_id


def test_continue_selects_only_lowest_pending_step(
    execution_service: TaskExecutionService,
) -> None:
    plan = execution_service.plan_service.create_task_plan(
        session_key="cli:s1",
        title="Two steps",
        steps=["Read README", "Summarize tests"],
    )

    result = execution_service.begin_next_step(session_key="cli:s1", request_id="req-1")

    assert result.attempt.step_id == plan.steps[0].step_id
    assert result.step.index == 1
    assert result.replayed is False


def test_continue_does_not_skip_failed_step(
    execution_service: TaskExecutionService,
) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="Two steps", steps=["Read", "Summarize"]
    )
    _fail_first_step(execution_service)

    with pytest.raises(TaskExecutionConflictError, match="explicit retry"):
        execution_service.begin_next_step(session_key="cli:s1", request_id="req-3")


def test_explicit_skip_allows_continue_after_failed_step(
    execution_service: TaskExecutionService,
) -> None:
    plan = execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="Two steps", steps=["Read", "Summarize"]
    )
    _fail_first_step(execution_service)
    execution_service.plan_service.update_step_status(
        session_key="cli:s1",
        task_id=plan.task_id,
        step_id=plan.steps[0].step_id,
        status="skipped",
        result_summary="user explicitly skipped failed step",
    )

    result = execution_service.begin_next_step(
        session_key="cli:s1", request_id="req-after-skip"
    )

    assert result.step.index == 2


def test_replay_is_returned_after_final_step_completed(
    execution_service: TaskExecutionService,
) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    original = execution_service.begin_next_step(
        session_key="cli:s1", request_id="req-final"
    )
    execution_service.start_attempt(session_key="cli:s1", attempt_id=original.attempt.attempt_id)
    _record_successful_work(execution_service, attempt_id=original.attempt.attempt_id)
    execution_service.finish_attempt(
        session_key="cli:s1",
        attempt_id=original.attempt.attempt_id,
        success=True,
        result_summary="Read README",
    )

    replay = execution_service.begin_next_step(session_key="cli:s1", request_id="req-final")

    assert replay.replayed is True
    assert replay.attempt.attempt_id == original.attempt.attempt_id
    assert execution_service.plan_service.get_active_task_plan(session_key="cli:s1") is None


@pytest.mark.parametrize("terminal", ["failed", "blocked"])
def test_replay_is_returned_after_terminal_attempt(
    execution_service: TaskExecutionService, terminal: str
) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    original = execution_service.begin_next_step(
        session_key="cli:s1", request_id="req-terminal"
    )
    if terminal == "failed":
        execution_service.start_attempt(
            session_key="cli:s1", attempt_id=original.attempt.attempt_id
        )
        execution_service.finish_attempt(
            session_key="cli:s1",
            attempt_id=original.attempt.attempt_id,
            success=False,
            result_summary="read failed",
            error_code="read_error",
        )
    else:
        execution_service.block_attempt(
            session_key="cli:s1",
            attempt_id=original.attempt.attempt_id,
            terminal_reason="turn_interrupted_outcome_unknown",
        )

    replay = execution_service.replay_request(
        session_key="cli:s1", request_id="req-terminal"
    )

    assert replay is not None
    assert replay.replayed is True
    assert replay.attempt.attempt_id == original.attempt.attempt_id


def test_retry_creates_next_attempt_without_separate_step_update(
    execution_service: TaskExecutionService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    first_id = _fail_first_step(execution_service)
    monkeypatch.setattr(
        execution_service.plan_service,
        "update_step_status",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("retry must not commit a separate step reset")
        ),
    )

    retried = execution_service.retry_step(
        session_key="cli:s1",
        step_id=plan.steps[0].step_id,
        request_id="req-retry",
    )

    assert retried.attempt.attempt_no == 2
    assert retried.attempt.status == "pending"
    assert [event.event_type for event in execution_service.inspect(
        session_key="cli:s1", attempt_id=first_id
    ).events] == ["attempt_claimed", "attempt_started", "attempt_failed"]


def test_find_retryable_step_selects_lowest_failed_or_recovery_blocked_step(
    execution_service: TaskExecutionService,
) -> None:
    plan = execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="Two steps", steps=["Read", "Summarize"]
    )
    _fail_first_step(execution_service)

    retryable = execution_service.find_retryable_step(session_key="cli:s1")

    assert retryable is not None
    assert retryable.step_id == plan.steps[0].step_id


def test_retry_accepts_approved_recovery_block_and_creates_next_attempt(
    execution_service: TaskExecutionService,
) -> None:
    plan = execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    first = execution_service.begin_next_step(
        session_key="cli:s1", request_id="req-blocked"
    )
    execution_service.block_attempt(
        session_key="cli:s1",
        attempt_id=first.attempt.attempt_id,
        terminal_reason="lease_expired_outcome_unknown",
        error_code="lease_expired",
    )

    retry = execution_service.retry_step(
        session_key="cli:s1",
        step_id=plan.steps[0].step_id,
        request_id="req-blocked-retry",
    )

    assert retry.attempt.attempt_no == 2
    assert retry.attempt.status == "pending"


def test_continue_does_not_skip_recovery_blocked_step_for_later_pending_step(
    execution_service: TaskExecutionService,
) -> None:
    plan = execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="Two steps", steps=["Read", "Summarize"]
    )
    first = execution_service.begin_next_step(
        session_key="cli:s1", request_id="req-blocked-first"
    )
    execution_service.block_attempt(
        session_key="cli:s1",
        attempt_id=first.attempt.attempt_id,
        terminal_reason="runtime_restarted_outcome_unknown",
    )

    with pytest.raises(
        TaskExecutionConflictError, match="blocked_step_requires_explicit_retry"
    ):
        execution_service.begin_next_step(
            session_key="cli:s1", request_id="req-ordinary-continue"
        )

    attempts = execution_service.store.list_execution_attempts(plan.task_id)
    assert [attempt.step_id for attempt in attempts] == [plan.steps[0].step_id]


def test_finish_success_requires_successful_work_event(
    execution_service: TaskExecutionService,
) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    claimed = execution_service.begin_next_step(session_key="cli:s1", request_id="req-1")
    execution_service.start_attempt(session_key="cli:s1", attempt_id=claimed.attempt.attempt_id)

    with pytest.raises(TaskExecutionConflictError, match="work event"):
        execution_service.finish_attempt(
            session_key="cli:s1",
            attempt_id=claimed.attempt.attempt_id,
            success=True,
            result_summary="done",
        )


def test_search_only_event_cannot_finish(execution_service: TaskExecutionService) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="Search", steps=["Find a tool"]
    )
    claimed = execution_service.begin_next_step(
        session_key="cli:s1", request_id="req-search"
    )
    execution_service.start_attempt(session_key="cli:s1", attempt_id=claimed.attempt.attempt_id)
    execution_service.record_tool_event(
        session_key="cli:s1",
        attempt_id=claimed.attempt.attempt_id,
        event=RuntimeToolEvent(
            event_type="tool_finished",
            tool_name="tool_search",
            tool_call_id="call-search",
            source_turn_id=11,
            tool_risk="read-only",
            tool_capabilities=(),
            counts_as_work=False,
            invoker_reached=True,
            invoker_succeeded=True,
            execution_status="success",
            result_ok=True,
            error_code="",
            arguments_hash="hash-search",
            result_preview="read_file",
        ),
    )

    with pytest.raises(TaskExecutionConflictError, match="work event"):
        execution_service.finish_attempt(
            session_key="cli:s1",
            attempt_id=claimed.attempt.attempt_id,
            success=True,
            result_summary="searched only",
        )


def test_record_event_rejects_untrusted_mapping(execution_service: TaskExecutionService) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    claimed = execution_service.begin_next_step(session_key="cli:s1", request_id="req-map")
    execution_service.start_attempt(session_key="cli:s1", attempt_id=claimed.attempt.attempt_id)

    with pytest.raises(TypeError, match="RuntimeToolEvent"):
        execution_service.record_tool_event(  # type: ignore[arg-type]
            session_key="cli:s1",
            attempt_id=claimed.attempt.attempt_id,
            event=cast(Any, {"event_type": "tool_finished"}),
        )


def test_defer_redacts_raw_arguments_before_persisting_reason(
    execution_service: TaskExecutionService,
) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    claimed = execution_service.begin_next_step(session_key="cli:s1", request_id="req-defer")

    deferred = execution_service.defer_attempt(
        session_key="cli:s1",
        attempt_id=claimed.attempt.attempt_id,
        tool_name="send_message",
        requested_arguments={"token": "super-secret", "recipient": "ops"},
        requested_capabilities=("network:send",),
        reason="authorization required",
    )

    assert deferred.attempt is not None
    assert deferred.attempt.status == "waiting_authorization"
    assert "super-secret" not in deferred.attempt.terminal_reason
    assert "arguments_hash=" in deferred.attempt.terminal_reason


def test_runtime_block_resets_step_without_marking_failed(
    execution_service: TaskExecutionService,
) -> None:
    plan = execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="Interrupted", steps=["Read README"]
    )
    claimed = execution_service.begin_next_step(session_key="cli:s1", request_id="req-block")
    execution_service.start_attempt(session_key="cli:s1", attempt_id=claimed.attempt.attempt_id)

    snapshot = execution_service.block_attempt(
        session_key="cli:s1",
        attempt_id=claimed.attempt.attempt_id,
        terminal_reason="turn_interrupted_outcome_unknown",
    )
    active = execution_service.plan_service.get_active_task_plan(session_key="cli:s1")

    assert snapshot.attempt is not None
    assert snapshot.attempt.status == "blocked"
    assert active is not None
    assert active.steps[0].status == "pending"
    assert plan.task_id == active.task_id


def test_runtime_block_persists_bounded_error_code_on_attempt_and_event(
    execution_service: TaskExecutionService,
) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="Interrupted", steps=["Read README"]
    )
    claimed = execution_service.begin_next_step(
        session_key="cli:s1", request_id="req-block-error"
    )
    error_code = "interrupted_" + "x" * 200

    snapshot = execution_service.block_attempt(
        session_key="cli:s1",
        attempt_id=claimed.attempt.attempt_id,
        terminal_reason="turn_interrupted_outcome_unknown",
        error_code=error_code,
    )

    assert snapshot.attempt is not None
    assert snapshot.attempt.error_code == error_code[:125] + "..."
    assert snapshot.events[-1].event_type == "attempt_blocked"
    assert snapshot.events[-1].error_code == error_code[:125] + "..."


def test_active_attempt_rejects_manual_update_completion_cancellation_and_replacement(
    execution_service: TaskExecutionService,
) -> None:
    plan = execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    execution_service.begin_next_step(session_key="cli:s1", request_id="req-active")

    with pytest.raises(TaskPlanConflictError, match="active execution attempt"):
        execution_service.plan_service.update_step_status(
            session_key="cli:s1",
            task_id=plan.task_id,
            step_id=plan.steps[0].step_id,
            status="completed",
        )
    with pytest.raises(TaskPlanConflictError, match="active execution attempt"):
        execution_service.plan_service.complete_task_plan(
            session_key="cli:s1", task_id=plan.task_id
        )
    with pytest.raises(TaskPlanConflictError, match="active execution attempt"):
        execution_service.plan_service.cancel_task_plan(
            session_key="cli:s1", task_id=plan.task_id
        )
    with pytest.raises(TaskPlanConflictError, match="active execution attempt"):
        execution_service.plan_service.create_task_plan(
            session_key="cli:s1",
            title="Replacement",
            steps=["Other"],
            replace_active=True,
        )


def test_start_and_event_mutations_reject_expired_or_foreign_owner(
    tmp_path: Path,
) -> None:
    now = datetime(2030, 7, 15, tzinfo=UTC)
    store = TaskPlanStore(tmp_path / "cas.db")
    plan_service = TaskPlanService(store)
    plan = plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    foreign = store.claim_execution_attempt(
        task_id=plan.task_id,
        step_id=plan.steps[0].step_id,
        session_key="cli:s1",
        request_id="req-foreign",
        idempotency_key="foreign-idempotency",
        owner_instance_id="runtime-other",
        lease_expires_at=(now + timedelta(minutes=1)).isoformat(),
    )
    service = TaskExecutionService(
        store=store,
        plan_service=plan_service,
        runtime_instance_id="runtime-1",
        config=TaskExecutionConfig(lease_seconds=60),
        clock=lambda: now,
    )

    with pytest.raises(TaskExecutionConflictError, match="start"):
        service.start_attempt(session_key="cli:s1", attempt_id=foreign.attempt.attempt_id)

    expired_store = TaskPlanStore(tmp_path / "expired.db")
    expired_service = TaskExecutionService(
        store=expired_store,
        plan_service=TaskPlanService(expired_store),
        runtime_instance_id="runtime-1",
        config=TaskExecutionConfig(lease_seconds=60),
        clock=lambda: now,
    )
    expired_service.plan_service.create_task_plan(
        session_key="cli:s1", title="Expired", steps=["Read README"]
    )
    expired = expired_service.begin_next_step(session_key="cli:s1", request_id="req-expired")
    expired_service._now = lambda: now + timedelta(minutes=2)  # type: ignore[method-assign]

    with pytest.raises(TaskExecutionConflictError, match="start"):
        expired_service.start_attempt(
            session_key="cli:s1", attempt_id=expired.attempt.attempt_id
        )


def test_final_success_completes_task_and_orchestrator_reports_claim_then_replay(
    execution_service: TaskExecutionService,
) -> None:
    execution_service.plan_service.create_task_plan(
        session_key="cli:s1", title="One step", steps=["Read README"]
    )
    orchestrator = TaskExecutionOrchestrator(execution_service)
    claimed = orchestrator.decide_continue(session_key="cli:s1", request_id="req-orch")
    assert claimed.action == "claimed"
    assert claimed.reason == "task_execution_step_claimed"
    assert claimed.snapshot.attempt is not None

    attempt_id = claimed.snapshot.attempt.attempt_id
    execution_service.start_attempt(session_key="cli:s1", attempt_id=attempt_id)
    _record_successful_work(execution_service, attempt_id=attempt_id)
    execution_service.finish_attempt(
        session_key="cli:s1",
        attempt_id=attempt_id,
        success=True,
        result_summary="Read README",
    )
    replay = orchestrator.decide_continue(session_key="cli:s1", request_id="req-orch")

    assert replay.action == "replayed"
    assert replay.reason == "task_execution_request_replayed"
    assert execution_service.plan_service.get_active_task_plan(session_key="cli:s1") is None
