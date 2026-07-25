from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

from agent.config_models import TaskExecutionConfig
from agent.policies.task_control_arbiter import TaskControlIntentArbiter
from agent.policies.task_execution_budget import TaskExecutionEventClassifier
from agent.policies.task_execution_contract import (
    TaskExecutionTurnContract,
    detect_task_execution_intent,
    infer_task_execution_contract,
)
from agent.policies.task_plan_contract import infer_task_plan_turn_decision
from agent.policies.tool_boundary import BoundaryExecutionDecision
from agent.policies.tool_approval import canonical_args_hash, summarize_arguments
from agent.policies.tool_approval_runtime import ToolApprovalRuntime
from agent.task_plan.execution_models import (
    RuntimeToolEvent,
    TaskExecutionAttempt,
    TaskExecutionSnapshot,
    TERMINAL_ATTEMPT_STATUSES,
)
from agent.task_plan.execution_redaction import (
    bounded_execution_preview,
    hash_execution_arguments,
    redact_execution_arguments,
)
from agent.task_plan.execution_service import (
    TaskExecutionConflictError,
    TaskExecutionService,
)
from agent.task_plan.request_identity import ensure_task_execution_request_id
from agent.tools.base import ToolResult, normalize_tool_result
from agent.tools.execution_context import ToolExecutionContext
from agent.tool_hooks.types import ToolExecutionResult

logger = logging.getLogger(__name__)

_CONTROL_CAPABILITIES = frozenset(
    {
        "task_execution.begin",
        "task_execution.inspect",
        "task_execution.finish",
        "task_execution.defer",
        "task_execution.abort",
    }
)


class TaskExecutionLeaseLostError(RuntimeError):
    pass


class TaskExecutionPersistenceError(RuntimeError):
    pass


@dataclass
class PreparedTaskExecutionTurn:
    session_key: str
    request_id: str
    user_text: str
    source_turn_id: int | None
    turn_metadata: dict[str, object]
    contract: TaskExecutionTurnContract
    snapshot: TaskExecutionSnapshot = field(
        default_factory=lambda: TaskExecutionSnapshot(attempt=None)
    )
    request_replayed: bool = False
    request_claimed_attempt: bool = False
    request_owned_attempt_id: str | None = None
    decision_reason: str = "no_task_execution_intent"
    protocol_corrections: int = 0
    execution_tool_names: set[str] = field(default_factory=set)
    finalized: bool = False
    lease_lost: bool = False

    @property
    def attempt_id(self) -> str | None:
        attempt = self.snapshot.attempt
        return attempt.attempt_id if attempt is not None else self.contract.attempt_id

    @property
    def has_claimed_attempt(self) -> bool:
        return self.owns_current_attempt

    @property
    def owns_current_attempt(self) -> bool:
        attempt = self.snapshot.attempt
        return bool(
            self.request_claimed_attempt
            and attempt is not None
            and self.request_owned_attempt_id == attempt.attempt_id
            and attempt.session_key == self.session_key
            and attempt.request_id == self.request_id
        )

    @property
    def attempt_is_active(self) -> bool:
        attempt = self.snapshot.attempt
        return bool(
            self.owns_current_attempt
            and attempt is not None
            and attempt.status in {"pending", "running"}
        )


@dataclass(frozen=True)
class RuntimeCallDecision:
    execute: bool
    execution_context: ToolExecutionContext | None = None
    result_payload: str = ""
    status: str = ""
    reason: str = ""
    final_only: bool = False


@dataclass(frozen=True)
class RuntimeFinalDecision:
    action: Literal["accept", "correct", "failed"]
    model_hint: str = ""


class TaskExecutionLeaseGuard:
    def __init__(
        self,
        service: object,
        *,
        session_key: str,
        attempt_id: str,
        lease_seconds: float,
        clock: Callable[[], datetime],
    ) -> None:
        self._service = service
        self._session_key = session_key
        self._attempt_id = attempt_id
        self._interval = lease_seconds / 3
        self._clock = clock
        self._task: asyncio.Task[None] | None = None
        self._conflict: BaseException | None = None

    def renew_now(self) -> object:
        try:
            return self._service.renew_attempt_lease(
                session_key=self._session_key,
                attempt_id=self._attempt_id,
            )
        except Exception as exc:
            self._conflict = exc
            raise TaskExecutionLeaseLostError(
                "task execution lease renewal failed"
            ) from exc

    async def __aenter__(self) -> TaskExecutionLeaseGuard:
        self._task = asyncio.create_task(self._renew_loop())
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        renewal_error: TaskExecutionLeaseLostError | None = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except TaskExecutionLeaseLostError as task_error:
                renewal_error = task_error
        if exc is not None:
            return None
        if renewal_error is not None:
            raise renewal_error
        if exc is None:
            self.renew_now()

    async def _renew_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            self.renew_now()


class TaskExecutionRuntimeCoordinator:
    def __init__(
        self,
        service: TaskExecutionService | None,
        *,
        config: TaskExecutionConfig,
        runtime_instance_id: str,
        clock: Callable[[], datetime],
        approval_runtime: ToolApprovalRuntime | None = None,
    ) -> None:
        self._service = service
        self._config = config
        self._runtime_instance_id = runtime_instance_id
        self._clock = clock
        self._approval_runtime = approval_runtime
        self._event_classifier = TaskExecutionEventClassifier()
        self.protocol_correction_count = 0

    def set_approval_runtime(
        self,
        approval_runtime: ToolApprovalRuntime | None,
    ) -> None:
        self._approval_runtime = approval_runtime

    def prepare_turn(
        self,
        *,
        msg: object,
        session_key: str,
    ) -> PreparedTaskExecutionTurn:
        request_id = ensure_task_execution_request_id(msg)  # type: ignore[arg-type]
        user_text = str(getattr(msg, "content", "") or "")
        raw_metadata = getattr(msg, "metadata", {}) or {}
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        source_turn_id = _optional_int(metadata.get("source_turn_id"))
        self.protocol_correction_count = 0
        requested = detect_task_execution_intent(user_text)
        metadata.update(
            {
                "task_execution_enabled": bool(self._config.enabled and self._service),
                "task_execution_provider_available": self._service is not None,
                "task_execution_requested": requested,
                "task_execution_runtime_instance_id": self._runtime_instance_id,
            }
        )
        inactive = TaskExecutionTurnContract.inactive()
        if not self._config.enabled:
            return PreparedTaskExecutionTurn(
                session_key=session_key,
                request_id=request_id,
                user_text=user_text,
                source_turn_id=source_turn_id,
                turn_metadata=metadata,
                contract=inactive,
                decision_reason="task_execution_disabled",
            )
        if self._service is None:
            return PreparedTaskExecutionTurn(
                session_key=session_key,
                request_id=request_id,
                user_text=user_text,
                source_turn_id=source_turn_id,
                turn_metadata=metadata,
                contract=inactive,
                decision_reason="task_execution_provider_unavailable",
            )

        replay = self._service.replay_request(
            session_key=session_key,
            request_id=request_id,
        )
        if replay is not None:
            snapshot = self._service.inspect(
                session_key=session_key,
                attempt_id=replay.attempt.attempt_id,
            )
            metadata.update(
                {
                    "request_replay_attempt_id": replay.attempt.attempt_id,
                    "has_active_task": False,
                    "active_task_execution_snapshot": snapshot,
                    "latest_retryable_step_id": None,
                }
            )
            contract = infer_task_execution_contract(user_text, metadata)
            metadata["task_execution_contract"] = contract
            return PreparedTaskExecutionTurn(
                session_key=session_key,
                request_id=request_id,
                user_text=user_text,
                source_turn_id=source_turn_id,
                turn_metadata=metadata,
                contract=contract,
                snapshot=snapshot,
                request_replayed=True,
                request_claimed_attempt=True,
                request_owned_attempt_id=replay.attempt.attempt_id,
                decision_reason="task_execution_request_replayed",
                finalized=True,
            )

        self._service.reconcile_attempts(now=self._clock(), session_key=session_key)
        plan = self._service.plan_service.get_active_task_plan(session_key=session_key)
        snapshot = self._service.inspect(session_key=session_key)
        retryable = (
            self._service.find_retryable_step(session_key=session_key)
            if plan is not None
            else None
        )
        metadata.update(
            {
                "has_active_task": plan is not None,
                "active_task_execution_snapshot": snapshot,
                "latest_retryable_step_id": (
                    retryable.step_id if retryable is not None else None
                ),
            }
        )
        plan_decision = infer_task_plan_turn_decision(
            user_text,
            has_active_task=plan is not None,
        )
        execution_contract = infer_task_execution_contract(user_text, metadata)
        if snapshot.attempt is not None and execution_contract.active:
            execution_contract = replace(
                execution_contract,
                attempt_id=snapshot.attempt.attempt_id,
            )
        control = TaskControlIntentArbiter().resolve(
            task_plan_contract=plan_decision.contract,
            task_execution_contract=execution_contract,
            user_text=user_text,
            metadata=metadata,
        )
        metadata["task_plan_contract"] = control.task_plan_contract
        metadata["task_execution_contract"] = control.task_execution_contract
        return PreparedTaskExecutionTurn(
            session_key=session_key,
            request_id=request_id,
            user_text=user_text,
            source_turn_id=source_turn_id,
            turn_metadata=metadata,
            contract=control.task_execution_contract,
            snapshot=snapshot,
            decision_reason=control.task_execution_contract.reason,
        )

    async def before_tool_call(
        self,
        turn: PreparedTaskExecutionTurn,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        tool_capabilities: frozenset[str],
        boundary_decision: BoundaryExecutionDecision | None,
    ) -> RuntimeCallDecision:
        if not turn.contract.active:
            return RuntimeCallDecision(execute=True)
        session_control = _is_unowned_session_control(tool_capabilities)
        if not session_control and not turn.owns_current_attempt:
            return RuntimeCallDecision(
                execute=False,
                result_payload=_error_payload("task_execution_request_not_owner"),
                status="blocked_by_task_execution_ownership",
                reason="task_execution_request_not_owner",
                final_only=True,
            )
        waiting_control_allowed = bool(
            turn.snapshot.attempt is not None
            and turn.snapshot.attempt.status == "waiting_authorization"
            and tool_capabilities & {"task_execution.inspect", "task_execution.abort"}
        )
        if (
            turn.snapshot.attempt is not None
            and not waiting_control_allowed
            and (
                turn.snapshot.attempt.status in TERMINAL_ATTEMPT_STATUSES
                or turn.snapshot.attempt.status == "waiting_authorization"
            )
        ):
            return RuntimeCallDecision(
                execute=False,
                result_payload=_error_payload("task_execution_batch_terminal_skip"),
                status="skipped_after_task_execution_terminal",
                reason="task_execution_batch_terminal_skip",
                final_only=True,
            )
        if (
            boundary_decision is not None
            and boundary_decision.reason == "task_execution_authorization_required"
        ):
            await self._persist_defer(
                turn,
                tool_name=tool_name,
                arguments=arguments,
                capabilities=tool_capabilities,
            )
            if turn.decision_reason == "defer_persistence_failed":
                return RuntimeCallDecision(
                    execute=False,
                    result_payload=_error_payload("defer_persistence_failed"),
                    status="blocked_by_task_execution_persistence",
                    reason="defer_persistence_failed",
                    final_only=True,
                )
            return RuntimeCallDecision(
                execute=False,
                result_payload=(boundary_decision.result_payload or ""),
                status="deferred_by_task_execution",
                reason=turn.decision_reason,
                final_only=True,
            )
        if boundary_decision is not None and not boundary_decision.execute:
            if boundary_decision.metadata.get("terminal_transition") == "failed":
                self._fail_attempt(turn, boundary_decision.reason)
            return RuntimeCallDecision(
                execute=False,
                result_payload=boundary_decision.result_payload or "",
                status=(
                    "blocked_by_tool_boundary"
                    if boundary_decision.action == "block"
                    else "soft_stopped_by_tool_boundary"
                ),
                reason=boundary_decision.reason,
                final_only=turn.finalized,
            )
        return RuntimeCallDecision(
            execute=True,
            execution_context=self._execution_context(
                turn,
                tool_capabilities=tool_capabilities,
            ),
        )

    async def after_tool_call(
        self,
        turn: PreparedTaskExecutionTurn,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        result: str | ToolResult,
        execution_result: ToolExecutionResult,
        registry_risk: str,
        registry_capabilities: frozenset[str],
    ) -> None:
        if not turn.contract.active:
            return
        turn.execution_tool_names.add(tool_name)
        if "task_execution.begin" in registry_capabilities:
            normalized = normalize_tool_result(result)
            result_ok, _ = _result_facts(normalized)
            begin_action = _begin_decision_action(normalized)
            if not (
                execution_result.status == "success"
                and execution_result.invoker_reached
                and execution_result.invoker_succeeded
                and result_ok
                and begin_action in {"claimed", "replayed"}
            ):
                turn.decision_reason = "task_execution_begin_not_committed"
                return
            replay = self._require_service().replay_request(
                session_key=turn.session_key,
                request_id=turn.request_id,
            )
            if replay is None:
                turn.decision_reason = "task_execution_begin_state_missing"
                return
            snapshot = self._require_service().inspect(
                session_key=turn.session_key,
                attempt_id=replay.attempt.attempt_id,
            )
            if snapshot.attempt is None or not _attempt_matches_turn(
                turn, snapshot.attempt
            ):
                turn.decision_reason = "task_execution_begin_state_mismatch"
                return
            if snapshot.attempt is not None and snapshot.attempt.status == "pending":
                snapshot = self._require_service().start_attempt(
                    session_key=turn.session_key,
                    attempt_id=snapshot.attempt.attempt_id,
                )
            turn.snapshot = snapshot
            turn.request_replayed = begin_action == "replayed"
            turn.request_claimed_attempt = True
            turn.request_owned_attempt_id = snapshot.attempt.attempt_id
            turn.finalized = False
            turn.decision_reason = (
                "task_execution_request_replayed"
                if begin_action == "replayed"
                else (
                    "task_execution_step_retry_claimed"
                    if turn.contract.action == "retry"
                    else "task_execution_step_claimed"
                )
            )
            turn.contract = self._contract_for_snapshot(turn)
            return

        if registry_capabilities & _CONTROL_CAPABILITIES:
            attempt_id = turn.attempt_id
            if attempt_id:
                turn.snapshot = self._require_service().inspect(
                    session_key=turn.session_key,
                    attempt_id=attempt_id,
                )
                turn.contract = self._contract_for_snapshot(turn)
                attempt = turn.snapshot.attempt
                if (
                    "task_execution.inspect" in registry_capabilities
                    and not turn.owns_current_attempt
                ):
                    turn.contract = _inspect_final_contract(
                        attempt_id,
                        matched_terms=turn.contract.matched_terms,
                    )
                    turn.finalized = True
                    turn.decision_reason = "task_execution_inspected"
                    return
                if attempt is not None and (
                    attempt.status in TERMINAL_ATTEMPT_STATUSES
                    or attempt.status == "waiting_authorization"
                ):
                    turn.finalized = True
                    turn.decision_reason = _attempt_reason(attempt)
            return

        attempt_id = turn.attempt_id
        if (
            attempt_id is None
            or turn.contract.phase != "work"
            or not turn.owns_current_attempt
        ):
            return
        normalized = normalize_tool_result(result)
        result_ok, error_code = _result_facts(normalized)
        facts = self._event_classifier.classify(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            registry_risk=registry_risk,
            invoker_reached=execution_result.invoker_reached,
            invoker_succeeded=execution_result.invoker_succeeded,
            execution_status=execution_result.status,
            result_ok=result_ok,
        )
        event = RuntimeToolEvent(
            event_type="tool_finished",
            tool_name=facts.tool_name,
            tool_call_id=facts.tool_call_id,
            source_turn_id=turn.source_turn_id,
            tool_risk=facts.tool_risk,
            tool_capabilities=tuple(sorted(registry_capabilities)),
            counts_as_work=facts.counts_as_work,
            invoker_reached=facts.invoker_reached,
            invoker_succeeded=facts.invoker_succeeded,
            execution_status=facts.execution_status,
            result_ok=facts.result_ok,
            error_code=error_code,
            arguments_hash=hash_execution_arguments(
                redact_execution_arguments(dict(arguments))
            ),
            result_preview=bounded_execution_preview(normalized.preview()),
        )
        try:
            self._require_service().record_tool_event(
                session_key=turn.session_key,
                attempt_id=attempt_id,
                event=event,
            )
        except TaskExecutionConflictError:
            turn.lease_lost = True
            self._recover_expired_owner(turn)
            raise TaskExecutionLeaseLostError(
                "late task execution result rejected after lease loss"
            )
        turn.snapshot = self._require_service().inspect(
            session_key=turn.session_key,
            attempt_id=attempt_id,
        )

    def handle_model_final(
        self,
        turn: PreparedTaskExecutionTurn,
    ) -> RuntimeFinalDecision:
        attempt = turn.snapshot.attempt
        if (
            not turn.owns_current_attempt
            or turn.contract.phase != "work"
            or attempt is None
            or attempt.status != "running"
        ):
            return RuntimeFinalDecision(action="accept")
        if turn.protocol_corrections == 0:
            turn.protocol_corrections = 1
            self.protocol_correction_count = 1
            return RuntimeFinalDecision(
                action="correct",
                model_hint=(
                    "Task execution is still running. Call finish_task_step_execution "
                    "exactly once before giving the final reply."
                ),
            )
        self._fail_attempt(turn, "protocol_finish_missing")
        return RuntimeFinalDecision(
            action="failed",
            model_hint=(
                "The execution attempt was failed because the required finish "
                "protocol was omitted. Give a concise final status without tools."
            ),
        )

    def finalize_turn(
        self,
        turn: PreparedTaskExecutionTurn,
        *,
        exit_kind: str,
        error: BaseException | None = None,
    ) -> None:
        del error
        if turn.finalized and not turn.attempt_is_active:
            return
        if not turn.owns_current_attempt:
            replay = self._require_service().replay_request(
                session_key=turn.session_key,
                request_id=turn.request_id,
            )
            if replay is None or not _attempt_matches_turn(turn, replay.attempt):
                return
            turn.snapshot = self._require_service().inspect(
                session_key=turn.session_key,
                attempt_id=replay.attempt.attempt_id,
            )
            turn.request_claimed_attempt = True
            turn.request_owned_attempt_id = replay.attempt.attempt_id
        if not turn.owns_current_attempt:
            return
        attempt = turn.snapshot.attempt
        if attempt is None:
            return
        if attempt.status in TERMINAL_ATTEMPT_STATUSES:
            turn.finalized = True
            return
        if attempt.status == "waiting_authorization":
            turn.finalized = True
            return
        if exit_kind in {"max_iterations", "protocol_finish_missing"}:
            reason = (
                "work_budget_exhausted"
                if exit_kind == "max_iterations"
                else "protocol_finish_missing"
            )
            self._fail_attempt(turn, reason)
            return
        try:
            turn.snapshot = self._require_service().block_attempt(
                session_key=turn.session_key,
                attempt_id=attempt.attempt_id,
                terminal_reason="turn_interrupted_outcome_unknown",
                error_code=exit_kind,
            )
            turn.decision_reason = "turn_interrupted_outcome_unknown"
            turn.finalized = True
        except TaskExecutionConflictError:
            self._recover_expired_owner(turn)

    def request_has_claimed_attempt(self, *, session_key: str, request_id: str) -> bool:
        if self._service is None:
            return False
        return (
            self._service.replay_request(
                session_key=session_key,
                request_id=request_id,
            )
            is not None
        )

    def lease_guard(
        self, turn: PreparedTaskExecutionTurn
    ) -> TaskExecutionLeaseGuard | _NullLeaseGuard:
        attempt = turn.snapshot.attempt
        if (
            not turn.owns_current_attempt
            or attempt is None
            or attempt.status not in {"pending", "running"}
        ):
            return _NullLeaseGuard()
        return TaskExecutionLeaseGuard(
            self._require_service(),
            session_key=turn.session_key,
            attempt_id=attempt.attempt_id,
            lease_seconds=self._config.lease_seconds,
            clock=self._clock,
        )

    def trace(self, turn: PreparedTaskExecutionTurn) -> dict[str, object]:
        attempt = turn.snapshot.attempt
        return {
            "attempt_id": attempt.attempt_id if attempt is not None else None,
            "action": turn.contract.action,
            "phase": turn.contract.phase,
            "status": attempt.status if attempt is not None else None,
            "work_tool_count": sum(
                event.counts_as_work for event in turn.snapshot.events
            ),
            "work_tool_budget": turn.contract.work_call_budget,
            "request_replayed": turn.request_replayed,
            "reason": turn.decision_reason,
        }

    async def _persist_defer(
        self,
        turn: PreparedTaskExecutionTurn,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        capabilities: frozenset[str],
    ) -> None:
        attempt_id = turn.attempt_id
        if attempt_id is None or not turn.owns_current_attempt:
            raise RuntimeError("authorization defer requires a claimed attempt")
        try:
            requested_arguments = self._authorization_request_metadata(
                turn=turn,
                attempt_id=attempt_id,
                tool_name=tool_name,
                arguments=arguments,
            )
            requested_capabilities = _authorization_capabilities(
                tool_name,
                capabilities,
            )
            turn.snapshot = self._require_service().defer_attempt(
                session_key=turn.session_key,
                attempt_id=attempt_id,
                tool_name=tool_name,
                requested_arguments=requested_arguments,
                requested_capabilities=requested_capabilities,
                reason="task_execution_authorization_required",
            )
            turn.decision_reason = "task_execution_authorization_required"
            turn.contract = self._contract_for_snapshot(turn)
            turn.finalized = True
            logger.info(
                "[task_execution_deferred] attempt_id=%s tool=%s",
                attempt_id,
                tool_name,
            )
        except Exception:
            try:
                turn.snapshot = self._require_service().block_attempt(
                    session_key=turn.session_key,
                    attempt_id=attempt_id,
                    terminal_reason="defer_persistence_failed",
                )
                turn.decision_reason = "defer_persistence_failed"
                turn.contract = self._contract_for_snapshot(turn)
                turn.finalized = True
            except Exception as block_error:
                raise TaskExecutionPersistenceError(
                    "task execution defer persistence failed"
                ) from block_error

    def _authorization_request_metadata(
        self,
        *,
        turn: PreparedTaskExecutionTurn,
        attempt_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, object]:
        raw_arguments = dict(arguments)
        approval_scope = "task_execution_step"
        policy_reason = "task_execution_authorization_required"
        metadata: dict[str, object] = {
            "approval_request_id": "",
            "expires_at": "",
            "approval_scope": approval_scope,
            "args_hash": canonical_args_hash(raw_arguments),
            "args_summary": summarize_arguments(raw_arguments),
            "policy_reason": policy_reason,
        }
        if self._approval_runtime is None:
            return metadata
        record = self._approval_runtime.record_defer_request(
            request_id=attempt_id,
            session_key=turn.session_key,
            channel=str(turn.turn_metadata.get("channel", "")),
            chat_id=str(turn.turn_metadata.get("chat_id", "")),
            source="task_execution",
            tool_name=tool_name,
            risk="write",
            approval_scope=approval_scope,
            policy_reason=policy_reason,
            arguments=raw_arguments,
        )
        metadata["approval_request_id"] = record.approval_request_id
        metadata["expires_at"] = record.expires_at
        metadata["args_hash"] = record.args_hash
        return metadata

    def _execution_context(
        self,
        turn: PreparedTaskExecutionTurn,
        *,
        tool_capabilities: frozenset[str],
    ) -> ToolExecutionContext:
        protected: dict[str, object] = {
            "_session_key": turn.session_key,
            "_task_execution_request_id": turn.request_id,
            "_task_execution_action": turn.contract.action,
            "_task_execution_target_step_id": turn.contract.target_step_id or "",
            "_task_execution_attempt_id": (
                turn.attempt_id
                if turn.owns_current_attempt
                or tool_capabilities
                in {
                    frozenset({"task_execution.inspect"}),
                    frozenset({"task_execution.abort"}),
                }
                else ""
            ),
            "_task_execution_read_only": (
                turn.owns_current_attempt and turn.contract.phase == "work"
            ),
        }
        if not tool_capabilities & _CONTROL_CAPABILITIES:
            protected["_task_execution_action"] = turn.contract.action
        return ToolExecutionContext(protected=protected)

    def _contract_for_snapshot(
        self, turn: PreparedTaskExecutionTurn
    ) -> TaskExecutionTurnContract:
        attempt = turn.snapshot.attempt
        if attempt is None:
            return TaskExecutionTurnContract.inactive()
        if attempt.status in TERMINAL_ATTEMPT_STATUSES:
            return _terminal_contract(attempt.attempt_id)
        if not turn.owns_current_attempt:
            if turn.contract.action == "inspect":
                return _inspect_final_contract(
                    attempt.attempt_id,
                    matched_terms=turn.contract.matched_terms,
                )
            return turn.contract
        if attempt.status == "waiting_authorization":
            return TaskExecutionTurnContract(
                active=True,
                action="inspect",
                phase="waiting_authorization",
                attempt_id=attempt.attempt_id,
                target_step_id=None,
                required_capabilities=frozenset({"task_execution.inspect"}),
                allowed_capabilities=frozenset(
                    {"task_execution.inspect", "task_execution.abort"}
                ),
                allowed_risks=frozenset(),
                work_call_budget=0,
                tool_search_budget=0,
                completion_capability="task_execution.inspect",
                reason="task_execution_authorization_required",
                matched_terms=turn.contract.matched_terms,
            )
        return TaskExecutionTurnContract(
            active=True,
            action=(
                turn.contract.action
                if turn.contract.action in {"continue", "retry"}
                else "continue"
            ),
            phase="work",
            attempt_id=attempt.attempt_id,
            target_step_id=(
                turn.contract.target_step_id
                if turn.contract.action == "retry"
                else None
            ),
            required_capabilities=frozenset({"task_execution.finish"}),
            allowed_capabilities=frozenset(
                {
                    "task_execution.finish",
                    "task_execution.defer",
                    "task_execution.abort",
                    "task_execution.inspect",
                }
            ),
            allowed_risks=frozenset({"read-only"}),
            work_call_budget=self._config.max_work_tool_calls,
            tool_search_budget=self._config.max_tool_search_calls,
            completion_capability="task_execution.finish",
            reason="attempt_running",
            matched_terms=turn.contract.matched_terms,
        )

    def _fail_attempt(self, turn: PreparedTaskExecutionTurn, reason: str) -> None:
        attempt_id = turn.attempt_id
        if attempt_id is None or not turn.owns_current_attempt:
            return
        turn.snapshot = self._require_service().finish_attempt(
            session_key=turn.session_key,
            attempt_id=attempt_id,
            success=False,
            result_summary=reason,
            error_code=reason,
            terminal_reason=reason,
        )
        turn.contract = self._contract_for_snapshot(turn)
        turn.decision_reason = reason
        turn.finalized = True

    def _recover_expired_owner(self, turn: PreparedTaskExecutionTurn) -> None:
        service = self._require_service()
        service.reconcile_attempts(now=self._clock(), session_key=turn.session_key)
        attempt_id = turn.attempt_id
        if attempt_id is not None:
            turn.snapshot = service.inspect(
                session_key=turn.session_key,
                attempt_id=attempt_id,
            )
            turn.contract = self._contract_for_snapshot(turn)
            attempt = turn.snapshot.attempt
            turn.decision_reason = _attempt_reason(attempt) if attempt else ""
        turn.finalized = True

    def _require_service(self) -> TaskExecutionService:
        if self._service is None:
            raise RuntimeError("task execution provider is unavailable")
        return self._service


class _NullLeaseGuard:
    async def __aenter__(self) -> _NullLeaseGuard:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


def _terminal_contract(attempt_id: str) -> TaskExecutionTurnContract:
    return TaskExecutionTurnContract(
        active=True,
        action="replay",
        phase="terminal",
        attempt_id=attempt_id,
        target_step_id=None,
        required_capabilities=frozenset(),
        allowed_capabilities=frozenset(),
        allowed_risks=frozenset(),
        work_call_budget=0,
        tool_search_budget=0,
        completion_capability=None,
        reason="runtime_request_replay",
        matched_terms=(),
    )


def _inspect_final_contract(
    attempt_id: str,
    *,
    matched_terms: tuple[str, ...],
) -> TaskExecutionTurnContract:
    return TaskExecutionTurnContract(
        active=True,
        action="inspect",
        phase="finish",
        attempt_id=attempt_id,
        target_step_id=None,
        required_capabilities=frozenset({"task_execution.inspect"}),
        allowed_capabilities=frozenset({"task_execution.inspect"}),
        allowed_risks=frozenset(),
        work_call_budget=0,
        tool_search_budget=0,
        completion_capability="task_execution.inspect",
        reason="task_execution_inspected",
        matched_terms=matched_terms,
    )


def _attempt_matches_turn(
    turn: PreparedTaskExecutionTurn,
    attempt: TaskExecutionAttempt,
) -> bool:
    return bool(
        attempt.session_key == turn.session_key
        and attempt.request_id == turn.request_id
    )


def _is_unowned_session_control(capabilities: frozenset[str]) -> bool:
    return capabilities in {
        frozenset({"task_execution.begin"}),
        frozenset({"task_execution.inspect"}),
        frozenset({"task_execution.abort"}),
    }


def _authorization_capabilities(
    tool_name: str,
    capabilities: frozenset[str],
) -> tuple[str, ...]:
    normalized = set(capabilities)
    if tool_name in {"write_file", "edit_file"}:
        normalized.add("file.write")
    if tool_name == "shell":
        normalized.add("shell.execute")
    return tuple(sorted(normalized))


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _result_facts(result: ToolResult) -> tuple[bool, str]:
    if result.ok is not None:
        return bool(result.ok), str(result.error_code or "")
    try:
        payload = json.loads(result.text)
    except (TypeError, ValueError):
        return False, str(result.error_code or "unstructured_result")
    if not isinstance(payload, dict):
        return False, "unstructured_result"
    return bool(payload.get("ok")), str(payload.get("error_code") or "")


def _begin_decision_action(result: ToolResult) -> str:
    try:
        payload = json.loads(result.text)
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        return ""
    return str(decision.get("action") or "")


def _attempt_reason(attempt: object) -> str:
    return str(getattr(attempt, "terminal_reason", "") or "").split(";", 1)[0]


def _error_payload(error_code: str) -> str:
    return json.dumps(
        {"ok": False, "error_code": error_code, "fallback_allowed": False},
        ensure_ascii=False,
    )
