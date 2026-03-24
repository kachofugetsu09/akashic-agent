from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal

from agent.subagent import SubAgent

AgentBackgroundJobKind = Literal["conversation_spawn"]
AgentBackgroundCompletionMode = Literal["message_bus", "direct_notify"]
AgentBackgroundPersistenceMode = Literal["ephemeral", "task_dir"]
AgentBackgroundStatus = Literal["completed", "incomplete", "error"]


@dataclass(frozen=True)
class AgentBackgroundJobSpec:
    job_id: str
    job_kind: AgentBackgroundJobKind
    label: str
    task: str
    max_iterations: int
    completion_mode: AgentBackgroundCompletionMode
    persistence_mode: AgentBackgroundPersistenceMode


@dataclass(frozen=True)
class AgentBackgroundJobResult:
    job_id: str
    job_kind: AgentBackgroundJobKind
    label: str
    status: AgentBackgroundStatus
    exit_reason: str
    result_summary: str
    started_at: str
    finished_at: str
    completion_mode: AgentBackgroundCompletionMode
    persistence_mode: AgentBackgroundPersistenceMode


class AgentBackgroundJobRunner:
    """Run an agent-type background job and normalize lifecycle results."""

    def __init__(self, agent_factory: Callable[[], SubAgent]) -> None:
        self._agent_factory = agent_factory

    async def run(
        self,
        spec: AgentBackgroundJobSpec,
        *,
        on_exception: Callable[[Exception], None] | None = None,
        error_result_summary: str | None = None,
    ) -> AgentBackgroundJobResult:
        started_at = datetime.now(timezone.utc)
        try:
            agent = self._agent_factory()
            result_summary = await agent.run(spec.task)
            exit_reason = getattr(agent, "last_exit_reason", "completed")
            status = self.status_from_exit_reason(exit_reason)
        except Exception as e:
            if on_exception is not None:
                on_exception(e)
            result_summary = (
                error_result_summary
                if error_result_summary is not None
                else f"后台任务执行失败：{e}"
            )
            exit_reason = "error"
            status = "error"
        finished_at = datetime.now(timezone.utc)
        return AgentBackgroundJobResult(
            job_id=spec.job_id,
            job_kind=spec.job_kind,
            label=spec.label,
            status=status,
            exit_reason=exit_reason,
            result_summary=result_summary,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            completion_mode=spec.completion_mode,
            persistence_mode=spec.persistence_mode,
        )

    @staticmethod
    def status_from_exit_reason(exit_reason: str) -> AgentBackgroundStatus:
        if exit_reason == "completed":
            return "completed"
        if exit_reason == "error":
            return "error"
        return "incomplete"
