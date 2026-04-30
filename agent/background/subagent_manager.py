from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.background.runtime import (
    AgentBackgroundJobRunner,
    AgentBackgroundJobSpec,
)
from agent.policies.delegation import SpawnDecision
from agent.provider import LLMProvider
from agent.subagent import SubAgent
from agent.background.subagent_profiles import (
    PROFILE_RESEARCH,
    SubagentRuntime,
    build_spawn_spec,
)
from bus.internal_events import (
    SpawnCompletionEvent,
)
from bus.events import SpawnCompletionItem
from bus.queue import MessageBus
from core.common.strategy_trace import build_strategy_trace_envelope
from core.net.http import HttpRequester
from prompts.background import build_spawn_subagent_prompt

logger = logging.getLogger(__name__)

_RESULT_MAX_CHARS = 12_000
_SYNC_RESULT_MAX_CHARS = 100_000
_SPAWN_MAX_ITERATIONS = 50
_SYNC_MAX_ITERATIONS = 10


@dataclass(frozen=True)
class RunningSubagentJob:
    job_id: str
    label: str
    task: str
    profile: str
    origin_channel: str
    origin_chat_id: str
    task_dir: str
    retry_count: int
    started_at: str
    status: str = "running"


class SubagentManager:
    """Manage background subagent jobs and announce completion to the main loop."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str,
        max_tokens: int,
        fetch_requester: HttpRequester,
        multimodal: bool = True,
    ) -> None:
        self._workspace = workspace
        self._bus = bus
        self._runtime = SubagentRuntime(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
        )
        self._fetch_requester = fetch_requester
        self._multimodal = multimodal
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._running_jobs: dict[str, RunningSubagentJob] = {}
        self._cancel_announced: set[str] = set()

    def add_tool_hooks(self, hooks: list[object]) -> None:
        object.__setattr__(self._runtime, "tool_hooks", list(hooks))

    def _spawn_jobs_dir(self) -> Path:
        root = self._workspace / "subagent-runs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _job_task_dir(self, job_id: str) -> Path:
        task_dir = self._spawn_jobs_dir() / job_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    async def spawn_sync(
        self,
        *,
        task: str,
        label: str | None,
        profile: str = PROFILE_RESEARCH,
    ) -> str:
        """同步执行 subagent，阻塞当前 turn 直到完成，结果作为 tool result 直接返回。

        适合：调研后需要立即回复用户的任务，预计 ≤ 10 次工具调用。
        """
        job_id = uuid.uuid4().hex[:8]
        display_label = (label or task[:30] or job_id).strip()
        task_dir = self._job_task_dir(job_id)

        logger.info(
            "[spawn_sync] started job_id=%s label=%r profile=%s",
            job_id,
            display_label,
            profile,
        )

        subagent = self._build_subagent(
            task_dir=task_dir,
            profile=profile,
            max_iterations=_SYNC_MAX_ITERATIONS,
        )
        try:
            result = await subagent.run(task)
            exit_reason = getattr(subagent, "last_exit_reason", None) or "completed"
        except Exception as e:
            logger.exception("[spawn_sync] subagent failed job_id=%s err=%s", job_id, e)
            result = f"执行出错：{e}"
            exit_reason = "error"

        truncated = result
        if len(truncated) > _SYNC_RESULT_MAX_CHARS:
            original_len = len(truncated)
            truncated = (
                truncated[:_SYNC_RESULT_MAX_CHARS]
                + f"\n...[结果已截断，原始长度 {original_len}]"
            )

        logger.info(
            "[spawn_sync] completed job_id=%s exit_reason=%s result_len=%d",
            job_id,
            exit_reason,
            len(truncated),
        )
        return f"[子任务「{display_label}」结果]\n退出原因: {exit_reason}\n\n{truncated}"

    async def spawn(
        self,
        *,
        task: str,
        label: str | None,
        origin_channel: str,
        origin_chat_id: str,
        decision: SpawnDecision | None = None,
        profile: str = PROFILE_RESEARCH,
        retry_count: int = 0,
    ) -> str:
        """创建后台 subagent 任务，并立即把控制权还给主 agent。"""
        job_id = uuid.uuid4().hex[:8]
        display_label = (label or task[:30] or job_id).strip()
        task_dir = self._job_task_dir(job_id)
        # 1. 先生成 job_id 和 trace，确保后台任务还没起时也能追踪来源。
        self._append_spawn_trace(
            job_id=job_id,
            payload={
                "phase": "started",
                "label": display_label,
                "task_dir": str(task_dir),
                "origin_channel": origin_channel,
                "origin_chat_id": origin_chat_id,
                "profile": profile,
                "retry_count": retry_count,
                "decision": _decision_payload(decision),
            },
        )
        # 2. 再把真正执行逻辑放到后台 task 中，避免阻塞当前会话。
        bg_task = asyncio.create_task(
            self._run_subagent(
                job_id=job_id,
                task=task,
                label=display_label,
                task_dir=task_dir,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
                decision=decision,
                profile=profile,
                retry_count=retry_count,
            ),
            name=f"spawn:{job_id}",
        )
        # 3. 最后登记运行中任务，并返回给主 agent 一段立即可回复用户的确认文本。
        self._running_tasks[job_id] = bg_task
        self._running_jobs[job_id] = RunningSubagentJob(
            job_id=job_id,
            label=display_label,
            task=task,
            profile=profile,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            task_dir=str(task_dir),
            retry_count=retry_count,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        bg_task.add_done_callback(lambda _: self._forget_running_job(job_id))
        logger.info(
            "[spawn] started job_id=%s label=%r profile=%s retry_count=%d origin=%s:%s reason=%s confidence=%s",
            job_id,
            display_label,
            profile,
            retry_count,
            origin_channel,
            origin_chat_id,
            decision.meta.reason_code if decision is not None else "-",
            decision.meta.confidence if decision is not None else "-",
        )
        return (
            f"已创建后台任务「{display_label}」（job_id={job_id}）。"
            "不要等待其完成；请直接向用户说明你已开始处理，完成后会继续回复。"
        )

    def get_running_count(self) -> int:
        return len(self._running_tasks)

    def list_running_jobs(self) -> list[dict[str, object]]:
        return [asdict(job) for job in self._running_jobs.values()]

    async def cancel(self, job_id: str) -> bool:
        task = self._running_tasks.get(job_id)
        if task is None or task.done():
            return False
        job = self._running_jobs.get(job_id)
        if job is not None:
            self._cancel_announced.add(job_id)
            await self._announce_cancelled_job(job)
        task.cancel()
        await asyncio.sleep(0)
        logger.info("[spawn] cancel requested job_id=%s", job_id)
        return True

    def _forget_running_job(self, job_id: str) -> None:
        self._running_tasks.pop(job_id, None)
        self._running_jobs.pop(job_id, None)
        self._cancel_announced.discard(job_id)

    async def _run_subagent(
        self,
        *,
        job_id: str,
        task: str,
        label: str,
        task_dir: Path,
        origin_channel: str,
        origin_chat_id: str,
        decision: SpawnDecision | None,
        profile: str = PROFILE_RESEARCH,
        retry_count: int = 0,
    ) -> None:
        """运行后台 subagent，并把统一结果协议回灌给主 agent。"""
        job_runner = AgentBackgroundJobRunner(
            lambda: self._build_subagent(task_dir=task_dir, profile=profile)
        )
        try:
            # 1. 先按统一 background job spec 执行 subagent，本层不直接碰 loop 细节。
            result = await job_runner.run(
                AgentBackgroundJobSpec(
                    job_id=job_id,
                    job_kind="conversation_spawn",
                    label=label,
                    task=task,
                    max_iterations=_SPAWN_MAX_ITERATIONS,
                    completion_mode="message_bus",
                    persistence_mode="ephemeral",
                ),
                on_exception=lambda e: logger.exception(
                    "[spawn] subagent failed job_id=%s err=%s", job_id, e
                ),
                error_result_summary=None,
            )
        except asyncio.CancelledError:
            if job_id not in self._cancel_announced:
                await self._announce_result(
                    job_id=job_id,
                    label=label,
                    task=task,
                    origin_channel=origin_channel,
                    origin_chat_id=origin_chat_id,
                    status="cancelled",
                    exit_reason="cancelled",
                    result="后台任务已按请求取消。",
                    decision=decision,
                    profile=profile,
                    retry_count=retry_count,
                )
            self._append_spawn_trace(
                job_id=job_id,
                payload={
                    "phase": "cancelled",
                    "task_dir": str(task_dir),
                    "status": "cancelled",
                    "exit_reason": "cancelled",
                    "profile": profile,
                    "retry_count": retry_count,
                    "decision": _decision_payload(decision),
                },
            )
            raise
        # 2. 再把统一结果协议转成 bus completion event，回到原会话。
        await self._announce_result(
            job_id=job_id,
            label=label,
            task=task,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            status=result.status,
            exit_reason=result.exit_reason,
            result=result.result_summary,
            decision=decision,
            profile=profile,
            retry_count=retry_count,
        )
        # 3. 最后补 completion trace，方便排查"为什么这个后台任务结束了"。
        self._append_spawn_trace(
            job_id=job_id,
            payload={
                "phase": "completed",
                "task_dir": str(task_dir),
                "job_kind": result.job_kind,
                "status": result.status,
                "exit_reason": result.exit_reason,
                "completion_mode": result.completion_mode,
                "persistence_mode": result.persistence_mode,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "profile": profile,
                "retry_count": retry_count,
                "decision": _decision_payload(decision),
            },
        )

    async def _announce_cancelled_job(self, job: RunningSubagentJob) -> None:
        await self._announce_result(
            job_id=job.job_id,
            label=job.label,
            task=job.task,
            origin_channel=job.origin_channel,
            origin_chat_id=job.origin_chat_id,
            status="cancelled",
            exit_reason="cancelled",
            result="后台任务已按请求取消。",
            decision=None,
            profile=job.profile,
            retry_count=job.retry_count,
        )

    def _build_subagent(
        self,
        *,
        task_dir: Path,
        profile: str = PROFILE_RESEARCH,
        max_iterations: int = _SPAWN_MAX_ITERATIONS,
    ) -> SubAgent:
        spec = build_spawn_spec(
            workspace=self._workspace,
            task_dir=task_dir,
            fetch_requester=self._fetch_requester,
            system_prompt=self._build_subagent_prompt(
                task_dir=task_dir, profile=profile
            ),
            max_iterations=max_iterations,
            profile=profile,
            multimodal=self._multimodal,
        )
        return spec.build(self._runtime)

    def _build_subagent_prompt(self, task_dir: Path, profile: str = PROFILE_RESEARCH) -> str:
        return build_spawn_subagent_prompt(self._workspace, task_dir, profile)

    async def _announce_result(
        self,
        *,
        job_id: str,
        label: str,
        task: str,
        origin_channel: str,
        origin_chat_id: str,
        status: str,
        exit_reason: str,
        result: str,
        decision: SpawnDecision | None,
        profile: str = PROFILE_RESEARCH,
        retry_count: int = 0,
    ) -> None:
        """把后台结果包装成内部事件，重新投回主 agent 的消息总线。"""
        payload_result = result
        # 1. 先裁剪过长结果，避免 completion event 把主会话和 trace 撑爆。
        if len(payload_result) > _RESULT_MAX_CHARS:
            original_len = len(payload_result)
            payload_result = (
                payload_result[:_RESULT_MAX_CHARS]
                + f"\n...[结果已截断，原始长度 {original_len}]"
            )
        # 2. 再把结果包成 typed inbox item，路由回原 channel/chat_id。
        item = SpawnCompletionItem(
            channel=origin_channel,
            chat_id=origin_chat_id,
            event=SpawnCompletionEvent(
                job_id=job_id,
                label=label,
                task=task,
                status=status,
                exit_reason=exit_reason,
                result=payload_result,
                retry_count=retry_count,
                profile=profile,
            ),
            decision=decision,
        )
        # 3. 最后发布到 bus，让主 agent 以同一会话身份继续回复用户。
        await self._bus.publish_inbound(item)
        logger.info(
            "[spawn] completed job_id=%s status=%s exit_reason=%s profile=%s retry_count=%d route=%s:%s decision_reason=%s",
            job_id,
            status,
            exit_reason,
            profile,
            retry_count,
            origin_channel,
            origin_chat_id,
            decision.meta.reason_code if decision is not None else "-",
        )

    def _append_spawn_trace(self, *, job_id: str, payload: dict[str, object]) -> None:
        try:
            memory_dir = self._workspace / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            trace_file = memory_dir / "spawn_trace.jsonl"
            line = {
                **build_strategy_trace_envelope(
                    trace_type="spawn",
                    source="agent.spawn",
                    subject_kind="job",
                    subject_id=job_id,
                    payload=payload,
                ),
                **payload,
                "job_id": job_id,
            }
            with trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[spawn] write trace failed job_id=%s err=%s", job_id, e)


def _decision_payload(decision: SpawnDecision | None) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "should_spawn": decision.should_spawn,
        "label": decision.label,
        "block_reason": decision.block_reason,
        "meta": {
            "source": decision.meta.source,
            "confidence": decision.meta.confidence,
            "reason_code": decision.meta.reason_code,
        },
    }
