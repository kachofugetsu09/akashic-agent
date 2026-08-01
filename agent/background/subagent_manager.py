from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

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
from agent.tool_hooks.base import ToolHook
from bus.internal_events import (
    SpawnCompletionEvent,
)
from bus.events import SpawnCompletionItem
from bus.queue import MessageBus
from core.common.strategy_trace import build_strategy_trace_envelope
from core.net.http import HttpRequester
from prompts.background import build_spawn_subagent_prompt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotLease

_RESULT_MAX_CHARS = 12_000
_SYNC_RESULT_MAX_CHARS = 100_000
_SPAWN_MAX_ITERATIONS = 50
_SYNC_MAX_ITERATIONS = 10
MAX_ACTIVE_SUBAGENTS = 3


class SubagentCapacityError(RuntimeError):
    """当前 subagent admission 已满，拒绝本次创建。"""

    def __init__(self, *, active: int, maximum: int) -> None:
        self.active = active
        self.maximum = maximum
        super().__init__(
            "subagent capacity reached: "
            f"active={active}, max={maximum}; current spawn rejected"
        )


@dataclass(frozen=True)
class _SubagentAdmissionLease:
    token: str
    owner: str
    admission: "_SubagentAdmission"

    def release(self) -> None:
        self.admission.release(self)


class _SubagentAdmission:
    """在同一事件循环内原子登记真实 subagent worker。"""

    def __init__(self, maximum: int = MAX_ACTIVE_SUBAGENTS) -> None:
        if maximum < 1:
            raise ValueError("subagent admission maximum must be positive")
        self._maximum = maximum
        self._active: dict[str, _SubagentAdmissionLease] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    def acquire(self, *, owner: str) -> _SubagentAdmissionLease:
        """在无 await 的临界区登记 worker，容量满时只拒绝当前调用。"""

        # 1. 事件循环内同步检查和登记，避免同步/后台路径竞态超卖。
        active = len(self._active)
        if active >= self._maximum:
            raise SubagentCapacityError(active=active, maximum=self._maximum)

        lease = _SubagentAdmissionLease(
            token=uuid.uuid4().hex,
            owner=owner,
            admission=self,
        )
        self._active[lease.token] = lease
        return lease

    def release(self, lease: _SubagentAdmissionLease) -> None:
        current = self._active.get(lease.token)
        if current is not lease:
            raise RuntimeError(
                f"subagent admission lease 已释放或不属于 owner={lease.owner}"
            )
        del self._active[lease.token]


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
    """管理后台子任务，并将完成事件送回原会话。"""

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
        self._sync_tasks: dict[str, asyncio.Task[tuple[str, str]]] = {}
        self._running_jobs: dict[str, RunningSubagentJob] = {}
        self._cancel_announced: set[str] = set()
        self._snapshot_release_tasks: set[asyncio.Task[None]] = set()
        self._admission = _SubagentAdmission()

    def add_tool_hooks(self, hooks: list[ToolHook]) -> None:
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
        """同步执行子任务，并将结果直接返回当前轮次。"""

        job_id = uuid.uuid4().hex[:8]
        display_label = (label or task[:30] or job_id).strip()
        lease = self._admission.acquire(owner=f"sync:{job_id}")
        worker: asyncio.Task[tuple[str, str]] | None = None
        callback_registered = False
        try:
            task_dir = self._job_task_dir(job_id)
            worker = asyncio.create_task(
                self._run_sync_subagent(
                    task=task,
                    task_dir=task_dir,
                    profile=profile,
                ),
                name=f"spawn_sync:{job_id}",
            )
            self._sync_tasks[job_id] = worker
            worker.add_done_callback(
                lambda done: self._finish_sync_job(job_id, lease, done)
            )
            callback_registered = True
        except BaseException:
            if not callback_registered and worker is not None:
                self._sync_tasks.pop(job_id, None)
                worker.cancel()
                _ = await asyncio.gather(worker, return_exceptions=True)
            lease.release()
            raise

        logger.info(
            "[spawn_sync] started job_id=%s label=%r profile=%s",
            job_id,
            display_label,
            profile,
        )

        try:
            assert worker is not None
            result, exit_reason = await asyncio.shield(worker)
        except asyncio.CancelledError:
            # 1. 取消真实 worker，并等待其 finally/cleanup 完成后再传播取消。
            worker.cancel()
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                logger.exception(
                    "[spawn_sync] cancelled worker cleanup failed job_id=%s err=%s",
                    job_id,
                    exc,
                )
            # 2. done callback 只在 worker 真实结束后释放 admission。
            logger.info(
                "[spawn_sync] wait cancelled; worker cleanup confirmed job_id=%s",
                job_id,
            )
            raise

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
        lease = self._admission.acquire(owner=f"background:{job_id}")
        task_dir: Path | None = None
        snapshot_lease: RuntimeSnapshotLease | None = None
        bg_task: asyncio.Task[None] | None = None
        callback_registered = False
        try:
            # 1. 先写追踪记录，确保后台任务创建失败时仍可定位
            task_dir = self._job_task_dir(job_id)
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
            # 2. 租用当前快照后启动后台任务，隔离后续热重载
            from agent.plugins.snapshot import lease_current_runtime_snapshot

            snapshot_lease = lease_current_runtime_snapshot()
            assert task_dir is not None
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
                    snapshot_lease=snapshot_lease,
                ),
                name=f"spawn:{job_id}",
            )
            # 3. 登记任务后立即返回，完成回调负责释放 admission/快照
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
            bg_task.add_done_callback(
                lambda done: self._finish_background_job(
                    job_id,
                    snapshot_lease,
                    lease,
                    done,
                )
            )
            callback_registered = True
        except BaseException:
            if not callback_registered and bg_task is not None:
                self._forget_running_job(job_id)
                bg_task.cancel()
                _ = await asyncio.gather(bg_task, return_exceptions=True)
            if (
                not callback_registered
                and snapshot_lease is not None
                and snapshot_lease.active
            ):
                try:
                    await snapshot_lease.release()
                except BaseException as exc:
                    logger.exception(
                        "[spawn] cleanup_degraded snapshot release failed "
                        "job_id=%s err=%s",
                        job_id,
                        exc,
                    )
            if not callback_registered:
                lease.release()
            raise
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
        return self._admission.active_count

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

    async def _run_sync_subagent(
        self,
        *,
        task: str,
        task_dir: Path,
        profile: str,
    ) -> tuple[str, str]:
        """运行同步 worker，并把可观察错误转换成当前协议结果。"""

        subagent = self._build_subagent(
            task_dir=task_dir,
            profile=profile,
            max_iterations=_SYNC_MAX_ITERATIONS,
        )
        try:
            result = await subagent.run(task)
            exit_reason = getattr(subagent, "last_exit_reason", None) or "completed"
        except Exception as exc:
            logger.exception("[spawn_sync] subagent failed err=%s", exc)
            result = f"执行出错：{exc}"
            exit_reason = "error"
        return result, exit_reason

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
        snapshot_lease: RuntimeSnapshotLease | None = None,
    ) -> None:
        """运行后台子任务，并按统一协议回传结果。"""
        if snapshot_lease is not None:
            from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

            async with snapshot_lease:
                token = bind_runtime_snapshot(snapshot_lease)
                try:
                    await self._run_subagent(
                        job_id=job_id,
                        task=task,
                        label=label,
                        task_dir=task_dir,
                        origin_channel=origin_channel,
                        origin_chat_id=origin_chat_id,
                        decision=decision,
                        profile=profile,
                        retry_count=retry_count,
                    )
                finally:
                    reset_runtime_snapshot(token)
            return
        job_runner = AgentBackgroundJobRunner(
            lambda: self._build_subagent(task_dir=task_dir, profile=profile)
        )
        try:
            # 1. 通过统一任务协议执行，不在管理器内复制推理循环
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
        # 2. 将结果转换为完成事件，送回原会话
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
        # 3. 记录最终状态，保留任务结束原因
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

    def _finish_background_job(
        self,
        job_id: str,
        snapshot_lease: RuntimeSnapshotLease | None,
        admission_lease: _SubagentAdmissionLease,
        task: asyncio.Task[None],
    ) -> None:
        self._forget_running_job(job_id)
        admission_lease.release()
        _consume_task_exception(task)
        if snapshot_lease is not None and snapshot_lease.active:
            release_task = asyncio.create_task(
                snapshot_lease.release(),
                name=f"spawn_snapshot_release:{job_id}",
            )
            self._snapshot_release_tasks.add(release_task)
            release_task.add_done_callback(
                lambda done: self._finish_snapshot_release(job_id, done)
            )

    def _finish_sync_job(
        self,
        job_id: str,
        admission_lease: _SubagentAdmissionLease,
        task: asyncio.Task[tuple[str, str]],
    ) -> None:
        self._sync_tasks.pop(job_id, None)
        admission_lease.release()
        _consume_task_exception(task)

    def _finish_snapshot_release(
        self,
        job_id: str,
        task: asyncio.Task[None],
    ) -> None:
        self._snapshot_release_tasks.discard(task)
        _consume_task_exception(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "[spawn] cleanup_degraded snapshot release failed job_id=%s err=%s",
                job_id,
                task.exception(),
            )

    async def shutdown(self) -> None:
        tasks = [*self._running_tasks.values(), *self._sync_tasks.values()]
        for task in tasks:
            _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)
        while self._snapshot_release_tasks:
            release_tasks = tuple(self._snapshot_release_tasks)
            self._snapshot_release_tasks.difference_update(release_tasks)
            _ = await asyncio.gather(
                *release_tasks,
                return_exceptions=True,
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
        """将后台结果包装为内部事件，并送回主 Agent 消息总线。"""
        payload_result = result
        # 1. 裁剪事件载荷，避免完成消息挤占主会话上下文
        if len(payload_result) > _RESULT_MAX_CHARS:
            original_len = len(payload_result)
            payload_result = (
                payload_result[:_RESULT_MAX_CHARS]
                + f"\n...[结果已截断，原始长度 {original_len}]"
            )
        # 2. 构建带原 channel/chat_id 的结构化事件
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
        # 3. 发布到消息总线，由主 Agent 继续原会话
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


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    """读取后台 task 的异常，避免取消等待后的未处理异常噪声。"""

    if task.cancelled():
        return
    _ = task.exception()
