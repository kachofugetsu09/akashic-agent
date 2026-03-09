from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from agent.background.runtime import (
    AgentBackgroundJobRunner,
    AgentBackgroundJobSpec,
)
from agent.policies.delegation import SpawnDecision
from agent.provider import LLMProvider
from agent.subagent import SubAgent
from agent.background.subagent_profiles import SubagentRuntime, build_spawn_spec
from bus.internal_events import (
    SpawnCompletionEvent,
    make_spawn_completion_message,
)
from bus.queue import MessageBus
from core.common.strategy_trace import build_strategy_trace_envelope
from core.net.http import HttpRequester

logger = logging.getLogger(__name__)

_RESULT_MAX_CHARS = 12_000


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
    ) -> None:
        self._workspace = workspace
        self._bus = bus
        self._runtime = SubagentRuntime(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
        )
        self._fetch_requester = fetch_requester
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        *,
        task: str,
        label: str | None,
        origin_channel: str,
        origin_chat_id: str,
        decision: SpawnDecision | None = None,
    ) -> str:
        """创建后台 subagent 任务，并立即把控制权还给主 agent。"""
        job_id = uuid.uuid4().hex[:8]
        display_label = (label or task[:30] or job_id).strip()
        # 1. 先生成 job_id 和 trace，确保后台任务还没起时也能追踪来源。
        self._append_spawn_trace(
            job_id=job_id,
            payload={
                "phase": "started",
                "label": display_label,
                "origin_channel": origin_channel,
                "origin_chat_id": origin_chat_id,
                "decision": _decision_payload(decision),
            },
        )
        # 2. 再把真正执行逻辑放到后台 task 中，避免阻塞当前会话。
        bg_task = asyncio.create_task(
            self._run_subagent(
                job_id=job_id,
                task=task,
                label=display_label,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
                decision=decision,
            ),
            name=f"spawn:{job_id}",
        )
        # 3. 最后登记运行中任务，并返回给主 agent 一段立即可回复用户的确认文本。
        self._running_tasks[job_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(job_id, None))
        logger.info(
            "[spawn] started job_id=%s label=%r origin=%s:%s reason=%s confidence=%s",
            job_id,
            display_label,
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

    async def _run_subagent(
        self,
        *,
        job_id: str,
        task: str,
        label: str,
        origin_channel: str,
        origin_chat_id: str,
        decision: SpawnDecision | None,
    ) -> None:
        """运行后台 subagent，并把统一结果协议回灌给主 agent。"""
        job_runner = AgentBackgroundJobRunner(self._build_subagent)
        # 1. 先按统一 background job spec 执行 subagent，本层不直接碰 loop 细节。
        result = await job_runner.run(
            AgentBackgroundJobSpec(
                job_id=job_id,
                job_kind="conversation_spawn",
                label=label,
                task=task,
                max_iterations=20,
                completion_mode="message_bus",
                persistence_mode="ephemeral",
            ),
            on_exception=lambda e: logger.exception(
                "[spawn] subagent failed job_id=%s err=%s", job_id, e
            ),
            error_result_summary=None,
        )
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
        )
        # 3. 最后补 completion trace，方便排查“为什么这个后台任务结束了”。
        self._append_spawn_trace(
            job_id=job_id,
            payload={
                "phase": "completed",
                "job_kind": result.job_kind,
                "status": result.status,
                "exit_reason": result.exit_reason,
                "completion_mode": result.completion_mode,
                "persistence_mode": result.persistence_mode,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "decision": _decision_payload(decision),
            },
        )

    def _build_subagent(self) -> SubAgent:
        spec = build_spawn_spec(
            workspace=self._workspace,
            fetch_requester=self._fetch_requester,
            system_prompt=self._build_subagent_prompt(),
            max_iterations=20,
        )
        return spec.build(self._runtime)

    def _build_subagent_prompt(self) -> str:
        workspace = str(self._workspace.expanduser().resolve())
        return (
            "你是主 agent 派生出的后台执行 agent。\n"
            "你的唯一目标是完成当前分配的任务，不要做额外延伸。\n"
            "\n"
            "规则：\n"
            "1. 只处理当前任务，不主动接新任务。\n"
            "2. 不直接与用户对话；你的结果会回传给主 agent。\n"
            "3. 禁止再创建后台任务。\n"
            "4. 你看不到主会话完整历史，只能基于当前任务行动。\n"
            "5. 若创建或修改了文件，最终结果必须明确写出文件路径。\n"
            "6. 若未完成，最终结果必须明确写：已完成什么、未完成什么、下一步建议。\n"
            "\n"
            f"工作区根目录：{workspace}\n"
            f"技能目录：{workspace}/skills/ （需要时可自行读取对应 SKILL.md）"
        )

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
        # 2. 再把结果包成 spawn completion message，路由回原 channel/chat_id。
        msg = make_spawn_completion_message(
            channel=origin_channel,
            chat_id=origin_chat_id,
            event=SpawnCompletionEvent(
                job_id=job_id,
                label=label,
                task=task,
                status=status,
                exit_reason=exit_reason,
                result=payload_result,
            ),
            decision=decision,
        )
        # 3. 最后发布到 bus，让主 agent 以同一会话身份继续回复用户。
        await self._bus.publish_inbound(msg)
        logger.info(
            "[spawn] completed job_id=%s status=%s exit_reason=%s route=%s:%s decision_reason=%s",
            job_id,
            status,
            exit_reason,
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
        "meta": {
            "source": decision.meta.source,
            "confidence": decision.meta.confidence,
            "reason_code": decision.meta.reason_code,
        },
    }
