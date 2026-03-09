from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from agent.provider import LLMProvider
from agent.subagent import SubAgent
from agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from agent.tools.shell import ShellTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from bus.events import InboundMessage
from bus.queue import MessageBus
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
        self._provider = provider
        self._workspace = workspace
        self._bus = bus
        self._model = model
        self._max_tokens = max_tokens
        self._fetch_requester = fetch_requester
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        *,
        task: str,
        label: str | None,
        origin_channel: str,
        origin_chat_id: str,
    ) -> str:
        job_id = uuid.uuid4().hex[:8]
        display_label = (label or task[:30] or job_id).strip()
        bg_task = asyncio.create_task(
            self._run_subagent(
                job_id=job_id,
                task=task,
                label=display_label,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
            ),
            name=f"spawn:{job_id}",
        )
        self._running_tasks[job_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(job_id, None))
        logger.info(
            "[spawn] started job_id=%s label=%r origin=%s:%s",
            job_id,
            display_label,
            origin_channel,
            origin_chat_id,
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
    ) -> None:
        try:
            agent = self._build_subagent()
            result = await agent.run(task)
            exit_reason = getattr(agent, "last_exit_reason", "completed")
            status = self._status_from_exit_reason(exit_reason)
            await self._announce_result(
                job_id=job_id,
                label=label,
                task=task,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
                status=status,
                exit_reason=exit_reason,
                result=result,
            )
        except Exception as e:
            logger.exception("[spawn] subagent failed job_id=%s err=%s", job_id, e)
            await self._announce_result(
                job_id=job_id,
                label=label,
                task=task,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
                status="error",
                exit_reason="error",
                result=f"后台任务执行失败：{e}",
            )

    def _build_subagent(self) -> SubAgent:
        tools = [
            ReadFileTool(allowed_dir=self._workspace),
            ListDirTool(allowed_dir=self._workspace),
            WriteFileTool(allowed_dir=self._workspace),
            EditFileTool(allowed_dir=self._workspace),
            WebSearchTool(),
            WebFetchTool(self._fetch_requester),
            ShellTool(),
        ]
        return SubAgent(
            provider=self._provider,
            model=self._model,
            tools=tools,
            system_prompt=self._build_subagent_prompt(),
            max_iterations=20,
            max_tokens=self._max_tokens,
        )

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

    @staticmethod
    def _status_from_exit_reason(exit_reason: str) -> str:
        if exit_reason == "completed":
            return "completed"
        if exit_reason == "error":
            return "error"
        return "incomplete"

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
    ) -> None:
        payload_result = result
        if len(payload_result) > _RESULT_MAX_CHARS:
            original_len = len(payload_result)
            payload_result = (
                payload_result[:_RESULT_MAX_CHARS]
                + f"\n...[结果已截断，原始长度 {original_len}]"
            )
        msg = InboundMessage(
            channel=origin_channel,
            sender="spawn",
            chat_id=origin_chat_id,
            content="[internal spawn completed]",
            metadata={
                "internal_event": "spawn_completed",
                "spawn": {
                    "job_id": job_id,
                    "label": label,
                    "task": task,
                    "status": status,
                    "exit_reason": exit_reason,
                    "result": payload_result,
                },
            },
        )
        await self._bus.publish_inbound(msg)
        logger.info(
            "[spawn] completed job_id=%s status=%s exit_reason=%s route=%s:%s",
            job_id,
            status,
            exit_reason,
            origin_channel,
            origin_chat_id,
        )
