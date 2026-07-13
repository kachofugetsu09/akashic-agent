"""
PeerAgentTool：将远程 A2A peer agent 包装成 akashic 工具。

行为：
- 调用前检查 peer agent 健康状态，未启动则冷启动
- 提交 A2A 任务后立即返回（fire & forget）
- 将 task_id + channel/chat_id 注册到 Poller，由 Poller 异步跟进
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from agent.peer_agent.card_resolver import AgentCard
from agent.tools.base import Tool
from core.net.http import HttpRequester, RequestBudget

if TYPE_CHECKING:
    from agent.peer_agent.poller import PeerAgentPoller
    from agent.peer_agent.process_manager import PeerProcessManager

logger = logging.getLogger(__name__)

_SUBMIT_TIMEOUT_S = 15.0


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class PeerAgentTool(Tool):
    """委托给远程 A2A peer agent 执行深度任务，立即返回，结果异步推送。"""

    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "用户的原始请求，原封不动传入，不要改写、扩展或补充细节",
            },
            "breadth": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "并行子问题数，控制调研广度，默认 1",
            },
            "rounds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "最大调研轮次，控制调研深度，默认 2",
            },
        },
        "required": ["goal"],
    }

    def __init__(
        self,
        card: AgentCard,
        process_manager: PeerProcessManager,
        poller: PeerAgentPoller,
        requester: HttpRequester,
    ) -> None:
        self._card = card
        self._pm = process_manager
        self._poller = poller
        self._requester = requester
        self._name = f"delegate_{_slugify(card.name)}"
        skill = card.primary_skill
        base_desc = (skill.description if skill else card.description) or (
            f"委托 {card.name} 执行深度调研任务，生成结构化长报告。"
            "适合：技术选型、论文调研、行业分析、竞品对比、综合报告。"
        )
        self._description = base_desc + "\n注意：任务异步执行，完成后系统会自动通知并总结报告给用户。"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, **kwargs: Any) -> str:
        goal: str = kwargs["goal"]
        breadth: int = int(kwargs.get("breadth", 1))
        rounds: int = int(kwargs.get("rounds", 2))

        # channel/chat_id 由 ToolRegistry.execute() 从 set_context 合并而来
        channel: str = kwargs.get("channel", "unknown")
        chat_id: str = kwargs.get("chat_id", "unknown")

        async with self._poller.submission_lease(self._card.name):
            # 1. 冷启动
            ready = await self._pm.ensure_ready(self._card.name)
            try:
                # 2. 提交 A2A 任务
                task_id = await self._submit_task(goal, breadth, rounds)

                # 3. 在同一 ownership 窗口登记到 Poller
                self._poller.register(
                    task_id=task_id,
                    agent_name=self._card.name,
                    agent_url=self._card.url,
                    channel=channel,
                    chat_id=chat_id,
                    goal=goal,
                )
            except BaseException as submit_error:
                try:
                    await self._reap_failed_cold_start(ready.started_by_call)
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "peer agent 提交失败且冷启动进程回收失败",
                        [submit_error, cleanup_error],
                    )
                raise

        logger.info(
            "[PeerAgentTool] 任务已提交 task_id=%s agent=%s channel=%s chat_id=%s",
            task_id, self._card.name, channel, chat_id,
        )

        return json.dumps(
            {
                "status": "submitted",
                "task_id": task_id,
                "agent": self._card.name,
                "message": (
                    "深度调研任务已在后台启动，通常需要3-10分钟。"
                    "完成后系统会自动通知你，届时请读取报告文件并向用户总结。"
                ),
            },
            ensure_ascii=False,
        )

    async def _reap_failed_cold_start(self, started_by_call: bool) -> None:
        """提交未登记时，仅回收本次调用独占的新进程。"""

        if started_by_call and not self._poller.has_pending(self._card.name):
            await self._pm.terminate(self._card.name)

    async def _submit_task(self, goal: str, breadth: int, rounds: int) -> str:
        """通过 A2A JSON-RPC 提交任务，返回 task_id。"""

        # 1. 构建非阻塞 A2A 请求
        payload = {
            "jsonrpc": "2.0",
            "id": "submit-1",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": goal}],
                    "metadata": {"breadth": breadth, "max_rounds": rounds},
                },
                "configuration": {"blocking": False},
            },
        }

        # 2. 发送请求并确认 HTTP 成功
        r = await self._requester.post(
            self._card.url,
            json=payload,
            budget=RequestBudget(total_timeout_s=_SUBMIT_TIMEOUT_S),
        )
        r.raise_for_status()
        raw_data: object = r.json()
        if not isinstance(raw_data, dict):
            raise RuntimeError(
                f"A2A message/send 返回了非对象响应："
                f"类型={type(raw_data).__name__}，值={raw_data!r}"
            )
        data = cast(dict[str, object], raw_data)

        if "error" in data:
            raise RuntimeError(f"A2A 错误: {data['error']}")

        # 3. 只接受服务端返回的真实 Task ID
        raw_result = data.get("result")
        if not isinstance(raw_result, dict):
            raise RuntimeError(
                f"A2A message/send 返回了无效 result："
                f"类型={type(raw_result).__name__}，值={raw_result!r}"
            )
        result = cast(dict[str, object], raw_result)
        task_id = result.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise RuntimeError(f"A2A message/send 未返回有效 Task ID：{task_id!r}")
        return task_id
