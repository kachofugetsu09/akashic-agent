from __future__ import annotations

from typing import Any

from agent.background.subagent_manager import SubagentManager
from agent.policies.delegation import DelegationPolicy
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
import logging

logger = logging.getLogger(__name__)


class SpawnTool(Tool):
    """Create a background subagent task bound to the current session."""

    def __init__(
        self,
        manager: SubagentManager,
        tool_registry: ToolRegistry,
        policy: DelegationPolicy | None = None,
    ) -> None:
        self._manager = manager
        self._tool_registry = tool_registry
        self._policy = policy or DelegationPolicy()

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return """\
把一个有界的多步任务交给独立 subagent 执行，主 agent 专注决策和用户沟通。

何时使用 spawn（同时满足所有条件）：
- 预计需要 4 步以上工具调用
- 可以完全独立完成，中途不需要用户确认
- 产出是报告 / 文件 / 分析结论，而非"立刻执行的行动"

何时不用 spawn：
- 只需 1–3 次工具调用 → 直接调用工具，更快
- 直接回答问题（查询 / 解释 / 计算）→ 直接回答
- 任务需要修改当前会话状态（写 session memory）→ 主 agent 自己做
- 任务需要和用户来回确认才能推进
- 用户说"发送/告诉/立即执行"——需要立即生效的行动

执行模式（run_in_background）：
- false（默认）：同步执行，主会话等待结果后直接回复用户；适合研究后需要立即回答的任务，预计 ≤ 10 次工具调用
- true：后台执行，主会话立即继续，结果完成后系统带回；适合独立长任务，预计 > 60 秒或 > 15 次工具调用

工具权限 profile：
- research（默认）：只读调研，可搜索 / 读文件 / 抓网页，无法执行命令或写文件；大多数场景选此
- scripting：执行型，可运行 shell 命令 / 在任务目录写文件，无法访问网络
- general：两者兼有，仅在任务明确需要"边调研边执行"时使用

如何写好 task 参数：
subagent 没有看过当前会话。像给刚进房间的同事写交接文档：
- 任务目标：一句话说清楚产出物是什么
- 关键约束：格式 / 范围 / 截止 / 不能做什么
- 关键上下文：用户相关偏好、当前状态摘要、已经试过什么
- 期望输出格式：文本报告 / Markdown / JSON / 写入文件

同步模式调用后主 agent 等待结果再回复用户。
后台模式调用后本轮只做简短确认，结果完成后系统会带回当前会话继续处理。\
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "交给 subagent 的完整任务描述。必须包含：\n"
                        "1. 任务目标（一句话，说清楚产出物）\n"
                        "2. 关键约束（格式 / 范围 / 截止）\n"
                        "3. 关键上下文（用户偏好、当前状态、已试过什么）\n"
                        "4. 期望输出格式"
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "3–5 字的任务短标签，用于状态显示",
                },
                "profile": {
                    "type": "string",
                    "enum": ["research", "scripting", "general"],
                    "description": (
                        "subagent 的工具权限配置：\n"
                        "- research（默认）：只读调研，可搜索 / 读文件 / 抓网页\n"
                        "- scripting：执行型，可运行 shell 命令 / 在任务目录写文件\n"
                        "- general：两者兼有，仅在明确需要时使用"
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "false（默认）：同步执行，主会话等待结果后直接回复用户。\n"
                        "true：后台执行，主会话立即继续，结果完成后系统带回。"
                    ),
                },
                "retry_count": {
                    "type": "integer",
                    "description": "当前后台任务已重试次数。首次调用为 0，重试时传 1。",
                    "minimum": 0,
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        profile: str = "research",
        run_in_background: bool = False,
        retry_count: int = 0,
        **_: Any,
    ) -> str:
        retry_count = max(0, int(retry_count))
        running_count = self._manager.get_running_count() if run_in_background else 0
        decision = self._policy.decide(
            task=task, label=label, running_count=running_count
        )
        logger.info(
            "[spawn] decision should_spawn=%s reason=%s confidence=%s source=%s label=%r profile=%s background=%s explicit_call=true",
            decision.should_spawn,
            decision.meta.reason_code,
            decision.meta.confidence,
            decision.meta.source,
            decision.label,
            profile,
            run_in_background,
        )

        if not decision.should_spawn:
            return f"任务被拦截：{decision.block_reason}"

        if run_in_background:
            ctx = self._tool_registry.get_context()
            channel = str(ctx.get("channel", "") or "").strip()
            chat_id = str(ctx.get("chat_id", "") or "").strip()
            if not channel or not chat_id:
                return "错误：当前会话上下文缺失，无法创建后台任务"
            return await self._manager.spawn(
                task=task,
                label=label,
                origin_channel=channel,
                origin_chat_id=chat_id,
                decision=decision,
                profile=profile,
                retry_count=retry_count,
            )

        return await self._manager.spawn_sync(
            task=task,
            label=label,
            profile=profile,
        )
