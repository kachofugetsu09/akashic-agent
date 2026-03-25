from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent.scheduler import LatencyTracker, SchedulerService
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.schedule import CancelScheduleTool, ListSchedulesTool, ScheduleTool


def build_scheduler(
    workspace: Path,
    push_tool: MessagePushTool,
    *,
    agent_loop_provider: Callable[[], Any] | None = None,
) -> SchedulerService:
    return SchedulerService(
        store_path=workspace / "schedules.json",
        push_tool=push_tool,
        agent_loop=None,
        agent_loop_provider=agent_loop_provider,
        tracker=LatencyTracker(),
    )


def register_scheduler_tools(
    tools: ToolRegistry,
    scheduler: SchedulerService,
) -> None:
    tools.register(ScheduleTool(scheduler), tags=["scheduling"], risk="write", search_keywords=["定时任务", "设置提醒", "计划任务", "cron", "延时执行", "timer"])
    tools.register(ListSchedulesTool(scheduler), tags=["scheduling"], risk="read-only", search_keywords=["查看定时任务", "定时列表", "提醒列表", "有哪些计划"])
    tools.register(CancelScheduleTool(scheduler), tags=["scheduling"], risk="write", search_keywords=["取消定时", "删除提醒", "取消任务", "cancel schedule"])
