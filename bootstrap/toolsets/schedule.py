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
    tools.register(ScheduleTool(scheduler), risk="write", search_hint="cron timer 延时执行")
    tools.register(ListSchedulesTool(scheduler), risk="read-only", search_hint="提醒列表 已有计划")
    tools.register(CancelScheduleTool(scheduler), risk="write", search_hint="删除提醒 取消任务")
