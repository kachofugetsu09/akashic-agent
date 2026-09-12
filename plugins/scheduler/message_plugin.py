from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context, RUNTIME_STARTED, RUNTIME_STOPPING, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.tasks import TASKS, Task
from agent.plugin_composition.timers import TIMERS



from plugins.delivery.plugin import DELIVERY


from plugins.tools.plugin import ALL_TOOLS, TOOLS, ToolView

from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts import Message

from .runtime import SchedulerRuntime
from .store import JobStore
from .inspection import SCHEDULER_INSPECTION, SchedulerInspectionProvider
from .tools import CancelInput, ListSchedules, ScheduleInput, ScheduleTool

REPLY_EXECUTE = ServiceKey[Callable[..., Awaitable[Message]]]("reply.execute.v1")


api_version = 3
name = "scheduler"
version = "4.0.0"
desc = "持久调度，按原触发恢复内部消息与最终通知"




inject = (
    TIMERS,
    TOOLS,
    ALL_TOOLS,
    DELIVERY,
    REPLY_EXECUTE,
)
workspace_files = ("schedules.json",)
_DISABLED_TOOLS = frozenset({"message_push", "recall_memory", "memorize", "remember_memory", "forget_memory"})


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=40, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)


async def apply(ctx: Context, config: Config) -> None:
    """注册工具不启动调度；旧 binding 直接重读同一文件，不依赖当前 runtime 指针。"""
    store = JobStore(ctx.workspace_file("schedules.json"))
    _ = await ctx.provide(SCHEDULER_INSPECTION, SchedulerInspectionProvider(store))
    watcher: asyncio.Task[None] | None = None
    tool_view = ToolView(())
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=desc)

    for action, schema, description in (
        ("schedule", ScheduleInput.model_json_schema(), "新增一次或周期定时任务"),
        ("cancel", CancelInput.model_json_schema(), "取消明确匹配的定时任务"),
    ):
        @asynccontextmanager
        async def open_tool(_state: object, action: Literal["schedule", "cancel"] = cast(Literal["schedule", "cancel"], action)) -> AsyncGenerator[ScheduleTool]:
            yield ScheduleTool(store, ctx.require(TASKS).open(ctx), action)

        _ = await catalog.register(
            ctx,
            name="schedule" if action == "schedule" else "cancel_schedule",
            description=description,
            parameters=schema,
            open=open_tool,
            idempotent=True,
        )

    @asynccontextmanager
    async def open_list(_state: object) -> AsyncGenerator[ListSchedules]:
        yield ListSchedules(store)

    _ = await catalog.register(
        ctx,
        name="list_schedules",
        description="查看当前定时任务",
        parameters={"type": "object", "properties": {}},
        open=open_list,
        idempotent=True,
        risk="read-only",
    )

    async def program(task: Task, reader: MessageReader) -> Message:
        bindings = ctx.require(BINDINGS)

        async def authorize(
            binding_id: str, arguments: Mapping[str, object]
        ) -> Mapping[str, object] | str:
            tool = cast(
                Mapping[str, object], bindings.describe(binding_id, TOOLS)["tool"]
            )
            if tool["name"] not in {ref.name for ref in tool_view.refs}:
                return '当前调度组合未授予此工具'
            return {"source": "scheduler", "session_id": reader.session_id}

        return await ctx.require(REPLY_EXECUTE)(
                         ctx, task, reader, 'scheduler',
                         authorize=authorize,
                         tool_view=tool_view,
                         max_output_tokens=config.max_output_tokens,
                         max_steps=config.max_steps,
                         exclude_materials=frozenset({'akasha', 'markdown_memory'}),
                     )

    async def start(_event: object) -> None:
        nonlocal tool_view, watcher
        tool_view = ctx.require(ALL_TOOLS)().without(_DISABLED_TOOLS)
        runtime = SchedulerRuntime(ctx, store, program)
        watcher = await ctx.spawn(runtime.follow(), name="scheduler")

    async def stop(_event: object) -> None:
        if watcher is not None:
            _ = watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
