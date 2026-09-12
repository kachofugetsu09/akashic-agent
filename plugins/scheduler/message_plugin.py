from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CHAT_MODELS, Context, RUNTIME_STARTED, RUNTIME_STOPPING, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.tasks import TASKS, Task
from agent.plugin_composition.timers import TIMERS
from plugins.content.plugin import CONTENT
from plugins.context.plugin import CONTEXT
from plugins.context.materials import MATERIALS
from plugins.conversation.program import run_reply
from plugins.delivery.plugin import DELIVERY
from plugins.models.projection import MODEL_CALLS
from plugins.react.plugin import REACT
from plugins.tools.api import Denied
from plugins.tools.plugin import ALL_TOOLS, TOOLS, ToolView
from plugins.turn_projection.plugin import TURN_PROJECTION
from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts import CallRef, Message

from .runtime import SchedulerRuntime
from .store import JobStore
from .inspection import SCHEDULER_INSPECTION, SchedulerInspectionProvider
from .tools import CancelInput, ListSchedules, ScheduleInput, ScheduleTool

api_version = 3
name = "scheduler"
version = "4.0.0"
desc = "持久调度，按原触发恢复内部消息与最终通知"


class ToolCleanup(Protocol):
    """Scheduler 只接收工具 owner 的窄收尾边界。"""

    def __call__(
        self,
        ctx: Context,
        reader: MessageReader,
        source: str,
        from_seq: int,
        *,
        task: Task,
        drain: Callable[[tuple[CallRef, ...]], Awaitable[None]],
    ) -> AbstractAsyncContextManager[None]: ...


TOOL_CLEANUP = ServiceKey[ToolCleanup]("tools.cleanup.v1")


inject = (
    TIMERS,
    TOOLS,
    ALL_TOOLS,
    CHAT_MODELS,
    CONTENT,
    CONTEXT,
    MATERIALS,
    REACT,
    MODEL_CALLS,
    TURN_PROJECTION,
    DELIVERY,
    TOOL_CLEANUP,
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
        ) -> Mapping[str, object]:
            tool = cast(
                Mapping[str, object], bindings.describe(binding_id, TOOLS)["tool"]
            )
            if tool["name"] not in {ref.name for ref in tool_view.refs}:
                raise Denied("当前调度组合未授予此工具")
            return {"source": "scheduler", "session_id": reader.session_id}

        return await run_reply(
            ctx,
            task,
            reader,
            "scheduler",
            models=ctx.require(CHAT_MODELS),
            content=ctx.require(CONTENT),
            context=ctx.require(CONTEXT),
            tools=ctx.require(TOOLS),
            cleanup=ctx.require(TOOL_CLEANUP),
            react=ctx.require(REACT),
            materials=ctx.require(MATERIALS),
            turn_projection=ctx.require(TURN_PROJECTION),
            read_call=ctx.require(MODEL_CALLS),
            authorize=authorize,
            tool_view=tool_view,
            max_output_tokens=config.max_output_tokens,
            max_steps=config.max_steps,
            exclude_materials=frozenset({"akasha", "markdown_memory"}),
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
