from __future__ import annotations

from importlib import import_module
import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    Context,
)
from plugins.ledger.contract import BINDINGS
from plugins.ledger.contract import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    MessageReader,
)
from agent.plugin_composition.tasks import TASKS, Task
from plugins.timer.contract import TIMERS
from plugins.ledger.contract import Message
from plugins.sources.contract import SourceGuard
from plugins.reply_program.contract import (
    REPLY_EXECUTE_V4 as REPLY_EXECUTE,
)

from .inputs import (
    ALL_TOOLS, CONTENT, CONVERSATION_COMPLETE, DELIVERY, DELIVERY_SENDERS, INPUT_ORIGIN,
    TOOL_LOADING_PRESENTATION, TOOLS, CallSource,
)
from .contract import SCHEDULER_INSPECTION
from .inspection import SchedulerInspectionProvider
from .runtime import SchedulerRuntime
from .schedule import SCHEDULE_MAX_ACTIVE_JOBS
from .store import Fire, JobStore
from core.common.file_io import run_file_io
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.model import ServiceKey
from plugins.ui.contract import Configuration, ModelSettings, mount_settings
from .tools import CancelInput, ListSchedules, ScheduleInput, ScheduleTool

api_version = 3
name = "scheduler"
version = "5.0.0"
desc = "持久调度：到点在创建任务的会话里执行，最终回复按原触发投递"


inject = (
    PLUGIN_CONFIG, BINDINGS, TASKS, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE,
    CONTENT,
    CONVERSATION_COMPLETE,
    DELIVERY_SENDERS,
    INPUT_ORIGIN,
    TIMERS,
    TOOLS,
    ALL_TOOLS,
    DELIVERY,
    REPLY_EXECUTE,
)
workspace_files = ("schedules.json",)
# 工具与所在会话一致；只去掉写长期记忆的工具，用户不在场时不改记忆。
_DISABLED_TOOLS = frozenset({"memorize", "remember_memory", "forget_memory"})
_MODE_PROMPT = (
    "当前是定时任务的自动执行，没有用户在实时等待。只完成任务要求的事，不要追问用户。"
    "工具产出的卡片和附件会直接显示在会话里；最后一条回复写清结论，它会作为通知推送给用户。"
)


SETTINGS = ServiceKey[Configuration]("scheduler.settings.v1")
SETTING_FIELDS = ("max_steps", "max_output_tokens", "max_active_jobs")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=40, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_active_jobs: int = Field(default=SCHEDULE_MAX_ACTIVE_JOBS, ge=1, le=200)


async def apply(ctx: Context) -> None:
    """注册工具不启动调度；旧 binding 直接重读同一文件，不依赖当前 runtime 指针。"""
    config = Config.model_validate(ctx.config)
    await mount_settings(ctx, SETTINGS, ModelSettings(ctx, Config, SETTING_FIELDS),
                         lambda: import_module(".settings_ui", __package__))
    store = JobStore(ctx.workspace_file("schedules.json"), max_active=config.max_active_jobs)
    _ = await ctx.provide(SCHEDULER_INSPECTION, SchedulerInspectionProvider(store))
    watcher: asyncio.Task[None] | None = None
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=desc)

    async def target(source: CallSource | None) -> tuple[str, str, str] | None:
        """任务属于发起调用的会话；投递渠道沿用该会话最近一条输入的来源，定时任务里沿用本次触发的目标。"""
        if source is None or not source.messages:
            return None
        message = source.messages[-1]
        if message.source.startswith("scheduler:"):
            fire = (await run_file_io(store.read)).fires.get(message.source.removeprefix("scheduler:"))
            return None if fire is None else (fire.job.session_id, fire.job.channel, fire.job.chat_id)
        reader = ctx.require(MESSAGE_CATALOG).reader(message.session_id)
        route = ctx.require(INPUT_ORIGIN)(reader, message.source, through_seq=message.seq)
        return None if route is None else (message.session_id, *route)

    for action, schema, description in (
        ("schedule", ScheduleInput.model_json_schema(), "在当前会话新增一次或周期定时任务"),
        ("cancel", CancelInput.model_json_schema(), "取消明确匹配的定时任务"),
    ):
        @asynccontextmanager
        async def open_tool(_state: object, action: Literal["schedule", "cancel"] = cast(Literal["schedule", "cancel"], action)) -> AsyncGenerator[ScheduleTool]:
            yield ScheduleTool(store, ctx.require(TASKS).open(ctx), action, target)

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
        parallel=True,
    )

    # 程序在目标会话的执行位里运行，入口仍以 scheduler 身份签发模型与消息 writer。
    @ctx.entrypoint
    async def program(task: Task, reader: MessageReader, check_admission: SourceGuard, fire: Fire) -> Message:
        """每次执行取得当前工具；展示与授权共用本次固定的目录。"""
        bindings = ctx.require(BINDINGS)
        # 1. STARTED 只表示本插件已启动，其他 provider 之后仍可注册或卸载。
        view = ctx.require(ALL_TOOLS)().without(_DISABLED_TOOLS)

        async def authorize(
            binding_id: str, arguments: Mapping[str, object]
        ) -> Mapping[str, object] | str:
            tool = cast(
                Mapping[str, object], bindings.describe(binding_id, TOOLS)["tool"]
            )
            if tool["name"] not in {ref.name for ref in view.refs}:
                return '当前调度组合未授予此工具'
            return {"source": fire.source, "session_id": reader.session_id}

        # 2. 展示追加的入口与业务工具共用这次执行的授权范围；模型跟随会话当前选择。
        with ctx.borrow(TOOL_LOADING_PRESENTATION) as present:
            presentation = None
            if present is not None:
                view, presentation = present(view)
            return await ctx.require(REPLY_EXECUTE)(
                ctx, task, reader, fire.source,
                check_admission=check_admission,
                authorize=authorize,
                tool_view=view,
                max_output_tokens=config.max_output_tokens,
                max_steps=config.max_steps,
                presentation=presentation,
                exclude_material_kinds=frozenset({'recall', 'profile'}),
                prompt_hints=(_MODE_PROMPT,),
                history="session" if fire.job.context == "session" else "own",
            )

    async def start(_event: object) -> None:
        nonlocal watcher
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
