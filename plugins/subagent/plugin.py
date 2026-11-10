from __future__ import annotations

from functools import partial

import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    Context,
    stop_spawned,
)
from plugins.ledger.contract import BINDINGS
from plugins.ledger.contract import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
    MessageReader,
)
from agent.plugin_composition.tasks import TASKS, Task
from plugins.ledger.contract import Message
from plugins.sources.contract import (
    SOURCE_CHECK_V2 as SOURCE_CHECK,
)
from plugins.reply_program.contract import (
    REPLY_EXECUTE_V4 as REPLY_EXECUTE,
)
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.model import ServiceKey
from plugins.models.contract import (
    CHAT_MODELS, MODEL_CATALOG, MODEL_SELECTION, ChatModelSelection, ModelChoice, ModelUse, model_options,
)
from plugins.ui.contract import Configuration, ModelSettings, mount_settings

from .inputs import (
    ALL_TOOLS,
    CHECK_ORIGIN,
    CONTENT,
    CONVERSATION_COMPLETE,
    DELIVERY,
    DELIVERY_SENDERS,
    REPLY_PROGRAM,
    TOOL_BIND_SAVED,
    TOOLS,
)
from .prompts import build_spawn_subagent_prompt
from .request import PROFILE_TOOLS, Request, SpawnInput
from .contract import SUBAGENT_PROGRAM, AGENT_RUNNERS
from .runners import Runners
from plugins.jobs.contract import JOB_NOTICES
from .runtime import DEFAULT_MAX_ACTIVE, DEFAULT_RESULT_CHARS, Subagents
from .tools import Manage, ManageInput, Spawn, select_model

api_version = 3
name = "subagent"
version = "4.0.0"
desc = "独立内部消息任务，固定工具权限并向父会话回传"


inject = (
    PLUGIN_CONFIG, CONTENT,
    CHAT_MODELS, MODEL_CATALOG, MODEL_SELECTION,
    CHECK_ORIGIN,
    TOOL_BIND_SAVED,
    BINDINGS,
    TASKS,
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
    TOOLS,
    ALL_TOOLS,
    CONVERSATION_COMPLETE,
    DELIVERY,
    DELIVERY_SENDERS,
    REPLY_PROGRAM,
    REPLY_EXECUTE, SOURCE_CHECK, JOB_NOTICES,
)
workspace_roots = ("subagent-runs",)
workspace_files = ("memory/SELF.md", "memory/spawn_trace.jsonl")


SETTINGS = ServiceKey[Configuration]("subagent.settings.v1")
# 子任务默认沿用派出它的那轮对话的模型；也可固定跟随默认或指定一个模型。
SUBAGENT_MODEL = ModelUse("subagent", "子任务", follows=("session", "default"))
SETTING_FIELDS = ("model", "max_steps", "max_output_tokens", "max_active", "result_chars")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=40, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_active: int = Field(default=DEFAULT_MAX_ACTIVE, ge=1, le=20)
    result_chars: int = Field(default=DEFAULT_RESULT_CHARS, ge=1000, le=1_000_000)
    model: ModelChoice = ModelChoice(follow="session")

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: ModelChoice) -> ModelChoice:
        return SUBAGENT_MODEL.check(value)


async def apply(ctx: Context) -> None:
    """工具提交请求，正式订阅者启动工作；归档只提供原程序和工具入口。"""
    config = Config.model_validate(ctx.config)
    await mount_settings(ctx, SETTINGS, ModelSettings(ctx, Config, SETTING_FIELDS,
                                                       status=model_options(ctx, {"model": SUBAGENT_MODEL})))
    watcher: asyncio.Task[None] | None = None
    jobs = Subagents(ctx, max_active=config.max_active, result_chars=config.result_chars)
    runners = Runners(ctx, max_active=config.max_active, result_chars=config.result_chars)
    await runners.recover()
    await ctx.provide(AGENT_RUNNERS, runners)
    from plugins.ui.contract import UI_SLOTS
    from .runner_ui import register_ui
    await ctx.inject((UI_SLOTS,), lambda child: register_ui(child, runners), name="runner-ui")
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=desc)

    # 子任务模型：调用参数显式指定 model/connection 时直接核对目录并固定；否则按设置（跟随对话取父调用选择）解析。
    async def child_model(messages: tuple[Message, ...], args: SpawnInput) -> ChatModelSelection | str:
        if any(value is not None and value.strip() != "" for value in (args.model, args.connection, args.reasoning_effort)):
            snapshot = ctx.require(MODEL_CATALOG).snapshot()
            return select_model(
                snapshot,
                model=args.model,
                connection=args.connection,
                reasoning_effort=args.reasoning_effort,
            )
        selection = ctx.require(MODEL_SELECTION)
        session = selection.read(messages)
        if session is None:
            metadata = ctx.require(MESSAGE_CATALOG).reader(messages[-1].session_id).metadata()
            session = selection.read_saved(metadata if metadata is not None else {})
        return await ctx.require(CHAT_MODELS).resolve(SUBAGENT_MODEL, config.model, session)

    @asynccontextmanager
    async def open_spawn(state: Mapping[str, object]) -> AsyncGenerator[Spawn]:
        if set(state) != {"tools", "senders"}:
            raise ValueError("子任务原能力选择损坏")
        for value in state.values():
            if not isinstance(value, Mapping) or any(not isinstance(identity, str) or not identity for identity in cast(Mapping[object, object], value).values()):
                raise ValueError("子任务原能力目录损坏")
        yield Spawn(ctx, cast(Mapping[str, str], state["tools"]), cast(Mapping[str, str], state["senders"]), jobs,
                    model=child_model)

    async def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("spawn 不接收额外 binding 配置")
        allowed = {name for names in PROFILE_TOOLS.values() for name in names}
        refs = ctx.require(ALL_TOOLS)()
        return {
            "tools": {
                ref.name: await catalog.bind_scoped(ref, ctx.require(BINDINGS))
                for ref in refs.refs
                if ref.name in allowed
            },
            "senders": dict(
                ctx.require(DELIVERY_SENDERS).bind_all(ctx.require(BINDINGS))
            ),
        }

    @asynccontextmanager
    async def open_manage(_state: Mapping[str, object]) -> AsyncGenerator[Manage]:
        yield Manage(jobs, catalog=ctx.require(MODEL_CATALOG).snapshot)

    _ = await catalog.register(
        ctx,
        name="spawn",
        description=(
            "把有界、多步且可独立完成的任务交给子任务。默认 general（可调研、读写文件、执行命令）；"
            "research 只读；scripting 读写与命令执行。子任务默认继承父会话工作目录（可通过 cwd 指定初始目录）。"
            "默认同步返回结果；长任务可用 run_in_background，完成后回传原会话。"
            "后台结果回报最多串联三层子任务，达到上限后应汇报已有结果并等待用户决定。"
            "可通过 model 与 connection 为本次子任务单独指定模型及所属连接渠道（不确定有哪些连接和模型时，先用 spawn_manage(action=\"models\") 查询）；留空时按子任务设置解析。"
            "任务需要用户交互或只是简单操作时直接处理。task 必须包含目标、约束、上下文和预期结果。"
        ),
        parameters=SpawnInput.model_json_schema(),
        open=open_spawn,
        capture=capture,
        idempotent=True,
    )
    _ = await catalog.register(
        ctx,
        name="spawn_manage",
        description="列出或取消已接纳的子任务，或查询子任务当前可用的连接渠道与模型列表（action=\"models\"）",
        parameters=ManageInput.model_json_schema(),
        open=open_manage,
        idempotent=True,
    )

    async def program(task: Task, reader: MessageReader, request: Request) -> Message:
        task_dir = ctx.workspace_root("subagent-runs") / request.job_id
        task_dir.mkdir(parents=True, exist_ok=True)
        async def authorize(binding_id: str, arguments: Mapping[str, object]) -> Mapping[str, object] | str:
            if binding_id not in request.tools.values():
                return '工具不属于子任务原 profile'
            return {"source": "subagent", "session_id": reader.session_id}

        return await ctx.require(REPLY_EXECUTE)(
                         ctx, task, reader, 'subagent',
                         check_admission=partial(ctx.require(SOURCE_CHECK), task, reader, "subagent", task.boundary_hint),
                         authorize=authorize,
                         tool_view=None,
                         fixed_bindings=request.tools,
                         max_output_tokens=config.max_output_tokens,
                         max_steps=config.max_steps,
                         exclude_material_kinds=frozenset({'recall'}),
                         prompt_hints=(build_spawn_subagent_prompt(
                             task_dir.parent.parent, task_dir, request.profile, cwd=request.cwd,
                         ),),
                     )

    _ = await ctx.provide(SUBAGENT_PROGRAM, program)

    # Retry the durable scan a bounded number of times; exhausted failures reach readiness.
    async def follow() -> None:
        for attempt in range(4):
            try:
                await jobs.follow()
                return
            except Exception as error:
                _ = ctx.report_incident("subagent-watcher", f"attempt={attempt + 1}: {error!r}")
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def start(_event: object) -> None:
        nonlocal watcher
        watcher = await ctx.spawn(follow(), name="subagent-messages")

    async def stop(_event: object) -> None:
        nonlocal watcher
        await stop_spawned(watcher)
        watcher = None

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
