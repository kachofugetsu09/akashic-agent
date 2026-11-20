from __future__ import annotations
from importlib import import_module


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
from plugins.ledger.contract import BINDINGS, OwnerTransaction
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
from plugins.models.contract import (
    CHAT_MODELS, MODEL_CATALOG, MODEL_SELECTION, ChatModelSelection, ModelChoice, ModelUse, model_options,
)
from plugins.ui.contract import UI, ModelSettings

from plugins.tools.contract import TOOL_LOADING_PRESENTATION, ToolPresentation, ToolView
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
from .request import DELEGATION_TOOLS, EXCLUDED_ALWAYS_TOOLS, Request, SpawnInput
from .contract import SUBAGENT_PROGRAM, AGENT_RUNNERS
from .runners import Runners
from plugins.jobs.contract import JOB_NOTICES
from .runtime import DEFAULT_MAX_ACTIVE, DEFAULT_MAX_DEPTH, DEFAULT_RESULT_CHARS, Subagents
from .tools import Manage, ManageInput, ReadInput, SendInput, Spawn, SubagentRead, SubagentSend, select_model

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


from .dashboard import SETTINGS, SUBAGENT_UI_SERVICE, UI_CONTRACT_DIGESTS
# 子任务默认沿用派出它的那轮对话的模型；也可固定跟随默认或指定一个模型。
SUBAGENT_MODEL = ModelUse("subagent", "子任务", follows=("session", "default"))
SETTING_FIELDS = ("model", "max_depth", "max_steps", "max_output_tokens", "max_active", "result_chars")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # 0 = 不限制总步数，和 ReAct 的 max_steps>0 才生效保持一致；
    # 入口不再硬编码上限，实际值只由设置决定。
    max_depth: int = Field(default=DEFAULT_MAX_DEPTH, ge=0, le=5)
    max_steps: int = Field(default=0, ge=0)
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
    await ctx.provide(SETTINGS, ModelSettings(ctx, Config, SETTING_FIELDS,
                                              status=model_options(ctx, {"model": SUBAGENT_MODEL})))
    watcher: asyncio.Task[None] | None = None
    runners: Runners | None = None
    jobs = Subagents(
        ctx,
        max_active=config.max_active,
        result_chars=config.result_chars,
        max_depth=config.max_depth,
        external_active_for_root=lambda root_id: 0 if runners is None else runners.active_for_root(root_id),
    )
    runners = Runners(
        ctx,
        max_active=config.max_active,
        result_chars=config.result_chars,
        resolve_root=lambda sid: jobs.resolve_lineage(sid)[0],
        tree_active_count=jobs.active_count_for_root,
        check_depth=jobs.check_depth,
        ensure_index=jobs.ensure_index,
        admit_external=jobs.admit_external,
    )
    await runners.recover()
    await ctx.provide(AGENT_RUNNERS, runners)
    await ctx.provide(SUBAGENT_UI_SERVICE, (jobs, runners))

    async def register_web_app(child: Context) -> None:
        _ = await child.require(UI).register(
            child,
            web="web_module.js",
            dashboard=lambda: import_module(".dashboard", __package__),
            requires=("shell.settings-plugins.v1", "conversation.tools.v2"),
            contract_digests=UI_CONTRACT_DIGESTS,
            configuration_key=SETTINGS,
        )

    _ = await ctx.inject((UI, SETTINGS, SUBAGENT_UI_SERVICE), register_web_app, name="subagent-web-ui")
    from plugins.ui.contract import UI_SLOTS
    from .runner_ui import register_ui
    await ctx.inject((UI_SLOTS,), lambda child: register_ui(child, runners, jobs), name="runner-ui")
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=desc, icon="git-fork")

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
        excluded = DELEGATION_TOOLS | EXCLUDED_ALWAYS_TOOLS
        refs = ctx.require(ALL_TOOLS)()
        return {
            "tools": {
                ref.name: await catalog.bind_scoped(ref, ctx.require(BINDINGS))
                for ref in refs.refs
                if ref.name not in excluded
            },
            "senders": dict(
                ctx.require(DELIVERY_SENDERS).bind_all(ctx.require(BINDINGS))
            ),
        }

    @asynccontextmanager
    async def open_manage(_state: Mapping[str, object]) -> AsyncGenerator[Manage]:
        yield Manage(
            jobs,
            catalog=ctx.require(MODEL_CATALOG).snapshot,
            external_jobs=None if runners is None else runners.active_jobs,
        )

    @asynccontextmanager
    async def open_send(_state: Mapping[str, object]) -> AsyncGenerator[SubagentSend]:
        yield SubagentSend(jobs)

    @asynccontextmanager
    async def open_read(_state: Mapping[str, object]) -> AsyncGenerator[SubagentRead]:
        yield SubagentRead(jobs)

    _ = await catalog.register(
        ctx,
        name="spawn",
        description=(
            "把有界、多步且可独立完成的任务交给子任务。默认继承父会话已授权的工具与工作目录（可通过 cwd 指定初始目录），"
            "需要只读调研时设 read_only=true。默认同步返回结果；长任务可用 run_in_background（支持并发启动多条并按 completion_wake 合并回传）。"
            "所有子任务受统一递归深度上限约束（默认 1 层）。"
            "可通过 model 与 connection 为本次子任务单独指定模型及所属连接渠道（不确定有哪些连接和模型时，先用 spawn_manage(action=\"models\") 查询）；留空时按子任务设置解析。"
            "任务需要用户交互或只是简单操作时直接处理。task 必须包含目标、约束、上下文和预期结果。"
        ),
        parameters=SpawnInput.model_json_schema(),
        open=open_spawn,
        capture=capture,
        idempotent=True,
        parallel=True,
    )
    _ = await catalog.register(
        ctx,
        name="spawn_manage",
        description="列出或取消已接纳的子任务，或查询子任务当前可用的连接渠道与模型列表（action=\"models\"）",
        parameters=ManageInput.model_json_schema(),
        open=open_manage,
        idempotent=True,
        parallel=True,
    )
    _ = await catalog.register(
        ctx,
        name="subagent_send",
        description="向当前会话树下已创建的子任务继续发送指令：运行中实时 Steer 插话，已结束则沿用原上下文继续执行",
        parameters=SendInput.model_json_schema(),
        open=open_send,
        idempotent=False,
        icon="send",
    )
    _ = await catalog.register(
        ctx,
        name="subagent_read",
        description="只读查看当前会话树下指定子任务的内部会话进展与输出消息",
        parameters=ReadInput.model_json_schema(),
        open=open_read,
        idempotent=True,
        parallel=True,
        icon="eye",
    )

    # 子任务沿用 prepare 时冻结的 binding；有分组加载展示时只追加 load_tools 入口，避免全量 schema 超出模型上限。
    async def frozen_menu(request: Request) -> tuple[Mapping[str, str], ToolPresentation | None]:
        with ctx.borrow(TOOL_LOADING_PRESENTATION) as present:
            if present is None:
                return request.tools, None
            refs = {ref.name: ref for ref in ctx.require(ALL_TOOLS)().refs}
            if not set(request.tools).issubset(refs):
                return request.tools, None
            view, presentation = present(ToolView(tuple(refs[name] for name in request.tools)))
            bindings = ctx.require(BINDINGS)
            entries = {
                ref.name: await catalog.bind_scoped(ref, bindings, configuration=presentation.configuration(ref.name))
                for ref in view.refs if ref.name not in request.tools
            }
            return {**request.tools, **entries}, presentation

    async def program(task: Task, reader: MessageReader, request: Request) -> Message:
        task_dir = ctx.workspace_root("subagent-runs") / request.job_id
        task_dir.mkdir(parents=True, exist_ok=True)
        fixed_bindings, presentation = await frozen_menu(request)
        allowed = frozenset(fixed_bindings.values())

        async def authorize(binding_id: str, arguments: Mapping[str, object]) -> Mapping[str, object] | str:
            if binding_id not in allowed:
                return '工具不属于子任务已固定权限边界'
            return {"source": "subagent", "session_id": reader.session_id}

        def check_admission(*, transaction: OwnerTransaction | None = None) -> None:
            ctx.require(SOURCE_CHECK)(task, reader, "subagent", task.boundary_hint, transaction=transaction)

        before_step, retry_wake = jobs.steer_binding(reader.session_id, task=task)
        return await ctx.require(REPLY_EXECUTE)(
                         ctx, task, reader, 'subagent',
                         check_admission=check_admission,
                         authorize=authorize,
                         tool_view=None,
                         fixed_bindings=fixed_bindings,
                         presentation=presentation,
                         before_step=before_step,
                         retry_wake=retry_wake,
                         max_output_tokens=config.max_output_tokens,
                         max_steps=config.max_steps,
                         exclude_material_kinds=frozenset({'recall'}),
                         prompt_hints=(build_spawn_subagent_prompt(
                             task_dir.parent.parent, task_dir, request.profile,
                             read_only=request.read_only, depth=request.depth,
                             max_depth=config.max_depth, cwd=request.cwd, fork=request.fork,
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
