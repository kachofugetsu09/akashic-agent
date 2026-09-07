from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager

from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CHAT_MODELS, Context, RUNTIME_STARTED, RUNTIME_STOPPING
from plugins.content.plugin import CONTENT
from plugins.context.plugin import CONTEXT
from plugins.context.materials import MATERIALS
from plugins.conversation.plugin import CONVERSATION
from plugins.conversation.program import run_reply
from plugins.models.projection import MODEL_CALLS
from plugins.delivery.plugin import DELIVERY
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.reply.api import REPLY_PROGRAM
from plugins.react.plugin import REACT
from plugins.tools.api import Denied
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from agent.plugin_composition.tasks import TASKS, Task
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION
from agent.plugin_composition.bindings import BINDINGS
from session.log import MessageReader
from session.message import Message

from .prompts import build_spawn_subagent_prompt
from .request import PROFILE_TOOLS, Request, SpawnInput
from .runtime import SUBAGENT_PROGRAM, Subagents
from .tools import Manage, ManageInput, Spawn

api_version = 3
name = "subagent"
version = "4.0.0"
desc = "独立内部消息任务，固定工具权限并向父会话回传"
inject = (BINDINGS, TASKS, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION, TOOLS, CHAT_MODELS, CONTENT, CONTEXT, MATERIALS, REACT, MODEL_CALLS, TURN_PROJECTION, CONVERSATION, DELIVERY, DELIVERY_SENDERS, REPLY_PROGRAM)
workspace_roots = ("subagent-runs",)
workspace_files = ("memory/SELF.md", "memory/spawn_trace.jsonl")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=40, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)


async def apply(ctx: Context, config: Config) -> None:
    """工具提交请求，正式订阅者启动工作；归档只提供原程序和工具入口。"""
    watcher: asyncio.Task[None] | None = None
    jobs = Subagents(ctx)

    @asynccontextmanager
    async def open_spawn(state: Mapping[str, object]) -> AsyncGenerator[Spawn]:
        if set(state) != {"tools", "senders"}:
            raise ValueError("子任务原能力选择损坏")
        for value in state.values():
            if not isinstance(value, Mapping) or any(not isinstance(identity, str) or not identity for identity in cast(Mapping[object, object], value).values()):
                raise ValueError("子任务原能力目录损坏")
        yield Spawn(ctx, cast(Mapping[str, str], state["tools"]), cast(Mapping[str, str], state["senders"]))

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("spawn 不接收额外 binding 配置")
        tools = ctx.require(TOOLS)
        allowed = {name for names in PROFILE_TOOLS.values() for name in names}
        return {"tools": {cast(str, item["name"]): tools.bind(cast(str, item["name"]), ctx.require(BINDINGS))
                for item in tools.descriptions() if item["name"] in allowed},
                "senders": dict(ctx.require(DELIVERY_SENDERS).bind_all(ctx.require(BINDINGS)))}

    @asynccontextmanager
    async def open_manage(_state: Mapping[str, object]) -> AsyncGenerator[Manage]:
        yield Manage(ctx)

    _ = await ctx.require(TOOLS).register(ctx, name="spawn", description=(
        "把有界、多步且可独立完成的任务交给子任务。research 只读；scripting 可在任务目录写文件、执行禁网命令；"
        "general 可调研与执行。默认同步返回结果；长任务可用 run_in_background，完成后回传原会话。"
        "任务需要用户交互或只是简单操作时直接处理。task 必须包含目标、约束、上下文和预期结果。"),
        parameters=SpawnInput.model_json_schema(), open=open_spawn, capture=capture, idempotent=True)
    _ = await ctx.require(TOOLS).register(ctx, name="spawn_manage", description="列出或取消已接纳的子任务",
        parameters=ManageInput.model_json_schema(), open=open_manage, idempotent=True)

    async def program(task: Task, reader: MessageReader, request: Request) -> Message:
        task_dir = ctx.workspace_root("subagent-runs") / request.job_id
        task_dir.mkdir(parents=True, exist_ok=True)
        async def authorize(binding_id: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
            if binding_id not in request.tools.values():
                raise Denied("工具不属于子任务原 profile")
            return {"source": "subagent", "session_id": reader.session_id}
        return await run_reply(ctx, task, reader, "subagent", models=ctx.require(CHAT_MODELS),
            content=ctx.require(CONTENT), context=ctx.require(CONTEXT), tools=ctx.require(TOOLS),
            react=ctx.require(REACT), materials=ctx.require(MATERIALS), turn_projection=ctx.require(TURN_PROJECTION),
            read_call=ctx.require(MODEL_CALLS), authorize=authorize, tool_names=tuple(request.tools),
            fixed_bindings=request.tools, max_output_tokens=config.max_output_tokens, max_steps=config.max_steps,
            exclude_materials=frozenset({"akasha"}),
            prompt_hints=(build_spawn_subagent_prompt(task_dir.parent.parent, task_dir, request.profile),))

    _ = await ctx.provide(SUBAGENT_PROGRAM, program)

    async def start(_event: object) -> None:
        nonlocal watcher
        watcher = asyncio.create_task(jobs.follow(), name="subagent-messages")

    async def stop(_event: object) -> None:
        nonlocal watcher
        if watcher is not None:
            _ = watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
            watcher = None

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
