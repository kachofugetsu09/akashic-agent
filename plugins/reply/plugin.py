from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CHAT_MODELS, RUNTIME_STARTED, RUNTIME_STOPPING, Context
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.tasks import Task
from plugins.content.plugin import CONTENT
from plugins.context.plugin import CONTEXT
from plugins.context.materials import MATERIALS
from plugins.conversation.plugin import CONVERSATION
from plugins.conversation.commands import CONVERSATION_COMMANDS
from plugins.conversation.program import run_reply
from plugins.models.projection import MODEL_CALLS
from plugins.react.plugin import REACT
from plugins.tools.api import Denied
from plugins.tools.menu import check_menu
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.log import MessageReader

from .follow import follow

api_version = 3
name = "reply"
version = "1.0.0"
desc = "跟随日志并组合默认回复；接纳、材料、模型与工具各有独立 owner"
inject = (CONVERSATION, CONVERSATION_COMMANDS, CHAT_MODELS, CONTENT, CONTEXT, MATERIALS, TOOLS, REACT, MODEL_CALLS, TURN_PROJECTION)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=40, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    tools: tuple[str, ...] | None = None


async def apply(ctx: Context, config: Config) -> None:
    """自动回复是普通可移除插件；正式启动后才读日志和接纳任务。"""
    watcher: asyncio.Task[None] | None = None
    names: tuple[str, ...] = ()
    allowed: frozenset[str] = frozenset()

    async def program(task: Task, reader: MessageReader, source: str) -> object:
        command = await ctx.require(CONVERSATION_COMMANDS)(task, reader, source)
        if command is not None:
            return command
        tools = ctx.require(TOOLS)
        bindings = ctx.require(BINDINGS)

        async def authorize(binding_id: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
            metadata = bindings.describe(binding_id, TOOLS)
            tool = cast(Mapping[str, object], metadata["tool"])
            if tool["name"] not in allowed:
                raise Denied("当前回复组合未授予此工具")
            return {"source": source, "session_id": reader.session_id}

        return await run_reply(
            ctx, task, reader, source, models=ctx.require(CHAT_MODELS),
            content=ctx.require(CONTENT), context=ctx.require(CONTEXT), tools=tools,
            react=ctx.require(REACT), materials=ctx.require(MATERIALS),
            turn_projection=ctx.require(TURN_PROJECTION),
            read_call=ctx.require(MODEL_CALLS), authorize=authorize,
            tool_names=names, max_output_tokens=config.max_output_tokens, max_steps=config.max_steps,
        )

    async def start(_event: object) -> None:
        nonlocal watcher, names, allowed
        available = {cast(str, item["name"]) for item in ctx.require(TOOLS).descriptions()}
        names = tuple(sorted(available)) if config.tools is None else config.tools
        check_menu(ctx.require(TOOLS), names)
        allowed = frozenset(names)
        catalog = ctx.require(MESSAGE_CATALOG)
        watcher = await ctx.spawn(follow(ctx, catalog, ctx.require(CONVERSATION), program), name="reply")

    async def stop(_event: object) -> None:
        if watcher is not None:
            _ = watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
