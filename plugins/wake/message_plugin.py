from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from functools import partial

from agent.plugin_composition import CHAT_MODELS, Context, RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS
from agent.plugin_composition.timers import TIMERS
from plugins.akasha.interest import SEMANTIC_INTEREST
from plugins.delivery.history import DELIVERY_READ
from plugins.drift.plugin import DRIFT_CHANGED
from plugins.content.api import ContentSchema
from plugins.content.plugin import CONTENT
from plugins.context.materials import MATERIALS
from plugins.context.plugin import CONTEXT
from plugins.delivery.plugin import DELIVERY
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.models.projection import MODEL_CALLS
from plugins.react.plugin import REACT
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION

from .api import Config, EVENTMAIL_WAKE, EVENTMAIL_DELIVERY, DRIFT_WAKE, DRIFT_DELIVERY, EVENTMAIL_CHANGED
from .program import run
from .runtime import Runtime
from .request import WAKE_PROGRAM, check_phase, check_request
from .tools import DecisionTool, SCHEMAS

api_version = 3
name = "wake"
version = "4.0.0"
desc = "内部消息完成初筛、调查与告警，真实送达后确认原职责"
inject = (BINDINGS, TASKS, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
          TOOLS, CHAT_MODELS, CONTENT, CONTEXT, MATERIALS, REACT, MODEL_CALLS, TURN_PROJECTION,
          DELIVERY, DELIVERY_SENDERS, EVENTMAIL_WAKE, EVENTMAIL_DELIVERY, DRIFT_WAKE, DRIFT_DELIVERY, TIMERS, SEMANTIC_INTEREST, DELIVERY_READ)


async def apply(ctx: Context, config: Config) -> None:
    """归档注册原程序和私有决定工具；消息与领域状态仅在正式来源执行时打开。"""
    _ = await ctx.require(CONTENT).register(ctx, ContentSchema(name="wake", content={
        "wake.request": check_request, "wake.phase": check_phase,
    }))
    descriptions = {"screen_content": "初筛本轮 Content 候选并写兴趣理由与调查问题",
                    "share_content": "提交本轮分享正文与采用的 Content 候选 ID；Drift 使用空 items",
                    "skip_content": "明确跳过本轮职责并说明原因", "share_alert": "提交原告警的用户通知正文"}
    for name, schema in SCHEMAS.items():
        @asynccontextmanager
        async def open_tool(_state: Mapping[str, object], name: str = name) -> AsyncGenerator[DecisionTool]:
            yield DecisionTool(name)
        _ = await ctx.require(TOOLS).register(ctx, name=name, description=descriptions[name],
            parameters=schema.model_json_schema(), open=open_tool, idempotent=True, public=False)
    _ = await ctx.provide(WAKE_PROGRAM, partial(run, ctx))

    runtime: Runtime | None = None
    watcher: asyncio.Task[None] | None = None

    async def start(_event: object) -> None:
        nonlocal runtime, watcher
        runtime = Runtime(ctx, config)
        watcher = await ctx.spawn(runtime.follow(), name="wake")

    async def stop(_event: object) -> None:
        if watcher is not None:
            _ = watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    def changed(_event: object) -> None:
        if runtime is not None:
            runtime.changed.set()

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
    _ = await ctx.on(EVENTMAIL_CHANGED, changed)
    _ = await ctx.on(DRIFT_CHANGED, changed)
