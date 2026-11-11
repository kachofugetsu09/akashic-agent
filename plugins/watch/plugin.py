from __future__ import annotations

import asyncio

from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING, Context, stop_spawned
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from plugins.plugin_settings.contract import SETTINGS_FORMS, declare_settings

from .contract import WATCH_EVENTS
from .runtime import Runtime
from .settings import FORM, Config

api_version = 3
name = "watch"
version = "1.0.0"
desc = "值班：把各个来源的事件收成同一种形状，留给之后的判断"
inject = (PLUGIN_CONFIG,)
workspace_roots = ("watch",)


async def apply(ctx: Context) -> None:
    """事件入口始终可用；只有配置了主对话，收事件的循环才启动。"""
    config = Config.model_validate(ctx.config)
    runtime = Runtime(ctx, config)
    _ = await ctx.provide(WATCH_EVENTS, runtime)
    _ = await ctx.inject((SETTINGS_FORMS, PLUGIN_CONFIG),
                         lambda child: declare_settings(child, Config, FORM), name="settings")
    watcher: asyncio.Task[None] | None = None

    async def start(_event: object) -> None:
        nonlocal watcher
        if config.target:
            watcher = await ctx.spawn(runtime.follow(), name="watch")

    async def stop(_event: object) -> None:
        await stop_spawned(watcher)

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
