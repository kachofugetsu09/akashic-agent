from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from functools import partial
from importlib import import_module

from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    Context,
    ServiceKey,
)
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
)
from agent.plugin_composition.tasks import TASKS
from agent.plugin_composition.timers import TIMERS
from agent.plugin_composition.ui import UI
from agent.plugin_contracts.models import MODEL_CONTENT, MODEL_SELECTION

from ._boundary import (
    ALL_TOOLS,
    CONTENT,
    DELIVERY,
    DELIVERY_READ,
    DELIVERY_SENDERS,
    DRIFT_CHANGED,
    SEMANTIC_INTEREST,
    TOOLS,
    ToolRef,
)
from .api import (
    DRIFT_DELIVERY,
    DRIFT_WAKE,
    EVENTMAIL_CHANGED,
    EVENTMAIL_DELIVERY,
    EVENTMAIL_WAKE,
    Config,
)
from .program import REPLY_EXECUTE, run
from .request import WAKE_PROGRAM, WAKE_TOOLS_VIEW, check_phase, check_request
from .runtime import DashboardView, Runtime
from .tools import SCHEMAS, DecisionTool

api_version = 3
name = "wake"
version = "4.0.0"
desc = "内部消息完成初筛、调查与告警，真实送达后确认原职责"
inject = (
    MODEL_SELECTION, MODEL_CONTENT,
    REPLY_EXECUTE,
    BINDINGS,
    TASKS,
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
    TOOLS,
    CONTENT,
    DELIVERY,
    DELIVERY_SENDERS,
    EVENTMAIL_WAKE,
    EVENTMAIL_DELIVERY,
    DRIFT_WAKE,
    DRIFT_DELIVERY,
    TIMERS,
    SEMANTIC_INTEREST,
    DELIVERY_READ,
    ALL_TOOLS,
)

WAKE_DASHBOARD = ServiceKey[Callable[[], DashboardView | None]]("wake.dashboard.v1")


async def _stop_watcher(watcher: asyncio.Task[None]) -> None:
    """Drain a watcher without replaying a completed business failure as stop failure."""

    cancel_requested = watcher.cancel() if not watcher.done() else False
    try:
        await watcher
    except asyncio.CancelledError:
        if asyncio.current_task().cancelling():
            raise
    except Exception:
        if cancel_requested:
            raise


async def apply(ctx: Context) -> None:
    """归档注册原程序和私有决定工具；消息与领域状态仅在正式来源执行时打开。"""

    config = Config.model_validate(ctx.config)
    _ = await ctx.require(CONTENT).register(
        ctx,
        {
            "name": "wake",
            "content": {
                "wake.request": check_request,
                "wake.phase": check_phase,
            },
        },
    )
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=desc)
    refs: list[ToolRef] = []
    descriptions = {
        "screen_content": "初筛本轮 Content 候选并写兴趣理由与调查问题",
        "share_content": "提交本轮分享正文与采用的 Content 候选 ID；Drift 使用空 items",
        "skip_content": "明确跳过本轮职责并说明原因",
        "share_alert": "提交原告警的用户通知正文",
    }
    for name, schema in SCHEMAS.items():
        @asynccontextmanager
        async def open_tool(_state: Mapping[str, object], name: str = name) -> AsyncGenerator[DecisionTool]:
            yield DecisionTool(name)

        refs.append(
            await catalog.register(
                ctx,
                name=name,
                description=descriptions[name],
                parameters=schema.model_json_schema(),
                open=open_tool,
                idempotent=True,
                public=False,
            )
        )
    _ = await ctx.provide(WAKE_TOOLS_VIEW, catalog.view(*refs))
    _ = await ctx.provide(WAKE_PROGRAM, partial(run, ctx))

    runtime: Runtime | None = None
    dashboard: DashboardView | None = None
    watcher: asyncio.Task[None] | None = None

    def current_dashboard() -> DashboardView | None:
        return dashboard

    _ = await ctx.provide(WAKE_DASHBOARD, current_dashboard)

    async def start(_event: object) -> None:
        nonlocal runtime, dashboard, watcher
        runtime = Runtime(ctx, config)
        dashboard = runtime.dashboard_view()
        watcher = await ctx.spawn(runtime.follow(), name="wake")

    async def stop(_event: object) -> None:
        nonlocal dashboard, runtime
        if watcher is not None:
            await _stop_watcher(watcher)
        dashboard = None
        runtime = None

    def changed(_event: object) -> None:
        if runtime is not None:
            runtime.changed.set()

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
    _ = await ctx.on(EVENTMAIL_CHANGED, changed)
    _ = await ctx.on(DRIFT_CHANGED, changed)
    _ = await ctx.inject((UI, WAKE_DASHBOARD), _register_ui, name="ui")


async def _register_ui(ctx: Context) -> None:
    """界面随 UI provider 换代，不牵动计算与持久状态。"""
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        dashboard=lambda: import_module(".dashboard", __package__),
        requires=("workbench.panels.v2",),
        provides=(),
        contract_digests={
            "workbench.panels.v2": "fb6417c9bf532c1fdb344767d06065d5d3293da85deb64eff1e8088889a33bcb",
        },
    )
